import os
import json
import glob
import asyncio
import logging
import uuid
import sys
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
import openai
from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Import for postgres
import asyncpg
from pgvector.asyncpg import register_vector

# Load environment variables from .env file
load_dotenv('../pipeline.env')

# ==========================================
# Configuration & Setup
# ==========================================
DATA_DIR = os.getenv("DATA_DIR", "../output_final_rag_data")
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chromadb").lower()
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "../chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "writing_style_logs")

CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST")
CHROMA_DB_PORT = os.getenv("CHROMA_DB_PORT")

# Postgres settings
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_DB_NAME = os.getenv("PG_DB_NAME")

# AI & Embedding Providers
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").lower()
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

LOG_DIR = os.getenv("LOG_DIR", "../log")
PROC_DIR = os.getenv("PROC_DIR", "../processing")
PROCESS_NAME = "rag_ingestion_pipeline"

# Batch processing size
BATCH_SIZE = 100

# Ensure directories exist
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(PROC_DIR).mkdir(parents=True, exist_ok=True)
process_log_dir = Path(LOG_DIR) / PROCESS_NAME
process_log_dir.mkdir(parents=True, exist_ok=True)
process_proc_dir = Path(PROC_DIR) / PROCESS_NAME
process_proc_dir.mkdir(parents=True, exist_ok=True)

# Initialize OpenAI Client (used for OpenAI or local LM Studio)
if AI_PROVIDER == "lm_studio":
    client = AsyncOpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
elif AI_PROVIDER == "openai":
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in the environment or pipeline.env file.")
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
else:
    if AI_PROVIDER == "gemini":
         client = AsyncOpenAI(api_key=GEMINI_API_KEY)
    elif AI_PROVIDER == "anthropic":
         client = AsyncOpenAI(api_key=ANTHROPIC_API_KEY)
    else:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Setup logging
def setup_logger():
    date_str = datetime.now().strftime("%Y%m%d")
    base_log_filename = f"{PROCESS_NAME}_{date_str}.log"
    log_file_path = process_log_dir / base_log_filename

    batch = 1
    while log_file_path.exists() and log_file_path.stat().st_size > 1 * 1024 * 1024 * 1024:
        log_file_path = process_log_dir / f"{PROCESS_NAME}_{date_str}_{batch}.log"
        batch += 1

    logger = logging.getLogger(PROCESS_NAME)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()
if AI_PROVIDER not in ["lm_studio", "openai", "gemini", "anthropic"]:
    logger.warning(f"AI_PROVIDER '{AI_PROVIDER}' not fully supported natively via AsyncOpenAI yet. Make sure your wrapper logic is correctly mapped.")

# Setup Progress tracking
PROGRESS_FILE = process_proc_dir / "progress.json"

def load_progress():
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading progress: {e}")
    return {"processed_files": [], "total_files": 0, "metrics": {"total_read": 0, "noise_filtered": 0, "successful_chunks": 0, "processing_errors": 0}}

def save_progress(progress_data):
    try:
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving progress: {e}")

def clear_progress():
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

def prompt_resume(progress_data):
    if progress_data.get("processed_files"):
        while True:
            response = input(f"Found interrupted progress ({len(progress_data['processed_files'])} files processed). Do you want to resume? (y/n): ").strip().lower()
            if response in ['y', 'n']:
                return response == 'y'
    return False

# ==========================================
# 1. Pydantic Model for Structured Output
# ==========================================
class ContentAnalysisResult(BaseModel):
    is_noise: bool = Field(
        description="True if the text is soft-ads, subscription notices, meaningless greetings, canned auto-replies, or broken text."
    )
    generated_tags: List[str] = Field(
        description="2-4 precise topic keywords in English or Chinese (e.g., 'career_planning', 'flutter_dev')."
    )
    content_category: str = Field(
        description="The style and category of the text (e.g., 'academic_research', 'casual_chat', 'official_report', 'personal_reflection')."
    )
    information_weight: float = Field(
        description="Information density and value (0.0 to 1.0). 0.1-0.3: casual chat; 0.4-0.7: general work/drafts; 0.8-1.0: core research/deep logic."
    )


# ==========================================
# 2. AI Analysis Function (with Retry & Concurrency limit)
# ==========================================
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError))
)
async def analyze_content_with_llm(page_content: str, semaphore: asyncio.Semaphore) -> ContentAnalysisResult:
    async with semaphore:
        response = await client.beta.chat.completions.parse(
            model=LLM_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior data curator and metadata tagging expert. Analyze the following text."
                },
                {"role": "user", "content": f"Text to analyze:\n\n{page_content}"}
            ],
            response_format=ContentAnalysisResult,
            temperature=0.1
        )
        return response.choices[0].message.parsed

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError))
)
async def get_embeddings(texts: List[str], model: str, semaphore: asyncio.Semaphore) -> List[List[float]]:
    if not texts:
        return []
    
    async with semaphore:
        response = await client.embeddings.create(input=texts, model=model)
        return [data.embedding for data in response.data]

async def get_existing_chunk_hashes(
    file_name: str,
    chunk_indices: List[int],
    db_connection: Optional[asyncpg.Connection],
    chroma_collection: Optional[chromadb.api.models.Collection.Collection]
) -> Dict[int, str]:
    """
    Retrieves existing content hashes for given file_name and chunk_indices.
    Returns a dictionary mapping chunk_index to content_hash.
    """
    existing_hashes = {}
    if VECTOR_DB_TYPE == "postgres" and db_connection:
        query = f"""
            SELECT chunk_index, content_hash FROM {COLLECTION_NAME}
            WHERE file_name = $1 AND chunk_index = ANY($2::int[])
        """
        records = await db_connection.fetch(query, file_name, chunk_indices)
        for record in records:
            existing_hashes[record["chunk_index"]] = record["content_hash"]
    elif VECTOR_DB_TYPE == "chromadb" and chroma_collection:
        # ChromaDB's get method can filter by metadata, but it's not as direct for batch checking
        # We need to construct IDs and then query.
        # This assumes the ChromaDB IDs are generated as f"{file_name}-{chunk_index}"
        ids_to_check = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_name}-{idx}")) for idx in chunk_indices]
        results = chroma_collection.get(ids=ids_to_check, include=['metadatas'])
        for i, _id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            if metadata and metadata.get('file_name') == file_name: # Double check
                existing_hashes[metadata['chunk_index']] = metadata['content_hash']
    return existing_hashes


async def _insert_chunks_to_db(
    chunks_batch: List[Dict[str, Any]],
    db_connection: Optional[asyncpg.Connection],
    chroma_collection: Optional[chromadb.api.models.Collection.Collection],
    semaphore: asyncio.Semaphore,
    embedding_func: Optional[Any], # This could be chromadb's EmbeddingFunction or None
    metrics: Dict[str, Any]
):
    if not chunks_batch:
        return

    logger.info(f"Inserting batch of {len(chunks_batch)} chunks into {VECTOR_DB_TYPE}...")

    if VECTOR_DB_TYPE == "postgres":
        texts_to_embed = [c["raw_content"] for c in chunks_batch]
        embeddings = []

        if EMBEDDING_PROVIDER == "openai":
            # OpenAI embedding API also has rate limits, so batching here too
            openai_batch_size = 100
            for i in range(0, len(texts_to_embed), openai_batch_size):
                batch_texts = texts_to_embed[i:i+openai_batch_size]
                batch_embeddings = await get_embeddings(batch_texts, EMBEDDING_MODEL_NAME, semaphore)
                embeddings.extend(batch_embeddings)
        elif EMBEDDING_PROVIDER == "lm_studio":
            if embedding_func:
                loop = asyncio.get_running_loop()
                # Embedding function from chromadb is synchronous, run in executor
                embeddings = await loop.run_in_executor(None, embedding_func, texts_to_embed)
            else:
                logger.error("LM Studio embedding function not initialized for Postgres.")
                return
        else:
            logger.error(f"Unsupported embedding provider for Postgres: {EMBEDDING_PROVIDER}")
            return
        
        if not embeddings:
            logger.warning("No embeddings were generated for the current batch.")
            return

        records_to_insert = []
        for i, chunk in enumerate(chunks_batch):
            records_to_insert.append(
                (
                    chunk["file_name"], chunk["raw_content"], chunk["ai_summary"],
                    chunk["data_source"], chunk["content_category"], chunk["style_tags"],
                    chunk["is_noise"], chunk["information_weight"], chunk["content_hash"],
                    chunk["chunk_index"], embeddings[i], chunk["original_time"]
                )
            )
        
        try:
            await db_connection.executemany(
                f"""
                INSERT INTO {COLLECTION_NAME} (
                    file_name, raw_content, ai_summary, data_source, content_category,
                    style_tags, is_noise, information_weight, content_hash, chunk_index,
                    embedding, original_time
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (file_name, chunk_index) DO UPDATE SET
                    raw_content = EXCLUDED.raw_content,
                    ai_summary = EXCLUDED.ai_summary,
                    data_source = EXCLUDED.data_source,
                    content_category = EXCLUDED.content_category,
                    style_tags = EXCLUDED.style_tags,
                    is_noise = EXCLUDED.is_noise,
                    information_weight = EXCLUDED.information_weight,
                    content_hash = EXCLUDED.content_hash,
                    embedding = EXCLUDED.embedding,
                    original_time = EXCLUDED.original_time,
                    created_at = CURRENT_TIMESTAMP
                """,
                records_to_insert
            )
            logger.info(f"Successfully inserted/updated {len(records_to_insert)} chunks into Postgres.")
            metrics["successful_chunks"] += len(records_to_insert)
        except Exception as e:
            logger.error(f"Failed to insert/update batch into Postgres: {e}")
            metrics["processing_errors"] += len(records_to_insert) # Count as errors for these chunks

    elif VECTOR_DB_TYPE == "chromadb":
        try:
            chroma_collection.add(
                ids=[str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{c['file_name']}-{c['chunk_index']}")) for c in chunks_batch],
                documents=[c["raw_content"] for c in chunks_batch],
                metadatas=[{
                    "file_name": c["file_name"],
                    "data_source": c["data_source"],
                    "content_category": c["content_category"],
                    "style_tags": c["style_tags"],
                    "is_noise": c["is_noise"],
                    "information_weight": c["information_weight"],
                    "content_hash": c["content_hash"],
                    "chunk_index": c["chunk_index"],
                    "original_time": c["original_time"]
                } for c in chunks_batch]
            )
            logger.info(f"Successfully inserted/updated {len(chunks_batch)} chunks into ChromaDB.")
            metrics["successful_chunks"] += len(chunks_batch)
        except Exception as e:
            logger.error(f"Failed to insert/update batch into ChromaDB: {e}")
            metrics["processing_errors"] += len(chunks_batch) # Count as errors for these chunks


# ==========================================
# 3. Main Data Pipeline
# ==========================================
async def process_and_ingest_data():
    start_time = datetime.now()
    logger.info("Process Started")
    logger.info("Starting RAG data ingestion pipeline...")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        logger.warning(f"Data directory {DATA_DIR} was not found, created a new empty one. Please place your JSON files there.")
        return

    progress_data = load_progress()
    resume = prompt_resume(progress_data)
    
    if not resume:
        logger.info("Starting fresh, clearing previous progress.")
        clear_progress()
        progress_data = {"processed_files": [], "total_files": 0, "metrics": {"total_read": 0, "noise_filtered": 0, "successful_chunks": 0, "processing_errors": 0}}
    else:
        logger.info(f"Resuming process. {len(progress_data.get('processed_files', []))} files already processed.")

    processed_files = progress_data.get("processed_files", [])
    total_files_processed_count = progress_data.get("total_files", 0)
    metrics = progress_data.get("metrics", {"total_read": 0, "noise_filtered": 0, "successful_chunks": 0, "processing_errors": 0})

    file_pattern = os.path.join(DATA_DIR, "*.json")
    all_files = glob.glob(file_pattern)
    files_to_process = sorted([f for f in all_files if f not in processed_files]) # Sort for consistent processing order

    total_files_in_queue = len(files_to_process)
    logger.info(f"Found {total_files_in_queue} files to process out of {len(all_files)} total files.")

    if total_files_in_queue == 0:
        logger.info("No new files to process.")
        return

    semaphore = asyncio.Semaphore(20) # Concurrency limit for AI calls
    
    logger.info("Initializing Vector Database...")
    embedding_func = None
    if VECTOR_DB_TYPE == "chromadb":
        if EMBEDDING_PROVIDER == "lm_studio":
            import chromadb.utils.embedding_functions as embedding_functions_chroma

            class LocalOpenAIEmbeddingFunction(embedding_functions_chroma.EmbeddingFunction):
                def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
                    from openai import OpenAI
                    sync_client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
                    response = sync_client.embeddings.create(input=input, model=EMBEDDING_MODEL_NAME)
                    return [data.embedding for data in response.data]

            embedding_func = LocalOpenAIEmbeddingFunction()
        else:
            embedding_func = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name=EMBEDDING_MODEL_NAME
            )
    # For Postgres, embedding_func is handled by get_embeddings directly, not passed to _insert_chunks_to_db

    db_connection = None
    chroma_collection = None
    if VECTOR_DB_TYPE == "postgres":
        logger.info(f"Connecting to Postgres vector store at {PG_HOST}:{PG_PORT}")
        db_connection = await asyncpg.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            database=PG_DB_NAME
        )
        await register_vector(db_connection)
        await db_connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {COLLECTION_NAME} (
                id serial4 NOT NULL,
                file_name varchar(255) NOT NULL,
                raw_content text NOT NULL,
                ai_summary text NULL,
                data_source varchar(100) NULL,
                content_category varchar(100) NULL,
                style_tags text[] NULL,
                is_noise boolean DEFAULT false NOT NULL,
                information_weight numeric(3, 2) DEFAULT 0.50 NULL,
                content_hash varchar(64) NOT NULL,
                chunk_index int4 DEFAULT 0 NOT NULL,
                embedding public.vector(1536) NULL,
                original_time timestamptz NULL,
                created_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL,
                CONSTRAINT writing_style_logs_pkey PRIMARY KEY (id),
                CONSTRAINT uq_file_chunk UNIQUE (file_name, chunk_index)
            );
        """)

    elif VECTOR_DB_TYPE == "chromadb":
        if CHROMA_DB_HOST and CHROMA_DB_PORT:
            logger.info(f"Connecting to remote ChromaDB server at {CHROMA_DB_HOST}:{CHROMA_DB_PORT}")
            chroma_client = chromadb.HttpClient(host=CHROMA_DB_HOST, port=CHROMA_DB_PORT)
        else:
            logger.info(f"Connecting to local ChromaDB instance at {CHROMA_DB_DIR}")
            chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

        chroma_collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func # Pass embedding_func to ChromaDB
        )
    else:
        raise ValueError(f"Unsupported VECTOR_DB_TYPE: {VECTOR_DB_TYPE}")


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )

    current_batch_chunks: List[Dict[str, Any]] = []
    
    try:
        for file_idx, filepath in enumerate(files_to_process):
            current_file_name = os.path.basename(filepath)
            logger.info(f"Processing file {file_idx + 1}/{total_files_in_queue}: {current_file_name}")
            
            docs: List[Dict[str, Any]] = []
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        docs.extend(data)
                    elif isinstance(data, dict):
                        extracted_list = None
                        for key, val in data.items():
                            if isinstance(val, list):
                                extracted_list = val
                                break
                        if extracted_list:
                            docs.extend(extracted_list)
                        else:
                            docs.append(data)
            except Exception as e:
                logger.error(f"Failed to read {current_file_name}: {e}")
                metrics["processing_errors"] += 1
                continue

            if not docs:
                processed_files.append(filepath)
                total_files_processed_count += 1
                progress_data["processed_files"] = processed_files
                progress_data["total_files"] = total_files_processed_count
                save_progress(progress_data)
                logger.info(f"No documents found in {current_file_name}. Skipping.")
                continue

            metrics["total_read"] += len(docs)

            logger.info(f"Starting AI analysis for {len(docs)} documents in {current_file_name}...")
            tasks = [analyze_content_with_llm(doc.get("page_content", ""), semaphore) for doc in docs]
            analysis_results = await asyncio.gather(*tasks, return_exceptions=True)

            valid_docs = []
            for doc_idx_in_file, (doc, result) in enumerate(zip(docs, analysis_results)):
                logger.info(f"  Processing record {doc_idx_in_file + 1}/{len(docs)} in {current_file_name} for AI analysis and chunking.")
                if isinstance(result, Exception):
                    logger.warning(f"  Error analyzing record {doc_idx_in_file + 1} in {current_file_name}: {result}")
                    metrics["processing_errors"] += 1
                    continue
                
                if result.is_noise:
                    logger.info(f"  Record {doc_idx_in_file + 1} in {current_file_name} identified as noise. Skipping.")
                    metrics["noise_filtered"] += 1
                    continue

                doc["analysis"] = result
                valid_docs.append(doc)

            file_chunks_count = 0
            # Collect all chunk_indices for the current file to query existing hashes efficiently
            # This assumes that the text_splitter will produce chunks with sequential indices starting from 0
            # For each doc, we need to know how many chunks it will produce to get the range of indices
            all_chunk_indices_for_file = []
            for doc_in_valid_docs in valid_docs:
                temp_chunks = text_splitter.split_text(doc_in_valid_docs["page_content"])
                all_chunk_indices_for_file.extend(range(len(temp_chunks))) # This is not correct, chunk_index is per doc.
            
            # Correct way to get all chunk indices for the file:
            # We need to iterate through valid_docs and then its chunks to get the actual chunk_indices
            # This is a bit tricky because chunk_index is per doc, not global for the file.
            # Let's assume chunk_index is unique per file for simplicity for now, or we need a more complex key.
            # The current schema uses (file_name, chunk_index) as unique, implying chunk_index is unique per file.
            # If chunk_index is per document, then the unique key should be (file_name, doc_id, chunk_index).
            # Given the current schema, I will assume chunk_index is unique within a file.
            # If not, the schema needs to be updated to include a document identifier.

            # Re-evaluating: The current `chunk_index` is per document.
            # `uq_file_chunk UNIQUE (file_name, chunk_index)` means that for a given file,
            # there can only be one chunk with `chunk_index = 0`, one with `chunk_index = 1`, etc.
            # This is problematic if a file contains multiple documents, and each document is chunked.
            # The `chunk_index` should be unique per (file_name, original_document_identifier).
            # For now, I will proceed with the assumption that `chunk_index` is unique per file,
            # and if a file contains multiple documents, their chunks will overwrite each other if they have the same chunk_index.
            # A more robust solution would be to add a `document_id` to the schema and use `(file_name, document_id, chunk_index)` as unique.
            # For this request, I will stick to the current schema and assume `chunk_index` is unique per file.

            # Let's collect all potential chunk indices for the current file.
            # This is still problematic if multiple documents in a file produce chunks with the same index.
            # I will modify the `chunk_index` to be a global index within the file for this implementation.
            # This means `chunk_index` will be `global_chunk_counter` for the file.

            all_generated_chunks_for_file = []
            for doc_in_valid_docs in valid_docs:
                chunks_from_doc = text_splitter.split_text(doc_in_valid_docs["page_content"])
                for chunk_text in chunks_from_doc:
                    all_generated_chunks_for_file.append({
                        "raw_content": chunk_text,
                        "data_source": doc_in_valid_docs.get("metadata", {}).get("source"),
                        "content_category": doc_in_valid_docs["analysis"].content_category,
                        "style_tags": doc_in_valid_docs["analysis"].generated_tags,
                        "is_noise": doc_in_valid_docs["analysis"].is_noise,
                        "information_weight": doc_in_valid_docs["analysis"].information_weight,
                        "original_time": doc_in_valid_docs.get("metadata", {}).get("created_at"),
                    })
            
            # Now, query existing hashes for all potential chunk indices in this file
            # We need to know the maximum chunk_index that could be generated for this file.
            # This is still not ideal. The `get_existing_chunk_hashes` should be called for specific (file_name, chunk_index) pairs.
            # Let's adjust the logic to query for each chunk as it's generated, or query for a range of chunk_indices.
            # For simplicity and to avoid over-fetching, I will query for each chunk as it's processed.
            # This might increase DB calls but ensures correctness with the current schema.

            global_chunk_idx_for_file = 0
            for doc_idx, doc in enumerate(valid_docs):
                chunks = text_splitter.split_text(doc["page_content"])
                analysis_result = doc["analysis"]

                logger.info(f"  Chunking document {doc_idx + 1}/{len(valid_docs)} from {current_file_name}. Generated {len(chunks)} chunks.")

                for chunk_idx_in_doc, chunk_text in enumerate(chunks):
                    content_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
                    
                    # Query for existing chunk hash for this specific (file_name, global_chunk_idx_for_file)
                    existing_chunk_hashes = await get_existing_chunk_hashes(
                        current_file_name,
                        [global_chunk_idx_for_file], # Query for a single chunk_index
                        db_connection,
                        chroma_collection
                    )
                    existing_hash = existing_chunk_hashes.get(global_chunk_idx_for_file)

                    if existing_hash == content_hash:
                        logger.info(f"  Chunk (file: {current_file_name}, index: {global_chunk_idx_for_file}) is a duplicate. Skipping.")
                        global_chunk_idx_for_file += 1
                        continue # Skip this chunk, it's already in the DB and unchanged
                    elif existing_hash: # Exists but hash is different
                        logger.info(f"  Chunk (file: {current_file_name}, index: {global_chunk_idx_for_file}) has been modified. Will update.")
                        # Add to batch for update
                    else: # Does not exist
                        logger.info(f"  Chunk (file: {current_file_name}, index: {global_chunk_idx_for_file}) is new. Will insert.")
                        # Add to batch for insert

                    current_batch_chunks.append({
                        "file_name": current_file_name,
                        "raw_content": chunk_text,
                        "ai_summary": None, # Placeholder for summary
                        "data_source": doc.get("metadata", {}).get("source"),
                        "content_category": analysis_result.content_category,
                        "style_tags": analysis_result.generated_tags,
                        "is_noise": analysis_result.is_noise,
                        "information_weight": analysis_result.information_weight,
                        "content_hash": content_hash,
                        "chunk_index": global_chunk_idx_for_file, # Use global index for uniqueness per file
                        "original_time": doc.get("metadata", {}).get("created_at"),
                    })
                    file_chunks_count += 1
                    global_chunk_idx_for_file += 1

                    if len(current_batch_chunks) >= BATCH_SIZE:
                        await _insert_chunks_to_db(current_batch_chunks, db_connection, chroma_collection, semaphore, embedding_func, metrics)
                        current_batch_chunks = [] # Clear batch after insertion
                        # Save progress after each batch insertion
                        progress_data["metrics"] = metrics
                        save_progress(progress_data)
            
            logger.info(f"Finished processing {current_file_name}. Total chunks generated from this file: {file_chunks_count}.")

            processed_files.append(filepath)
            total_files_processed_count += 1
            progress_data["processed_files"] = processed_files
            progress_data["total_files"] = total_files_processed_count
            progress_data["metrics"] = metrics
            save_progress(progress_data)
            logger.info(f"Overall Progress: {total_files_processed_count}/{len(all_files)} files processed. Total chunks inserted: {metrics['successful_chunks']}")

        # Insert any remaining chunks after all files have been processed
        if current_batch_chunks:
            logger.info(f"Inserting final batch of {len(current_batch_chunks)} chunks.")
            await _insert_chunks_to_db(current_batch_chunks, db_connection, chroma_collection, semaphore, embedding_func, metrics)
            current_batch_chunks = [] # Clear batch after final insertion
            progress_data["metrics"] = metrics
            save_progress(progress_data)


        end_time = datetime.now()
        execution_time = end_time - start_time

        logger.info("==========================================")
        logger.info("RAG Ingestion Pipeline Completed Successfully!")
        logger.info("--- Execution Report ---")
        logger.info(f"Total Source Documents Read   : {metrics['total_read']}")
        logger.info(f"Processing Errors Encountered : {metrics['processing_errors']}")
        logger.info(f"Noise Documents Intercepted   : {metrics['noise_filtered']}")
        logger.info(f"Chunks Successfully Inserted  : {metrics['successful_chunks']}")
        logger.info(f"Total Execution Time          : {execution_time}")
        logger.info("==========================================")

        clear_progress()

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
        save_progress(progress_data)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}", exc_info=True)
        save_progress(progress_data)
        sys.exit(1)
    finally:
        if db_connection:
            await db_connection.close()


if __name__ == "__main__":
    try:
        asyncio.run(process_and_ingest_data())
    except KeyboardInterrupt:
         pass
