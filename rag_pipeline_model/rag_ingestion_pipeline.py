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

async def parse_rag_ready_json(json_data: Dict[str, Any], file_name: str) -> List[Dict[str, Any]]:
    """
    Parses JSON data from various 'rag_ready' formats into a flat list of documents.
    Each document will have 'page_content' and a 'metadata' dictionary.
    """
    documents = []

    # Helper to standardize metadata
    def standardize_metadata(raw_metadata: Dict[str, Any], default_source: str = "unknown") -> Dict[str, Any]:
        standardized = {}
        # Map timestamp to original_time
        if "timestamp" in raw_metadata:
            standardized["original_time"] = raw_metadata.pop("timestamp")
        
        # Map source to data_source
        if "source" in raw_metadata:
            standardized["data_source"] = raw_metadata.pop("source")
        elif "data_source" not in standardized:
            standardized["data_source"] = default_source
        
        # Add any remaining metadata
        standardized.update(raw_metadata)
        return standardized

    # --- Handle chat_html_to_rag.py and meta_data_to_rag.py (messages part) ---
    if "messages" in json_data and isinstance(json_data["messages"], list):
        for thread in json_data["messages"]:
            if not isinstance(thread, dict): continue
            thread_id = thread.get("thread_id", str(uuid.uuid4()))
            thread_metadata = standardize_metadata(thread.get("metadata", {}), default_source="chat_html")
            
            if "conversation" in thread and isinstance(thread["conversation"], list):
                for message in thread["conversation"]:
                    if not isinstance(message, dict): continue
                    msg_page_content = message.get("page_content")
                    if not msg_page_content: continue

                    msg_metadata = standardize_metadata(message.get("metadata", {}), default_source=thread_metadata.get("data_source", "chat_html"))
                    
                    # Merge thread-level and message-level metadata
                    final_metadata = {
                        **thread_metadata,
                        **msg_metadata,
                        "thread_id": thread_id,
                        "message_id": message.get("message_id", str(uuid.uuid4()))
                    }
                    documents.append({"page_content": msg_page_content, "metadata": final_metadata})

    # --- Handle gmail_to_rag.py ---
    elif "email_threads" in json_data and isinstance(json_data["email_threads"], list):
        for thread in json_data["email_threads"]:
            if not isinstance(thread, dict): continue
            thread_id = thread.get("thread_id", str(uuid.uuid4()))
            subject = thread.get("subject", "No Subject")
            thread_metadata = standardize_metadata(thread.get("metadata", {}), default_source="gmail")

            if "conversation" in thread and isinstance(thread["conversation"], list):
                for message in thread["conversation"]:
                    if not isinstance(message, dict): continue
                    msg_page_content = message.get("page_content")
                    if not msg_page_content: continue

                    msg_metadata = standardize_metadata(message.get("metadata", {}), default_source=thread_metadata.get("data_source", "gmail"))
                    
                    # Merge thread-level and message-level metadata
                    final_metadata = {
                        **thread_metadata,
                        **msg_metadata,
                        "thread_id": thread_id,
                        "subject": subject,
                        "message_id": message.get("message_id", str(uuid.uuid4()))
                    }
                    documents.append({"page_content": msg_page_content, "metadata": final_metadata})

    # --- Handle meta_data_to_rag.py (posts part) ---
    elif "posts" in json_data and isinstance(json_data["posts"], list):
        for post in json_data["posts"]:
            if not isinstance(post, dict): continue
            post_id = post.get("post_id", str(uuid.uuid4()))
            post_page_content = post.get("page_content")
            post_metadata = standardize_metadata(post.get("metadata", {}), default_source="meta_post")

            if post_page_content:
                final_metadata = {**post_metadata, "post_id": post_id}
                documents.append({"page_content": post_page_content, "metadata": final_metadata})
            
            # Handle comments within posts
            if "comments" in post and isinstance(post["comments"], list):
                for comment in post["comments"]:
                    if not isinstance(comment, dict): continue
                    comment_page_content = comment.get("page_content")
                    if not comment_page_content: continue

                    comment_metadata = standardize_metadata(comment.get("metadata", {}), default_source=post_metadata.get("data_source", "meta_comment"))
                    
                    final_metadata = {
                        **post_metadata, # Inherit post metadata
                        **comment_metadata, # Override with comment metadata if keys overlap
                        "post_id": post_id,
                        "comment_id": comment.get("comment_id", str(uuid.uuid4()))
                    }
                    documents.append({"page_content": comment_page_content, "metadata": final_metadata})
    
    # --- Fallback for simple list of documents or single document ---
    elif isinstance(json_data, list):
        for item in json_data:
            if isinstance(item, dict) and "page_content" in item:
                documents.append({"page_content": item["page_content"], "metadata": standardize_metadata(item.get("metadata", {}), default_source="unknown_list")})
    elif isinstance(json_data, dict) and "page_content" in json_data:
        documents.append({"page_content": json_data["page_content"], "metadata": standardize_metadata(json_data.get("metadata", {}), default_source="unknown_single")})
    else:
        logger.warning(f"File {file_name} has an unrecognized JSON structure. No documents extracted.")

    return documents


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
                    # Use the new structured parsing function
                    docs = await parse_rag_ready_json(data, current_file_name)
            except Exception as e:
                logger.error(f"Failed to read or parse {current_file_name}: {e}")
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

            logger.info(f"Starting incremental AI analysis for {len(docs)} documents in {current_file_name}...")
            
            # 設定 AI 處理的子批次大小 (例如每 20 筆做一次結算)
            AI_SUB_BATCH_SIZE = 20
            global_chunk_idx_for_file = 0 # Reset for each file
            file_chunks_count = 0

            for i in range(0, len(docs), AI_SUB_BATCH_SIZE):
                sub_docs = docs[i : i + AI_SUB_BATCH_SIZE]
                logger.info(f"  Processing batch {i//AI_SUB_BATCH_SIZE + 1}/{(len(docs)-1)//AI_SUB_BATCH_SIZE + 1}...")
                
                tasks = [analyze_content_with_llm(d.get("page_content", ""), semaphore) for d in sub_docs]
                sub_results = await asyncio.gather(*tasks, return_exceptions=True)

                for doc_idx_in_sub, (doc, result) in enumerate(zip(sub_docs, sub_results)):
                    current_idx = i + doc_idx_in_sub + 1
                    if isinstance(result, Exception):
                        logger.warning(f"  Error analyzing record {current_idx}: {result}")
                        metrics["processing_errors"] += 1
                        continue
                    
                    if result.is_noise:
                        metrics["noise_filtered"] += 1
                        continue

                    # 立即進行切片處理
                    chunks = text_splitter.split_text(doc["page_content"])
                    
                    # 優化：針對此文件的這組 Chunks 一次查詢雜湊 (減少 DB 往返)
                    chunk_indices_to_check = list(range(global_chunk_idx_for_file, global_chunk_idx_for_file + len(chunks)))
                    existing_hashes = await get_existing_chunk_hashes(
                        current_file_name, chunk_indices_to_check, db_connection, chroma_collection
                    )

                    for chunk_text in chunks:
                        content_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
                        existing_hash = existing_hashes.get(global_chunk_idx_for_file)

                        if existing_hash == content_hash:
                            global_chunk_idx_for_file += 1
                            continue
                        
                        current_batch_chunks.append({
                            "file_name": current_file_name,
                            "raw_content": chunk_text,
                            "ai_summary": None,
                            "data_source": doc.get("metadata", {}).get("data_source"),
                            "content_category": result.content_category,
                            "style_tags": result.generated_tags,
                            "is_noise": result.is_noise,
                            "information_weight": result.information_weight,
                            "content_hash": content_hash,
                            "chunk_index": global_chunk_idx_for_file,
                            "original_time": doc.get("metadata", {}).get("original_time"),
                        })
                        file_chunks_count += 1
                        global_chunk_idx_for_file += 1

                        # 每達到 100 筆 Chunk 寫入一次資料庫
                        if len(current_batch_chunks) >= BATCH_SIZE:
                            await _insert_chunks_to_db(current_batch_chunks, db_connection, chroma_collection, semaphore, embedding_func, metrics)
                            current_batch_chunks = []
                            progress_data["metrics"] = metrics
                            save_progress(progress_data)

                logger.info(f"  Completed through record {min(i + AI_SUB_BATCH_SIZE, len(docs))}/{len(docs)}")
            
            logger.info(f"Finished file {current_file_name}. Chunks added: {file_chunks_count}")

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
