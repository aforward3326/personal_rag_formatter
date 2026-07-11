import os
import json
import glob
import asyncio
import logging
import uuid
import sys
import hashlib
from datetime import datetime
from typing import List, Dict, Any

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
    total_files = progress_data.get("total_files", 0)
    metrics = progress_data.get("metrics", {"total_read": 0, "noise_filtered": 0, "successful_chunks": 0, "processing_errors": 0})

    file_pattern = os.path.join(DATA_DIR, "*.json")
    all_files = glob.glob(file_pattern)
    files_to_process = [f for f in all_files if f not in processed_files]

    total_docs = len(files_to_process)
    logger.info(f"Found {total_docs} files to process out of {len(all_files)} total files.")

    if total_docs == 0:
        logger.info("No new files to process.")
        return

    semaphore = asyncio.Semaphore(20)
    
    logger.info("Initializing Vector Database...")
    embedding_func = None
    if VECTOR_DB_TYPE == "chromadb" or EMBEDDING_PROVIDER == "lm_studio":
        if EMBEDDING_PROVIDER == "lm_studio":
            import chromadb.utils.embedding_functions as embedding_functions

            class LocalOpenAIEmbeddingFunction(embedding_functions.EmbeddingFunction):
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

    db_connection = None
    collection = None
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
                CONSTRAINT uq_file_chunk UNIQUE (content_hash, chunk_index)
            );
        """)

    elif VECTOR_DB_TYPE == "chromadb":
        if CHROMA_DB_HOST and CHROMA_DB_PORT:
            logger.info(f"Connecting to remote ChromaDB server at {CHROMA_DB_HOST}:{CHROMA_DB_PORT}")
            chroma_client = chromadb.HttpClient(host=CHROMA_DB_HOST, port=CHROMA_DB_PORT)
        else:
            logger.info(f"Connecting to local ChromaDB instance at {CHROMA_DB_DIR}")
            chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func
        )
    else:
        raise ValueError(f"Unsupported VECTOR_DB_TYPE: {VECTOR_DB_TYPE}")


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )

    try:
        for filepath in files_to_process:
            logger.info(f"Processing file: {filepath}")
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
                logger.error(f"Failed to read {filepath}: {e}")
                metrics["processing_errors"] += 1
                continue

            if not docs:
                processed_files.append(filepath)
                progress_data["processed_files"] = processed_files
                save_progress(progress_data)
                continue

            metrics["total_read"] += len(docs)

            logger.info(f"Starting AI analysis for {len(docs)} documents in {os.path.basename(filepath)}...")
            tasks = [analyze_content_with_llm(doc.get("page_content", ""), semaphore) for doc in docs]
            analysis_results = await asyncio.gather(*tasks, return_exceptions=True)

            valid_docs = []
            for doc, result in zip(docs, analysis_results):
                if isinstance(result, Exception):
                    logger.warning(f"Error analyzing document: {result}")
                    metrics["processing_errors"] += 1
                    continue

                doc["analysis"] = result
                valid_docs.append(doc)


            chunks_to_insert = []
            for doc_idx, doc in enumerate(valid_docs):
                chunks = text_splitter.split_text(doc["page_content"])
                analysis_result = doc["analysis"]

                for chunk_idx, chunk_text in enumerate(chunks):
                    content_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
                    
                    chunks_to_insert.append({
                        "file_name": os.path.basename(filepath),
                        "raw_content": chunk_text,
                        "ai_summary": None, # Placeholder for summary
                        "data_source": doc.get("metadata", {}).get("source"),
                        "content_category": analysis_result.content_category,
                        "style_tags": analysis_result.generated_tags,
                        "is_noise": analysis_result.is_noise,
                        "information_weight": analysis_result.information_weight,
                        "content_hash": content_hash,
                        "chunk_index": chunk_idx,
                        "original_time": doc.get("metadata", {}).get("created_at"),
                    })

            metrics["successful_chunks"] += len(chunks_to_insert)

            if chunks_to_insert:
                if VECTOR_DB_TYPE == "postgres":
                    texts_to_embed = [c["raw_content"] for c in chunks_to_insert]
                    embeddings = []

                    if EMBEDDING_PROVIDER == "openai":
                        batch_size = 100
                        for i in range(0, len(texts_to_embed), batch_size):
                            batch_texts = texts_to_embed[i:i+batch_size]
                            batch_embeddings = await get_embeddings(batch_texts, EMBEDDING_MODEL_NAME, semaphore)
                            embeddings.extend(batch_embeddings)

                    elif EMBEDDING_PROVIDER == "lm_studio":
                        if embedding_func:
                            loop = asyncio.get_running_loop()
                            embeddings = await loop.run_in_executor(None, embedding_func, texts_to_embed)
                        else:
                            logger.error("LM Studio embedding function not initialized.")
                            continue
                    else:
                        logger.error(f"Unsupported embedding provider for Postgres: {EMBEDDING_PROVIDER}")
                        continue
                    
                    if not embeddings:
                        logger.warning("No embeddings were generated.")
                        continue

                    records_to_insert = []
                    for i, chunk in enumerate(chunks_to_insert):
                        records_to_insert.append(
                            (
                                chunk["file_name"], chunk["raw_content"], chunk["ai_summary"],
                                chunk["data_source"], chunk["content_category"], chunk["style_tags"],
                                chunk["is_noise"], chunk["information_weight"], chunk["content_hash"],
                                chunk["chunk_index"], embeddings[i], chunk["original_time"]
                            )
                        )
                    
                    await db_connection.executemany(
                        f"""
                        INSERT INTO {COLLECTION_NAME} (
                            file_name, raw_content, ai_summary, data_source, content_category,
                            style_tags, is_noise, information_weight, content_hash, chunk_index,
                            embedding, original_time
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        ON CONFLICT (content_hash, chunk_index) DO NOTHING
                        """,
                        records_to_insert
                    )
                    logger.info(f"Inserted {len(chunks_to_insert)} chunks into Postgres for file {os.path.basename(filepath)}.")

                elif VECTOR_DB_TYPE == "chromadb":
                    batch_size = 500
                    for i in range(0, len(chunks_to_insert), batch_size):
                        batch = chunks_to_insert[i: i + batch_size]
                        collection.add(
                            ids=[c["id"] for c in batch],
                            documents=[c["text"] for c in batch],
                            metadatas=[c["metadata"] for c in batch]
                        )
                    logger.info(f"Inserted {len(chunks_to_insert)} chunks into Vector DB for file {os.path.basename(filepath)}.")

            processed_files.append(filepath)
            total_files += 1
            progress_data["processed_files"] = processed_files
            progress_data["total_files"] = total_files
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
