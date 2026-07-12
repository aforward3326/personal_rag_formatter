import os
import sys
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple

# Load environment variables from .env file FIRST.
load_dotenv('../pipeline.env')

# Get API keys and provider settings from environment.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AI_PROVIDER_ENV = os.getenv("AI_PROVIDER", "openai").lower()
EMBEDDING_PROVIDER_ENV = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "project")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")

# Conditionally import the new Gemini library.
try:
    # Use the new SDK structure
    from google import genai
except ImportError:
    genai = None

# 1. Access via the types module
from google.genai import types

# Default Generation Config for single/streaming requests
config = types.GenerateContentConfig(
    temperature=0.2,
    top_p=0.95,
    max_output_tokens=512, # Lowered from 1024, as we only expect JSON metadata
)


# This will hold the configured client for Google AI
gemini_client: Optional[genai.Client] = None

def configure_generative_ai() -> Optional[genai.Client]:
    """
    Configures and returns a Google Generative AI client based on the environment provider.
    Supports 'gemini' for Google AI Studio and 'vertex' for Vertex AI.
    """
    # Determine the primary provider, giving precedence to the AI provider.
    provider = AI_PROVIDER_ENV
    if provider not in ["gemini", "vertex"]:
        provider = EMBEDDING_PROVIDER_ENV

    if provider not in ["gemini", "vertex"]:
        return None # Not using a Google provider.

    if not genai:
        raise ImportError("google.genai is not installed. Please install it with 'pip install google-genai'")

    if provider == "vertex":
        print("Configuring for Vertex AI...")
        
        # Fail-fast check for Google Cloud Application Default Credentials
        try:
            import google.auth
            from google.auth.exceptions import DefaultCredentialsError
            google.auth.default()
        except ImportError:
            pass # Let the genai client handle it if google.auth is missing
        except DefaultCredentialsError as e:
            raise RuntimeError("Google Cloud Application Default Credentials not found. Please run `gcloud auth application-default login` or set the GOOGLE_APPLICATION_CREDENTIALS environment variable.") from e

        # For Vertex, instantiate the client with project and location.
        client = genai.Client(vertexai=True, project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        print(f"Vertex AI client created for project '{VERTEX_PROJECT_ID}' in '{VERTEX_LOCATION}'.")
        return client
    
    elif provider == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY must be set in the environment for 'gemini' provider.")
        print("Configuring for Google AI Studio (Gemini)...")
        # For AI Studio, create the client explicitly with the API key.
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("Google AI Studio (Gemini) client created.")
        return client
    
    return None

# Execute the configuration at startup.
try:
    gemini_client = configure_generative_ai()
except RuntimeError as e:
    print(f"\n[Configuration Error] {e}\n")
    sys.exit(1)

# Import Google Cloud Storage for Vertex AI batch processing
try:
    from google.cloud import storage
except ImportError:
    storage = None

# Now import all other dependencies
import glob
import asyncio
import logging
import time
import uuid
import hashlib
from datetime import datetime
from urllib.parse import urljoin
from collections import defaultdict
import re

# Optional dependencies for Batch API mode
from pydantic import BaseModel, Field, ValidationError
import openai
from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, RetryError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import asyncpg
from pgvector.asyncpg import register_vector
try:
    import tiktoken
    # Cache encoding globally to avoid overhead of loading it for every chunk
    TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    tiktoken = None
    TIKTOKEN_ENCODING = None
try:
    from tqdm.asyncio import tqdm as async_tqdm
    from tqdm import tqdm
except ImportError:
    # Provide a dummy tqdm if it's not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable
    async_tqdm = tqdm


# ==========================================
# Configuration & Setup
# ==========================================
DATA_DIR = os.getenv("DATA_DIR", "../output_final_rag_data")
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "postgres").lower()
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "../chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "writing_style_logs")

# Batch Processing Configuration
USE_BATCH_API = os.getenv("USE_BATCH_API", "false").lower() == "true"
BATCH_MAX_LINES = int(os.getenv("BATCH_MAX_LINES", "50000"))
BATCH_MODEL_NAME = "gemini-1.5-flash-001"
API_BATCH_SIZE = int(os.getenv("API_BATCH_SIZE", "10"))
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# Test Mode Configuration
TEST_MODE = os.getenv("RAG_TEST_MODE", "false").lower() == "true"
TEST_HIDE_CONTENT = os.getenv("RAG_TEST_HIDE_CONTENT", "false").lower() == "true"

# Postgres settings
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_DB_NAME = os.getenv("PG_DB_NAME")

# AI & Embedding Providers
AI_PROVIDER = AI_PROVIDER_ENV
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
EMBEDDING_PROVIDER = EMBEDDING_PROVIDER_ENV
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# Other API Keys & Endpoints
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
ANYTHINGLLM_BASE_URL = os.getenv("ANYTHINGLLM_BASE_URL")
ANYTHINGLLM_API_KEY = os.getenv("ANYTHINGLLM_API_KEY")
ANYTHINGLLM_WORKSPACE_SLUG = os.getenv("ANYTHINGLLM_WORKSPACE_SLUG")

LOG_DIR = os.getenv("LOG_DIR", "../log")
PROC_DIR = os.getenv("PROC_DIR", "../processing")
CACHE_DIR = os.getenv("CACHE_DIR", "../cache")
LLM_CLEANED_DIR = os.getenv("LLM_CLEANED_DIR", "../llm_cleaned_data")
ERROR_DIR = os.getenv("ERROR_DIR", "../error_data")
MY_EMAIL = os.getenv("MY_EMAIL", "default@example.com")
MY_IDENTITIES_MATRIX = os.getenv("MY_IDENTITIES_MATRIX", "[]")
PROCESS_NAME = "rag_ingestion_pipeline"
BATCH_SIZE = 100

RUN_TIME_STR = datetime.now().strftime("%Y%m%d%H")

# Directory structures: ../{file_type}/{process_name}/
PROCESS_CACHE_DIR = os.path.join(CACHE_DIR, PROCESS_NAME)
PROCESS_CLEANED_DIR = os.path.join(LLM_CLEANED_DIR, PROCESS_NAME)
PROCESS_ERROR_DIR = os.path.join(ERROR_DIR, PROCESS_NAME)
BATCH_MANIFEST_FILE = os.path.join(PROCESS_CACHE_DIR, "batch_manifest.json")
BATCH_JOB_FILE = os.path.join(PROC_DIR, "active_gemini_batch_job.txt")

# Setup logging
def setup_logger(is_test_mode: bool = False):
    process_log_dir = Path(LOG_DIR) / PROCESS_NAME
    process_log_dir.mkdir(parents=True, exist_ok=True)
    
    log_suffix = "_test" if is_test_mode else ""
    base_log_filename = f"{PROCESS_NAME}_{RUN_TIME_STR}_{log_suffix}.log"
    
    log_file_path = process_log_dir / base_log_filename
    logger = logging.getLogger(PROCESS_NAME)
    logger.setLevel(logging.INFO)
    
    # Avoid adding handlers if they already exist
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    if is_test_mode:
        logger.warning("TEST MODE is enabled. No data will be sent to AI or saved to the database.")
        if TEST_HIDE_CONTENT:
            logger.warning("TEST_HIDE_CONTENT is enabled. Document content will not be shown in the log.")
        if not tiktoken:
            logger.warning("`tiktoken` library not found. Token counts will be rough estimates. Install with `pip install tiktoken` for better accuracy.")
            
    return logger

logger = setup_logger(is_test_mode=TEST_MODE)

# ==========================================
# Pydantic Model for Structured Output
# ==========================================
class AnalysisDetail(BaseModel):
    is_ad: bool = Field(description="True if the text is an advertisement.")
    ad_reason: str = Field(default="", description="Reason for marking as ad.")
    is_noise: bool = Field(description="True if the text is meaningless noise or completely irrelevant to [AUTHOR_ME].")
    noise_reason: str = Field(default="", description="Reason for marking as noise.")
    ai_summary: str = Field(description="An ultra-concise 5-10 word summary of the valid content, or empty string if noise/ad.")
    content_category: str = Field(description="The category of the text (e.g., 'casual_chat', 'professional_email', 'article', 'meeting_notes').")
    style_tags: List[str] = Field(description="Exactly 3-5 highly relevant topic/style keywords in English or Chinese for vector search precision, or empty list if noise/ad.")
    information_weight: float = Field(default=0.5, description="Information density and value (0.0 to 1.0).")
    original_time: Optional[str] = Field(default=None, description="The primary or earliest timestamp found in the content (Strictly ISO 8601 format, e.g., '2024-05-20T14:30:00Z'), or null if not applicable.")
    speaker: Optional[str] = Field(default=None, description="The primary speaker or sender of the valid content (e.g., '[AUTHOR_ME]' or '[SPEAKER_1]'), or null if not applicable.")
    data_source: Optional[str] = Field(default=None, description="Inferred source of the data (e.g., 'messenger', 'email', 'line_chat', 'slack'), or null.")
    needs_human_review: bool = Field(default=False, description="True if AI is uncertain about PII masking, relevance, or whether it should be marked as noise/ad.")
    human_review_reason: str = Field(default="", description="Brief explanation of why human review is needed, otherwise empty.")

class ChunkAnalysisDetail(BaseModel):
    chunk_id: str = Field(description="The unique identifier of the chunk from the input array.")
    analysis: AnalysisDetail = Field(description="The analysis result for this specific chunk.")

class MiniBatchAnalysisResult(BaseModel):
    results: List[ChunkAnalysisDetail] = Field(description="List of analysis results matching the input chunks.")

MINI_BATCH_SYSTEM_PROMPT_TEMPLATE = """
You are an expert data analysis system focused on extreme efficiency. The user is `[AUTHOR_ME]` (email: {MY_EMAIL}).
You will receive a JSON array of text chunks. Your task is to analyze EACH chunk and return a JSON array of results with MINIMAL token usage.

JSON output format must strictly match:
{{
  "results": [
    {{
      "chunk_id": "string (MUST match input)",
      "analysis": {{
         "is_ad": boolean, "ad_reason": "string (max 10 words)",
         "is_noise": boolean, "noise_reason": "string (max 10 words)",
         "ai_summary": "string (ultra-concise, 5-10 words, or \"\" if ad/noise)",
         "content_category": "string",
         "style_tags": ["tag1", "tag2", "tag3"],
         "information_weight": float,
         "original_time": "ISO 8601 or null",
         "speaker": "string or null",
         "data_source": "string or null",
         "needs_human_review": boolean, "human_review_reason": "string (max 15 words)"
      }}
    }}
  ]
}}

**CRITICAL RULES FOR EFFICIENCY**:
1.  **Process All**: Return one result for EVERY input chunk.
2.  **BE CONCISE**: Every string you generate must be as short as possible.
3.  **AD/NOISE RULE**: If `is_ad` or `is_noise` is true, you MUST make `ai_summary` an empty string `""` and `style_tags` an empty list `[]`. The `ad_reason` or `noise_reason` should be very brief (e.g., "spam link," "system message").
4.  **SUMMARY**: `ai_summary` must be 5-10 words MAXIMUM.
5.  **TAGS**: Provide 3-5 `style_tags` MAXIMUM.
6.  **JSON ONLY**: Your entire output must be only the JSON object. No markdown, no explanations.
"""

SYSTEM_PROMPT_TEMPLATE = """
You are an expert data analysis system focused on extreme efficiency. The user is `[AUTHOR_ME]` (email: {MY_EMAIL}).
Your task is to analyze the provided text and extract ONLY a JSON object with MINIMAL token usage.

JSON output format:
{{
    "is_ad": boolean, "ad_reason": "string (max 10 words)",
    "is_noise": boolean, "noise_reason": "string (max 10 words)",
    "ai_summary": "string (ultra-concise, 5-10 words, or \"\" if ad/noise)",
    "content_category": "string",
    "style_tags": ["tag1", "tag2", "tag3"],
    "information_weight": float,
    "original_time": "ISO 8601 or null",
    "speaker": "string or null",
    "data_source": "string or null",
    "needs_human_review": boolean, "human_review_reason": "string (max 15 words)"
}}

**CRITICAL RULES FOR EFFICIENCY**:
1.  **BE CONCISE**: Every string you generate must be as short as possible.
2.  **AD/NOISE RULE**: If `is_ad` or `is_noise` is true, you MUST make `ai_summary` an empty string `""` and `style_tags` an empty list `[]`. The `ad_reason` or `noise_reason` should be very brief (e.g., "spam link," "system message").
3.  **SUMMARY**: `ai_summary` must be 5-10 words MAXIMUM.
4.  **TAGS**: Provide 3-5 `style_tags` MAXIMUM.
5.  **JSON ONLY**: Your entire output must be only the JSON object. No markdown, no explanations.
"""

# ==========================================
# AI Client Initialization
# ==========================================
openai_client: Optional[AsyncOpenAI] = None
embedding_client: Optional[AsyncOpenAI] = None

if not TEST_MODE:
    if AI_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set for 'openai' provider.")
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    elif AI_PROVIDER == "lm_studio":
        openai_client = AsyncOpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
    elif AI_PROVIDER == "anythingllm":
        if not ANYTHINGLLM_BASE_URL or not ANYTHINGLLM_API_KEY:
            raise ValueError("ANYTHINGLLM_BASE_URL and ANYTHINGLLM_API_KEY must be set for 'anythingllm' provider.")
        base_url = ANYTHINGLLM_BASE_URL
        if ANYTHINGLLM_WORKSPACE_SLUG:
            if not base_url.endswith('/'): base_url += '/'
            base_url = urljoin(base_url, f"workspace/{ANYTHINGLLM_WORKSPACE_SLUG}/")
        openai_client = AsyncOpenAI(base_url=base_url, api_key=ANYTHINGLLM_API_KEY)
    elif AI_PROVIDER not in ["gemini", "vertex"]:
        api_key_map = {"anthropic": ANTHROPIC_API_KEY}
        api_key = api_key_map.get(AI_PROVIDER)
        if not api_key:
            raise ValueError(f"API key for '{AI_PROVIDER}' not found or provider is not supported.")
        openai_client = AsyncOpenAI(api_key=api_key)

    # Configure embedding client
    if EMBEDDING_PROVIDER not in ["gemini", "vertex"]:
        if EMBEDDING_PROVIDER == AI_PROVIDER:
            embedding_client = openai_client
        else:
            logger.info(f"Configuring separate embedding client for provider: {EMBEDDING_PROVIDER}")
            if EMBEDDING_PROVIDER == "openai":
                if not OPENAI_API_KEY:
                    raise ValueError("OPENAI_API_KEY must be set for 'openai' embedding provider.")
                embedding_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            elif EMBEDDING_PROVIDER == "lm_studio":
                embedding_client = AsyncOpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
            elif EMBEDDING_PROVIDER == "anythingllm":
                if not ANYTHINGLLM_BASE_URL or not ANYTHINGLLM_API_KEY:
                    raise ValueError("ANYTHINGLLM_BASE_URL and ANYTHINGLLM_API_KEY must be set for 'anythingllm' embedding provider.")
                base_url = ANYTHINGLLM_BASE_URL
                if ANYTHINGLLM_WORKSPACE_SLUG:
                    if not base_url.endswith('/'): base_url += '/'
                    base_url = urljoin(base_url, f"workspace/{ANYTHINGLLM_WORKSPACE_SLUG}/")
                embedding_client = AsyncOpenAI(base_url=base_url, api_key=ANYTHINGLLM_API_KEY)

# ==========================================
# AI Analysis & Embedding Functions
# ==========================================
@retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(10), retry=retry_if_exception_type(Exception))
async def analyze_with_llm_for_cleaning(chunk_data: Dict, semaphore: asyncio.Semaphore) -> Tuple[AnalysisDetail, int, int]:
    async with semaphore:
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(MY_EMAIL=MY_EMAIL)
        user_content = f"[Overlap Context]\n{chunk_data.get('overlap_context', '')}\n\n[Main Content]\n{chunk_data.get('main_content', '')}"
        in_tokens, out_tokens = 0, 0

        if AI_PROVIDER in ["gemini", "vertex"]:
            if not gemini_client:
                raise RuntimeError("Gemini client not configured. Check your environment variables.")
            
            response = await gemini_client.aio.models.generate_content(
                model=LLM_MODEL_NAME.replace("models/", ""),
                contents=f"{system_prompt}\n\n{user_content}",
                generation_config=types.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=AnalysisDetail,
                ),
                safety_settings=[
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                ]
            )
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                in_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                out_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            try:
                cleaned_text = response.text.strip()
                if cleaned_text.startswith("```"):
                    cleaned_text = cleaned_text.strip("`").removeprefix("json").strip()
                
                return AnalysisDetail.model_validate_json(cleaned_text), in_tokens, out_tokens
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse Gemini response: {response.text}. Error: {e}")
                raise
        else:
            response = await openai_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=512
            )
            if response.usage:
                in_tokens = response.usage.prompt_tokens
                out_tokens = response.usage.completion_tokens
            
            analysis = AnalysisDetail.model_validate_json(response.choices[0].message.content)
            return analysis, in_tokens, out_tokens

@retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(10), retry=retry_if_exception_type(Exception))
async def analyze_batch_with_llm_for_cleaning(batch_payload: List[Dict], semaphore: asyncio.Semaphore) -> Tuple[MiniBatchAnalysisResult, int, int]:
    async with semaphore:
        system_prompt = MINI_BATCH_SYSTEM_PROMPT_TEMPLATE.format(MY_EMAIL=MY_EMAIL)
        user_content = json.dumps(batch_payload, ensure_ascii=False)

        in_tokens, out_tokens = 0, 0

        if AI_PROVIDER in ["gemini", "vertex"]:
            if not gemini_client:
                raise RuntimeError("Gemini client not configured. Check your environment variables.")

            response = await gemini_client.aio.models.generate_content(
                model=LLM_MODEL_NAME.replace("models/", ""),
                contents=f"{system_prompt}\n\n{user_content}",
                generation_config=types.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=MiniBatchAnalysisResult,
                    max_output_tokens=8192,
                    temperature=0.2,
                ),
                safety_settings=[
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
                ]
            )

            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                in_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                out_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)

            try:
                cleaned_text = response.text.strip()
                if cleaned_text.startswith("```"):
                    cleaned_text = cleaned_text.strip("`").removeprefix("json").strip()
                return MiniBatchAnalysisResult.model_validate_json(cleaned_text), in_tokens, out_tokens
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse Gemini batch response: {response.text}. Error: {e}")
                raise
        else:
            response = await openai_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=4096
            )
            if response.usage:
                in_tokens = response.usage.prompt_tokens
                out_tokens = response.usage.completion_tokens
            
            analysis = MiniBatchAnalysisResult.model_validate_json(response.choices[0].message.content)
            return analysis, in_tokens, out_tokens

@retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(10), retry=retry_if_exception_type(Exception))
async def get_embeddings(texts: List[str], model: str, semaphore: asyncio.Semaphore) -> List[List[float]]:
    if not texts:
        return []
    
    async with semaphore:
        if EMBEDDING_PROVIDER in ["gemini", "vertex"]:
            if not gemini_client:
                raise RuntimeError("Gemini client not configured. Check your environment variables.")
            
            response = await gemini_client.aio.models.embed_content(
                model=model.replace("models/", ""),
                contents=texts,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return [e.values for e in response.embeddings]
        else:
            if not embedding_client:
                raise RuntimeError(f"Embedding client for provider '{EMBEDDING_PROVIDER}' is not configured. Check your .env file and provider settings.")

            model_to_use = "" if EMBEDDING_PROVIDER == "anythingllm" else model
            response = await embedding_client.embeddings.create(input=texts, model=model_to_use)
            return [data.embedding for data in response.data]

async def get_existing_chunk_hashes(file_name: str, chunk_indices: List[int], db_connection: Optional[asyncpg.Connection], chroma_collection: Optional[chromadb.api.models.Collection.Collection]) -> Dict[int, str]:
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
        ids_to_check = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_name}-{idx}")) for idx in chunk_indices]
        results = chroma_collection.get(ids=ids_to_check, include=['metadatas'])
        if results and results['ids']:
            for i, _id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                if metadata and metadata.get('file_name') == file_name:
                    existing_hashes[metadata['chunk_index']] = metadata['content_hash']
    return existing_hashes

async def _insert_chunks_to_db(chunks_batch: List[Dict[str, Any]], db_connection: Optional[asyncpg.Connection], chroma_collection: Optional[chromadb.api.models.Collection.Collection], semaphore: asyncio.Semaphore, metrics: Dict[str, Any], is_recovery: bool = False):
    if not chunks_batch:
        return

    texts_to_embed = [c["raw_content"] for c in chunks_batch]
    embeddings = await get_embeddings(texts_to_embed, EMBEDDING_MODEL_NAME, semaphore)

    if not embeddings:
        logger.warning("No embeddings were generated for the current batch.")
        return
        
    def validate_iso8601(ts_str):
        if not ts_str: return None
        try:
            datetime.fromisoformat(str(ts_str).replace('Z', '+00:00'))
            return str(ts_str)
        except ValueError:
            return None
    
    if VECTOR_DB_TYPE == "postgres":
        if not db_connection:
            logger.error("Postgres connection is missing. Cannot insert data.")
            return
            
        query = f"""
            INSERT INTO {COLLECTION_NAME} (
                file_name, group_title, chunk_index, raw_content, content_hash,
                ai_summary, content_category, style_tags, information_weight,
                needs_human_review, human_review_reason, embedding,
                original_time, speaker, data_source
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                $13::text::timestamptz, $14, $15
            )
        """
        records = [
            (
                chunk.get("file_name", chunk.get("group_title", "unknown")),
                chunk.get("group_title"),
                chunk.get("chunk_index"),
                chunk.get("raw_content"),
                chunk.get("content_hash") or hashlib.sha256(str(chunk.get("raw_content", "")).encode('utf-8')).hexdigest(),
                chunk.get("ai_summary"),
                chunk.get("content_category"),
                chunk.get("style_tags"),
                chunk.get("information_weight"),
                chunk.get("needs_human_review"),
                chunk.get("human_review_reason"),
                embeddings[i],
                validate_iso8601(chunk.get("original_time")),
                chunk.get("speaker"),
                chunk.get("data_source")
            ) for i, chunk in enumerate(chunks_batch)
        ]
        await db_connection.executemany(query, records)
        
    elif VECTOR_DB_TYPE == "chromadb":
        if not chroma_collection:
            logger.error("ChromaDB collection is missing. Cannot insert data.")
            return
            
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks_batch):
            group_title = str(chunk.get("group_title", "unknown"))
            chunk_idx = chunk.get("chunk_index", 0)
            content_hash = chunk.get("content_hash") or hashlib.sha256(str(chunk.get("raw_content", "")).encode('utf-8')).hexdigest()
            
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{group_title}-{chunk_idx}-{content_hash}"))
            ids.append(chunk_id)
            documents.append(chunk.get("raw_content", ""))
            
            metadata = {
                "file_name": str(chunk.get("file_name", chunk.get("group_title", "unknown"))),
                "group_title": group_title,
                "chunk_index": chunk_idx,
                "content_hash": content_hash,
                "ai_summary": str(chunk.get("ai_summary", "")),
                "content_category": str(chunk.get("content_category", "")),
                "style_tags": ",".join(chunk.get("style_tags", [])),
                "information_weight": float(chunk.get("information_weight", 0.0)),
                "needs_human_review": str(chunk.get("needs_human_review", False)).lower(),
                "human_review_reason": str(chunk.get("human_review_reason", "")),
                "original_time": str(chunk.get("original_time", "")),
                "speaker": str(chunk.get("speaker", "")),
                "data_source": str(chunk.get("data_source", ""))
            }
            metadatas.append(metadata)
            
        chroma_collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

async def parse_rag_ready_json(json_data: Dict[str, Any], file_name: str) -> List[Dict[str, Any]]:
    documents = []
    def standardize_metadata(raw_metadata: Dict[str, Any], default_source: str = "unknown") -> Dict[str, Any]:
        standardized = {}
        if "timestamp" in raw_metadata:
            standardized["original_time"] = raw_metadata.pop("timestamp")
        if "source" in raw_metadata:
            standardized["data_source"] = raw_metadata.pop("source")
        elif "data_source" not in standardized:
            standardized["data_source"] = default_source
        standardized.update(raw_metadata)
        return standardized

    if "corpus" in json_data and isinstance(json_data["corpus"], list):
        for item in json_data["corpus"]:
            if isinstance(item, dict) and "page_content" in item:
                documents.append({
                    "page_content": item["page_content"],
                    "metadata": standardize_metadata(item.get("metadata", {}), default_source="corpus_file")
                })
        return documents

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
                    final_metadata = {**thread_metadata, **msg_metadata, "thread_id": thread_id, "message_id": message.get("message_id", str(uuid.uuid4()))}
                    documents.append({"page_content": msg_page_content, "metadata": final_metadata})
        return documents

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
                    final_metadata = {**thread_metadata, **msg_metadata, "thread_id": thread_id, "subject": subject, "message_id": message.get("message_id", str(uuid.uuid4()))}
                    documents.append({"page_content": msg_page_content, "metadata": final_metadata})
        return documents

    elif "posts" in json_data and isinstance(json_data["posts"], list):
        for post in json_data["posts"]:
            if not isinstance(post, dict): continue
            post_id = post.get("post_id", str(uuid.uuid4()))
            post_page_content = post.get("page_content")
            post_metadata = standardize_metadata(post.get("metadata", {}), default_source="meta_post")

            if post_page_content:
                final_metadata = {**post_metadata, "post_id": post_id}
                documents.append({"page_content": post_page_content, "metadata": final_metadata})
            
            if "comments" in post and isinstance(post["comments"], list):
                for comment in post["comments"]:
                    if not isinstance(comment, dict): continue
                    comment_page_content = comment.get("page_content")
                    if not comment_page_content: continue
                    comment_metadata = standardize_metadata(comment.get("metadata", {}), default_source=post_metadata.get("data_source", "meta_comment"))
                    final_metadata = {**post_metadata, **comment_metadata, "post_id": post_id, "comment_id": comment.get("comment_id", str(uuid.uuid4()))}
                    documents.append({"page_content": comment_page_content, "metadata": final_metadata})
        return documents
    
    elif isinstance(json_data, list):
        for item in json_data:
            if isinstance(item, dict) and "page_content" in item:
                documents.append({"page_content": item["page_content"], "metadata": standardize_metadata(item.get("metadata", {}), default_source="unknown_list")})
        return documents
    elif isinstance(json_data, dict) and "page_content" in json_data:
        documents.append({"page_content": json_data["page_content"], "metadata": standardize_metadata(json_data.get("metadata", {}), default_source="unknown_single")})
        return documents
    else:
        logger.warning(f"File {file_name} has an unrecognized JSON structure. No documents extracted.")

    return documents

def create_sliding_window_chunks(text: str, group_id: str, max_tokens: int = 2500, overlap_tokens: int = 500) -> List[Dict]:
    chunks = []
    if not tiktoken or not TIKTOKEN_ENCODING:
        max_chars = max_tokens * 4
        overlap_chars = overlap_tokens * 4
        start_idx = 0
        while start_idx < len(text):
            end_idx = min(start_idx + max_chars, len(text))
            main_content = text[start_idx:end_idx]
            overlap_content = ""
            if start_idx > 0:
                overlap_start = max(0, start_idx - overlap_chars)
                overlap_content = text[overlap_start:start_idx]
            chunks.append({"group_id": group_id, "main_content": main_content, "overlap_context": overlap_content, "tokens_count": len(main_content) // 4})
            start_idx += max_chars
        return chunks
        
    tokens = TIKTOKEN_ENCODING.encode(text)
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        main_tokens = tokens[start_idx:end_idx]
        main_content = TIKTOKEN_ENCODING.decode(main_tokens)
        overlap_content = ""
        if start_idx > 0:
            overlap_start = max(0, start_idx - overlap_tokens)
            overlap_tokens_list = tokens[overlap_start:start_idx]
            overlap_content = TIKTOKEN_ENCODING.decode(overlap_tokens_list)
        chunks.append({"group_id": group_id, "main_content": main_content, "overlap_context": overlap_content, "tokens_count": len(main_tokens)})
        start_idx += max_tokens
    return chunks

async def process_single_cache(file_path: str, semaphore: asyncio.Semaphore, metrics: Dict, cancel_event: asyncio.Event = None) -> Tuple[int, int]:
    if cancel_event and cancel_event.is_set():
        return 0, 0
        
    processing_path = file_path.replace(".pending", ".processing")
    done_path = processing_path.replace(".processing", ".done")
    error_path = processing_path.replace(".processing", ".error")
    
    try:
        os.rename(file_path, processing_path)
    except OSError as e:
        logger.warning(f"Failed to rename {file_path} to .processing: {e}")
        metrics["processing_errors"] += 1
        return 0, 0
        
    try:
        with open(processing_path, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)
            
        analysis, in_tokens, out_tokens = await analyze_with_llm_for_cleaning(chunk_data, semaphore)
        
        is_invalid = analysis.is_ad or analysis.is_noise
        
        if not is_invalid:
            out_filename = os.path.basename(processing_path).replace(".json.processing", ".json")
            out_filepath = os.path.join(PROCESS_CLEANED_DIR, out_filename)
            
            match = re.search(r"_chunk_(\d+)\.json", out_filename)
            chunk_index = int(match.group(1)) if match else 0
            
            final_output = {
                "file_name": chunk_data.get("file_name", chunk_data.get("group_id", "unknown")),
                "group_title": chunk_data.get("group_id"),
                "chunk_index": chunk_index,
                "raw_content": chunk_data.get("main_content", ""),
                "content_hash": hashlib.sha256(str(chunk_data.get("main_content", "")).encode('utf-8')).hexdigest(),
                "ai_summary": analysis.ai_summary,
                "content_category": analysis.content_category,
                "style_tags": analysis.style_tags,
                "information_weight": analysis.information_weight,
                "original_time": analysis.original_time,
                "speaker": analysis.speaker,
                "data_source": analysis.data_source,
                "needs_human_review": analysis.needs_human_review,
                "human_review_reason": analysis.human_review_reason
            }
            with open(out_filepath, "w", encoding="utf-8") as out_f:
                json.dump(final_output, out_f, ensure_ascii=False, indent=2)
            metrics["successful_chunks"] += 1
        else:
            metrics["noise_filtered"] += 1
            
        os.rename(processing_path, done_path)
        
        if AI_PROVIDER in ["gemini", "vertex"]:
            await asyncio.sleep(5)
            
        return in_tokens, out_tokens
    except Exception as e:
        if isinstance(e, RetryError) and e.last_attempt:
            e = f"Max retries reached. Underlying error: {repr(e.last_attempt.exception())}"
        logger.error(f"Error processing {processing_path}: {e}")
        metrics["processing_errors"] += 1
        try:
            os.rename(processing_path, error_path)
        except OSError:
            pass
        return 0, 0
        
async def process_batch_cache(file_paths: List[str], semaphore: asyncio.Semaphore, metrics: Dict, cancel_event: asyncio.Event = None) -> Tuple[int, int]:
    if cancel_event and cancel_event.is_set():
        return 0, 0
        
    batch_payload = []
    chunk_data_map = {}
    
    for file_path in file_paths:
        processing_path = file_path.replace(".pending", ".processing")
        try:
            os.rename(file_path, processing_path)
        except OSError as e:
            logger.warning(f"Failed to rename {file_path} to .processing: {e}")
            metrics["processing_errors"] += 1
            continue
            
        try:
            with open(processing_path, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)
                
            chunk_id = os.path.basename(processing_path)
            chunk_data_map[chunk_id] = {"path": processing_path, "data": chunk_data}
            
            batch_payload.append({
                "chunk_id": chunk_id,
                "main_content": chunk_data.get("main_content", ""),
                "overlap_context": chunk_data.get("overlap_context", "")
            })
        except Exception as e:
            logger.error(f"Failed to read {processing_path}: {e}")
            metrics["processing_errors"] += 1
            os.rename(processing_path, processing_path.replace(".processing", ".error"))

    if not batch_payload:
        return 0, 0
        
    try:
        batch_result, in_tokens, out_tokens = await analyze_batch_with_llm_for_cleaning(batch_payload, semaphore)
        
        results_dict = {res.chunk_id: res.analysis for res in batch_result.results}
        
        for chunk_id, info in chunk_data_map.items():
            processing_path = info["path"]
            chunk_data = info["data"]
            done_path = processing_path.replace(".processing", ".done")
            error_path = processing_path.replace(".processing", ".error")
            
            analysis = results_dict.get(chunk_id)
            
            if analysis:
                is_invalid = analysis.is_ad or analysis.is_noise
                if not is_invalid:
                    out_filename = chunk_id.replace(".json.processing", ".json")
                    out_filepath = os.path.join(LLM_CLEANED_DIR, out_filename)
                    
                    match = re.search(r"_chunk_(\d+)\.json", out_filename)
                    chunk_index = int(match.group(1)) if match else 0
                    
                    final_output = {
                        "file_name": chunk_data.get("file_name", chunk_data.get("group_id", "unknown")),
                        "group_title": chunk_data.get("group_id"),
                        "chunk_index": chunk_index,
                        "raw_content": chunk_data.get("main_content", ""),
                        "content_hash": hashlib.sha256(str(chunk_data.get("main_content", "")).encode('utf-8')).hexdigest(),
                        "ai_summary": analysis.ai_summary,
                        "content_category": analysis.content_category,
                        "style_tags": analysis.style_tags,
                        "information_weight": analysis.information_weight,
                        "original_time": analysis.original_time,
                        "speaker": analysis.speaker,
                        "data_source": analysis.data_source,
                        "needs_human_review": analysis.needs_human_review,
                        "human_review_reason": analysis.human_review_reason
                    }
                    with open(out_filepath, "w", encoding="utf-8") as out_f:
                        json.dump(final_output, out_f, ensure_ascii=False, indent=2)
                    metrics["successful_chunks"] += 1
                else:
                    metrics["noise_filtered"] += 1
                
                os.rename(processing_path, done_path)
            else:
                logger.warning(f"Analysis missing for {chunk_id} in batch response. Marking as error.")
                metrics["processing_errors"] += 1
                os.rename(processing_path, error_path)
                
        return in_tokens, out_tokens
        
    except Exception as e:
        logger.error(f"Error processing batch of size {len(batch_payload)}: {e}")
        metrics["processing_errors"] += len(batch_payload)
        for info in chunk_data_map.values():
            try:
                os.rename(info["path"], info["path"].replace(".processing", ".error"))
            except OSError:
                pass
        return 0, 0

async def ingest_cleaned_files(file_paths: List[str], semaphore: asyncio.Semaphore, metrics: Dict[str, Any]):
    if not file_paths:
        return

    logger.info(f"Found {len(file_paths)} cleaned files to ingest into the database.")
    metrics["chunks_to_ingest"] = len(file_paths)

    db_conn = None
    chroma_coll = None
    try:
        if VECTOR_DB_TYPE == "postgres":
            db_conn = await asyncpg.connect(host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, database=PG_DB_NAME)
            await register_vector(db_conn)
        elif VECTOR_DB_TYPE == "chromadb":
            chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            chroma_coll = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

        chunks_with_paths = []
        for cf_path in file_paths:
            try:
                with open(cf_path, "r", encoding="utf-8") as f:
                    chunks_with_paths.append({"data": json.load(f), "path": cf_path})
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Could not read or parse {cf_path}, skipping: {e}")
                metrics["ingestion_errors"] += 1

        archive_run_dir = os.path.join(LLM_CLEANED_DIR, "archive", RUN_TIME_STR)
        os.makedirs(archive_run_dir, exist_ok=True)

        for i in tqdm(range(0, len(chunks_with_paths), BATCH_SIZE), desc="Ingesting to DB"):
            batch_with_paths = chunks_with_paths[i : i + BATCH_SIZE]
            batch_chunks = [item["data"] for item in batch_with_paths]
            batch_paths = [item["path"] for item in batch_with_paths]

            try:
                await _insert_chunks_to_db(batch_chunks, db_conn, chroma_coll, semaphore, metrics, is_recovery=True)
                metrics["ingested_chunks"] += len(batch_chunks)

                for path in batch_paths:
                    try:
                        archive_path = os.path.join(archive_run_dir, os.path.basename(path))
                        os.rename(path, archive_path)
                    except OSError as e:
                        logger.error(f"Failed to archive {os.path.basename(path)}: {e}")
                        metrics["archive_errors"] += 1

            except Exception as e:
                logger.error(f"Error inserting batch into DB: {e}", exc_info=True)
                metrics["ingestion_errors"] += len(batch_chunks)

    except Exception as e:
        logger.critical(f"A critical error occurred during the ingestion process: {e}", exc_info=True)
    finally:
        if db_conn:
            await db_conn.close()

async def handle_existing_cleaned_files(metrics: Dict):
    cleaned_files = [f for f in glob.glob(os.path.join(LLM_CLEANED_DIR, "*.json")) if os.path.isfile(f)]
    if not cleaned_files:
        return

    logger.info(f"Found {len(cleaned_files)} unprocessed files in {LLM_CLEANED_DIR}.")

    if sys.stdin.isatty():
        try:
            choice = input("Do you want to import these files into the database now? (y/n) [Default: n]: ").strip().lower()
        except EOFError:
            choice = 'n'
    else:
        logger.info("Non-interactive environment. Skipping import of existing cleaned files.")
        choice = 'n'

    if choice == 'y':
        logger.info("Starting import process for existing cleaned files...")
        semaphore = asyncio.Semaphore(5)
        await ingest_cleaned_files(cleaned_files, semaphore, metrics)
        logger.info("Import of existing files complete. The program will now exit.")
        sys.exit(0)
    else:
        logger.warning("Existing cleaned files will not be imported.")
        if sys.stdin.isatty():
            try:
                clear_choice = input(f"Do you want to DELETE these {len(cleaned_files)} files and continue? (y/n) [Default: n]: ").strip().lower()
            except EOFError:
                clear_choice = 'n'
        else:
            clear_choice = 'n'

        if clear_choice == 'y':
            logger.info(f"User confirmed. Deleting {len(cleaned_files)} files...")
            for f in cleaned_files:
                try:
                    os.remove(f)
                except OSError as e:
                    logger.error(f"Failed to delete file {f}: {e}")
        else:
            logger.info("Aborting pipeline to allow manual handling of existing cleaned files.")
            sys.exit(0)

async def run_batch_pipeline(metrics: Dict):
    """Prepares and submits a Gemini Batch API job."""
    if not gemini_client:
        logger.error("Gemini client is not configured. Cannot run batch pipeline.")
        return

    pending_files = glob.glob(os.path.join(PROCESS_CACHE_DIR, "*.pending"))
    if not pending_files:
        logger.info("No pending chunk files found to process in batch mode.")
        return

    if os.path.exists(BATCH_JOB_FILE):
        with open(BATCH_JOB_FILE, "r") as f:
            active_job_name = f.read().strip()
        logger.error(f"An active batch job seems to exist: {active_job_name}")
        logger.error(f"Please check its status with '--mode import_batch_results --job-name {active_job_name}' before starting a new one.")
        sys.exit(1)

    logger.info(f"Found {len(pending_files)} pending chunks. Preparing for Gemini Batch API.")

    batch_input_path = os.path.join(PROC_DIR, f"gemini_batch_input_{RUN_TIME_STR}.jsonl")
    manifest_data = {}
    ordered_chunk_keys = []

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(MY_EMAIL=MY_EMAIL)
    # Create a plain dictionary for JSON serialization in the JSONL batch file
    generation_config_dict = {
        "response_mime_type": "application/json",
        "response_schema": AnalysisDetail.model_json_schema(),
        "temperature": 0.2,
        "max_output_tokens": 512,
    }

    try:
        with open(batch_input_path, "w", encoding="utf-8") as f_out:
            for file_path in tqdm(pending_files, desc="Preparing Batch Input"):
                chunk_filename = os.path.basename(file_path)
                try:
                    with open(file_path, "r", encoding="utf-8") as f_in:
                        chunk_data = json.load(f_in)

                    user_content = f"[Overlap Context]\n{chunk_data.get('overlap_context', '')}\n\n[Main Content]\n{chunk_data.get('main_content', '')}"

                    if AI_PROVIDER_ENV == "vertex":
                        # Vertex AI expects each line to be a prediction instance.
                        # System instructions are part of the instance for Gemini models on Vertex.
                        request_payload = {
                            "contents": [{"role": "user", "parts": [{"text": user_content}]}],
                            "system_instruction": {"parts": [{"text": system_prompt}]},
                        }
                    else:
                        # Google AI Studio (Gemini) expects a key/request structure.
                        request_payload = {
                            "key": chunk_filename,
                            "request": {
                                "contents": [{"role": "user", "parts": [{"text": user_content}]}],
                                "system_instruction": {"parts": [{"text": system_prompt}]},
                            "generation_config": generation_config_dict,
                            }
                        }
                    
                    f_out.write(json.dumps(request_payload, ensure_ascii=False) + "\n")
                    manifest_data[chunk_filename] = {"path": file_path, "data": chunk_data}
                    ordered_chunk_keys.append(chunk_filename)
                    metrics["chunks_prepared_for_batch"] += 1

                except Exception as e:
                    logger.error(f"Failed to process chunk {chunk_filename} for batch: {e}")
                    metrics["processing_errors"] += 1

        logger.info(f"Batch input file created at: {batch_input_path}")

        if AI_PROVIDER_ENV == "vertex":
            if not GCS_BUCKET_NAME:
                logger.error("GCS_BUCKET_NAME environment variable must be set for Vertex AI batch processing.")
                sys.exit(1)
            if not storage:
                logger.error("google-cloud-storage is required for Vertex AI. Please install it.")
                sys.exit(1)
                
            logger.info("Uploading batch input file to Google Cloud Storage (Vertex AI mode)...")
            storage_client = storage.Client(project=VERTEX_PROJECT_ID)
            bucket = storage_client.bucket(GCS_BUCKET_NAME)
            gcs_blob_name = f"batch_inputs/rag_ingestion_batch_{RUN_TIME_STR}.jsonl"
            blob = bucket.blob(gcs_blob_name)
            blob.upload_from_filename(batch_input_path)
            
            input_uri = f"gs://{GCS_BUCKET_NAME}/{gcs_blob_name}"
            output_uri = f"gs://{GCS_BUCKET_NAME}/batch_outputs/job_{RUN_TIME_STR}/"
            logger.info(f"File uploaded successfully to GCS: {input_uri}")

            # For Vertex, generation config is passed as model parameters in a plain dict.
            batch_config = {
                'dest': output_uri,
                'model_parameters': generation_config_dict
            }

            logger.info(f"Creating batch job with model: {BATCH_MODEL_NAME}...")
            batch_job = gemini_client.batches.create(
                model=BATCH_MODEL_NAME.replace("models/", ""),
                src=input_uri,
                config=batch_config
            )
        else:
            # Upload the file to Gemini File API
            logger.info("Uploading batch input file to Gemini File API...")
            uploaded_file = gemini_client.files.upload(file=batch_input_path, config={'display_name': f"rag-ingestion-batch-{RUN_TIME_STR}"})
            logger.info(f"File uploaded successfully: {uploaded_file.name} ({uploaded_file.display_name})")

            # Create the batch job
            logger.info(f"Creating batch job with model: {BATCH_MODEL_NAME}...")
            batch_job = gemini_client.batches.create(
                model=f"models/{BATCH_MODEL_NAME}",
                requests=uploaded_file.name
            )

        logger.info(f"Batch job created successfully! Job Name: {batch_job.name}")
        logger.info("You can monitor the job status and import results later using:")
        logger.info(f"python {os.path.basename(__file__)} --mode import_batch_results --job-name {batch_job.name}")

        # Persist job name and manifest, and update file states
        with open(BATCH_JOB_FILE, "w") as f:
            f.write(batch_job.name)
        
        full_manifest = {
            "map": manifest_data,
            "order": ordered_chunk_keys
        }
        with open(BATCH_MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(full_manifest, f, ensure_ascii=False, indent=2)

        for key, info in manifest_data.items():
            os.rename(info["path"], info["path"].replace(".pending", ".processing"))

    except Exception as e:
        logger.critical(f"A critical error occurred during batch job submission: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up the local JSONL file after upload
        if os.path.exists(batch_input_path):
            os.remove(batch_input_path)

async def import_batch_results(job_name: str, metrics: Dict, semaphore: asyncio.Semaphore):
    """Polls a Gemini Batch API job and imports the results."""
    if not gemini_client:
        logger.error("Gemini client is not configured. Cannot import batch results.")
        return

    logger.info(f"Starting to import results for batch job: {job_name}")
    
    try:
        # Poll for job completion
        job = gemini_client.batches.get(name=job_name)
        while job.state in {types.JobState.JOB_STATE_PENDING, types.JobState.JOB_STATE_RUNNING}:
            logger.info(f"Job '{job.name}' is currently in state: {job.state.name}. Waiting 60 seconds...")
            time.sleep(60)
            job = gemini_client.batches.get(name=job_name)

        if job.state != types.JobState.JOB_STATE_SUCCEEDED:
            logger.error(f"Job '{job.name}' did not succeed. Final state: {job.state.name}")
            logger.error(f"Error details: {job.error}")
            # TODO: Handle renaming .processing files to .error based on manifest
            sys.exit(1)

        if AI_PROVIDER_ENV == "vertex":
            output_uri = job.config['dest']
            logger.info(f"Job '{job.name}' succeeded! Downloading results from GCS: {output_uri}...")
            storage_client = storage.Client(project=VERTEX_PROJECT_ID)
            # 解析出 bucket 與 prefix
            bucket_name = output_uri.replace("gs://", "").split("/")[0]
            prefix = "/".join(output_uri.replace("gs://", "").split("/")[1:])
            bucket = storage_client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=prefix))
            
            result_file_content = ""
            for blob in blobs:
                if blob.name.endswith(".jsonl"):
                    result_file_content += blob.download_as_text() + "\n"
        else:
            logger.info(f"Job '{job.name}' succeeded! Downloading results from {job.dest.file_name}...")
            result_file_content = gemini_client.files.download(name=job.dest.file_name).content.decode("utf-8")
        
        logger.info("Loading manifest to map results back to original chunks...")
        with open(BATCH_MANIFEST_FILE, "r", encoding="utf-8") as f:
            full_manifest = json.load(f)
            manifest_data = full_manifest.get("map", full_manifest) # Handle old and new manifest format
            ordered_chunk_keys = full_manifest.get("order", [])

        cleaned_chunks_for_db = []
        result_lines = result_file_content.strip().split("\n")
        for i, line in enumerate(tqdm(result_lines, desc="Processing Batch Results")):
            if not line.strip(): continue
            result = json.loads(line)

            analysis_data = None
            chunk_key = None

            if AI_PROVIDER_ENV == "vertex":
                if i < len(ordered_chunk_keys):
                    chunk_key = ordered_chunk_keys[i]
                    # Vertex output is typically {"instance":..., "prediction":...}
                    analysis_data = result.get("prediction")
                else:
                    logger.warning(f"Result line {i+1} is out of bounds for the manifest's ordered keys. Skipping.")
            else: # Gemini API
                chunk_key = result.get("key")
                analysis_data = result.get("response")

            if chunk_key not in manifest_data:
                logger.warning(f"Received result for unknown key '{chunk_key}'. Skipping.")
                continue

            chunk_info = manifest_data[chunk_key]
            chunk_data = chunk_info["data"]
            try:
                analysis = AnalysisDetail.model_validate(analysis_data)
                # ... (rest of the logic from process_batch_cache)
                is_invalid = analysis.is_ad or analysis.is_noise
                if not is_invalid:
                    # Create the final output structure and save to LLM_CLEANED_DIR
                    out_filename = chunk_key.replace(".json.pending", ".json")
                    out_filepath = os.path.join(PROCESS_CLEANED_DIR, out_filename)
                    
                    match = re.search(r"_chunk_(\d+)\.json", out_filename)
                    chunk_index = int(match.group(1)) if match else 0
                    
                    final_output = {
                        "file_name": chunk_data.get("file_name", chunk_data.get("group_id", "unknown")),
                        "group_title": chunk_data.get("group_id"),
                        "chunk_index": chunk_index,
                        "raw_content": chunk_data.get("main_content", ""),
                        "content_hash": hashlib.sha256(str(chunk_data.get("main_content", "")).encode('utf-8')).hexdigest(),
                        **analysis.model_dump()
                    }
                    with open(out_filepath, "w", encoding="utf-8") as out_f:
                        json.dump(final_output, out_f, ensure_ascii=False, indent=2)
                    metrics["successful_chunks"] += 1
                else:
                    metrics["noise_filtered"] += 1
                
                os.rename(chunk_info["path"].replace(".pending", ".processing"), chunk_info["path"].replace(".pending", ".done"))
            except (ValidationError, KeyError) as e:
                logger.error(f"Failed to parse or validate result for key '{chunk_key}': {e}")
                logger.error(f"Problematic result data: {analysis_data}")
                metrics["processing_errors"] += 1
                os.rename(chunk_info["path"].replace(".pending", ".processing"), chunk_info["path"].replace(".pending", ".error"))

        # Ingest all newly created cleaned files
        cleaned_files = [f for f in glob.glob(os.path.join(LLM_CLEANED_DIR, "*.json")) if os.path.isfile(f)]
        await ingest_cleaned_files(cleaned_files, semaphore, metrics)

    except Exception as e:
        logger.critical(f"A critical error occurred during batch result import: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up job tracking files
        for f in [BATCH_JOB_FILE, BATCH_MANIFEST_FILE]:
            if os.path.exists(f):
                os.remove(f)
        logger.info("Batch import process finished. Cleaned up tracking files.")

async def process_and_ingest_data():
    start_time = datetime.now()
    logger.info("Process Started: RAG data ingestion pipeline...")

    metrics = defaultdict(int)

    import argparse
    parser = argparse.ArgumentParser(description="RAG Ingestion Pipeline")
    parser.add_argument("--mode", type=str, default="ingest", choices=["ingest", "import_batch_results"], 
                        help="Execution mode: 'ingest' to process files, 'import_batch_results' for Gemini Batch API.")
    parser.add_argument("--job-name", type=str, help="The Gemini Batch API job name to import results from.")
    args, _ = parser.parse_known_args()

    semaphore = asyncio.Semaphore(5) # Used for DB ingestion

    for d in [DATA_DIR, PROCESS_CACHE_DIR, PROCESS_CLEANED_DIR, PROCESS_ERROR_DIR, PROC_DIR, os.path.join(LLM_CLEANED_DIR, "archive")]:
        os.makedirs(d, exist_ok=True)

    if USE_BATCH_API:
        if AI_PROVIDER not in ["gemini", "vertex"]:
            logger.error(f"Batch API mode is only supported for 'gemini' or 'vertex' providers. Current provider: {AI_PROVIDER}")
            sys.exit(1)
        
        logger.info("Gemini Batch API mode is enabled.")

        if args.mode == "import_batch_results":
            job_name = args.job_name
            if not job_name:
                if os.path.exists(BATCH_JOB_FILE):
                    with open(BATCH_JOB_FILE, "r") as f:
                        job_name = f.read().strip()
                else:
                    logger.error("Must provide --job-name or have an active job file to import results.")
                    sys.exit(1)
            await import_batch_results(job_name, metrics, semaphore)
        elif args.mode == "ingest":
            # First, create all the pending chunk files
            await create_pending_chunks(metrics)
            # Then, run the batch pipeline to submit them
            await run_batch_pipeline(metrics)
        return
    else:
        logger.info("Standard (real-time) API mode is enabled.")
        await handle_existing_cleaned_files(metrics)

        if not any(f.endswith('.json') for f in os.listdir(DATA_DIR)):
            logger.warning(f"Data directory {DATA_DIR} is empty. Nothing to do.")
            return

        existing_caches = glob.glob(os.path.join(PROCESS_CACHE_DIR, "*"))
        if existing_caches:
            if sys.stdin.isatty():
                try:
                    choice = input(f"Found {len(existing_caches)} cache files. Continue or restart? (y/n) [y]: ").lower()
                    if choice == 'n':
                        for f in existing_caches: os.remove(f)
                except EOFError: pass
        
        orphaned_files = glob.glob(os.path.join(PROCESS_CACHE_DIR, "*.processing")) + glob.glob(os.path.join(PROCESS_CACHE_DIR, "*.error"))
        if orphaned_files:
            logger.info(f"Reverting {len(orphaned_files)} orphaned files to .pending...")
            for f in orphaned_files:
                os.rename(f, f.replace(".processing", ".pending").replace(".error", ".pending"))

        await create_pending_chunks(metrics)

        if not TEST_MODE:
            pending_files = glob.glob(os.path.join(PROCESS_CACHE_DIR, "*.pending"))
            if pending_files:
                logger.info(f"Found {len(pending_files)} pending chunks. Starting LLM analysis...")
                tasks = []
                cancel_event = asyncio.Event()
                # Group files into batches for process_batch_cache
                for i in range(0, len(pending_files), API_BATCH_SIZE):
                    batch_files = pending_files[i:i + API_BATCH_SIZE]
                    tasks.append(process_batch_cache(batch_files, semaphore, metrics, cancel_event))
                
                # Use async_tqdm to show progress for the async tasks
                results = await async_tqdm.gather(*tasks, desc="Analyzing Chunks")
                
                for in_tokens, out_tokens in results:
                    metrics["total_input_tokens"] += in_tokens
                    metrics["total_output_tokens"] += out_tokens
                    
            cleaned_files = [f for f in glob.glob(os.path.join(LLM_CLEANED_DIR, "*.json")) if os.path.isfile(f)]
            await ingest_cleaned_files(cleaned_files, semaphore, metrics)
                    
            logger.info("Cleaning up cache files...")
            for ext in ["*.pending", "*.processing", "*.done", "*.error"]:
                for f in glob.glob(os.path.join(PROCESS_CACHE_DIR, ext)):
                    try: os.remove(f)
                    except OSError: pass

    end_time = datetime.now()
    total_time = end_time - start_time
    logger.info("==========================================")
    logger.info("RAG Ingestion Pipeline Completed")
    logger.info("==========================================")
    logger.info(f"Total Execution Time: {total_time}")
    
    if USE_BATCH_API:
        logger.info(f"Mode: BATCH API ({args.mode})")
        logger.info(f"LLM Provider: {AI_PROVIDER.upper()}, Model: {BATCH_MODEL_NAME}")
        if args.mode == "ingest":
            logger.info("---")
            logger.info(f"File Processing:")
            logger.info(f"  - Total Files Read: {metrics['total_files']}")
            logger.info(f"  - Documents Extracted: {metrics['total_documents']}")
            logger.info(f"  - Chunks Generated: {metrics['chunks_created']}")
            logger.info(f"  - Chunks Prepared for Batch: {metrics['chunks_prepared_for_batch']}")
            logger.info(f"  - Processing Errors (pre-batch): {metrics['processing_errors']}")
        elif args.mode == "import_batch_results":
            logger.info("---")
            logger.info(f"LLM Analysis (from Batch):")
            logger.info(f"  - Chunks Successfully Analyzed: {metrics['successful_chunks']}")
            logger.info(f"  - Chunks Filtered (Noise/Ad): {metrics['noise_filtered']}")
            logger.info(f"  - Processing Errors (post-batch): {metrics['processing_errors']}")
            logger.info("---")
            logger.info(f"Database Ingestion ({VECTOR_DB_TYPE.upper()}):")
            logger.info(f"  - Chunks Queued for Ingestion: {metrics['chunks_to_ingest']}")
            logger.info(f"  - Chunks Successfully Ingested: {metrics['ingested_chunks']}")
            logger.info(f"  - Ingestion Errors: {metrics['ingestion_errors']}")
            logger.info(f"  - Archiving Errors: {metrics['archive_errors']}")
    elif TEST_MODE:
        logger.info(f"Mode: TEST RUN")
        logger.info(f"Files Scanned: {metrics['total_files']}")
        logger.info(f"Documents Analyzed: {metrics['total_documents']}")
    else:
        logger.info(f"Mode: Standard (Real-time) Ingestion")
        logger.info(f"LLM Provider: {AI_PROVIDER.upper()}, Model: {LLM_MODEL_NAME}")
        logger.info(f"Embedding Provider: {EMBEDDING_PROVIDER.upper()}, Model: {EMBEDDING_MODEL_NAME}")
        logger.info("---")
        logger.info(f"File Processing:")
        logger.info(f"  - Total Files Read: {metrics['total_files']}")
        logger.info(f"  - Documents Extracted: {metrics['total_documents']}")
        logger.info(f"  - Chunks Generated: {metrics['chunks_created']}")
        logger.info("---")
        logger.info(f"LLM Analysis:")
        logger.info(f"  - Chunks Successfully Analyzed: {metrics['successful_chunks']}")
        logger.info(f"  - Chunks Filtered (Noise/Ad): {metrics['noise_filtered']}")
        logger.info(f"  - Processing Errors: {metrics['processing_errors']}")
        logger.info("---")
        logger.info(f"Token Usage:")
        logger.info(f"  - Input Tokens: {metrics['total_input_tokens']:,}")
        logger.info(f"  - Output Tokens: {metrics['total_output_tokens']:,}")
        logger.info(f"  - Total Tokens: {metrics['total_input_tokens'] + metrics['total_output_tokens']:,}")
        logger.info("---")
        logger.info(f"Database Ingestion ({VECTOR_DB_TYPE.upper()}):")
        logger.info(f"  - Chunks Queued for Ingestion: {metrics['chunks_to_ingest']}")
        logger.info(f"  - Chunks Successfully Ingested: {metrics['ingested_chunks']}")
        logger.info(f"  - Ingestion Errors: {metrics['ingestion_errors']}")
        logger.info(f"  - Archiving Errors: {metrics['archive_errors']}")
    logger.info("==========================================")

async def create_pending_chunks(metrics: Dict):
    """Parses source files and creates .pending chunk files in the cache directory."""
    files_to_process = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    metrics["total_files"] = len(files_to_process)
    logger.info(f"Found {metrics['total_files']} files to process.")

    if not files_to_process:
        return

    for filepath in tqdm(files_to_process, desc="Parsing Source Files"):
        current_file_name = os.path.basename(filepath)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            docs = await parse_rag_ready_json(data, current_file_name)
            metrics["total_documents"] += len(docs)
        except Exception as e:
            logger.error(f"Failed to read/parse {current_file_name}: {e}")
            metrics["file_errors"] += 1
            continue

        if not docs: continue
        
        if TEST_MODE:
            logger.info(f"[TEST MODE] Would create chunks for {len(docs)} documents from {current_file_name}")
            continue

        grouped_docs = defaultdict(list)
        for doc in docs:
            meta = doc.get("metadata", {})
            group_id = meta.get("thread_id") or meta.get("post_id") or meta.get("subject") or current_file_name
            grouped_docs[group_id].append(doc["page_content"])
            
        for group_id, contents in grouped_docs.items():
            full_text = "\n".join(contents)
            chunks = create_sliding_window_chunks(full_text, group_id, max_tokens=2500, overlap_tokens=500)
            
            for i, chunk_data in enumerate(chunks):
                metrics["chunks_created"] += 1
                chunk_data["file_name"] = current_file_name
                safe_group_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(group_id))
                cache_filename = f"{PROCESS_NAME}_{RUN_TIME_STR}_{safe_group_id}_chunk_{i}.json.pending"
                cache_file = os.path.join(PROCESS_CACHE_DIR, cache_filename)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(chunk_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    try:
        asyncio.run(process_and_ingest_data())
    except KeyboardInterrupt:
         logger.info("\nProcess interrupted by user. Exiting.")
         sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected critical error occurred: {e}", exc_info=True)
        sys.exit(1)