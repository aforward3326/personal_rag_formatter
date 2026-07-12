import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# --- Project Root ---
# Define the project root to make all path management robust and absolute.
# This assumes this config.py is at src/rag_pipeline/config.py
PROJECT_ROOT = Path(__file__).parent.parent.parent

# --- Load Environment Variables ---
# Load variables from .env file for local development
# This allows for centralized configuration management.
load_dotenv(dotenv_path=PROJECT_ROOT / "pipeline.env")


# --- Core Paths ---
# All data-related paths are now organized under a single `data` directory.
DATA_ROOT = PROJECT_ROOT / "data"
LOG_ROOT = PROJECT_ROOT / "log" # Or move to data/log if preferred
PROCESSING_ROOT = PROJECT_ROOT / "processing" # Temporary processing files

DATA_DIR = os.getenv("DATA_DIR", str(DATA_ROOT / "input"))
OUTPUT_DIR = DATA_ROOT / "output"
LOG_DIR = os.getenv("LOG_DIR", str(LOG_ROOT))
PROC_DIR = os.getenv("PROC_DIR", str(PROCESSING_ROOT))
CACHE_DIR = os.getenv("CACHE_DIR", str(PROJECT_ROOT / "cache"))
BATCH_PROC_DIR = Path(PROC_DIR) / "batch"
TEMPLATE_DIR = DATA_ROOT / "templates" / "rag_pipeline"

# --- Process-Specific Paths ---
PROCESS_NAME = "rag_ingestion_pipeline"
RUN_TIME_STR = datetime.now().strftime("%Y%m%d%H")

LLM_CLEANED_DIR = OUTPUT_DIR / "cleaned"
PROCESS_CACHE_DIR = os.path.join(CACHE_DIR, PROCESS_NAME)
PROCESS_CLEANED_DIR = os.path.join(LLM_CLEANED_DIR, PROCESS_NAME)
PROCESS_ERROR_DIR = OUTPUT_DIR / "error" / PROCESS_NAME
BATCH_MANIFEST_FILE = os.path.join(PROCESS_CACHE_DIR, "batch_manifest.json")
BATCH_JOB_FILE = os.path.join(PROC_DIR, "active_batch_job.txt")
ARCHIVE_DIR = OUTPUT_DIR / "archive"

# --- AI & Embedding Providers ---
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai").lower()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# --- Vector Database ---
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "postgres").lower()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_knowledge")

# Chroma settings
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", str(PROJECT_ROOT / "chroma_db"))

# Postgres settings
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_DB_NAME = os.getenv("PG_DB_NAME")

# --- Batch Processing ---
USE_BATCH_API = os.getenv("USE_BATCH_API", "false").lower() == "true"
BATCH_MAX_LINES = int(os.getenv("BATCH_MAX_LINES", "50000"))
BATCH_MODEL_NAME = os.getenv("BATCH_MODEL_NAME", "gemini-1.5-flash-001")
API_BATCH_SIZE = int(os.getenv("API_BATCH_SIZE", "10"))
DB_BATCH_SIZE = int(os.getenv("DB_BATCH_SIZE", "100"))
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# --- User, Anonymization & Prompt Settings ---
MY_EMAIL = os.getenv("MY_EMAIL", "your.email@example.com")

# Default JSON strings for identity and sensitive word mapping.
# These are loaded as strings and should be parsed into dictionaries in the application logic.
DEFAULT_IDENTITIES = json.dumps({
    "Your Name": "[AUTHOR_ME]",
    "Your Nickname": "[AUTHOR_ME]",
    "your.email@example.com": "[AUTHOR_ME]"
})
DEFAULT_SENSITIVE_WORDS = json.dumps({
    "某某科技股份有限公司": "[REDACTED_COMPANY]",
    "內部機密專案X": "[REDACTED_PROJECT]",
    "台北市信義區市府路1號": "[REDACTED_ADDRESS]"
})

MY_IDENTITIES_MATRIX = os.getenv("MY_IDENTITIES_MATRIX", DEFAULT_IDENTITIES)
SENSITIVE_WORDS_MAP = os.getenv("SENSITIVE_WORDS_MAP", DEFAULT_SENSITIVE_WORDS)

# --- Test Mode Settings ---
RAG_TEST_MODE = os.getenv("RAG_TEST_MODE", "false").lower() == "true"
RAG_TEST_HIDE_CONTENT = os.getenv("RAG_TEST_HIDE_CONTENT", "false").lower() == "true"


# --- Prompt Templates ---
# Note: The loading logic is now simplified as the path is absolute.
def load_prompt_template(file_path: Path) -> str:
    """Loads a prompt template from the specified file path."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # This will be handled by the logger in the main application
        raise

SYSTEM_PROMPT_TEMPLATE = load_prompt_template(TEMPLATE_DIR / "system_prompt.txt")
MINI_BATCH_SYSTEM_PROMPT_TEMPLATE = load_prompt_template(TEMPLATE_DIR / "mini_batch_system_prompt.txt")
