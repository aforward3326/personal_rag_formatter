from pathlib import Path
import os

# --- Project Root ---
# Assumes this config.py is at src/text_formatting/config.py
PROJECT_ROOT = Path(__file__).parent.parent.parent

# --- Load Environment Variables ---
# Using the same .env file from the project root
from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / "pipeline.env")

# --- Shared Paths ---
# These paths are common across different formatters
DATA_ROOT = PROJECT_ROOT / "data"
RUNTIME_ROOT = PROJECT_ROOT / "_runtime"

# Default input and output directories
ORG_DATA_DIR = DATA_ROOT / "org_data"
OUTPUT_DIR = DATA_ROOT / "output_clean_data"
LOG_DIR = RUNTIME_ROOT / "log"
PROC_DIR = RUNTIME_ROOT / "processing"

# --- Gmail Formatter Settings ---
GMAIL_INPUT_MBOX_DIR = Path(os.getenv("MBOX_DIR_PATH", str(ORG_DATA_DIR / "mail_data")))
GMAIL_OUTPUT_FILE = OUTPUT_DIR / "gmail_corpus_rag_ready.json"
GMAIL_MY_EMAIL = os.getenv("MY_EMAIL", "example@example.com")

# --- Chat HTML Formatter Settings ---
CHAT_INPUT_HTML_DIR = Path(os.getenv("CHAT_HTML_DIR", str(ORG_DATA_DIR / "chat_data")))
CHAT_OUTPUT_FILE = OUTPUT_DIR / "chat_messages_rag_ready.json"

# --- Meta Formatter Settings ---
META_INPUT_DIR = Path(os.getenv("META_ORG_DATA_DIR", str(ORG_DATA_DIR))) # Meta data is often in the root of org_data
META_OUTPUT_FILE = OUTPUT_DIR / "meta_rag_ready_data.json"

# --- Corpus Builder Settings ---
CORPUS_INPUT_DIR = Path(os.getenv("CORPUS_INPUT_DIR", str(ORG_DATA_DIR / "word_data")))
CORPUS_OUTPUT_FILE = OUTPUT_DIR / "personal_corpus_rag_ready.json"

# --- General Settings ---
# Batch size for writing output files
BATCH_WRITE_SIZE = int(os.getenv("FORMATTER_BATCH_SIZE", "100"))
