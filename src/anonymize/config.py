from pathlib import Path
import os
import json

# --- Project Root ---
# Assumes this config.py is at src/anonymize/config.py
PROJECT_ROOT = Path(__file__).parent.parent.parent

# --- Load Environment Variables ---
# Using the same .env file from the project root
from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / "pipeline.env")

# --- Core Paths ---
# Default to a structured layout within the project's data directory
DATA_ROOT = PROJECT_ROOT / "data"
RUNTIME_ROOT = PROJECT_ROOT / "_runtime"

INPUT_DIR = Path(os.getenv("ANONYMIZE_INPUT_DIR", str(DATA_ROOT / "output_clean_data")))
OUTPUT_DIR = Path(os.getenv("ANONYMIZE_OUTPUT_DIR", str(DATA_ROOT / "output_mask_data")))
LOG_DIR = Path(os.getenv("LOG_DIR", str(RUNTIME_ROOT / "log")))
PROC_DIR = Path(os.getenv("PROC_DIR", str(RUNTIME_ROOT / "processing")))

# --- Process-Specific Settings ---
PROCESS_NAME = "anonymize_corpus"
PROGRESS_FILE = PROC_DIR / PROCESS_NAME / "progress.json"

# --- Anonymization Rules & Keys ---
# Keys in the JSON structure that should not be masked
EXCLUDED_KEYS = {
    "doc_id", "id", "uuid", "file_format", "doc_type", "last_modified", 
    "created_at", "is_ocr", "timestamp", "content_type", "message_id"
}

# Keys that identify a speaker and should be tracked
SPEAKER_TRACKING_KEYS = {
    "sender_name", "sender", "receiver", "author", "creator", "from", "to"
}

# --- Default Data for Anonymization Strategies ---
# These serve as fallbacks if not provided in the environment.
DEFAULT_IDENTITIES = json.dumps({
    "Your Name": "[AUTHOR_ME]",
    "Your Nickname": "[AUTHOR_ME]",
    "your.email@example.com": "[AUTHOR_ME]"
})
DEFAULT_SENSITIVE_WORDS = json.dumps({
    "ExampleCompany Ltd.": "[REDACTED_COMPANY]",
    "SecretProjectX": "[REDACTED_PROJECT]",
    "123 Example Street, City": "[REDACTED_ADDRESS]"
})

# Load from environment or use the default
# The application logic will be responsible for parsing these JSON strings.
MY_IDENTITIES_MATRIX_JSON = os.getenv("MY_IDENTITIES_MATRIX", DEFAULT_IDENTITIES)
SENSITIVE_WORDS_MAP_JSON = os.getenv("SENSITIVE_WORDS_MAP", DEFAULT_SENSITIVE_WORDS)

# --- Presidio Configuration ---
# Configuration for the Presidio NLP engine
PRESIDIO_NLP_CONFIG = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "zh", "model_name": "zh_core_web_sm"},
        {"lang_code": "en", "model_name": "en_core_web_sm"}
    ],
    "ner_model_configuration": {
        "labels_to_ignore": ["CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "TIME", "MONEY"]
    }
}

# Mapping from Presidio entity types to custom mask tokens
PRESIDIO_OPERATORS = {
    "PERSON": {"type": "replace", "new_value": "[REDACTED_PERSON]"},
    "LOCATION": {"type": "replace", "new_value": "[REDACTED_LOCATION]"},
    "ORGANIZATION": {"type": "replace", "new_value": "[REDACTED_ORG]"}
}
