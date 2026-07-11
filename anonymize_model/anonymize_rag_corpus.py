import os
import json
import re
import logging
from datetime import datetime
from pathlib import Path
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('../pipeline.env')

# --- Configuration Section ---
INPUT_DIR = os.getenv("ANONYMIZE_INPUT_DIR", "../output_clean_data")
OUTPUT_DIR = os.getenv("ANONYMIZE_OUTPUT_DIR", "../output_mask_data")
LOG_DIR = os.getenv("LOG_DIR", "../log")
PROC_DIR = os.getenv("PROC_DIR", "../processing")
PROCESS_NAME = "anonymize_rag_corpus"

# Ensure directories exist
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(PROC_DIR).mkdir(parents=True, exist_ok=True)
process_log_dir = Path(LOG_DIR) / PROCESS_NAME
process_log_dir.mkdir(parents=True, exist_ok=True)
process_proc_dir = Path(PROC_DIR) / PROCESS_NAME
process_proc_dir.mkdir(parents=True, exist_ok=True)

# Setup logging (English, No Emojis)
def setup_logger():
    date_str = datetime.now().strftime("%Y%m%d")
    base_log_filename = f"{PROCESS_NAME}_{date_str}.log"
    log_file_path = process_log_dir / base_log_filename

    # Handle max 1GB log size and batching
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

# Setup Progress tracking
PROGRESS_FILE = process_proc_dir / "progress.json"

def load_progress():
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading progress: {e}")
    return {"processed_files": [], "total_files": 0}

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

# --- Third-party packages block (Presidio & spaCy) ---
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    HAS_PRESIDIO = True
except ImportError as e:
    HAS_PRESIDIO = False
    logger.error(f"Missing Presidio or spaCy packages: {e}.")
    logger.error("Please run: pip install presidio-analyzer presidio-anonymizer spacy")
    logger.error("And download models: python -m spacy download zh_core_web_sm && python -m spacy download en_core_web_sm")

# Excluded keys for structural metadata (Do not mask these)
EXCLUDED_KEYS = {
    "doc_id", "id", "uuid", "file_format", "doc_type", "last_modified", "created_at", "is_ocr", "timestamp", "content_type", "message_id"
}

# Speaker tracking keys (will assign [SPEAKER_N] tags to these)
SPEAKER_TRACKING_KEYS = {
    "sender_name", "sender", "receiver", "author", "creator", "from", "to"
}

# Default identity matrix if not provided in .env
DEFAULT_IDENTITIES_MATRIX = {
    "YourName": "[AUTHOR_ME]",
    "your_email@gmail.com": "[AUTHOR_ME]",
    "YourNickname": "[AUTHOR_ME]"
}

# Default sensitive words map if not provided in .env
DEFAULT_SENSITIVE_WORDS_MAP = {
    "ExampleCompany Ltd.": "[REDACTED_COMPANY]",
    "SecretProjectX": "[REDACTED_PROJECT]",
    "123 Example Street, City": "[REDACTED_ADDRESS]"
}

# 1. Identity Matrix (Targeting the author's personal identities)
# Read from pipeline.env, default to DEFAULT_IDENTITIES_MATRIX if not provided
MY_IDENTITIES_MATRIX = DEFAULT_IDENTITIES_MATRIX
try:
    env_identities = os.getenv("MY_IDENTITIES_MATRIX", "{}")
    if env_identities.strip() and env_identities != "{}":
        MY_IDENTITIES_MATRIX = json.loads(env_identities)
except Exception as e:
    logger.warning(f"Failed to parse MY_IDENTITIES_MATRIX from .env. Using default example dict. Error: {e}")

# 2. Custom Sensitive Words Map (Targeting specific terms or addresses)
# Read from pipeline.env, default to DEFAULT_SENSITIVE_WORDS_MAP if not provided
SENSITIVE_WORDS_MAP = DEFAULT_SENSITIVE_WORDS_MAP
try:
    env_sensitive_words = os.getenv("SENSITIVE_WORDS_MAP", "{}")
    if env_sensitive_words.strip() and env_sensitive_words != "{}":
        SENSITIVE_WORDS_MAP = json.loads(env_sensitive_words)
except Exception as e:
    logger.warning(f"Failed to parse SENSITIVE_WORDS_MAP from .env. Using default example dict. Error: {e}")

# 3. Regex Compilation Rules
REGEX_RULES = [
    (re.compile(r'(?<!\d)(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?!\d)'), '[REDACTED_IP]'),
    (re.compile(r'https?://[^\s<>"\']+|(?:www\.)[^\s<>"\']+'), '[REDACTED_URL]'),
    (re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'), '[REDACTED_EMAIL]'),
    (re.compile(r'(?:\+886\s*|0)9\d{2}[\s\-]?\d{3}[\s\-]?\d{3}'), '[REDACTED_PHONE]'),
    (re.compile(r'(?:\+886\s*|0)[2-8][\s\-]?\d{3,4}[\s\-]?\d{4}'), '[REDACTED_PHONE]'),
    (re.compile(r'(?<![A-Za-z0-9])[A-Za-z][1289]\d{8}(?![0-9])'), '[REDACTED_ID]'),
    (re.compile(r'(?<!\d)\d{8}(?!\d)'), '[REDACTED_ID]')
]

# --- Global Presidio Initialization ---
analyzer = None
anonymizer = None

def get_output_file_path(output_dir: Path, base_filename: str) -> Path:
    """Generate output file path, appending timestamp and batch number if exists"""
    timestamp = datetime.now().strftime("%Y%m%d%H")
    name, ext = os.path.splitext(base_filename)
    
    base_new_filename = f"{name}_{timestamp}{ext}"
    output_new_path = output_dir / base_new_filename
    
    if not output_new_path.exists():
        return output_new_path
    
    batch = 1
    while True:
        new_filename = f"{name}_{timestamp}_{batch}{ext}"
        new_path = output_dir / new_filename
        if not new_path.exists():
            return new_path
        batch += 1

def init_presidio():
    """Initialize Presidio Analyzer and Anonymizer with bilingual support (zh, en)"""
    global analyzer, anonymizer
    if not HAS_PRESIDIO or analyzer:
        return
        
    logger.info("Initializing Presidio NLP engine (spaCy)... This might take a few seconds.")
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "zh", "model_name": "zh_core_web_sm"},
            {"lang_code": "en", "model_name": "en_core_web_sm"}
        ],
        "ner_model_configuration": {
            "labels_to_ignore": ["CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "TIME", "MONEY"]
        }
    }
    
    try:
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["zh", "en"])
        anonymizer = AnonymizerEngine()
        logger.info("Presidio NLP engine initialized successfully!")
    except Exception as e:
        logger.error(f"Presidio initialization failed. Please ensure spaCy language models are downloaded. Error: {e}")
        analyzer = None

def apply_general_masking(text: str) -> str:
    """Apply Regex, custom dictionary, and Presidio NLP masking to general text"""
    if not isinstance(text, str) or not text.strip():
        return text

    # Priority 1: Replace author's own identity
    for identity, mask_token in MY_IDENTITIES_MATRIX.items():
        text = text.replace(identity, mask_token)

    # Priority 2: Custom Sensitive Words
    for sensitive_word, mask_token in SENSITIVE_WORDS_MAP.items():
        text = text.replace(sensitive_word, mask_token)

    # Priority 3: Regex Pattern Matching
    for pattern, replacement in REGEX_RULES:
        text = pattern.sub(replacement, text)

    # Priority 4: Presidio NLP Processing
    if analyzer and anonymizer:
        try:
            analyzer_results = analyzer.analyze(text=text, language='zh')
            operators = {
                "PERSON": OperatorConfig("replace", {"new_value": "[REDACTED_PERSON]"}),
                "LOCATION": OperatorConfig("replace", {"new_value": "[REDACTED_LOCATION]"}),
                "ORGANIZATION": OperatorConfig("replace", {"new_value": "[REDACTED_ORG]"})
            }
            anonymized_result = anonymizer.anonymize(text=text, analyzer_results=analyzer_results, operators=operators)
            text = anonymized_result.text
        except Exception as e:
            logger.debug(f"Error occurred during Presidio processing: {e}")

    return text

def anonymize_document_with_speaker_tracking(doc: dict) -> dict:
    """Anonymize a single document while tracking speakers within the document scope."""
    # speaker_map is only valid during the lifecycle of a single document
    speaker_map = {}

    def mask_recursive_helper(data, key_name=None):
        nonlocal speaker_map

        if isinstance(data, str):
            # 1. Check if it's an excluded structural key
            if key_name in EXCLUDED_KEYS:
                return data

            # 2. Check if it's a speaker tracking key
            if key_name in SPEAKER_TRACKING_KEYS:
                original_name = data.strip()
                
                # Is it the author?
                if original_name in MY_IDENTITIES_MATRIX:
                    return "[AUTHOR_ME]"
                
                # Has this person spoken before in this document?
                if original_name in speaker_map:
                    return speaker_map[original_name]
                
                # New speaker, assign a new ID
                new_speaker_id = f"[SPEAKER_{len(speaker_map) + 1}]"
                speaker_map[original_name] = new_speaker_id
                return new_speaker_id

            # 3. Otherwise, treat as general text
            return apply_general_masking(data)
        
        elif isinstance(data, dict):
            return {k: mask_recursive_helper(v, key_name=k) for k, v in data.items()}
        
        elif isinstance(data, list):
            return [mask_recursive_helper(item, key_name=key_name) for item in data]
        
        else:
            return data

    return mask_recursive_helper(doc)

def save_batch(data, output_file_path):
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully saved batched anonymized file: {output_file_path}")
    except Exception as e:
        logger.error(f"Error saving batch JSON file {output_file_path}: {e}")

def process_anonymization():
    """Main function: recursively traverse, process JSON files and output to target directory"""
    start_time = datetime.now()
    logger.info("Process Started")
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)

    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input directory not found: {INPUT_DIR}. Please prepare source data first!")
        return
        
    progress_data = load_progress()
    resume = prompt_resume(progress_data)
    
    if not resume:
        logger.info("Starting fresh, clearing previous progress.")
        clear_progress()
        progress_data = {"processed_files": [], "total_files": 0}
    else:
        logger.info(f"Resuming process. {len(progress_data.get('processed_files', []))} files already processed.")

    processed_files = progress_data.get("processed_files", [])
    total_files = progress_data.get("total_files", 0)

    init_presidio()

    logger.info(f"Starting anonymization process. Input Directory: {INPUT_DIR} | Output Directory: {OUTPUT_DIR}")

    try:
        for file_path in input_path.rglob('*.json'):
            rel_path = file_path.relative_to(input_path)
            str_rel_path = str(rel_path)
            
            if str_rel_path in processed_files:
                continue
                
            total_files += 1
            
            base_output_file_name = f"{file_path.stem}_mask{file_path.suffix}"
            
            # Ensure target subdirectories exist
            target_dir = output_path / rel_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing file: {file_path.name}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Assuming files might be huge, if it has a list, we might want to batch write.
                # However, anonymize output format usually matches input. The request says "有output json檔的每100筆進行批次寫入"
                batch_size = 100
                
                if isinstance(data, dict):
                    # Find any list to batch if applicable
                    list_key = None
                    list_data = None
                    for key, val in data.items():
                        if isinstance(val, list):
                            list_key = key
                            list_data = val
                            break
                            
                    if list_data and len(list_data) > batch_size:
                        batch_items = []
                        batch_count = 1
                        for item in list_data:
                            batch_items.append(anonymize_document_with_speaker_tracking(item))
                            if len(batch_items) >= batch_size:
                                # Create a copy of original data with batch
                                batch_data = data.copy()
                                batch_data[list_key] = batch_items
                                output_file_path = get_output_file_path(target_dir, base_output_file_name)
                                save_batch(batch_data, output_file_path)
                                batch_items = []
                                batch_count += 1
                        if batch_items:
                            batch_data = data.copy()
                            batch_data[list_key] = batch_items
                            output_file_path = get_output_file_path(target_dir, base_output_file_name)
                            save_batch(batch_data, output_file_path)
                    elif isinstance(data, list):
                        batch_items = []
                        for item in data:
                            batch_items.append(anonymize_document_with_speaker_tracking(item))
                            if len(batch_items) >= batch_size:
                                output_file_path = get_output_file_path(target_dir, base_output_file_name)
                                save_batch(batch_items, output_file_path)
                                batch_items = []
                        if batch_items:
                            output_file_path = get_output_file_path(target_dir, base_output_file_name)
                            save_batch(batch_items, output_file_path)
                    else:
                        data = anonymize_document_with_speaker_tracking(data)
                        output_file_path = get_output_file_path(target_dir, base_output_file_name)
                        save_batch(data, output_file_path)
                else:
                    data = anonymize_document_with_speaker_tracking(data)
                    output_file_path = get_output_file_path(target_dir, base_output_file_name)
                    save_batch(data, output_file_path)

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON format in file {file_path.name}, skipped: {e}")
            except Exception as e:
                logger.error(f"Unexpected error occurred while processing {file_path.name}: {e}")
                
            processed_files.append(str_rel_path)
            progress_data["processed_files"] = processed_files
            progress_data["total_files"] = total_files
            save_progress(progress_data)

        end_time = datetime.now()
        execution_time = end_time - start_time
        
        logger.info("-" * 40)
        logger.info("Anonymization parsing completed - Final Report:")
        logger.info(f"Total files processed: {total_files}")
        logger.info(f"Total Execution Time: {execution_time}")
        logger.info("-" * 40)

        clear_progress()
        logger.info("Process Finished Successfully.")

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
        save_progress(progress_data)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}", exc_info=True)
        save_progress(progress_data)
        sys.exit(1)

if __name__ == "__main__":
    process_anonymization()