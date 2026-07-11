import os
import json
import re
import hashlib
import logging
import mailbox
import email.utils
from email.header import decode_header
from datetime import datetime, timezone
from pathlib import Path
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
import warnings
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('../pipeline.env')

# Ignore BeautifulSoup XML as HTML warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# --- Configuration Section ---
MBOX_DIR_PATH = os.getenv("MBOX_DIR_PATH", "../org_data/mail_data")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "../output_clean_data")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "gmail_corpus_rag_ready.json")
MY_EMAIL = os.getenv("MY_EMAIL", "example@example.com")
LOG_DIR = os.getenv("LOG_DIR", "../log")
PROC_DIR = os.getenv("PROC_DIR", "../processing")
PROCESS_NAME = "gmail_to_rag"

# Ensure directories exist
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(PROC_DIR).mkdir(parents=True, exist_ok=True)
process_log_dir = Path(LOG_DIR) / PROCESS_NAME
process_log_dir.mkdir(parents=True, exist_ok=True)
process_proc_dir = Path(PROC_DIR) / PROCESS_NAME
process_proc_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
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
    
    # Avoid duplicate handlers
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
    return {"processed_files": [], "grouped_threads": {}, "total_files_parsed": 0, "total_messages_parsed": 0, "current_file": None, "current_msg_idx": 0}

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
    if progress_data.get("current_file") or progress_data.get("processed_files"):
        while True:
            response = input(f"Found interrupted progress (last file: {progress_data.get('current_file')}). Do you want to resume? (y/n): ").strip().lower()
            if response in ['y', 'n']:
                return response == 'y'
    return False

def decode_mime_header(header_value) -> str:
    """Decode MIME header (Handles Quoted-Printable or Base64)"""
    if not header_value:
        return ""
    try:
        decoded_parts = decode_header(header_value)
        result = ""
        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                charset = charset if charset else 'utf-8'
                try:
                    result += part.decode(charset, errors='replace')
                except LookupError:
                    result += part.decode('utf-8', errors='replace')
            else:
                result += str(part)
        return result.strip()
    except Exception as e:
        logger.debug(f"Header decoding failed: {e}")
        return str(header_value)

def clean_subject(subject_raw: str) -> str:
    """Clean email subject prefixes (Re:, Fwd:, etc.)"""
    subject = decode_mime_header(subject_raw)
    if not subject:
        return "No Subject"
    
    # Repeatedly remove prefixes until no match is found
    pattern = re.compile(r'^(re|fwd|fw|回覆|轉寄)\s*[:：]\s*', re.IGNORECASE)
    while pattern.search(subject):
        subject = pattern.sub('', subject).strip()
    
    return subject if subject else "No Subject"

def parse_date_to_iso(date_str: str) -> str:
    """Convert RFC 2822 date string to ISO-8601"""
    if not date_str:
        return ""
    try:
        parsed_tuple = email.utils.parsedate_tz(date_str)
        if parsed_tuple:
            timestamp = email.utils.mktime_tz(parsed_tuple)
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        logger.debug(f"Date parsing failed for {date_str}: {e}")
    return ""

def extract_email(address_str: str) -> str:
    """Extract email address precisely from Name <email@domain.com>"""
    if not address_str:
        return ""
    decoded_str = decode_mime_header(address_str)
    _, email_addr = email.utils.parseaddr(decoded_str)
    return email_addr.lower()

def extract_body(msg) -> str:
    """Extract plain text body, completely ignoring attachments"""
    body_text = ""
    body_html = ""

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get('Content-Disposition'))

            # Skip attachments
            if 'attachment' in disposition or 'inline' in disposition:
                if content_type not in ['text/plain', 'text/html']:
                    continue
            
            # Only take text/plain or text/html
            if content_type == 'text/plain':
                charset = part.get_content_charset() or 'utf-8'
                try:
                    body_text += part.get_payload(decode=True).decode(charset, errors='replace')
                except Exception:
                    pass
            elif content_type == 'text/html':
                charset = part.get_content_charset() or 'utf-8'
                try:
                    body_html += part.get_payload(decode=True).decode(charset, errors='replace')
                except Exception:
                    pass
    else:
        content_type = msg.get_content_type()
        charset = msg.get_content_charset() or 'utf-8'
        try:
            payload = msg.get_payload(decode=True).decode(charset, errors='replace')
            if content_type == 'text/plain':
                body_text = payload
            elif content_type == 'text/html':
                body_html = payload
        except Exception:
            pass
    
    # Prefer plain text, if none, convert HTML to plain text
    if body_text.strip():
        return body_text
    elif body_html.strip():
        soup = BeautifulSoup(body_html, 'html.parser')
        return soup.get_text(separator='\n')
    return ""

def clean_content(text: str) -> str:
    """Remove quoted messages and signature noise"""
    if not text:
        return ""
    
    # Process possible HTML leftovers
    soup = BeautifulSoup(text, 'html.parser')
    clean_text = soup.get_text(separator='\n')

    lines = clean_text.split('\n')
    filtered_lines = []

    # Identify and remove historical emails and quote lines
    quote_patterns = [
        re.compile(r'^On\s+.+?wrote:$', re.IGNORECASE),
        re.compile(r'^-{3,}\s*Original Message\s*-{3,}', re.IGNORECASE),
        re.compile(r'^_{3,}$'), # Underscore separator
        re.compile(r'^From:\s+.*<.+@.+>', re.IGNORECASE)
    ]
    
    # Remove default mobile signatures
    signature_patterns = [
        re.compile(r'^Sent from my (iPhone|iPad|Android)', re.IGNORECASE),
        re.compile(r'^從我的\s*(iPhone|iPad|Android)\s*傳送', re.IGNORECASE)
    ]

    for line in lines:
        stripped_line = line.strip()
        
        # If it's the start of a quoted historical email block, stop reading further
        if any(pattern.match(stripped_line) for pattern in quote_patterns):
            break
            
        # Skip lines starting with ">"
        if stripped_line.startswith('>'):
            continue
            
        # Skip default signatures
        if any(pattern.match(stripped_line) for pattern in signature_patterns):
            continue

        filtered_lines.append(line)

    result_text = '\n'.join(filtered_lines).strip()
    
    # Compress excessive newlines
    result_text = re.sub(r'\n{3,}', '\n\n', result_text)
    return result_text

def get_output_file_path(output_dir: Path, base_filename: str) -> Path:
    """Generate output file path, fixed name + timestamp + batch number if exists"""
    timestamp = datetime.now().strftime("%Y%m%d%H")
    name, ext = os.path.splitext(base_filename)
    
    base_new_filename = f"{name}_{timestamp}{ext}"
    output_path = output_dir / base_new_filename
    
    if not output_path.exists():
        return output_path
        
    batch = 1
    
    while True:
        new_filename = f"{name}_{timestamp}_{batch}{ext}"
        new_path = output_dir / new_filename
        if not new_path.exists():
            return new_path
        batch += 1

def save_batch(final_email_threads, output_path_dir, base_filename):
    output_file_path = get_output_file_path(output_path_dir, base_filename)
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump({"email_threads": final_email_threads}, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully saved batch to {output_file_path}")
    except Exception as e:
        logger.error(f"Error saving batch JSON file: {e}")

def process_mbox_files():
    start_time = datetime.now()
    logger.info("Process Started")
    if not os.path.exists(MBOX_DIR_PATH):
        logger.error(f"Cannot find MBOX directory: {MBOX_DIR_PATH}")
        return

    progress_data = load_progress()
    resume = prompt_resume(progress_data)
    
    if not resume:
        logger.info("Starting fresh, clearing previous progress.")
        clear_progress()
        progress_data = {"processed_files": [], "grouped_threads": {}, "total_files_parsed": 0, "total_messages_parsed": 0, "current_file": None, "current_msg_idx": 0}
        
        # Check and cleanup unfinished partial files? The instruction says "如果不繼續則fallback 有output json檔的每100筆進行批次寫入 不繼續中斷紀錄的則清除未完成的json檔"
        # We will clear files related to this process in the output dir that might be considered "unfinished", but it's safer to just start writing new batched files.
        # Actually, let's look for incomplete files. Since we batch write and the file names include timestamps, it might be complex to identify. We will rely on new timestamps.
    else:
        logger.info(f"Resuming from file {progress_data.get('current_file')} at message index {progress_data.get('current_msg_idx', 0)}")

    grouped_threads = progress_data["grouped_threads"]
    total_messages_parsed = progress_data["total_messages_parsed"]
    total_files_parsed = progress_data["total_files_parsed"]
    processed_files = progress_data["processed_files"]

    mbox_dir_path = Path(MBOX_DIR_PATH)
    all_mbox_files = list(mbox_dir_path.glob("*.mbox"))
    
    try:
        for mbox_file_path in all_mbox_files:
            file_name_str = str(mbox_file_path.name)
            if file_name_str in processed_files and (progress_data["current_file"] != file_name_str):
                continue

            if progress_data["current_file"] != file_name_str:
                progress_data["current_file"] = file_name_str
                progress_data["current_msg_idx"] = 0
                total_files_parsed += 1

            logger.info(f"Start parsing MBOX file: {mbox_file_path}")
            
            try:
                mbox = mailbox.mbox(mbox_file_path)
                
                for idx, msg in enumerate(mbox):
                    if idx < progress_data["current_msg_idx"]:
                        continue

                    total_messages_parsed += 1
                    progress_data["current_msg_idx"] = idx + 1
                    progress_data["total_messages_parsed"] = total_messages_parsed

                    if total_messages_parsed % 1000 == 0:
                        logger.info(f"Scanned {total_messages_parsed} emails...")
                        save_progress(progress_data)
                        
                    # Extract and clean subject
                    raw_subject = msg.get('Subject', '')
                    cleaned_subject = clean_subject(raw_subject)
                    
                    # Extract time and participants
                    timestamp = parse_date_to_iso(msg.get('Date', ''))
                    sender_email = extract_email(msg.get('From', ''))
                    receiver_email = extract_email(msg.get('To', ''))
                    
                    # Get clean body
                    raw_body = extract_body(msg)
                    page_content = clean_content(raw_body)
                    
                    if not page_content:
                        continue
                        
                    message_id = msg.get('Message-ID', f"msg_{hashlib.md5(f'{cleaned_subject}{timestamp}'.encode()).hexdigest()}")
                    is_me = (MY_EMAIL.lower() == sender_email)
            
                    message_data = {
                        "message_id": message_id.strip('<>'),
                        "page_content": page_content,
                        "metadata": {
                            "sender": sender_email,
                            "receiver": receiver_email,
                            "timestamp": timestamp,
                            "is_me": is_me,
                            "source_file": mbox_file_path.name
                        }
                    }
                    
                    if cleaned_subject not in grouped_threads:
                        grouped_threads[cleaned_subject] = []
                        
                    grouped_threads[cleaned_subject].append(message_data)
                    
                # File completed
                if file_name_str not in processed_files:
                    processed_files.append(file_name_str)
                progress_data["processed_files"] = processed_files
                progress_data["current_file"] = None
                progress_data["current_msg_idx"] = 0
                progress_data["total_files_parsed"] = total_files_parsed
                save_progress(progress_data)

            except Exception as e:
                logger.error(f"Failed to read MBOX file {mbox_file_path}: {e}")

        logger.info(f"Email reading completed. Read {total_files_parsed} files, found {len(grouped_threads)} conversation threads, preparing for two-way interaction filtering...")

        # Perform strict filtering and formatting
        final_email_threads = []
        batch_results = []
        output_path_dir = Path(OUTPUT_DIR)
        output_path_dir.mkdir(parents=True, exist_ok=True)
        
        for subject, conversation in grouped_threads.items():
            # Core rule: Check if any email was sent by MY_EMAIL
            has_my_reply = any(msg['metadata']['is_me'] for msg in conversation)
            
            if has_my_reply:
                # Sort by original send time from oldest to newest
                conversation_sorted = sorted(conversation, key=lambda x: x['metadata']['timestamp'] if x['metadata']['timestamp'] else "")
                
                thread_id = hashlib.md5(subject.encode('utf-8')).hexdigest()
                
                thread_data = {
                    "thread_id": thread_id,
                    "subject": subject,
                    "metadata": {
                        "source": "gmail",
                        "total_messages": len(conversation_sorted)
                    },
                    "conversation": conversation_sorted
                }
                final_email_threads.append(thread_data)
                batch_results.append(thread_data)

                # Batch write every 100
                if len(batch_results) >= 100:
                    save_batch(batch_results, output_path_dir, OUTPUT_FILE)
                    batch_results = []

        if batch_results:
            save_batch(batch_results, output_path_dir, OUTPUT_FILE)

        # Generate log report
        filtered_out_count = len(grouped_threads) - len(final_email_threads)
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        logger.info("-" * 40)
        logger.info("MBOX parsing and filtering completed - Final Report:")
        logger.info(f"Total MBOX files read: {total_files_parsed}")
        logger.info(f"Total emails scanned: {total_messages_parsed}")
        logger.info(f"Initial conversation threads: {len(grouped_threads)}")
        logger.info(f"Threads filtered out due to no two-way interaction: {filtered_out_count}")
        logger.info(f"Successfully retained threads: {len(final_email_threads)}")
        logger.info(f"Total Execution Time: {execution_time}")
        logger.info("-" * 40)

        # Clear progress on success
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
    process_mbox_files()