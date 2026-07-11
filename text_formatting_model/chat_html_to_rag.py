import os
import json
import uuid
import re
import logging
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('../pipeline.env')

# ==========================================
# Configuration & Setup
# ==========================================
BASE_DIR = os.getenv("CHAT_HTML_DIR", "../org_data/chat_data")
OUTPUT_FILE = os.getenv("CHAT_HTML_OUTPUT", "../output_clean_data/chat_messages_rag_ready.json")
LOG_DIR = os.getenv("LOG_DIR", "../log")
PROC_DIR = os.getenv("PROC_DIR", "../processing")
PROCESS_NAME = "chat_html_to_rag"

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
    return {"processed_files": [], "total_html_files": 0}

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

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def get_output_file_path(output_path: str) -> Path:
    """Generate output file path, appending timestamp and batch number if exists"""
    base_path = Path(output_path)
    if not base_path.exists():
        return base_path
        
    timestamp = datetime.now().strftime("%Y%m%d%H")
    name = base_path.stem
    ext = base_path.suffix
    directory = base_path.parent
    
    base_new_filename = f"{name}_{timestamp}{ext}"
    output_new_path = directory / base_new_filename
    
    if not output_new_path.exists():
        return output_new_path
    
    batch = 1
    while True:
        new_filename = f"{name}_{timestamp}_{batch}{ext}"
        new_path = directory / new_filename
        if not new_path.exists():
            return new_path
        batch += 1

def save_batch(final_data, output_path_dir, base_filename):
    output_file_path = get_output_file_path(output_path_dir / base_filename)
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump({"messages": final_data}, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully saved batch to {output_file_path}")
    except Exception as e:
        logger.error(f"Error saving batch JSON file: {e}")

def clean_time_string(time_str: str) -> str:
    """Attempt to parse time string, return stripped string as fallback."""
    if not time_str:
        return ""
    time_str = time_str.strip()
    return time_str

def extract_content_and_media(msg_element) -> tuple:
    """Extract text content, media attachments, and content type."""
    content_type = "text"
    media_attachments = []

    # Find potential media tags
    for img in msg_element.find_all("img"):
        src = img.get("src")
        if src:
            media_attachments.append(src)
        content_type = "image"

    for video in msg_element.find_all("video"):
        src = video.get("src") or (video.find("source").get("src") if video.find("source") else None)
        if src:
            media_attachments.append(src)
        content_type = "video"

    for audio in msg_element.find_all("audio"):
        src = audio.get("src") or (audio.find("source").get("src") if audio.find("source") else None)
        if src:
            media_attachments.append(src)
        content_type = "voice"

    for link in msg_element.find_all("a"):
        href = link.get("href")
        if href and not href.startswith("#"):
            href_lower = href.lower()
            # Defensive check: if link is media file
            if any(href_lower.endswith(ext) for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]):
                content_type = "video"
            elif any(href_lower.endswith(ext) for ext in [".mp3", ".m4a", ".wav", ".ogg", ".aac"]):
                content_type = "voice"
            else:
                if not media_attachments:
                    content_type = "link"
            media_attachments.append(href)

    # Extract plain text
    text_content = msg_element.get_text(separator=' ', strip=True)

    # Assist typing by text features
    if content_type in ["text", "link"]:
        # Remove invisible chars and lower
        clean_text = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff]+', '', text_content).strip()
        check_lower = clean_text.lower()
        
        if any(k in check_lower for k in ["[相片]", "[圖片]", "[貼圖]"]) or "傳送了貼圖" in check_lower or "image omitted" in check_lower:
            content_type = "image"
        elif any(k in check_lower for k in ["[語音訊息]", "[語音]"]) or "audio omitted" in check_lower:
            content_type = "voice"
        elif "[影片]" in check_lower or "video omitted" in check_lower:
            content_type = "video"

    return content_type, media_attachments, text_content

# ---------------------------------------------------------
# HTML Parsing Logic
# ---------------------------------------------------------

def parse_html_chat(file_path: Path) -> dict:
    """Parse single HTML file and return structured thread data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None

    # Filter out layout elements
    for tag in soup(["nav", "aside", "header", "footer", "script", "style"]):
        tag.decompose()

    # Determine source and thread id
    source = "unknown"
    thread_id = file_path.stem

    title_text = (soup.title.string or "").strip() if soup.title else ""
    folder_name = file_path.parent.name
    file_name = file_path.name

    # LINE characteristics
    if "line" in title_text.lower() or "line" in folder_name.lower() or "line" in file_name.lower():
        source = "line"

    # WhatsApp characteristics
    elif "whatsapp" in title_text.lower() or "whatsapp" in folder_name.lower() or "whatsapp" in file_name.lower():
        source = "whatsapp"
        if title_text and title_text.lower() != "whatsapp messages":
            thread_id = title_text

    # Extract LINE target
    if source == "line":
        to_element = soup.find(string=re.compile(r"^TO:\s*(.*)"))
        if to_element:
            match = re.search(r"^TO:\s*(.*)", to_element)
            if match:
                thread_id = match.group(1).strip()

    conversation = []

    if source == "line":
        current_time = ""
        current_sender = "Unknown"

        container = soup.find(class_='content') or soup.find(id='container') or soup

        for element in container.children:
            if not hasattr(element, 'name'):
                continue

            if element.name == 'p':
                if element.get('align') == 'center' and re.search(r'\d{4}/\d{2}/\d{2}', element.text):
                    current_time = clean_time_string(element.text)
                    continue

                if element.get('align') == 'left' and not element.get('class'):
                    text = element.text.strip()
                    if text and not text.startswith("Account:") and not text.startswith("TO:"):
                        current_sender = text
                    continue
                
                if current_sender in ["LINE購物"] or "Line 銀行" in current_sender:
                    continue

                classes = element.get('class', [])
                if any('triangle' in c for c in classes):
                    content_type, media_attachments, text_content = extract_content_and_media(element)
                    
                    if content_type in ["image", "video", "voice"] or not text_content.strip() or "html" in text_content.lower():
                        continue
                        
                    conversation.append({
                        "message_id": str(uuid.uuid4()),
                        "page_content": text_content,
                        "metadata": {
                            "sender_name": current_sender,
                            "timestamp": current_time,
                            "content_type": content_type,
                            "media_attachments": media_attachments
                        }
                    })

            elif element.name == 'table':
                if current_sender in ["LINE購物"] or "Line 銀行" in current_sender:
                    continue
                classes = element.get('class', [])
                if any('triangle' in c for c in classes):
                    content_type, media_attachments, text_content = extract_content_and_media(element)

                    if content_type in ["image", "video", "voice"] or not text_content.strip() or "html" in text_content.lower():
                        continue
                        
                    conversation.append({
                        "message_id": str(uuid.uuid4()),
                        "page_content": text_content,
                        "metadata": {
                            "sender_name": current_sender,
                            "timestamp": current_time,
                            "content_type": content_type,
                            "media_attachments": media_attachments
                        }
                    })

    elif source == "whatsapp":
        msg_blocks = soup.find_all(attrs={"class": re.compile(r"message|msg", re.I)})
        for block in msg_blocks:
            sender_el = block.find(attrs={"class": re.compile(r"name|sender|author", re.I)})
            sender_name = sender_el.get_text(strip=True) if sender_el else "Unknown"

            if "Line 銀行" in sender_name:
                continue

            time_el = block.find(attrs={"class": re.compile(r"time|date|timestamp", re.I)})
            time_str = time_el.get_text(strip=True) if time_el else ""

            content_el = block.find(attrs={"class": re.compile(r"text|content|body", re.I)})
            if not content_el:
                content_el = block

            content_type, media_attachments, text_content = extract_content_and_media(content_el)

            if content_el == block:
                if sender_name and text_content.startswith(sender_name):
                    text_content = text_content[len(sender_name):].strip()
                if time_str and text_content.endswith(time_str):
                    text_content = text_content[:-len(time_str)].strip()

            if content_type in ["image", "video", "voice"] or not text_content.strip() or "html" in text_content.lower():
                continue

            conversation.append({
                "message_id": str(uuid.uuid4()),
                "page_content": text_content,
                "metadata": {
                    "sender_name": sender_name,
                    "timestamp": clean_time_string(time_str),
                    "content_type": content_type,
                    "media_attachments": media_attachments
                }
            })

    else:
        # Fallback
        msg_blocks = soup.find_all(attrs={"class": re.compile(r"message|msg|chat|bubble", re.I)})
        if not msg_blocks:
            msg_blocks = soup.find_all(["div", "p", "li"])

        for block in msg_blocks:
            sender_el = block.find(attrs={"class": re.compile(r"name|sender|author|user", re.I)})
            sender_name = sender_el.get_text(strip=True) if sender_el else "Unknown"

            if "Line 銀行" in sender_name:
                continue

            time_el = block.find(attrs={"class": re.compile(r"time|date|timestamp", re.I)})
            time_str = time_el.get_text(strip=True) if time_el else ""

            content_el = block.find(attrs={"class": re.compile(r"text|content|body", re.I)})
            if not content_el:
                content_el = block

            content_type, media_attachments, text_content = extract_content_and_media(content_el)

            if content_el == block:
                if sender_name and text_content.startswith(sender_name):
                    text_content = text_content[len(sender_name):].strip()
                if time_str and text_content.endswith(time_str):
                    text_content = text_content[:-len(time_str)].strip()

            if content_type in ["image", "video", "voice"] or not text_content.strip() or "html" in text_content.lower():
                continue

            conversation.append({
                "message_id": str(uuid.uuid4()),
                "page_content": text_content,
                "metadata": {
                    "sender_name": sender_name,
                    "timestamp": clean_time_string(time_str),
                    "content_type": content_type,
                    "media_attachments": media_attachments
                }
            })

    return {
        "thread_id": thread_id,
        "metadata": {
            "source": source
        },
        "conversation": conversation
    }


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    start_time = datetime.now()
    logger.info("Process Started")
    base_dir = Path(BASE_DIR)
    output_path = Path(OUTPUT_FILE)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_filename = output_path.name

    if not base_dir.exists():
        logger.error(f"Directory {base_dir} not found. Please check the folder path.")
        return

    progress_data = load_progress()
    resume = prompt_resume(progress_data)
    
    if not resume:
        logger.info("Starting fresh, clearing previous progress.")
        clear_progress()
        progress_data = {"processed_files": [], "total_html_files": 0}
    else:
        logger.info(f"Resuming process. {len(progress_data.get('processed_files', []))} files already processed.")

    processed_files = progress_data.get("processed_files", [])
    total_html_files = progress_data.get("total_html_files", 0)

    logger.info(f"Scanning HTML chat logs in {base_dir}...")
    
    batch_results = []
    
    try:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith('.html'):
                    file_path = Path(root) / file
                    file_rel_path = str(file_path.relative_to(base_dir))

                    if file_rel_path in processed_files:
                        continue
                    
                    total_html_files += 1
                    logger.info(f"Processing: {file_rel_path}")

                    thread_data = parse_html_chat(file_path)
                    if thread_data and thread_data["conversation"]:
                        batch_results.append(thread_data)
                    
                    processed_files.append(file_rel_path)
                    progress_data["processed_files"] = processed_files
                    progress_data["total_html_files"] = total_html_files
                    save_progress(progress_data)

                    if len(batch_results) >= 100:
                        save_batch(batch_results, output_dir, base_filename)
                        batch_results = []

        if batch_results:
            save_batch(batch_results, output_dir, base_filename)

        end_time = datetime.now()
        execution_time = end_time - start_time
        
        logger.info("-" * 40)
        logger.info("HTML parsing completed - Final Report:")
        logger.info(f"Total HTML files processed: {total_html_files}")
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
    main()