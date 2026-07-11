import os
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('../pipeline.env')

# ==========================================
# Configuration & Setup
# ==========================================
BASE_DIR = os.getenv("META_ORG_DATA_DIR", "../org_data")
OUTPUT_FILE = os.getenv("META_OUTPUT_FILE", "../output_clean_data/meta_rag_ready_data.json")
LOG_DIR = os.getenv("LOG_DIR", "../log")
PROC_DIR = os.getenv("PROC_DIR", "../processing")
PROCESS_NAME = "meta_data_to_rag"

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
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully saved batch to {output_file_path}")
    except Exception as e:
        logger.error(f"Error saving batch JSON file: {e}")

def fix_meta_encoding(text: str) -> str:
    """Fix Latin-1 encoding issues common in Meta export data"""
    if not isinstance(text, str):
        return ""
    try:
        return text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text

def convert_to_iso8601(timestamp) -> str:
    """Convert Unix Timestamp to ISO-8601 string"""
    if not timestamp:
        return ""
    try:
        ts = float(timestamp)
        # If timestamp is more than 10 digits, it's likely milliseconds
        if ts > 1e11:
            ts /= 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except (ValueError, TypeError):
        return ""

def determine_content_and_media(item_data: dict) -> tuple:
    """Analyze folder content, return content_type and media_attachments"""
    content_type = "text"
    media_attachments = []

    # Search for all possible attachment fields
    attachments = item_data.get("attachments", []) if isinstance(item_data, dict) else []
    photos = item_data.get("photos", []) if isinstance(item_data, dict) else []
    videos = item_data.get("videos", []) if isinstance(item_data, dict) else []
    audio_files = item_data.get("audio_files", []) if isinstance(item_data, dict) else []
    share = item_data.get("share", {}) if isinstance(item_data, dict) else {}

    if photos or videos or audio_files or share:
        # Found directly in the first layer
        if photos: content_type = "image"
        if videos: content_type = "video"
        if audio_files: content_type = "voice"
        if share: content_type = "link"

        for p in (photos if isinstance(photos, list) else []): 
            if isinstance(p, dict): media_attachments.append(p.get("uri", ""))
        for v in (videos if isinstance(videos, list) else []): 
            if isinstance(v, dict): media_attachments.append(v.get("uri", ""))
        for a in (audio_files if isinstance(audio_files, list) else []): 
            if isinstance(a, dict): media_attachments.append(a.get("uri", ""))
        if isinstance(share, dict) and share.get("link"): 
            media_attachments.append(share.get("link"))
    elif attachments and isinstance(attachments, list):
        # Handle deep attachments structure in Meta data
        for att in attachments:
            if not isinstance(att, dict): continue
            data_list = att.get("data", [])
            if not isinstance(data_list, list): continue
            for d in data_list:
                if not isinstance(d, dict): continue
                if "media" in d and isinstance(d["media"], dict):
                    uri = d["media"].get("uri", "")
                    if uri: media_attachments.append(uri)
                    content_type = "video" if ".mp4" in uri.lower() else "image"
                elif "external_context" in d and isinstance(d["external_context"], dict):
                    url = d["external_context"].get("url", "")
                    if url: media_attachments.append(url)
                    content_type = "link"
                elif "audio_file" in d and isinstance(d["audio_file"], dict):
                    uri = d["audio_file"].get("uri", "")
                    if uri: media_attachments.append(uri)
                    content_type = "voice"

    # If there are multiple types, could label as mixed
    if len(set(media_attachments)) > 1 and content_type != "text":
        pass  # Keep primary type based on last match, or change to "mixed" if needed

    return content_type, [m for m in media_attachments if m]

# ---------------------------------------------------------
# Parsing Functions
# ---------------------------------------------------------

def parse_posts(file_path: Path, source: str) -> list:
    """Parse post data"""
    posts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Meta posts are usually wrapped in a list or a specific key
            raw_posts = data if isinstance(data, list) else data.get("posts", [])
            if not isinstance(raw_posts, list):
                if isinstance(data, dict):
                    raw_posts = data.get("data", [])
                else:
                    raw_posts = []

            for rp in raw_posts:
                if not isinstance(rp, dict): continue
                timestamp = rp.get("timestamp") or rp.get("creation_timestamp")
                # Extract post text
                post_data_list = rp.get("data", [])
                page_content = ""
                if isinstance(post_data_list, list):
                    for pd in post_data_list:
                        if isinstance(pd, dict) and "post" in pd:
                            page_content = fix_meta_encoding(pd.get("post", ""))

                # If no data.post, try to find title directly
                if not page_content and "title" in rp:
                    page_content = fix_meta_encoding(rp.get("title", ""))

                content_type, media_attachments = determine_content_and_media(rp)

                # Skip if no content and no media
                if not page_content and not media_attachments:
                    continue

                posts.append({
                    "post_id": str(uuid.uuid4()),  # Assign unique ID
                    "page_content": page_content,
                    "metadata": {
                        "source": source,
                        "timestamp": convert_to_iso8601(timestamp),
                        "content_type": content_type,
                        "media_attachments": media_attachments
                    },
                    "comments": []  # Reserve empty array for relation mapping
                })
    except Exception as e:
        logger.error(f"Error parsing post {file_path}: {e}")
    return posts

def parse_comments(file_path: Path) -> list:
    """Parse comment data"""
    comments = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, list):
                raw_comments = data
            elif isinstance(data, dict):
                raw_comments = data.get("comments_v2", data.get("comments", []))
            else:
                raw_comments = []

            if not isinstance(raw_comments, list):
                raw_comments = []

            for rc in raw_comments:
                if not isinstance(rc, dict): continue
                timestamp = rc.get("timestamp")
                data_list = rc.get("data", [])
                page_content = ""
                author = ""

                if isinstance(data_list, list):
                    for d in data_list:
                        if isinstance(d, dict) and "comment" in d:
                            comment_data = d.get("comment")
                            if isinstance(comment_data, dict):
                                page_content = fix_meta_encoding(comment_data.get("comment", ""))
                                author = fix_meta_encoding(comment_data.get("author", ""))
                            elif isinstance(comment_data, str):
                                page_content = fix_meta_encoding(comment_data)

                if not page_content and "title" in rc:
                    page_content = fix_meta_encoding(rc.get("title", ""))

                content_type, _ = determine_content_and_media(rc)

                if page_content:
                    comments.append({
                        "comment_id": str(uuid.uuid4()),
                        "page_content": page_content,
                        "metadata": {
                            "author": author or "unknown",
                            "timestamp": convert_to_iso8601(timestamp),
                            "content_type": content_type
                        },
                        "target_title": fix_meta_encoding(rc.get("title", ""))  # Temp storage for post matching
                    })
    except Exception as e:
        logger.error(f"Error parsing comments {file_path}: {e}")
    return comments

def parse_messages(file_path: Path, source: str) -> dict:
    """Parse message/thread data"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return None

            participants = []
            for p in data.get("participants", []):
                if isinstance(p, dict):
                    participants.append(fix_meta_encoding(p.get("name", "")))
            
            raw_messages = data.get("messages", [])
            if not isinstance(raw_messages, list):
                raw_messages = []

            conversation = []
            for rm in raw_messages:
                if not isinstance(rm, dict): continue
                sender_name = fix_meta_encoding(rm.get("sender_name", ""))
                timestamp = rm.get("timestamp_ms")
                page_content = fix_meta_encoding(rm.get("content", ""))

                content_type, media_attachments = determine_content_and_media(rm)

                # Mark as share_link if it's a share
                if "share" in rm:
                    content_type = "share_link"

                if not page_content and not media_attachments:
                    continue

                conversation.append({
                    "message_id": str(uuid.uuid4()),
                    "page_content": page_content,
                    "metadata": {
                        "sender_name": sender_name,
                        "timestamp": convert_to_iso8601(timestamp),
                        "content_type": content_type,
                        "media_attachments": media_attachments
                    }
                })

            if conversation:
                # Adjust source name for Messenger/IG Direct
                msg_source = source + "_messenger" if source == "facebook" else (source + "_direct" if source == "instagram" else source)

                return {
                    "thread_id": str(uuid.uuid4()),
                    "metadata": {
                        "source": msg_source,
                        "participants": participants
                    },
                    "conversation": conversation[::-1]  # Reverse array for chronological order
                }
    except Exception as e:
        logger.error(f"Error parsing messages {file_path}: {e}")
    return None

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    start_time = datetime.now()
    logger.info("Process Started")
    base_dir = Path(BASE_DIR)
    
    # Ensure output directory exists
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
        progress_data = {"processed_files": [], "total_files": 0}
    else:
        logger.info(f"Resuming process. {len(progress_data.get('processed_files', []))} files already processed.")

    processed_files = progress_data.get("processed_files", [])
    total_files = progress_data.get("total_files", 0)

    batch_posts = []
    batch_messages = []
    all_comments = []

    try:
        # 1. Iterate through files
        for root, _, files in os.walk(base_dir):
            root_path = Path(root)

            # Determine Source
            source = "unknown"
            if "facebook_data" in root_path.parts:
                source = "facebook"
            elif "instagram_data" in root_path.parts:
                source = "instagram"
            elif "threads_data" in root_path.parts:
                source = "threads"

            for file in files:
                if not file.endswith('.json'):
                    continue

                file_path = root_path / file
                file_rel_path = str(file_path.relative_to(base_dir))

                if file_rel_path in processed_files:
                    continue

                total_files += 1
                logger.info(f"Processing: {file_rel_path}")

                file_name = file.lower()
                
                # Divert based on filename and folder path
                if "message" in file_name or "inbox" in root_path.parts:
                    thread_data = parse_messages(file_path, source)
                    if thread_data:
                        batch_messages.append(thread_data)

                elif "comment" in file_name:
                    parsed_comments = parse_comments(file_path)
                    all_comments.extend(parsed_comments)

                elif "post" in file_name or "profile" in file_name:
                    parsed_posts = parse_posts(file_path, source)
                    batch_posts.extend(parsed_posts)

                processed_files.append(file_rel_path)
                progress_data["processed_files"] = processed_files
                progress_data["total_files"] = total_files
                save_progress(progress_data)

                # Batch writing logic
                if len(batch_posts) >= 100 or len(batch_messages) >= 100:
                    # Resolve comments for current batch of posts
                    for comment in all_comments:
                        target_title = comment.get("target_title", "")
                        matched = False
                        if target_title:
                            for post in batch_posts:
                                if post["page_content"] and post["page_content"][:20] in target_title:
                                    post["comments"].append(comment)
                                    matched = True
                                    break
                        if not matched:
                            # Not matching in this batch, keep it in all_comments for next batch or final processing
                            pass
                            
                    save_batch({"posts": batch_posts, "messages": batch_messages}, output_dir, base_filename)
                    batch_posts = []
                    batch_messages = []
                    # Keep all_comments? It's tricky to map comments if posts are split across batches. 
                    # Assuming we keep unmatched comments until the end.

        # Process remaining comments for final batch
        if batch_posts or batch_messages or all_comments:
            for comment in all_comments:
                target_title = comment.pop("target_title", "")
                matched = False
                if target_title:
                    for post in batch_posts:
                        if post["page_content"] and post["page_content"][:20] in target_title:
                            post["comments"].append(comment)
                            matched = True
                            break
                if not matched:
                    batch_posts.append({
                        "post_id": str(uuid.uuid4()),
                        "page_content": "Orphaned Comment",
                        "metadata": {
                            "source": "comments",
                            "timestamp": comment["metadata"]["timestamp"],
                            "content_type": "text",
                            "media_attachments": []
                        },
                        "comments": [comment]
                    })
            save_batch({"posts": batch_posts, "messages": batch_messages}, output_dir, base_filename)

        end_time = datetime.now()
        execution_time = end_time - start_time
        
        logger.info("-" * 40)
        logger.info("Meta data parsing completed - Final Report:")
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
    main()