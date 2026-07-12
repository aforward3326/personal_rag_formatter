import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

def get_output_file_path(output_dir: Path, base_filename: str) -> Path:
    """
    Generates a unique output file path to avoid overwriting.
    Appends a timestamp and, if needed, a batch number.
    Example: my_file.json -> my_file_2023102714.json -> my_file_2023102714_1.json
    """
    timestamp = datetime.now().strftime("%Y%m%d%H")
    name, ext = base_filename.rsplit('.', 1)
    
    base_new_filename = f"{name}_{timestamp}.{ext}"
    output_new_path = output_dir / base_new_filename
    
    if not output_new_path.exists():
        return output_new_path
    
    batch = 1
    while True:
        new_filename = f"{name}_{timestamp}_{batch}.{ext}"
        new_path = output_dir / new_filename
        if not new_path.exists():
            return new_path
        batch += 1

def save_json_batch(data: List[Dict[str, Any]], output_dir: Path, base_filename: str, batch_key: str):
    """
    Saves a list of data into a batched JSON file, wrapped in a root key.
    
    :param data: The list of dictionary data to save.
    :param output_dir: The directory to save the file in.
    :param base_filename: The base name for the output file.
    :param batch_key: The root key to wrap the data list in (e.g., "email_threads", "posts").
    """
    if not data:
        return
        
    output_file_path = get_output_file_path(output_dir, base_filename)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump({batch_key: data}, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully saved batch of {len(data)} items to {output_file_path.name}")
    except IOError as e:
        logging.error(f"Error saving batch JSON file {output_file_path}: {e}")

def load_progress(progress_file: Path) -> Dict[str, Any]:
    """Loads the progress tracking file."""
    if progress_file and progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error loading progress from {progress_file}: {e}")
    return {}

def save_progress(progress_file: Path, progress_data: Dict[str, Any]):
    """Saves the progress tracking file."""
    if not progress_file: return
    try:
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    except IOError as e:
        logging.error(f"Error saving progress to {progress_file}: {e}")

def clear_progress(progress_file: Path):
    """Deletes the progress file if it exists."""
    if not progress_file: return
    try:
        if progress_file.exists():
            progress_file.unlink()
    except OSError as e:
        logging.error(f"Error clearing progress file {progress_file}: {e}")

def prompt_resume(progress_data: dict, logger: logging.Logger) -> bool:
    """Prompts the user to resume based on loaded progress data."""
    if progress_data and (progress_data.get("processed_files") or progress_data.get("current_file")):
        try:
            response = input(f"Found interrupted progress. Resume? (y/n): ").strip().lower()
            return response == 'y'
        except (EOFError, KeyboardInterrupt):
            logger.warning("Input not available, not resuming.")
            return False
    return False
