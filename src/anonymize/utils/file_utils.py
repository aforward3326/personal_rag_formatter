import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_progress(progress_file: Path) -> Dict[str, Any]:
    """Loads the progress tracking file."""
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading progress from {progress_file}: {e}")
    return {"processed_files": [], "total_files": 0}

def save_progress(progress_file: Path, progress_data: Dict[str, Any]):
    """Saves the progress tracking file."""
    try:
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    except IOError as e:
        logger.error(f"Error saving progress to {progress_file}: {e}")

def clear_progress(progress_file: Path):
    """Deletes the progress file if it exists."""
    try:
        if progress_file.exists():
            progress_file.unlink()
    except OSError as e:
        logger.error(f"Error clearing progress file {progress_file}: {e}")

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
    
    # If the base timestamped file exists, start adding a batch number
    batch = 1
    while True:
        new_filename = f"{name}_{timestamp}_{batch}.{ext}"
        new_path = output_dir / new_filename
        if not new_path.exists():
            return new_path
        batch += 1

def save_json_file(data: Any, output_file_path: Path):
    """Saves data to a JSON file."""
    try:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully saved anonymized file: {output_file_path.name}")
    except IOError as e:
        logger.error(f"Error saving JSON file {output_file_path}: {e}")
