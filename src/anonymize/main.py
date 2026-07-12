import logging
import sys
import json
from datetime import datetime
from pathlib import Path

from . import config
from .factory import AnonymizerFactory
from .utils import file_utils

# --- Logger Setup ---
def setup_logger():
    """Configures the main logger for the application."""
    log_dir = config.LOG_DIR / config.PROCESS_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"{config.PROCESS_NAME}_{date_str}.log"
    
    # Basic logger configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    # Return the logger instance for the current module
    return logging.getLogger(__name__)

logger = setup_logger()

# --- Main Application Logic ---
def main():
    """
    Main function to run the anonymization pipeline.
    Orchestrates file discovery, processing, and progress tracking.
    """
    start_time = datetime.now()
    logger.info("Anonymization process started.")

    # --- Directory and Progress Setup ---
    config.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    progress_data = file_utils.load_progress(config.PROGRESS_FILE)
    
    if progress_data.get("processed_files"):
        resume = input(f"Found interrupted progress. Resume? (y/n): ").strip().lower()
        if resume != 'y':
            logger.info("Starting fresh, clearing previous progress.")
            file_utils.clear_progress(config.PROGRESS_FILE)
            progress_data = {"processed_files": [], "total_files": 0}
    
    processed_files = set(progress_data.get("processed_files", []))

    # --- Processor Initialization ---
    logger.info("Initializing anonymization processor...")
    try:
        anonymizer = AnonymizerFactory.create_processor()
    except Exception as e:
        logger.critical(f"Failed to create anonymizer processor: {e}", exc_info=True)
        sys.exit(1)
    logger.info("Processor initialized successfully.")

    # --- File Processing Loop ---
    logger.info(f"Scanning for .json files in: {config.INPUT_DIR}")
    files_to_process = list(config.INPUT_DIR.rglob('*.json'))
    
    if not files_to_process:
        logger.warning("No .json files found in the input directory. Exiting.")
        return

    total_files_count = len(files_to_process)
    logger.info(f"Found {total_files_count} files to process.")

    try:
        for file_path in files_to_process:
            relative_path_str = str(file_path.relative_to(config.INPUT_DIR))
            
            if relative_path_str in processed_files:
                logger.debug(f"Skipping already processed file: {file_path.name}")
                continue

            logger.info(f"Processing file: {file_path.name}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # The core logic is now a single call to the processor
                anonymized_data = anonymizer.process_document(data)

                # Determine output path
                output_subdir = config.OUTPUT_DIR / file_path.relative_to(config.INPUT_DIR).parent
                base_filename = f"{file_path.stem}_masked{file_path.suffix}"
                output_path = file_utils.get_output_file_path(output_subdir, base_filename)

                # Save the processed file
                file_utils.save_json_file(anonymized_data, output_path)

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in {file_path.name}, skipping.")
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}", exc_info=True)

            # Update and save progress after each file
            processed_files.add(relative_path_str)
            progress_data["processed_files"] = list(processed_files)
            progress_data["total_files"] = total_files_count
            file_utils.save_progress(config.PROGRESS_FILE, progress_data)

        # --- Final Report ---
        end_time = datetime.now()
        logger.info("-" * 40)
        logger.info("Anonymization pipeline completed.")
        logger.info(f"Total files processed: {len(processed_files)}")
        logger.info(f"Total execution time: {end_time - start_time}")
        logger.info("-" * 40)

        # Clean up progress file on successful completion
        file_utils.clear_progress(config.PROGRESS_FILE)

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Progress saved.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # This allows the script to be run directly, e.g., python -m src.anonymize.main
    main()
