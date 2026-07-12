import logging
import sys
from pathlib import Path
from src.rag_pipeline import config

def setup_logger() -> logging.Logger:
    """
    Configures and returns a logger for the application.
    
    The logger will output to both a file and the console. It avoids
    adding duplicate handlers if called multiple times.
    """
    process_log_dir = Path(config.LOG_DIR) / config.PROCESS_NAME
    process_log_dir.mkdir(parents=True, exist_ok=True)
    
    base_log_filename = f"{config.PROCESS_NAME}_{config.RUN_TIME_STR}.log"
    log_file_path = process_log_dir / base_log_filename
    
    logger = logging.getLogger(config.PROCESS_NAME)
    logger.setLevel(logging.INFO)
    
    # Avoid adding handlers if they already exist
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

# Initialize and export the logger instance
logger = setup_logger()