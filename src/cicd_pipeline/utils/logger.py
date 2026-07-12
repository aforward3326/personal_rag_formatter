import logging
import sys
from pathlib import Path
from ..config import ProjectConfig

def setup_logger(config: ProjectConfig) -> logging.Logger:
    """
    Configures and returns the root logger for the application.
    Outputs to both console and a timestamped file, similar to the rag_pipeline.
    """
    # Create the log storage directory
    process_log_dir = Path(config.log_dir) / config.project_name
    process_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a log file path with a timestamp
    log_file_path = process_log_dir / f"{config.project_name}_{config.run_time_str}.log"
    
    # Configure the root logger so all PipelineStep class loggers can inherit these handlers
    logger = logging.getLogger()
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
        
        # File Handler - Save logs to a file
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console Handler - Output to the terminal
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    logger.info(f"Logger configured with level: {config.log_level}. Logging to {log_file_path}")
    
    return logger