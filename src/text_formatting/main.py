import argparse
import logging
from datetime import datetime

from . import config
from .factory import FormatterFactory

def setup_logger():
    """Configures a basic logger for the main application."""
    log_dir = config.LOG_DIR / "text_formatting"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"formatter_main_{date_str}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """
    Main entry point for the text formatting module.
    
    Parses command-line arguments to select and run the desired
    formatting strategy.
    """
    logger = setup_logger()
    
    parser = argparse.ArgumentParser(
        description="Text Formatting Pipeline for RAG."
    )
    parser.add_argument(
        "--formatter",
        type=str,
        required=True,
        choices=['gmail', 'chat', 'meta', 'corpus'],
        help="The name of the formatter strategy to run."
    )
    
    args = parser.parse_args()
    
    logger.info(f"Attempting to run formatter: '{args.formatter}'")
    
    # Use the factory to get the correct strategy instance
    formatter = FormatterFactory.create_formatter(args.formatter)
    
    if formatter:
        try:
            # Execute the main processing logic of the selected strategy
            formatter.process()
            logger.info(f"Formatter '{args.formatter}' completed successfully.")
        except Exception as e:
            logger.critical(
                f"A critical error occurred while running the '{args.formatter}' formatter.",
                exc_info=True
            )
    else:
        logger.error(f"Could not create formatter for '{args.formatter}'. Aborting.")

if __name__ == "__main__":
    # This allows the script to be run as a module:
    # python -m src.text_formatting.main --formatter gmail
    main()
