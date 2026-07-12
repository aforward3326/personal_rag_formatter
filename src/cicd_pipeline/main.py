import os
import sys
import logging
import argparse
from .config import ProjectConfig
from .utils.logger import setup_logger
from .executor import PipelineExecutor
from .pipelines.code_rag import CodeRAGPipeline

def main():
    """
    Main entry point for the CI/CD pipeline execution engine.
    """
    parser = argparse.ArgumentParser(description="CI/CD Pipeline Executor")
    parser.add_argument(
        "--pipeline",
        type=str,
        default="code_rag",
        choices=['code_rag'], # Add more pipeline names here as they are created
        help="The name of the pipeline to run."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume pipeline from the last failed step using a checkpoint."
    )
    parser.add_argument(
        "--resume-batch-id",
        type=str,
        default=None,
        help="Resume pipeline from a specific Batch Job ID (e.g., cicd_1234abcd)."
    )
    args = parser.parse_args()

    if args.resume_batch_id:
        os.environ["RESUME_BATCH_ID"] = args.resume_batch_id

    try:
        # Load and validate configuration from environment
        config = ProjectConfig()
        
        # Set up the logger with the loaded configuration
        setup_logger(config)

        # --- Pipeline Factory (simple version) ---
        # In the future, this can be expanded into a more formal factory pattern
        if args.pipeline == "code_rag":
            pipeline_to_run = CodeRAGPipeline(config)
        else:
            raise ValueError(f"Unknown pipeline name: {args.pipeline}")
        
        # Initialize and run the selected pipeline
        executor = PipelineExecutor(pipeline=pipeline_to_run, resume=args.resume)
        executor.run()

    except Exception as e:
        # A fallback logger in case config loading itself fails
        logging.basicConfig()
        logger = logging.getLogger(__name__)
        logger.critical(f"A critical error occurred during pipeline execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
