import os
import sys
import asyncio
import argparse
import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables from the root .env file BEFORE importing any project modules
# This makes the environment ready for all subsequent imports.
load_dotenv()

# Now, import project modules that rely on the configured environment
from src.rag_pipeline import config
from src.rag_pipeline.utils.logger import logger
from src.rag_pipeline.services import ingestion_service, batch_service

def _initialize_directories():
    """Creates all necessary directories for the pipeline to run."""
    dirs_to_create = [
        config.DATA_DIR,
        config.PROCESS_CACHE_DIR,
        config.PROCESS_CLEANED_DIR,
        config.PROCESS_ERROR_DIR,
        config.PROC_DIR,
        config.ARCHIVE_DIR,
        config.TEMPLATE_DIR,
        config.BATCH_PROC_DIR
    ]
    for d in dirs_to_create:
        # Use the Path object's mkdir for a more modern approach
        Path(d).mkdir(parents=True, exist_ok=True)

def _print_summary(metrics: defaultdict, start_time: datetime, mode: str):
    """Prints a formatted summary of the pipeline execution."""
    end_time = datetime.now()
    total_time = end_time - start_time
    
    logger.info("==========================================")
    logger.info("RAG Ingestion Pipeline Completed")
    logger.info("==========================================")
    logger.info(f"Total Execution Time: {total_time}")
    
    if config.USE_BATCH_API:
        logger.info(f"Mode: BATCH API ({mode})")
        logger.info(f"LLM Provider: {config.AI_PROVIDER.upper()}, Model: {config.BATCH_MODEL_NAME}")
        if mode == "submit":
            logger.info("---")
            logger.info(f"File Processing:")
            logger.info(f"  - Total Files Read: {metrics['total_files']}")
            logger.info(f"  - Documents Extracted: {metrics['total_documents']}")
            logger.info(f"  - Chunks Generated: {metrics['chunks_created']}")
            logger.info(f"  - Chunks Prepared for Batch: {metrics['chunks_prepared_for_batch']}")
            logger.info(f"  - Processing Errors (pre-batch): {metrics['processing_errors']}")
        elif mode in ["import_results", "ingest_db", "recover"]:
            if mode == "import_results":
                logger.info("---")
                logger.info(f"LLM Analysis (from Batch):")
                logger.info(f"  - Chunks Successfully Analyzed: {metrics['successful_chunks']}")
                logger.info(f"  - Chunks Filtered (Noise/Ad): {metrics['noise_filtered']}")
                logger.info(f"  - Processing Errors (post-batch): {metrics['processing_errors']}")
            logger.info("---")
            logger.info(f"Database Ingestion ({config.VECTOR_DB_TYPE.upper()}):")
            logger.info(f"  - Chunks Queued for Ingestion: {metrics['chunks_to_ingest']}")
            logger.info(f"  - Chunks Successfully Ingested: {metrics['ingested_chunks']}")
            logger.info(f"  - Ingestion Errors: {metrics['ingestion_errors']}")
            logger.info(f"  - Archiving Errors: {metrics['archive_errors']}")
    else:
        logger.info(f"Mode: Standard (Real-time) Ingestion")
        logger.info(f"LLM Provider: {config.AI_PROVIDER.upper()}, Model: {config.LLM_MODEL_NAME}")
        logger.info(f"Embedding Provider: {config.EMBEDDING_PROVIDER.upper()}, Model: {config.EMBEDDING_MODEL_NAME}")
        logger.info("---")
        logger.info(f"File Processing & Chunking:")
        logger.info(f"  - Total Files Read: {metrics['total_files']}")
        logger.info(f"  - Documents Extracted: {metrics['total_documents']}")
        logger.info(f"  - Chunks Generated: {metrics['chunks_created']}")
        logger.info("---")
        logger.info(f"LLM Analysis:")
        logger.info(f"  - Chunks Successfully Analyzed: {metrics['successful_chunks']}")
        logger.info(f"  - Chunks Filtered (Noise/Ad): {metrics['noise_filtered']}")
        logger.info(f"  - Processing Errors: {metrics['processing_errors']}")
        logger.info("---")
        logger.info(f"Token Usage:")
        logger.info(f"  - Input Tokens: {metrics['total_input_tokens']:,}")
        logger.info(f"  - Output Tokens: {metrics['total_output_tokens']:,}")
        logger.info(f"  - Total Tokens: {metrics['total_input_tokens'] + metrics['total_output_tokens']:,}")
        logger.info("---")
        logger.info(f"Database Ingestion ({config.VECTOR_DB_TYPE.upper()}):")
        logger.info(f"  - Chunks Queued for Ingestion: {metrics['chunks_to_ingest']}")
        logger.info(f"  - Chunks Successfully Ingested: {metrics['ingested_chunks']}")
        logger.info(f"  - Ingestion Errors: {metrics['ingestion_errors']}")
        logger.info(f"  - Archiving Errors: {metrics['archive_errors']}")
    logger.info("==========================================")

async def main():
    """Main entry point for the RAG ingestion pipeline."""
    start_time = datetime.now()
    logger.info("Process Started: RAG data ingestion pipeline...")

    metrics = defaultdict(int)
    
    parser = argparse.ArgumentParser(description="RAG Ingestion Pipeline")
    
    if config.USE_BATCH_API:
        parser.add_argument("--mode", type=str, default="submit", choices=["submit", "import_results", "ingest_db", "recover"],
                            help="Execution mode for Batch API: 'submit', 'import_results', 'ingest_db', or 'recover' for orphaned jsonl files.")
        parser.add_argument("--job-name", type=str, help="The Batch API job name to import results from.")
    else:
        parser.add_argument("--mode", type=str, default="ingest", choices=["ingest"],
                            help="Execution mode for Standard API: 'ingest' processes and ingests files in real-time.")

    args = parser.parse_args()

    _initialize_directories()
    
    api_semaphore = asyncio.Semaphore(config.API_BATCH_SIZE)

    if config.USE_BATCH_API:
        logger.info("Batch API mode is enabled.")
        if args.mode == "submit":
            await ingestion_service.create_pending_chunks(metrics)
            await batch_service.run_batch_submission_pipeline(metrics)
        
        elif args.mode == "import_results":
            job_name = args.job_name
            if not job_name:
                if os.path.exists(config.BATCH_JOB_FILE):
                    with open(config.BATCH_JOB_FILE, "r") as f:
                        job_name = f.read().strip()
                else:
                    logger.error("Must provide --job-name or have an active job file to import results.")
                    sys.exit(1)
            await batch_service.import_batch_results(job_name, metrics, api_semaphore)
            
        elif args.mode == "ingest_db":
            logger.info("Executing standalone Database Ingestion...")
            await ingestion_service.ingest_cleaned_files_to_db(api_semaphore, metrics)
            
        elif args.mode == "recover":
            logger.info("Executing Emergency Recovery Mode...")
            await batch_service.recover_all_orphaned_results(metrics, api_semaphore)
            
    else: # Standard, real-time API mode
        logger.info("Standard (real-time) API mode is enabled.")
        
        if not ingestion_service.handle_existing_files_prompt():
            return # User chose to abort

        await ingestion_service.create_pending_chunks(metrics)

        pending_files = glob.glob(os.path.join(config.PROCESS_CACHE_DIR, "*.pending"))
        if pending_files:
            logger.info(f"Found {len(pending_files)} pending chunks. Starting LLM analysis...")
            
            tasks = []
            for i in range(0, len(pending_files), config.API_BATCH_SIZE):
                batch_files = pending_files[i:i + config.API_BATCH_SIZE]
                tasks.append(ingestion_service.process_chunk_batch(batch_files, api_semaphore, metrics))
            
            try:
                from tqdm.asyncio import tqdm as async_tqdm
                results = await async_tqdm.gather(*tasks, desc="Analyzing Chunks")
            except ImportError:
                results = await asyncio.gather(*tasks)

            for in_tokens, out_tokens in results:
                metrics["total_input_tokens"] += in_tokens
                metrics["total_output_tokens"] += out_tokens
        
        await ingestion_service.ingest_cleaned_files_to_db(api_semaphore, metrics)
        
        logger.info("Cleaning up cache files...")
        for ext in ["*.pending", "*.processing", "*.done", "*.error"]:
            for f in glob.glob(os.path.join(config.PROCESS_CACHE_DIR, ext)):
                try: os.remove(f)
                except OSError: pass

    _print_summary(metrics, start_time, args.mode)

def run():
    """
    This wrapper function allows the script to be executed as a console script
    if defined in pyproject.toml.
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected critical error occurred in main: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run()