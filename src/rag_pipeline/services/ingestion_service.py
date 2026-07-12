import os
import sys
import json
import glob
import asyncio
import hashlib
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
from pydantic import ValidationError

from src.rag_pipeline import config
from src.rag_pipeline.utils.logger import logger
from src.rag_pipeline.utils import file_parser, chunking
from src.rag_pipeline.db import crud
from src.rag_pipeline.db.connection import db_manager
from src.rag_pipeline.db.schemas import DBChunk
from src.rag_pipeline.services import analysis_service

async def create_pending_chunks(metrics: defaultdict):
    """Parses source files and creates .pending chunk files in the cache directory."""
    files_to_process = sorted(Path(config.DATA_DIR).glob("*.json"))
    metrics["total_files"] = len(files_to_process)
    logger.info(f"Found {metrics['total_files']} files to process in '{config.DATA_DIR}'.")

    if not files_to_process:
        return

    for filepath in tqdm(files_to_process, desc="Parsing Source Files"):
        current_file_name = filepath.name
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            docs = file_parser.parse_rag_ready_json(data, current_file_name)
            metrics["total_documents"] += len(docs)
        except Exception as e:
            logger.error(f"Failed to read/parse {current_file_name}: {e}")
            metrics["file_errors"] += 1
            continue

        if not docs:
            continue

        grouped_docs = defaultdict(list)
        for doc in docs:
            meta = doc.get("metadata", {})
            group_id = meta.get("thread_id") or meta.get("post_id") or meta.get("subject") or current_file_name
            grouped_docs[group_id].append(doc["page_content"])

        for group_id, contents in grouped_docs.items():
            full_text = "\n".join(contents)
            chunks = chunking.create_sliding_window_chunks(full_text, group_id)

            for i, chunk_data in enumerate(chunks):
                metrics["chunks_created"] += 1
                chunk_data["file_name"] = current_file_name
                safe_group_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(group_id))
                cache_filename = f"{config.PROCESS_NAME}_{config.RUN_TIME_STR}_{safe_group_id}_chunk_{i}.json.pending"
                cache_file = Path(config.PROCESS_CACHE_DIR) / cache_filename
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(chunk_data, f, ensure_ascii=False, indent=2)

async def process_chunk_batch(
    file_paths: List[Path],
    semaphore: asyncio.Semaphore,
    metrics: defaultdict
) -> Tuple[int, int]:
    """Processes a batch of pending chunk files using the real-time API."""
    batch_payload = []
    chunk_data_map = {}

    for file_path in file_paths:
        processing_path = file_path.with_suffix(".processing")
        try:
            os.rename(file_path, processing_path)
        except OSError as e:
            logger.warning(f"Failed to rename {file_path} to .processing: {e}")
            metrics["processing_errors"] += 1
            continue

        try:
            with open(processing_path, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)

            chunk_id = processing_path.name
            chunk_data_map[chunk_id] = {"path": processing_path, "data": chunk_data}

            batch_payload.append({
                "chunk_id": chunk_id,
                "main_content": chunk_data.get("main_content", ""),
                "overlap_context": chunk_data.get("overlap_context", "")
            })
        except Exception as e:
            logger.error(f"Failed to read {processing_path}: {e}")
            metrics["processing_errors"] += 1
            os.rename(processing_path, processing_path.with_suffix(".error"))

    if not batch_payload:
        return 0, 0

    try:
        batch_result, in_tokens, out_tokens = await analysis_service.analyze_batch_of_chunks_for_cleaning(batch_payload, semaphore)
        results_dict = {res.chunk_id: res.analysis for res in batch_result.results}

        # Create a list of tasks for _handle_analysis_result
        tasks = []
        for chunk_id, info in chunk_data_map.items():
            task = _handle_analysis_result(chunk_id, results_dict.get(chunk_id), info, metrics, semaphore)
            tasks.append(task)

        # Run all summarization and processing tasks concurrently
        await asyncio.gather(*tasks)

        return in_tokens, out_tokens

    except Exception as e:
        logger.error(f"Critical error processing batch of size {len(batch_payload)}: {e}")
        metrics["processing_errors"] += len(batch_payload)
        for info in chunk_data_map.values():
            try:
                os.rename(info["path"], info["path"].with_suffix(".error"))
            except OSError:
                pass
        return 0, 0

async def _handle_analysis_result(chunk_id: str, analysis: Any, chunk_info: Dict, metrics: defaultdict, semaphore: asyncio.Semaphore):
    """Helper to process the analysis result for a single chunk."""
    processing_path = Path(chunk_info["path"])
    chunk_data = chunk_info["data"]
    
    if analysis:
        is_invalid = analysis.is_ad or analysis.is_noise
        if not is_invalid:
            try:
                # Directly use the ai_summary generated during Batch LLM analysis to save additional LLM API costs
                summary = analysis.ai_summary

                out_filename = processing_path.with_suffix(".json").name
                out_filepath = Path(config.PROCESS_CLEANED_DIR) / out_filename

                match = re.search(r"_chunk_(\d+)\.json", out_filename)
                chunk_index = int(match.group(1)) if match else 0

                final_output = DBChunk(
                    file_name=chunk_data.get("file_name", chunk_data.get("group_id", "unknown")),
                    group_title=chunk_data.get("group_id"),
                    chunk_index=chunk_index,
                    raw_content=chunk_data.get("main_content", ""),
                    content_hash=hashlib.sha256(str(chunk_data.get("main_content", "")).encode('utf-8')).hexdigest(),
                    **analysis.model_dump()
                )
                with open(out_filepath, "w", encoding="utf-8") as out_f:
                    json.dump(final_output.model_dump(), out_f, ensure_ascii=False, indent=2)
                metrics["successful_chunks"] += 1
            except Exception as e:
                logger.error(f"Error during summarization or file writing for {chunk_id}: {e}")
                metrics["processing_errors"] += 1
                if processing_path.exists():
                    os.rename(processing_path, processing_path.with_suffix(".error"))
                return
        else:
            metrics["noise_filtered"] += 1

            if processing_path.exists():
                os.rename(processing_path, processing_path.with_suffix(".done"))
    else:
        logger.warning(f"Analysis missing for {chunk_id} in batch response. Marking as error.")
        metrics["processing_errors"] += 1
        if processing_path.exists():
            os.rename(processing_path, processing_path.with_suffix(".error"))

async def ingest_cleaned_files_to_db(semaphore: asyncio.Semaphore, metrics: defaultdict):
    """Finds cleaned JSON files and ingests them into the vector database."""
    cleaned_files = list(Path(config.PROCESS_CLEANED_DIR).glob("*.json"))
    if not cleaned_files:
        logger.info(f"No cleaned files found in {config.PROCESS_CLEANED_DIR} to ingest.")
        return

    logger.info(f"Found {len(cleaned_files)} cleaned files to ingest into the database.")
    metrics["chunks_to_ingest"] = len(cleaned_files)

    try:
        await db_manager.connect()
        db_connection = db_manager.get_connection()
        if not db_connection:
            raise ConnectionError("Failed to get a valid database connection.")

        chunks_with_paths = []
        for cf_path in cleaned_files:
            try:
                with open(cf_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # The 'embedding' field is not present in the JSON, so we exclude it from validation
                    chunks_with_paths.append({"data": DBChunk.model_validate(data), "path": cf_path})
            except (json.JSONDecodeError, IOError, ValidationError) as e:
                logger.error(f"Could not read or parse {cf_path}, skipping: {e}")
                metrics["ingestion_errors"] += 1

        archive_run_dir = Path(config.ARCHIVE_DIR) / config.RUN_TIME_STR
        archive_run_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(0, len(chunks_with_paths), config.DB_BATCH_SIZE), desc="Ingesting to DB"):
            batch_with_paths = chunks_with_paths[i : i + config.DB_BATCH_SIZE]
            batch_chunks = [item["data"] for item in batch_with_paths]
            batch_paths = [item["path"] for item in batch_with_paths]

            try:
                # Use ai_summary to generate Embeddings (HyDE architecture)
                texts_to_embed = [c.ai_summary for c in batch_chunks]
                embeddings = await analysis_service.get_embeddings_for_texts(texts_to_embed, semaphore)
                
                if embeddings and len(embeddings) == len(batch_chunks):
                    await crud.insert_chunks_batch(batch_chunks, embeddings, db_connection)
                    metrics["ingested_chunks"] += len(batch_chunks)

                    for path in batch_paths:
                        try:
                            archive_path = archive_run_dir / Path(path).name
                            os.rename(path, archive_path)
                        except OSError as e:
                            logger.error(f"Failed to archive {Path(path).name}: {e}")
                            metrics["archive_errors"] += 1
                else:
                    logger.warning(f"Embedding generation failed for a batch. Skipping DB insertion.")
                    metrics["ingestion_errors"] += len(batch_chunks)

            except Exception as e:
                logger.error(f"Error inserting batch into DB: {e}", exc_info=True)
                metrics["ingestion_errors"] += len(batch_chunks)

    except Exception as e:
        logger.critical(f"A critical error occurred during the ingestion process: {e}", exc_info=True)
    finally:
        await db_manager.close()

def handle_existing_files_prompt():
    """
    Checks for existing cleaned or cached files and prompts the user for action.
    Returns True if the pipeline should continue, False otherwise.
    """
    cleaned_files = list(Path(config.PROCESS_CLEANED_DIR).glob("*.json"))
    if cleaned_files:
        logger.warning(f"Found {len(cleaned_files)} unprocessed files in the cleaned data directory.")
        if sys.stdin.isatty():
            choice = input("These files have been processed by the LLM but not ingested. Do you want to DELETE them and continue? (y/n) [Default: n]: ").strip().lower()
            if choice == 'y':
                logger.info(f"User confirmed. Deleting {len(cleaned_files)} files...")
                for f in cleaned_files:
                    try: os.remove(f)
                    except OSError as e: logger.error(f"Failed to delete file {f}: {e}")
            else:
                logger.info("Aborting pipeline. Please handle or move the existing cleaned files before restarting.")
                return False
        else:
            logger.error("Non-interactive environment detected with existing cleaned files. Aborting.")
            return False

    cached_files = list(Path(config.PROCESS_CACHE_DIR).glob("*"))
    if cached_files:
        logger.warning(f"Found {len(cached_files)} files in the cache directory.")
        if sys.stdin.isatty():
            choice = input("Do you want to clear the cache and restart, or continue processing? (clear/continue) [Default: continue]: ").strip().lower()
            if choice == 'clear':
                logger.info("Clearing cache...")
                for f in cached_files: os.remove(f)
        else:
            logger.info("Non-interactive environment. Continuing with existing cache.")

    orphaned_files = list(Path(config.PROCESS_CACHE_DIR).glob("*.processing")) + list(Path(config.PROCESS_CACHE_DIR).glob("*.error"))
    if orphaned_files:
        logger.info(f"Reverting {len(orphaned_files)} orphaned cache files to .pending state...")
        for f in orphaned_files:
            os.rename(f, f.with_suffix(".pending"))
            
    return True