import os
import sys
import json
import time
import uuid
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from pydantic import ValidationError
from google.cloud import storage
import hashlib

from src.rag_pipeline import config
from src.rag_pipeline.utils.logger import logger
from src.rag_pipeline.llm.schemas import AnalysisDetail
from src.rag_pipeline.services import ingestion_service, analysis_service

def _gcs_upload(bucket_name: str, source_file: str, dest_blob_name: str) -> str:
    """Uploads a local file to GCS and returns the GCS URI."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(source_file)
    gcs_uri = f"gs://{bucket_name}/{dest_blob_name}"
    logger.info(f"Uploaded input file to GCS: {gcs_uri}")
    return gcs_uri

def _gcs_download_and_cleanup(bucket_name: str, prefix: str, local_dir: str, file_prefix: str = "") -> list:
    """Downloads files from a GCS prefix, optionally renames them, cleans GCS, and returns local paths."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        
    logger.info("Downloading results from GCS...")
    downloaded_files = []
    for i, blob in enumerate(blobs):
        if blob.name.endswith("/"): continue
        ext = Path(blob.name).suffix
        new_filename = f"{file_prefix}_{i}{ext}" if file_prefix else os.path.basename(blob.name)
        local_path = os.path.join(local_dir, new_filename)
        blob.download_to_filename(local_path)
        logger.info(f" - Downloaded: {local_path}")
        downloaded_files.append(local_path)
        
    logger.info("Cleaning up GCS output files...")
    for blob in blobs:
        blob.delete()
    
    return downloaded_files

def _calculate_and_log_tokens(output_files: list, metrics: defaultdict):
    """Parses token usage from downloaded result files."""
    total_input, total_output = 0, 0
    for file_path in output_files:
        if not file_path.endswith(".jsonl"): continue
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    usage = data.get("response", {}).get("usageMetadata", {})
                    total_input += usage.get("promptTokenCount", 0)
                    total_output += usage.get("candidatesTokenCount", 0)
                except json.JSONDecodeError:
                    continue
    
    metrics["total_input_tokens"] += total_input
    metrics["total_output_tokens"] += total_output
    logger.info(f"Batch Token Usage - Input: {total_input}, Output: {total_output}, Total: {total_input + total_output}")

async def run_batch_submission_pipeline(metrics: defaultdict):
    """
    Prepares and submits a Vertex AI batch job for all pending chunks, then saves state for later import.
    """
    if not analysis_service.llm_client:
        logger.error("LLM client is not configured. Cannot run batch pipeline.")
        return

    pending_files = list(Path(config.PROCESS_CACHE_DIR).glob("*.pending"))
    if not pending_files:
        logger.info("No pending chunk files found to process in batch mode.")
        return

    logger.info(f"Found {len(pending_files)} pending chunks. Starting batch submission.")

    batch_run_id = f"{config.RUN_TIME_STR}_{uuid.uuid4().hex[:8]}"
    local_input_file = Path(config.PROC_DIR) / f"batch_input_{batch_run_id}.jsonl"
    gcs_input_blob = f"batch_input/{local_input_file.name}"
    gcs_output_prefix = f"batch_output/{batch_run_id}/"
    
    manifest_data = {}
    hash_to_chunk_key = {}

    try:
        # 1. Prepare local batch input file
        with open(local_input_file, "w", encoding="utf-8") as f_out:
            for file_path in tqdm(pending_files, desc="Preparing Batch Input"):
                chunk_filename = file_path.name.replace(".pending", "")
                with open(file_path, "r", encoding="utf-8") as f_in:
                    chunk_data = json.load(f_in)

                user_content = f"[Main Content]\n{chunk_data.get('main_content', '')}"
                if chunk_data.get('overlap_context'):
                    user_content = f"[Overlap Context]\n{chunk_data.get('overlap_context')}\n\n" + user_content
                
                # Format for Vertex AI Batch Prediction
                request_obj = {
                    "request": {
                        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
                        "systemInstruction": {"role": "system", "parts": [{"text": config.SYSTEM_PROMPT_TEMPLATE}]},
                        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}
                    }
                }
                f_out.write(json.dumps(request_obj) + "\n")
                manifest_data[chunk_filename] = {"path": str(file_path), "data": chunk_data}
                
                # Create a stable hash to map the response back securely regardless of order
                content_hash = hashlib.md5(user_content.encode('utf-8')).hexdigest()
                hash_to_chunk_key[content_hash] = chunk_filename
                metrics["chunks_prepared_for_batch"] += 1

        # 2. Upload to GCS
        gcs_input_uri = _gcs_upload(config.GCS_BUCKET_NAME, str(local_input_file), gcs_input_blob)

        # 3. Submit Batch Job
        logger.info(f"Submitting Batch Job for model: {config.BATCH_MODEL_NAME}...")
        job = analysis_service.llm_client.submit_batch_job(
            model_name=config.BATCH_MODEL_NAME,
            input_gcs_uri=gcs_input_uri,
            output_gcs_uri_prefix=f"gs://{config.GCS_BUCKET_NAME}/{gcs_output_prefix}"
        )
        logger.info(f"Batch Job created successfully. Job ID: {job.name}")

        # 4. Persist job state for later result import
        with open(config.BATCH_JOB_FILE, "w", encoding="utf-8") as f:
            f.write(job.name)
            
        manifest_payload = {
            "hash_to_chunk_key": hash_to_chunk_key,
            "manifest_data": manifest_data,
            "gcs_output_prefix": gcs_output_prefix,
            "batch_run_id": batch_run_id
        }
        with open(config.BATCH_MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, ensure_ascii=False, indent=2)
            
        logger.info("Batch submission complete. You can import results later using --mode import_results")

    except Exception as e:
        logger.critical(f"A critical error occurred during the batch submission pipeline: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Cleaning up temporary local input files... (GCS input preserved for Vertex AI)")
        if os.path.exists(local_input_file):
            os.remove(local_input_file)


async def import_batch_results(job_name: str, metrics: defaultdict, semaphore):
    """
    Checks job status and downloads/processes results from GCS if completed.
    """
    if not analysis_service.llm_client:
        logger.error("LLM client is not configured. Cannot import batch results.")
        return
        
    logger.info(f"Checking status for batch job: {job_name}")
    job = analysis_service.llm_client.get_batch_job_status(name=job_name)
    state_str = str(job.state).upper()
    logger.info(f"Current Job State: {state_str}")
    
    if "SUCCEEDED" not in state_str:
        if "FAILED" in state_str or "CANCELLED" in state_str:
            error_msg = getattr(job, 'error', 'No specific error message available on the job object.')
            logger.error(f"Vertex AI Batch Job FAILED! Cloud Error: {error_msg}")
            logger.error("Please check the Google Cloud Console (Vertex AI -> Batch Predictions) for detailed logs.")
            
            # Clean up tracking files and GCS input file on failure
            if os.path.exists(config.BATCH_MANIFEST_FILE):
                try:
                    with open(config.BATCH_MANIFEST_FILE, "r", encoding="utf-8") as f:
                        failed_run_id = json.load(f).get("batch_run_id")
                    if failed_run_id:
                        storage.Client().bucket(config.GCS_BUCKET_NAME).blob(f"batch_input/batch_input_{failed_run_id}.jsonl").delete()
                except Exception:
                    pass
                os.remove(config.BATCH_MANIFEST_FILE)

            # Remove the active job tracking file so the user can submit a new job
            if os.path.exists(config.BATCH_JOB_FILE):
                os.remove(config.BATCH_JOB_FILE)
        else:
            logger.warning(f"Job is not yet successfully completed (State: {state_str}). Cannot import results at this time.")
        return
        
    if not os.path.exists(config.BATCH_MANIFEST_FILE):
        logger.error(f"Manifest file not found at {config.BATCH_MANIFEST_FILE}. Cannot map results back to original chunks.")
        return
        
    with open(config.BATCH_MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest_payload = json.load(f)
        
    manifest_data = manifest_payload["manifest_data"]
    gcs_output_prefix = manifest_payload["gcs_output_prefix"]
    batch_run_id = manifest_payload["batch_run_id"]
    hash_to_chunk_key = manifest_payload.get("hash_to_chunk_key", {})
    
    try:
        output_files = [str(f) for f in Path(config.BATCH_PROC_DIR).glob(f"{batch_run_id}*.jsonl")]
        
        # 1. Check for local files first; download from GCS only if none exist
        if not output_files:
            _gcs_download_and_cleanup(
                config.GCS_BUCKET_NAME, 
                gcs_output_prefix, 
                str(config.BATCH_PROC_DIR),
                file_prefix=batch_run_id
            )
            output_files = [str(f) for f in Path(config.BATCH_PROC_DIR).glob(f"{batch_run_id}*.jsonl")]
        else:
            logger.info(f"Found {len(output_files)} existing local JSONL files. Skipping GCS download.")

        _calculate_and_log_tokens(output_files, metrics)
        
        # Create an idempotency tracker to replace reliance on .pending files
        tracker_file = Path(config.BATCH_PROC_DIR) / f"{batch_run_id}_processed_keys.json"
        processed_keys = set()
        if tracker_file.exists():
            try:
                with open(tracker_file, "r") as tf:
                    processed_keys = set(json.load(tf))
            except Exception:
                pass

        # 3. Process each result file
        for file_path in output_files:
            if not file_path.endswith(".jsonl"): continue
            with open(file_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(tqdm(f, desc=f"Processing {os.path.basename(file_path)}")):
                    if not line.strip(): continue
                    try:
                        result = json.loads(line)
                        
                        # Identify chunk_key robustly
                        chunk_key = None
                        if hash_to_chunk_key:
                            req_text = result.get("request", {}).get("contents", [{}])[0].get("parts", [{}])[0].get("text", "")
                            content_hash = hashlib.md5(req_text.encode('utf-8')).hexdigest()
                            chunk_key = hash_to_chunk_key.get(content_hash)
                            
                        if not chunk_key:
                            logger.warning(f"Could not map batch result back to original chunk using hash. Hash not found for request text starting with: '{req_text[:50]}...'. Skipping.")
                            continue
                            
                        # Safety check: verify if this data has already been successfully repacked
                        if chunk_key in processed_keys:
                            logger.debug(f"Skipping chunk '{chunk_key}': Already unpacked in a previous attempt.")
                            continue
                            
                        # Handle Vertex API individual chunk errors
                        if "error" in result:
                            logger.error(f"Vertex AI Batch API returned an error for chunk {chunk_key}: {result['error']}")
                            metrics["processing_errors"] += 1
                            processed_keys.add(chunk_key)  # Mark as processed to prevent infinite retry of erroneous data
                            continue

                        analysis_json = result.get("response", {}).get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
                        clean_json_text = analysis_json.strip().replace("```json", "").replace("```", "").strip()
                        analysis_data = json.loads(clean_json_text)
                        
                        await ingestion_service._handle_analysis_result(chunk_key, AnalysisDetail.model_validate(analysis_data), manifest_data[chunk_key], metrics, semaphore)

                        # Successfully repacked, add to the processed list
                        processed_keys.add(chunk_key)

                    except (ValidationError, json.JSONDecodeError, KeyError, IndexError) as e:
                        logger.error(f"Failed to parse or validate result for key '{chunk_key}': {e}")
                        metrics["processing_errors"] += 1
                        processed_keys.add(chunk_key)
                        
            # Save progress after processing each .jsonl file
            with open(tracker_file, "w", encoding="utf-8") as tf:
                json.dump(list(processed_keys), tf)
        
        await ingestion_service.ingest_cleaned_files_to_db(semaphore, metrics)
        
        # 4. Successfully ingested, now safely archive JSONL files
        archive_batch_dir = Path(config.ARCHIVE_DIR) / config.RUN_TIME_STR / "batch_results"
        archive_batch_dir.mkdir(parents=True, exist_ok=True)
        for f in output_files:
            try:
                os.rename(f, archive_batch_dir / Path(f).name)
            except OSError as e:
                logger.warning(f"Could not archive batch result {f}: {e}")
                
        if tracker_file.exists(): os.remove(tracker_file) # 全部順利完成後移除 tracker

        # Clean up tracking files after successful import
        os.remove(config.BATCH_MANIFEST_FILE)
        if os.path.exists(config.BATCH_JOB_FILE):
            os.remove(config.BATCH_JOB_FILE)

    except (RuntimeError, Exception) as e:
        logger.critical(f"A critical error occurred during result import: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up GCS input file after successful import
        try:
            storage.Client().bucket(config.GCS_BUCKET_NAME).blob(f"batch_input/batch_input_{batch_run_id}.jsonl").delete()
        except Exception as e:
            logger.warning(f"Could not clean up GCS input blob: {e}")
        
        logger.info("Batch result import finished.")

async def process_orphan_jsonl(file_path: str, metrics: defaultdict, semaphore):
    """
    Emergency recovery: Parses a standalone predictions.jsonl file without a manifest,
    reconstructs dummy metadata, and forces ingestion to the database.
    """
    logger.info(f"Starting emergency recovery from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    recovery_id = uuid.uuid4().hex[:6]
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Recovering")):
            if not line.strip(): continue
            try:
                result = json.loads(line)
                req_text = result.get("request", {}).get("contents", [{}])[0].get("parts", [{}])[0].get("text", "")
                
                # Restore original content from the submitted Prompt
                main_content = req_text.split("[Main Content]\n")[-1].strip()
                
                analysis_json = result.get("response", {}).get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
                clean_json_text = analysis_json.strip().replace("```json", "").replace("```", "").strip()
                analysis_data = json.loads(clean_json_text)

                chunk_key = f"recovered_chunk_{i}_{uuid.uuid4().hex[:8]}"
                
                # Because the Manifest is lost, mock the Metadata so the system can successfully write to the DB
                fake_manifest_data = {
                    "path": f"recovered_{recovery_id}_chunk_{i}.json",
                    "data": {
                        "main_content": main_content,
                        "group_id": f"recovered_batch_{recovery_id}",
                        "file_name": "predictions.jsonl (Recovered)"
                    }
                }
                
                await ingestion_service._handle_analysis_result(
                    chunk_key, 
                    AnalysisDetail.model_validate(analysis_data), 
                    fake_manifest_data, 
                    metrics, 
                    semaphore
                )
            except Exception as e:
                logger.error(f"Failed to recover row {i}: {e}")
                metrics["processing_errors"] += 1
                
    logger.info("Recovery parsing complete. Triggering database ingestion...")
    await ingestion_service.ingest_cleaned_files_to_db(semaphore, metrics)
    
    archive_batch_dir = Path(config.ARCHIVE_DIR) / config.RUN_TIME_STR / "batch_results"
    archive_batch_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(file_path, archive_batch_dir / Path(file_path).name)
    except OSError as e:
        logger.warning(f"Could not archive recovered file: {e}")
        
    logger.info("Emergency recovery and database ingestion finished successfully!")

async def recover_all_orphaned_results(metrics: defaultdict, semaphore):
    """Scans the batch processing directory for any orphan .jsonl files and recovers them."""
    orphan_files = list(Path(config.BATCH_PROC_DIR).glob("*.jsonl"))
    if not orphan_files:
        logger.info(f"No orphan .jsonl files found in {config.BATCH_PROC_DIR} to recover.")
        return
        
    logger.info(f"Found {len(orphan_files)} orphan .jsonl files. Starting bulk recovery...")
    for file_path in orphan_files:
        await process_orphan_jsonl(str(file_path), metrics, semaphore)