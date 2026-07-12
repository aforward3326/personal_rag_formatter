import os
import json
import time
import uuid
import hashlib
from typing import Dict, Any
from google.cloud import storage

from .base import PipelineStep
from .processing.chunker import CodeChunkerAndSanitizer
from .processing.metadata_extractor import CodeMetadataExtractor
from .processing.llm_analyzer import LLMAnalyzer
from .processing.embedder import Embedder

class BatchCodeProcessingStep(PipelineStep):
    """
    A CI/CD pipeline step that processes code files using Vertex AI Batch API.
    It compiles all code chunks, submits a single batch job, waits for completion,
    and ingests the results.
    """
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.chunker = CodeChunkerAndSanitizer()
        self.metadata_extractor = CodeMetadataExtractor()
        self.analyzer = LLMAnalyzer(
            provider=self.config.llm_provider,
            model_name=self.config.batch_model_name,
            api_key=self.config.gemini_api_key if self.config.llm_provider == "gemini" else self.config.lm_studio_api_key,
            base_url=self.config.lm_studio_base_url,
            google_cloud_project=self.config.google_cloud_project,
            google_cloud_location=self.config.google_cloud_location
        )
        self.embedder = Embedder(
            provider=self.config.embedding_provider,
            model_name=self.config.embedding_model_name,
            base_url=self.config.lm_studio_base_url,
            api_key=self.config.lm_studio_api_key
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        repo_path = context.get('repo_local_path')
        commit_hash = context.get('commit_hash')
        if not repo_path or not commit_hash or not self.config.gcs_bucket_name:
            raise ValueError("Context lacks required fields or 'gcs_bucket_name' is missing in config.")

        # Ensure the workspace directory exists to prevent FileNotFoundError when writing files
        os.makedirs(self.config.base_workspace_dir, exist_ok=True)

        resume_batch_id = getattr(self.config, 'resume_batch_id', None)
        batch_run_id = resume_batch_id if resume_batch_id else f"cicd_{uuid.uuid4().hex[:8]}"
        local_input_file = os.path.join(self.config.base_workspace_dir, f"batch_input_{batch_run_id}.jsonl")
        
        hash_to_metadata = {}
        total_chunks = 0

        # 1. Generate Batch Request JSONL (If resuming, we only rebuild the metadata mapping)
        self.logger.info("Extracting code and building chunk metadata...")
        f_out = None
        if not resume_batch_id:
            f_out = open(local_input_file, "w", encoding="utf-8")
            
        try:
            for root, _, files in os.walk(repo_path):
                if '.git' in root: continue
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, repo_path)
                    chunks = self.chunker.process_file(file_path)
                    
                    for chunk in chunks:
                        meta = self.metadata_extractor.extract(chunk['content'], chunk['language'])
                        system_prompt = 'Analyze the code and return a JSON with "summary_en", "summary_zh", and "tags" (string array).'
                        text_content = f'Code:\n```{chunk["content"]}```'
                        
                        request_obj = {
                            "request": {
                                "contents": [{"role": "user", "parts": [{"text": text_content}]}],
                                "systemInstruction": {"role": "system", "parts": [{"text": system_prompt}]},
                                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1}
                            }
                        }
                        if f_out:
                            f_out.write(json.dumps(request_obj) + "\n")
                        
                        req_hash = hashlib.md5(text_content.encode('utf-8')).hexdigest()
                        hash_to_metadata[req_hash] = {"chunk": chunk, "meta": meta, "relative_path": relative_path}
                        total_chunks += 1
        finally:
            if f_out:
                f_out.close()

        if total_chunks == 0:
            self.logger.info("No code chunks found to process.")
            context['processed_records'] = []
            return context

        storage_client = storage.Client(project=self.config.google_cloud_project)
        bucket = storage_client.bucket(self.config.gcs_bucket_name)

        if not resume_batch_id:
            # 2. Upload to GCS
            gcs_input_blob = f"cicd_batch_input/input_{batch_run_id}.jsonl"
            gcs_input_uri = f"gs://{self.config.gcs_bucket_name}/{gcs_input_blob}"
            gcs_output_prefix = f"gs://{self.config.gcs_bucket_name}/cicd_batch_output/{batch_run_id}/"
            
            self.logger.info(f"Uploading batch file ({total_chunks} requests) to GCS...")
            bucket.blob(gcs_input_blob).upload_from_filename(local_input_file)

            # 3. Submit Batch Job
            self.logger.info("Submitting Vertex AI Batch Job...")
            job = self.analyzer.submit_batch_job(
                model_name=self.config.batch_model_name,
                input_gcs_uri=gcs_input_uri,
                output_gcs_uri_prefix=gcs_output_prefix
            )
            self.logger.info(f"Batch Job ID: {job.name}. Waiting for completion (this may take a few minutes)...")

            # 4. Poll until completion
            while True:
                status = self.analyzer.get_batch_job_status(job.name)
                state_str = str(status.state).upper()
                if "SUCCEEDED" in state_str:
                    self.logger.info("Batch job completed successfully!")
                    break
                elif "FAILED" in state_str or "CANCELLED" in state_str:
                    raise RuntimeError(f"Vertex Batch Job failed: {getattr(status, 'error', 'Unknown Error')}")
                
                self.logger.info(f"Job state: {state_str}. Polling again in 30 seconds...")
                time.sleep(30)
        else:
            self.logger.info(f"Resume mode activated for batch ID {batch_run_id}. Skipping job submission and polling.")

        # 5. Download and process results
        processed_records = []
        blobs = list(bucket.list_blobs(prefix=f"cicd_batch_output/{batch_run_id}/"))
        
        local_res_paths = []
        
        if blobs:
            self.logger.info("Downloading results from GCS...")
            for blob in blobs:
                if not blob.name.endswith(".jsonl"): continue
                # Prefix file name with batch ID to easily locate it later if GCS is deleted
                local_res_path = os.path.join(self.config.base_workspace_dir, f"{batch_run_id}_{os.path.basename(blob.name)}")
                blob.download_to_filename(local_res_path)
                local_res_paths.append(local_res_path)
        else:
            self.logger.info("No result files found in GCS. Checking local workspace...")
            for f in os.listdir(self.config.base_workspace_dir):
                if f.startswith(batch_run_id) and f.endswith(".jsonl") and "batch_input" not in f:
                    local_res_paths.append(os.path.join(self.config.base_workspace_dir, f))
                    
        if not local_res_paths:
            raise FileNotFoundError(f"Could not find any output results for batch ID {batch_run_id} in GCS or locally.")

        self.logger.info(f"Parsing {len(local_res_paths)} local result files...")
        for local_res_path in local_res_paths:
            with open(local_res_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    if not line.strip(): continue
                    res = json.loads(line)
                    
                    if "error" in res: continue
                    
                    request_data = res.get("request", {})
                    response_data = res.get("response", {})
                    req_text = request_data.get("contents", [{}])[0].get("parts", [{}])[0].get("text", "")
                    req_hash = hashlib.md5(req_text.encode('utf-8')).hexdigest()
                    meta_info = hash_to_metadata.get(req_hash)
                    if not meta_info: continue
                    
                    analysis_json = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
                    clean_json = analysis_json.strip().replace("```json", "").replace("```", "").strip()
                    try:
                        llm_analysis = json.loads(clean_json)
                        
                        # Handle cases where the LLM wraps the response in a JSON array
                        if isinstance(llm_analysis, list) and len(llm_analysis) > 0:
                            llm_analysis = llm_analysis[0]
                            
                        # Ensure it's a dictionary before attempting to use .get()
                        if not isinstance(llm_analysis, dict):
                            self.logger.warning(f"Unexpected JSON structure (not a dict) for {meta_info['relative_path']}. Skipping this chunk.")
                            continue
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse LLM response for {meta_info['relative_path']}. Skipping this chunk.")
                        continue
                    
                    # Embed & Build Record
                    text_to_embed = llm_analysis.get('summary_en', '') or meta_info['chunk']['content']
                    embedding_vector = self.embedder.embed(text_to_embed)
                    
                    hash_input = f"{self.config.git_url}|{self.config.branch}|{meta_info['relative_path']}|{meta_info['meta']['node_name']}".encode('utf-8')
                    # Assemble the final record, ensuring a consistent structure
                    record = {
                        "id": hashlib.md5(hash_input).hexdigest(),
                        "content": meta_info['chunk']['content'],
                        "repository": self.config.git_url,
                        "branch": self.config.branch,
                        "commit_hash": commit_hash,
                        "file_path": meta_info['relative_path'],
                        "language": meta_info['chunk']['language'],
                        "node_type": meta_info['meta']['node_type'],
                        "node_name": meta_info['meta']['node_name'],
                        "start_line": meta_info['chunk']['start_line'],
                        "end_line": meta_info['chunk']['end_line'],
                        "summary_zh": llm_analysis.get('summary_zh', ''),
                        "summary_en": llm_analysis.get('summary_en', ''),
                        "tags": llm_analysis.get('tags', []),
                        "dependencies": meta_info['meta']['dependencies'],
                        "embedding": embedding_vector
                    }
                    processed_records.append(record)

        context['processed_records'] = processed_records
        return context