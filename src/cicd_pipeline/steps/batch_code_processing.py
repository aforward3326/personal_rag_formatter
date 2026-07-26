import os
import json
import time
import uuid
import hashlib
import asyncio
from typing import Dict, Any, List, Optional, Tuple

from google.cloud import storage
import aiohttp

from .base import PipelineStep
from .processing.chunker import CodeChunkerAndSanitizer
from .processing.metadata_extractor import CodeMetadataExtractor
from .processing.llm_analyzer import LLMAnalyzer
from .processing.embedder import Embedder

class BatchCodeProcessingStep(PipelineStep):
    """
    A CI/CD pipeline step that distributes code processing tasks between 
    Vertex AI (Gemini) and a local LLM (e.g., LM Studio), based on file characteristics.
    It processes work in parallel and unifies the results for ingestion.
    """
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.chunker = CodeChunkerAndSanitizer()
        self.metadata_extractor = CodeMetadataExtractor()
        
        # Analyzer for Google Vertex AI (Gemini)
        self.gemini_analyzer = LLMAnalyzer(
            provider="gemini",
            model_name=self.config.batch_model_name,
            api_key=self.config.gemini_api_key,
            google_cloud_project=self.config.google_cloud_project,
            google_cloud_location=self.config.google_cloud_location
        )
        
        # Analyzer for Local LLM (LM Studio)
        self.local_analyzer = LLMAnalyzer(
            provider="lm_studio",
            model_name=self.config.local_model_name, # Assumes a new config for local model
            base_url=self.config.lm_studio_base_url,
            api_key=self.config.lm_studio_api_key
        )
        
        self.embedder = Embedder(
            provider=self.config.embedding_provider,
            model_name=self.config.embedding_model_name,
            base_url=self.config.lm_studio_base_url,
            api_key=self.config.lm_studio_api_key
        )
        
        self.excluded_dirs = {
            '.git', 'node_modules', 'venv', 'env', '.venv', '__pycache__', 
            '.pytest_cache', '.tox', 'dist', 'build', 'vendor', 'migrations', 'alembic'
        }

    def _is_boilerplate_file(self, file_path: str) -> Optional[Tuple[str, str, List[str]]]:
        """
        Checks if a file is boilerplate. If so, returns a tuple of 
        (summary_en, summary_zh, tags) for static generation. Case-insensitive.
        """
        path_lower = file_path.lower()
        if any(p in path_lower for p in ['/dto/', '/entity/', '/vo/', '/po/', '/models/', '/schemas/']):
            return ("Data Transfer Object or Schema Definition", "資料傳輸對象或結構定義", ["data-schema", "backend-model"])
        basename = os.path.basename(path_lower)
        if basename in ['__init__.py', 'wsgi.py', 'asgi.py', 'manage.py', 'setup.py']:
            return ("Python Boilerplate or Configuration File", "Python 樣板或設定檔", ["python-config", "boilerplate"])
        if path_lower.endswith('.d.ts') or any(p in path_lower for p in ['/interfaces/', '/types/']):
            return ("Frontend Type Definition or Interface", "前端類型定義或介面", ["typescript-types", "frontend-interface"])
        if path_lower.endswith(('.g.dart', '.freezed.dart')):
            return ("Generated Dart Code", "Dart 自動生成程式碼", ["dart-generated", "code-generation"])
        return None

    def _get_target_llm(self, file_path: str) -> str:
        """
        Determines which LLM to target based on file path and type.
        Returns 'gemini' for complex/core files, 'local' for others.
        """
        path_lower = file_path.lower()
        
        # Core backend logic, infrastructure, and critical configs go to Gemini
        core_patterns = [
            '/service/', '/controller/', '/manager/', '/usecase/', '/repository/', '/impl/',
            'dockerfile', 'jenkinsfile', '.tf', '.proto', '/grpc/', '/api/'
        ]
        if any(p in path_lower for p in core_patterns) or os.path.basename(path_lower) in ['pom.xml', 'build.gradle']:
            return 'gemini'
            
        # UI components, tests, and simple utilities go to Local LLM
        local_patterns = ['.vue', '.jsx', '.tsx', '/utils/', '/helpers/', '/test/', '.spec.js', '_test.go']
        if any(p in path_lower for p in local_patterns):
            return 'local'
            
        # Default to local for cost-saving
        return 'local'

    def _prepare_and_distribute_work(self, repo_path: str, changed_files: Optional[List[str]], commit_hash: str) -> Tuple[List[Dict], Dict, List[Dict], Dict, List[Dict]]:
        """
        Distributes files to Gemini, Local LLM, or bypasses as boilerplate.
        """
        gemini_requests, local_requests, boilerplate_records = [], [], []
        gemini_hash_map, local_hash_map = {}, {}

        changed_files_set = set(changed_files) if changed_files is not None else None
        log_msg = f"Processing {len(changed_files_set)} changed files." if changed_files_set is not None else "Processing all files."
        self.logger.info(f"Incremental check: {log_msg}")

        for root, dirs, files in os.walk(repo_path, topdown=True):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            for file in files:
                file_path = os.path.join(root, file)
                if changed_files_set is not None and file_path not in changed_files_set:
                    continue

                relative_path = os.path.relpath(file_path, repo_path)
                
                if boilerplate_data := self._is_boilerplate_file(relative_path):
                    content = self.chunker.read_file_content(file_path)
                    if not content: continue
                    meta = self.metadata_extractor.extract(content, self.chunker.get_language(file_path))
                    summary_en, summary_zh, tags = boilerplate_data
                    embedding_vector = self.embedder.embed(summary_en)
                    record = self._create_record(content, relative_path, meta, {"summary_en": summary_en, "summary_zh": summary_zh, "tags": tags}, embedding_vector, 1, len(content.splitlines()), commit_hash)
                    boilerplate_records.append(record)
                    continue

                target_llm = self._get_target_llm(relative_path)
                chunks = self.chunker.process_file(file_path)
                
                for chunk in chunks:
                    meta = self.metadata_extractor.extract(chunk['content'], chunk['language'])
                    system_prompt = "Analyze the code. Return a JSON with 'summary_en' (under 35 words), 'summary_zh' (under 35 words, in Traditional Chinese), and 'tags' (3-5 string array). Respond with JSON only."
                    text_content = f'File: {relative_path}\nCode:\n```{chunk["content"]}```'
                    
                    if target_llm == 'gemini':
                        req = {"request": {"contents": [{"role": "user", "parts": [{"text": text_content}]}], "systemInstruction": {"role": "system", "parts": [{"text": system_prompt}]}, "generationConfig": {"responseMimeType": "application/json", "temperature": 0.0, "maxOutputTokens": 150}}}
                        gemini_requests.append(req)
                        gemini_hash_map[hashlib.md5(text_content.encode('utf-8')).hexdigest()] = {"chunk": chunk, "meta": meta, "relative_path": relative_path}
                    else: # 'local'
                        req = {"model": self.config.local_model_name, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": text_content}], "temperature": 0.0, "max_tokens": 150}
                        local_requests.append(req)
                        local_hash_map[id(req)] = {"chunk": chunk, "meta": meta, "relative_path": relative_path}

        return gemini_requests, gemini_hash_map, local_requests, local_hash_map, boilerplate_records

    async def _execute_gemini_batch(self, requests: List[Dict], hash_map: Dict, commit_hash: str) -> List[Dict]:
        """Submits a batch job to Vertex AI and processes the results."""
        if not requests: return []
        
        batch_run_id = f"gemini_{uuid.uuid4().hex[:8]}"
        local_input_file = os.path.join(self.config.base_workspace_dir, f"batch_input_{batch_run_id}.jsonl")
        
        with open(local_input_file, "w", encoding="utf-8") as f:
            for req in requests: f.write(json.dumps(req) + "\n")

        storage_client = storage.Client(project=self.config.google_cloud_project)
        bucket = storage_client.bucket(self.config.gcs_bucket_name)
        gcs_input_blob = f"cicd_batch_input/{os.path.basename(local_input_file)}"
        gcs_input_uri = f"gs://{self.config.gcs_bucket_name}/{gcs_input_blob}"
        gcs_output_prefix = f"gs://{self.config.gcs_bucket_name}/cicd_batch_output/{batch_run_id}/"
        
        self.logger.info(f"Uploading Gemini batch file ({len(requests)} requests) to GCS...")
        bucket.blob(gcs_input_blob).upload_from_filename(local_input_file)

        self.logger.info("Submitting Vertex AI Batch Job...")
        job = self.gemini_analyzer.submit_batch_job(self.config.batch_model_name, gcs_input_uri, gcs_output_prefix)
        self.logger.info(f"Gemini Job Name: {job.name}. Polling for completion...")

        while True:
            status = self.gemini_analyzer.get_batch_job_status(job.name)
            if "SUCCEEDED" in str(status.state).upper(): break
            if "FAILED" in str(status.state).upper() or "CANCELLED" in str(status.state).upper():
                raise RuntimeError(f"Vertex Batch Job failed: {getattr(status, 'error', 'Unknown Error')}")
            self.logger.info(f"Gemini Job state: {status.state}. Polling again in 60 seconds...")
            await asyncio.sleep(60)

        processed_records = []
        for blob in bucket.list_blobs(prefix=f"cicd_batch_output/{batch_run_id}/"):
            if not blob.name.endswith(".jsonl"): continue
            for line in blob.download_as_text().splitlines():
                if not line.strip(): continue
                res = json.loads(line)
                if "error" in res: continue
                req_text = res.get("request", {}).get("contents", [{}])[0].get("parts", [{}])[0].get("text", "")
                meta_info = hash_map.get(hashlib.md5(req_text.encode('utf-8')).hexdigest())
                if not meta_info: continue
                
                analysis_str = res.get("response", {}).get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
                try:
                    llm_analysis = json.loads(analysis_str)
                    if isinstance(llm_analysis, list): llm_analysis = llm_analysis[0]
                    if not isinstance(llm_analysis, dict): continue
                    
                    text_to_embed = llm_analysis.get('summary_en', '') or meta_info['chunk']['content']
                    embedding = self.embedder.embed(text_to_embed)
                    record = self._create_record(meta_info['chunk']['content'], meta_info['relative_path'], meta_info['meta'], llm_analysis, embedding, meta_info['chunk']['start_line'], meta_info['chunk']['end_line'], commit_hash)
                    processed_records.append(record)
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Could not parse Gemini response for {meta_info['relative_path']}: {e}")
        
        self.logger.info(f"Gemini batch processing complete. Processed {len(processed_records)} records.")
        return processed_records

    async def _execute_local_batch(self, requests: List[Dict], hash_map: Dict, commit_hash: str) -> List[Dict]:
        """Processes a batch of requests using a local LLM endpoint."""
        if not requests: return []
        
        processed_records = []
        semaphore = asyncio.Semaphore(self.config.local_llm_concurrency or 4) # Add concurrency config
        
        async def process_one(session, request):
            async with semaphore:
                try:
                    async with session.post(self.local_analyzer.base_url, json=request, timeout=120) as response:
                        response.raise_for_status()
                        res_json = await response.json()
                        analysis_str = res_json.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                        
                        meta_info = hash_map.get(id(request))
                        if not meta_info: return

                        llm_analysis = json.loads(analysis_str)
                        if isinstance(llm_analysis, list): llm_analysis = llm_analysis[0]
                        if not isinstance(llm_analysis, dict): return

                        text_to_embed = llm_analysis.get('summary_en', '') or meta_info['chunk']['content']
                        embedding = self.embedder.embed(text_to_embed)
                        record = self._create_record(meta_info['chunk']['content'], meta_info['relative_path'], meta_info['meta'], llm_analysis, embedding, meta_info['chunk']['start_line'], meta_info['chunk']['end_line'], commit_hash)
                        processed_records.append(record)
                except Exception as e:
                    self.logger.error(f"Local LLM request failed: {e}")

        async with aiohttp.ClientSession() as session:
            tasks = [process_one(session, req) for req in requests]
            self.logger.info(f"Starting local LLM batch processing for {len(tasks)} requests...")
            await asyncio.gather(*tasks)

        self.logger.info(f"Local LLM batch processing complete. Processed {len(processed_records)} records.")
        return processed_records

    def _create_record(self, content: str, relative_path: str, meta: Dict, llm_analysis: Dict, embedding_vector: List[float], start_line: int, end_line: int, commit_hash: str) -> Dict:
        """Helper to assemble a final record for database ingestion."""
        hash_input = f"{self.config.git_url}|{self.config.branch}|{relative_path}|{meta.get('node_name', '')}|{start_line}".encode('utf-8')
        return {"id": hashlib.md5(hash_input).hexdigest(), "content": content, "repository": self.config.git_url, "branch": self.config.branch, "commit_hash": commit_hash, "file_path": relative_path, "language": meta.get('language', 'unknown'), "node_type": meta.get('node_type', 'file'), "node_name": meta.get('node_name', os.path.basename(relative_path)), "start_line": start_line, "end_line": end_line, "summary_zh": llm_analysis.get('summary_zh', ''), "summary_en": llm_analysis.get('summary_en', ''), "tags": llm_analysis.get('tags', []), "dependencies": meta.get('dependencies', []), "embedding": embedding_vector}

    async def _run_async(self, context: Dict[str, Any]) -> Dict[str, Any]:
        repo_path = context.get('repo_local_path')
        commit_hash = context.get('commit_hash')
        changed_files = context.get('changed_files')

        if not repo_path or not commit_hash:
            raise ValueError("Context lacks 'repo_local_path' or 'commit_hash'.")

        os.makedirs(self.config.base_workspace_dir, exist_ok=True)
        
        gemini_reqs, gemini_map, local_reqs, local_map, boilerplate_recs = self._prepare_and_distribute_work(repo_path, changed_files, commit_hash)
        
        self.logger.info(f"Work distribution: Gemini ({len(gemini_reqs)}), Local LLM ({len(local_reqs)}), Boilerplate ({len(boilerplate_recs)})")

        gemini_task = self._execute_gemini_batch(gemini_reqs, gemini_map, commit_hash)
        local_task = self._execute_local_batch(local_reqs, local_map, commit_hash)
        
        gemini_results, local_results = await asyncio.gather(gemini_task, local_task)
        
        all_processed_records = boilerplate_recs + gemini_results + local_results
        self.logger.info(f"All processing complete. Total records to be ingested: {len(all_processed_records)}")
        
        context['processed_records'] = all_processed_records
        return context

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for the asynchronous execution."""
        try:
            return asyncio.run(self._run_async(context))
        except Exception as e:
            self.logger.error(f"An unexpected error occurred in BatchCodeProcessingStep: {e}", exc_info=True)
            raise
