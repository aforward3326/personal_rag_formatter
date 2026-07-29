import os
import json
import uuid
import hashlib
from typing import Dict, Any, List

from .base import PipelineStep
from .processing.chunker import CodeChunkerAndSanitizer
from ..model.factory import AIProviderFactory
from .processing.embedder import Embedder
from ..utils.routing import HybridRouter
from ..utils.context_builder import build_repo_context
from ..utils.constants import SYSTEM_PROMPT_TEMPLATE

class BatchCodeProcessingStep(PipelineStep):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.chunker = CodeChunkerAndSanitizer()

        self.standard_client = AIProviderFactory.create_llm(
            provider=config.standard_ai_provider,
            api_key=config.standard_api_key,
            base_url=config.standard_base_url
        )
        self.thinking_client = AIProviderFactory.create_llm(
            provider=config.thinking_ai_provider,
            api_key=config.thinking_api_key,
            base_url=config.thinking_base_url,
            project=config.google_cloud_project,
            location=config.google_cloud_location
        )

        self.embedder = Embedder(
            provider=self.config.embedding_provider,
            model_name=self.config.embedding_model_name,
            base_url=self.config.embedding_base_url,
            api_key=self.config.embedding_api_key
        )
        
        self.router = HybridRouter(self.embedder)
        self.BATCH_JOB_FILE = os.path.join(self.config.base_workspace_dir, "batch_jobs.json")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.resume_batch_id:
            # When resuming, we don't need to re-prepare requests, just check status
            return self.resume(context)
        else:
            return self.submit(context)

    def submit(self, context: Dict[str, Any]) -> Dict[str, Any]:
        repo_path = context.get('repo_local_path')
        if not repo_path:
            raise ValueError("Context must contain 'repo_local_path'.")

        repo_context = build_repo_context(repo_path)
        full_system_prompt = f"{repo_context}\n\n{SYSTEM_PROMPT_TEMPLATE}" if repo_context else SYSTEM_PROMPT_TEMPLATE

        standard_requests, thinking_requests = self._prepare_requests(repo_path, full_system_prompt)

        jobs = {}
        if standard_requests:
            job_id = self._submit_job(standard_requests, "standard", self.standard_client, full_system_prompt, is_thinking_mode=False)
            jobs[job_id] = {"mode": "standard", "status": "pending"}
        if thinking_requests:
            job_id = self._submit_job(thinking_requests, "thinking", self.thinking_client, full_system_prompt, is_thinking_mode=True)
            jobs[job_id] = {"mode": "thinking", "status": "pending"}

        with open(self.BATCH_JOB_FILE, 'w') as f:
            json.dump(jobs, f)

        self.logger.info(f"Submitted batch jobs: {jobs}")
        # In a real scenario, this would exit and a separate process would handle resumption.
        # For this simulation, we'll just proceed to the resume step.
        context['processed_records'] = [] # Ensure this is initialized
        return self.resume(context)

    def resume(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not os.path.exists(self.BATCH_JOB_FILE):
            self.logger.warning("Batch job file not found. Nothing to resume.")
            context['processed_records'] = []
            return context

        with open(self.BATCH_JOB_FILE, 'r') as f:
            jobs = json.load(f)

        all_results = []
        for job_id, job_info in jobs.items():
            if job_info.get('status') == 'completed':
                continue

            client = self.standard_client if job_info['mode'] == 'standard' else self.thinking_client
            output_dir = os.path.join(self.config.base_workspace_dir, job_id)
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                results_path = client.retrieve_batch_results(job_id, output_dir)
                self.logger.info(f"Processing results for job {job_id} from {results_path}")
                # Placeholder: Actual result processing would happen here
                # all_results.extend(self._process_batch_results(results_path))
                job_info['status'] = 'completed'
            except Exception as e:
                self.logger.error(f"Failed to retrieve or process results for job {job_id}. Error: {e}", exc_info=True)
                job_info['status'] = 'failed'

        with open(self.BATCH_JOB_FILE, 'w') as f:
            json.dump(jobs, f)

        context['processed_records'] = all_results
        return context

    def _prepare_requests(self, repo_path: str, system_prompt: str) -> (List[Dict], List[Dict]):
        standard_requests = []
        thinking_requests = []

        for root, _, files in os.walk(repo_path):
            if '.git' in root: continue
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except (IOError, OSError):
                    continue

                use_thinking_client = self.config.routing_strategy == 'always_thinking' or \
                                     (self.config.routing_strategy == 'auto' and self.router.requires_deep_thinking(relative_path, content))

                chunks = self.chunker.process_file(file_path, content)
                for chunk in chunks:
                    request_body = {
                        "custom_id": f"{relative_path}::{chunk['start_line']}",
                        "method": "POST",
                        "url": "/v1/chat/completions", # Mock URL for batch API
                        "body": {
                            "model": self.config.thinking_model_name if use_thinking_client else self.config.standard_model_name,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": chunk['content']}
                            ],
                            "response_format": {"type": "json_object"}
                        }
                    }
                    if use_thinking_client:
                        thinking_requests.append(request_body)
                    else:
                        standard_requests.append(request_body)
        return standard_requests, thinking_requests

    def _submit_job(self, requests: List[Dict], mode: str, client, system_prompt: str, is_thinking_mode: bool) -> str:
        batch_id = f"{mode}_{uuid.uuid4().hex[:8]}"
        input_file_path = os.path.join(self.config.base_workspace_dir, f"{batch_id}_input.jsonl")
        
        with open(input_file_path, 'w') as f:
            for req in requests:
                f.write(json.dumps(req) + '\n')

        job_name_prefix = f"rag_ingestion_{mode}"
        model_name = self.config.thinking_model_name if is_thinking_mode else self.config.standard_model_name
        
        cache_id = client.create_context_cache(system_prompt, ttl_hours=1)

        job_id = client.submit_batch_job(
            input_file_path=input_file_path,
            job_name_prefix=job_name_prefix,
            system_prompt=system_prompt,
            model_name=model_name,
            config={},
            cache_id=cache_id,
            is_thinking_mode=is_thinking_mode
        )
        return job_id
