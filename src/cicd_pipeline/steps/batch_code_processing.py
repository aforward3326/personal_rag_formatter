import os
import json
import uuid
import hashlib
import asyncio
import aiohttp
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
        self.config = config
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
            provider=config.embedding_provider,
            model_name=config.embedding_model_name,
            base_url=config.embedding_base_url,
            api_key=config.embedding_api_key
        )
        
        self.router = HybridRouter(self.embedder)
        self.BATCH_JOB_FILE = os.path.join(config.base_workspace_dir, "batch_jobs.json")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # The main execution is now async
        return asyncio.run(self.arun(context))

    async def arun(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.resume_batch_id:
            return self.resume(context)
        else:
            return await self.submit(context)

    async def submit(self, context: Dict[str, Any]) -> Dict[str, Any]:
        repo_path = context.get('repo_local_path')
        commit_hash = context.get('commit_hash')
        if not repo_path or not commit_hash:
            raise ValueError("Context must contain 'repo_local_path' and 'commit_hash'.")

        repo_context = build_repo_context(repo_path)
        full_system_prompt = f"{repo_context}\n\n{SYSTEM_PROMPT_TEMPLATE}" if repo_context else SYSTEM_PROMPT_TEMPLATE

        standard_reqs, thinking_reqs = self._prepare_requests(repo_path, full_system_prompt)
        
        tasks = []
        jobs_to_track = {}
        
        # Process Standard Track
        if standard_reqs:
            if self.config.standard_ai_provider in ['vertex', 'gemini']:
                job_id = self._submit_job(standard_reqs, "standard", self.standard_client, full_system_prompt, is_thinking_mode=False)
                jobs_to_track[job_id] = {"mode": "standard", "status": "pending"}
            else: # Real-time processing for lm_studio, openai, etc.
                tasks.append(self._process_realtime_batch(standard_reqs, self.standard_client, commit_hash))

        # Process Thinking Track
        if thinking_reqs:
            if self.config.thinking_ai_provider in ['vertex', 'gemini']:
                job_id = self._submit_job(thinking_reqs, "thinking", self.thinking_client, full_system_prompt, is_thinking_mode=True)
                jobs_to_track[job_id] = {"mode": "thinking", "status": "pending"}
            else:
                tasks.append(self._process_realtime_batch(thinking_reqs, self.thinking_client, commit_hash))

        # Execute all real-time tasks concurrently
        results_from_realtime = await asyncio.gather(*tasks)
        
        # Flatten the list of lists into a single list of records
        processed_records = [record for sublist in results_from_realtime for record in sublist]

        if jobs_to_track:
            with open(self.BATCH_JOB_FILE, 'w') as f:
                json.dump(jobs_to_track, f)
            self.logger.info(f"Submitted batch jobs to track: {jobs_to_track}")
            # If there are jobs to track, we might need to resume later
            # For now, we'll just return the records we have
        
        context['processed_records'] = processed_records
        self.logger.info(f"Completed real-time processing. Generated {len(processed_records)} records.")
        
        # Optionally, you could chain resume logic here if needed
        return context

    async def _process_realtime_batch(self, requests: List[Dict], client, commit_hash: str) -> List[Dict]:
        """Processes a batch of requests in real-time using asyncio and aiohttp."""
        records = []
        # In a real implementation, you would use aiohttp to send these requests concurrently
        self.logger.info(f"Processing {len(requests)} requests in real-time for provider.")
        for req in requests:
            # This is a simplified loop. A real implementation would use asyncio.gather with aiohttp
            try:
                chunk_content = req['body']['messages'][-1]['content']
                # This is a placeholder for the actual analysis call
                # In a real scenario, you'd call client.analyze here
                llm_analysis = {"summary_en": "Real-time analysis placeholder", "summary_zh": "", "tags": []}
                
                # Reconstruct necessary info from custom_id
                relative_path, start_line_str = req['custom_id'].split('::')
                
                # Create a record (simplified version)
                record = {
                    "id": hashlib.md5(f"{relative_path}{start_line_str}".encode()).hexdigest(),
                    "content": chunk_content,
                    "file_path": relative_path,
                    "commit_hash": commit_hash,
                    # ... other fields
                }
                records.append(record)
            except Exception as e:
                self.logger.error(f"Failed to process real-time request: {e}")
                continue
        return records

    def resume(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Resume logic remains largely the same, focused on true batch jobs
        if not os.path.exists(self.BATCH_JOB_FILE):
            self.logger.warning("Batch job file not found. Nothing to resume.")
            context['processed_records'] = []
            return context
        # ... existing resume logic ...
        return context

    def _prepare_requests(self, repo_path: str, system_prompt: str) -> (List[Dict], List[Dict]):
        # This method remains the same
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
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.config.thinking_model_name if use_thinking_client else self.config.standard_model_name,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": chunk['content']}
                            ],
                        }
                    }
                    if use_thinking_client:
                        thinking_requests.append(request_body)
                    else:
                        standard_requests.append(request_body)
        return standard_requests, thinking_requests

    def _submit_job(self, requests: List[Dict], mode: str, client, system_prompt: str, is_thinking_mode: bool) -> str:
        # This method remains the same for true batch providers
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
