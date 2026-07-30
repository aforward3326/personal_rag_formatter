import os
import json
import uuid
import hashlib
import asyncio
from typing import Dict, Any, List

from .base import PipelineStep
from .processing.chunker import CodeChunkerAndSanitizer
from .processing.metadata_extractor import CodeMetadataExtractor
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
        self.metadata_extractor = CodeMetadataExtractor()

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

        standard_reqs, thinking_reqs = self._prepare_requests(repo_path, full_system_prompt, commit_hash)
        
        tasks = []
        jobs_to_track = {}
        
        if standard_reqs:
            if self.config.standard_ai_provider in ['vertex', 'gemini']:
                job_id = self._submit_job([r['api_request'] for r in standard_reqs], "standard", self.standard_client, full_system_prompt, is_thinking_mode=False)
                jobs_to_track[job_id] = {"mode": "standard", "status": "pending"}
            else:
                tasks.append(self._process_realtime_batch(standard_reqs, self.standard_client, is_thinking_mode=False))

        if thinking_reqs:
            if self.config.thinking_ai_provider in ['vertex', 'gemini']:
                job_id = self._submit_job([r['api_request'] for r in thinking_reqs], "thinking", self.thinking_client, full_system_prompt, is_thinking_mode=True)
                jobs_to_track[job_id] = {"mode": "thinking", "status": "pending"}
            else:
                tasks.append(self._process_realtime_batch(thinking_reqs, self.thinking_client, is_thinking_mode=True))

        results_from_realtime = await asyncio.gather(*tasks)
        processed_records = [record for sublist in results_from_realtime for record in sublist]

        if jobs_to_track:
            with open(self.BATCH_JOB_FILE, 'w') as f:
                json.dump(jobs_to_track, f)
            self.logger.info(f"Submitted batch jobs to track: {jobs_to_track}")
        
        context['processed_records'] = processed_records
        self.logger.info(f"Completed real-time processing. Generated {len(processed_records)} records.")
        return context

    async def _process_realtime_batch(self, requests: List[Dict], client, is_thinking_mode: bool) -> List[Dict]:
        """Processes a batch of requests concurrently in real-time for providers like LM Studio."""
        
        async def process_one(req_data: Dict):
            try:
                api_request = req_data['api_request']
                llm_analysis, _, _ = await client.analyze_batch(
                    req_data['metadata']['chunk'],
                    api_request['body']['messages'][0]['content'], # system prompt
                    api_request['body']['model'],
                    None,
                    is_thinking_mode=is_thinking_mode
                )

                if not llm_analysis: # If analysis failed and returned empty dict
                    return None

                metadata = req_data['metadata']
                chunk = metadata['chunk']
                combined_analysis = {**metadata['regex_analysis'], **llm_analysis}
                embedding_vector = self.embedder.embed(chunk['content'])[0]

                return {
                    "id": metadata['chunk_id'],
                    "content": chunk['content'],
                    "repository": self.config.git_url,
                    "branch": self.config.branch,
                    "commit_hash": metadata['commit_hash'],
                    "file_path": metadata['relative_path'],
                    "language": chunk['language'],
                    "node_type": combined_analysis.get('node_type', 'unknown'),
                    "node_name": combined_analysis.get('node_name', ''),
                    "start_line": chunk['start_line'],
                    "end_line": chunk['end_line'],
                    "summary_zh": combined_analysis.get('summary_zh', ''),
                    "summary_en": combined_analysis.get('summary_en', ''),
                    "tags": combined_analysis.get('tags', []),
                    "dependencies": combined_analysis.get('dependencies', []),
                    "embedding": embedding_vector
                }
            except Exception as e:
                self.logger.error(f"Failed to process real-time request for {req_data.get('metadata', {}).get('relative_path', 'unknown')}: {e}", exc_info=True)
                return None

        self.logger.info(f"Processing {len(requests)} requests in real-time for provider {client.__class__.__name__}...")
        tasks = [process_one(req) for req in requests]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results from failed tasks
        return [record for record in results if record is not None]

    def resume(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Resume logic needs to be fully implemented to process results from GCS
        return context

    def _prepare_requests(self, repo_path: str, system_prompt: str, commit_hash: str) -> (List[Dict], List[Dict]):
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
                    regex_analysis = self.metadata_extractor.extract(chunk['content'], chunk['language'])
                    hash_input = f"{self.config.git_url}|{self.config.branch}|{relative_path}|{regex_analysis.get('node_name', '')}|{chunk['start_line']}".encode('utf-8')
                    chunk_id = hashlib.md5(hash_input).hexdigest()

                    model_name = self.config.thinking_model_name if use_thinking_client else self.config.standard_model_name
                    
                    api_request = {
                        "custom_id": f"{relative_path}::{chunk['start_line']}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model_name,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": chunk['content']}
                            ],
                            "response_format": {"type": "json_object"},
                        }
                    }
                    
                    full_request_data = {
                        "api_request": api_request,
                        "metadata": {
                            "chunk": chunk,
                            "chunk_id": chunk_id,
                            "commit_hash": commit_hash,
                            "relative_path": relative_path,
                            "regex_analysis": regex_analysis
                        }
                    }
                    
                    if use_thinking_client:
                        thinking_requests.append(full_request_data)
                    else:
                        standard_requests.append(full_request_data)
        return standard_requests, thinking_requests

    def _submit_job(self, requests: List[Dict], mode: str, client, system_prompt: str, is_thinking_mode: bool) -> str:
        # This method remains the same
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
