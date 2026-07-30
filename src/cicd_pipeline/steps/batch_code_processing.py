import os
import json
import uuid
import hashlib
import asyncio
from typing import Dict, Any, List, Optional

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
            provider=config.standard_ai_provider, api_key=config.standard_api_key, base_url=config.standard_base_url
        )
        self.thinking_client = AIProviderFactory.create_llm(
            provider=config.thinking_ai_provider, api_key=config.thinking_api_key, base_url=config.thinking_base_url,
            project=config.google_cloud_project, location=config.google_cloud_location
        )
        self.embedder = Embedder(
            provider=config.embedding_provider, model_name=config.embedding_model_name,
            base_url=config.embedding_base_url, api_key=config.embedding_api_key
        )
        self.router = HybridRouter(self.embedder)
        self.BATCH_JOB_FILE = os.path.join(config.base_workspace_dir, "batch_jobs.json")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return asyncio.run(self.arun(context))

    async def arun(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.resume_batch_id:
            return self.resume(context)
        return await self.submit(context)

    async def submit(self, context: Dict[str, Any]) -> Dict[str, Any]:
        repo_path, commit_hash = context.get('repo_local_path'), context.get('commit_hash')
        if not repo_path or not commit_hash:
            raise ValueError("Context must contain 'repo_local_path' and 'commit_hash'.")

        repo_context = build_repo_context(repo_path)
        full_system_prompt = f"{repo_context}\n\n{SYSTEM_PROMPT_TEMPLATE}" if repo_context else SYSTEM_PROMPT_TEMPLATE

        work_items = self._prepare_work_items(repo_path, commit_hash)
        
        realtime_tasks, batch_items_standard, batch_items_thinking = [], [], []
        
        for item in work_items:
            is_thinking = item['use_thinking_client']
            provider = self.config.thinking_ai_provider if is_thinking else self.config.standard_ai_provider
            
            if provider in ['vertex', 'gemini']:
                (batch_items_thinking if is_thinking else batch_items_standard).append(item)
            else:
                client = self.thinking_client if is_thinking else self.standard_client
                realtime_tasks.append(self._execute_realtime_analysis(item, client, full_system_prompt))

        jobs_to_track = {}
        if batch_items_standard:
            job_id = self._submit_batch_job(batch_items_standard, "standard", self.standard_client, full_system_prompt, False)
            jobs_to_track[job_id] = {"mode": "standard", "status": "pending"}
        if batch_items_thinking:
            job_id = self._submit_batch_job(batch_items_thinking, "thinking", self.thinking_client, full_system_prompt, True)
            jobs_to_track[job_id] = {"mode": "thinking", "status": "pending"}

        realtime_results = await asyncio.gather(*realtime_tasks)
        processed_records = [record for record in realtime_results if record]

        if jobs_to_track:
            with open(self.BATCH_JOB_FILE, 'w') as f: json.dump(jobs_to_track, f)
            self.logger.info(f"Submitted batch jobs to track: {jobs_to_track}")
        
        context['processed_records'] = processed_records
        self.logger.info(f"Completed real-time processing. Generated {len(processed_records)} records.")
        return context

    async def _execute_realtime_analysis(self, work_item: Dict, client, system_prompt: str) -> Optional[Dict]:
        """Executes analysis for a single work item and returns a complete record."""
        metadata = work_item['metadata']
        chunk = metadata['chunk']
        try:
            model_name = self.config.thinking_model_name if work_item['use_thinking_client'] else self.config.standard_model_name

            llm_analysis, _, _ = await client.analyze_batch(
                chunk, system_prompt, model_name, None, is_thinking_mode=work_item['use_thinking_client']
            )

            # CRITICAL CHECK: If analysis fails, llm_analysis will be empty. Do not proceed.
            if not llm_analysis:
                self.logger.warning(f"LLM analysis returned empty for {metadata['relative_path']}. Skipping record.")
                return None

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
            self.logger.error(f"Failed to process real-time item for {metadata['relative_path']}: {e}", exc_info=True)
            return None

    def _prepare_work_items(self, repo_path: str, commit_hash: str) -> List[Dict]:
        work_items = []
        for root, _, files in os.walk(repo_path):
            if '.git' in root: continue
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
                except (IOError, OSError): continue

                use_thinking_client = self.config.routing_strategy == 'always_thinking' or \
                                     (self.config.routing_strategy == 'auto' and self.router.requires_deep_thinking(relative_path, content))

                chunks = self.chunker.process_file(file_path, content)
                for chunk in chunks:
                    regex_analysis = self.metadata_extractor.extract(chunk['content'], chunk['language'])
                    hash_input = f"{self.config.git_url}|{self.config.branch}|{relative_path}|{regex_analysis.get('node_name', '')}|{chunk['start_line']}".encode('utf-8')
                    
                    work_items.append({
                        "use_thinking_client": use_thinking_client,
                        "metadata": {
                            "chunk": chunk,
                            "chunk_id": hashlib.md5(hash_input).hexdigest(),
                            "commit_hash": commit_hash,
                            "relative_path": relative_path,
                            "regex_analysis": regex_analysis
                        }
                    })
        return work_items

    def _submit_batch_job(self, work_items: List[Dict], mode: str, client, system_prompt: str, is_thinking_mode: bool) -> str:
        batch_id = f"{mode}_{uuid.uuid4().hex[:8]}"
        input_file_path = os.path.join(self.config.base_workspace_dir, f"{batch_id}_input.jsonl")
        model_name = self.config.thinking_model_name if is_thinking_mode else self.config.standard_model_name

        with open(input_file_path, 'w') as f:
            for item in work_items:
                api_request = {
                    "custom_id": f"{item['metadata']['relative_path']}::{item['metadata']['chunk']['start_line']}",
                    "method": "POST", "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": item['metadata']['chunk']['content']}
                        ],
                        "response_format": {"type": "json_object"},
                    }
                }
                f.write(json.dumps(api_request) + '\n')

        cache_id = client.create_context_cache(system_prompt, ttl_hours=1)
        return client.submit_batch_job(
            input_file_path=input_file_path, job_name_prefix=f"rag_ingestion_{mode}",
            system_prompt=system_prompt, model_name=model_name, config={},
            cache_id=cache_id, is_thinking_mode=is_thinking_mode
        )

    def resume(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # This needs to be fully implemented to process results from GCS and create full records.
        self.logger.info("Resume function is a stub and needs full implementation.")
        context['processed_records'] = []
        return context
