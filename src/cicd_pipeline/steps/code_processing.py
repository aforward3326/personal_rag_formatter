import os
import hashlib
from typing import Dict, Any, List
from .base import PipelineStep
from .processing.chunker import CodeChunkerAndSanitizer
from .processing.metadata_extractor import CodeMetadataExtractor
from ..model.factory import AIProviderFactory
from .processing.embedder import Embedder
from ..utils.routing import HybridRouter
from ..utils.context_builder import build_repo_context
from ..utils.constants import SYSTEM_PROMPT_TEMPLATE

class CodeProcessingStep(PipelineStep):
    """
    A comprehensive pipeline step that processes all code files in a repository.
    It chunks files, extracts metadata, performs LLM analysis, and generates embeddings.
    This step uses a Hybrid Router to intelligently route analysis tasks.
    """
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.chunker = CodeChunkerAndSanitizer()
        self.metadata_extractor = CodeMetadataExtractor()

        self.standard_client = AIProviderFactory.create_llm(
            provider=config.standard_ai_provider,
            api_key=config.standard_ai_key,
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
        
        # Initialize the Hybrid Router with the embedder
        self.router = HybridRouter(self.embedder)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        repo_path = context.get('repo_local_path')
        commit_hash = context.get('commit_hash')
        if not repo_path or not commit_hash:
            raise ValueError("Context must contain 'repo_local_path' and 'commit_hash'.")

        # Build global repository context
        repo_context = build_repo_context(repo_path)
        full_system_prompt = f"{repo_context}\n\n{SYSTEM_PROMPT_TEMPLATE}" if repo_context else SYSTEM_PROMPT_TEMPLATE
        
        processed_records = []
        file_count = 0
        
        # Create a single context cache for the run with the full prompt
        cache_id = self.thinking_client.create_context_cache(full_system_prompt, ttl_hours=1)

        for root, _, files in os.walk(repo_path):
            if '.git' in root: continue
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                file_count += 1
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except (IOError, OSError):
                    self.logger.warning(f"Could not read file {relative_path}, skipping.")
                    continue

                chunks = self.chunker.process_file(file_path, content)
                if not chunks: continue
                    
                self.logger.info(f"Processing file ({file_count}): {relative_path} [{len(chunks)} chunks]")

                # Use the new Hybrid Router for decision making
                use_thinking_client = self.config.routing_strategy == 'always_thinking' or \
                                     (self.config.routing_strategy == 'auto' and self.router.requires_deep_thinking(relative_path, content))

                for i, chunk in enumerate(chunks):
                    self.logger.debug(f"Processing chunk {i+1}/{len(chunks)} from {relative_path}")
                    try:
                        record = self._process_single_chunk(chunk, commit_hash, relative_path, use_thinking_client, cache_id, full_system_prompt)
                        processed_records.append(record)
                    except Exception as e:
                        self.logger.error(f"Failed to process a chunk in {relative_path}. Skipping. Error: {e}", exc_info=True)
                        continue
        
        self.logger.info(f"Finished processing {file_count} files, generating {len(processed_records)} records.")
        context['processed_records'] = processed_records
        return context

    def _process_single_chunk(self, chunk: Dict[str, Any], commit_hash: str, relative_path: str, use_thinking_client: bool, cache_id: str, system_prompt: str) -> Dict[str, Any]:
        # 1. Fast metadata extraction
        regex_analysis = self.metadata_extractor.extract(chunk['content'], chunk['language'])
        
        # 2. Intelligent LLM analysis
        client_to_use = self.thinking_client if use_thinking_client else self.standard_client
        model_to_use = self.config.thinking_model_name if use_thinking_client else self.config.standard_model_name
        
        self.logger.debug(f"Using {'THINKING' if use_thinking_client else 'STANDARD'} client for {relative_path}")
        
        llm_analysis, _, _ = client_to_use.analyze(
            chunk, 
            system_prompt, 
            model_to_use, 
            None, 
            cache_id=cache_id, 
            is_thinking_mode=use_thinking_client
        )

        # 3. Combine results
        combined_analysis = {**regex_analysis, **llm_analysis}
        
        # 4. Create deterministic ID
        hash_input = f"{self.config.git_url}|{self.config.branch}|{relative_path}|{combined_analysis.get('node_name', '')}|{chunk['start_line']}".encode('utf-8')
        chunk_id = hashlib.md5(hash_input).hexdigest()
        
        # 5. Generate embedding for the chunk content itself
        text_to_embed = chunk['content']
        # The embed method now returns a list of vectors. Since we pass one text, we get one vector.
        embedding_vector = self.embedder.embed(text_to_embed)[0]
        
        # 6. Assemble the final record
        return {
            "id": chunk_id,
            "content": chunk['content'],
            "repository": self.config.git_url,
            "branch": self.config.branch,
            "commit_hash": commit_hash,
            "file_path": relative_path,
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
