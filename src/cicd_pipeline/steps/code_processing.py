import os
import hashlib
from typing import Dict, Any, List
from .base import PipelineStep
from .processing.chunker import CodeChunkerAndSanitizer
from .processing.metadata_extractor import CodeMetadataExtractor
from .processing.llm_analyzer import LLMAnalyzer
from .processing.embedder import Embedder

class CodeProcessingStep(PipelineStep):
    """
    A comprehensive pipeline step that processes all code files in a repository.
    It chunks files, extracts metadata, performs LLM analysis, and generates embeddings.
    """
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # Initialize all helper components
        self.chunker = CodeChunkerAndSanitizer()
        self.metadata_extractor = CodeMetadataExtractor()
        self.analyzer = LLMAnalyzer(
            provider=self.config.llm_provider,
            model_name=self.config.llm_model_name,
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
        """
        Walks through the repository, processes each file, and adds a list
        of processed records to the context.

        :param context: Must contain 'repo_local_path' and 'commit_hash'.
        :return: Updated context with 'processed_records'.
        """
        repo_path = context.get('repo_local_path')
        commit_hash = context.get('commit_hash')
        if not repo_path or not commit_hash:
            raise ValueError("Context must contain 'repo_local_path' and 'commit_hash'.")

        processed_records = []
        file_count = 0
        
        for root, _, files in os.walk(repo_path):
            if '.git' in root: continue
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                file_count += 1
                
                chunks = self.chunker.process_file(file_path)
                if not chunks: continue
                    
                self.logger.info(f"Processing file ({file_count}): {relative_path} [{len(chunks)} chunks]")

                for i, chunk in enumerate(chunks):
                    self.logger.debug(f"Processing chunk {i+1}/{len(chunks)} from {relative_path}")
                    try:
                        record = self._process_single_chunk(chunk, commit_hash, relative_path)
                        processed_records.append(record)
                    except Exception as e:
                        self.logger.error(f"Failed to process a chunk in {relative_path}. Skipping. Error: {e}")
                        continue
        
        self.logger.info(f"Finished processing {file_count} files, generating {len(processed_records)} records.")
        context['processed_records'] = processed_records
        return context

    def _process_single_chunk(self, chunk: Dict[str, Any], commit_hash: str, relative_path: str) -> Dict[str, Any]:
        """Processes a single code chunk through all analysis stages."""
        # 1. Fast metadata extraction
        regex_analysis = self.metadata_extractor.extract(chunk['content'], chunk['language'])
        
        # 2. High-level LLM analysis
        llm_analysis = self.analyzer.analyze(chunk['content'])
        
        # 3. Combine results
        combined_analysis = {**regex_analysis, **llm_analysis}
        
        # 4. Create deterministic ID
        hash_input = f"{self.config.git_url}|{self.config.branch}|{relative_path}|{combined_analysis['node_name']}".encode('utf-8')
        chunk_id = hashlib.md5(hash_input).hexdigest()
        
        # 5. Generate embedding
        text_to_embed = combined_analysis.get('summary_en', '') or chunk['content']
        embedding_vector = self.embedder.embed(text_to_embed)
        
        # 6. Assemble the final record
        return {
            "id": chunk_id,
            "content": chunk['content'],
            "repository": self.config.git_url,
            "branch": self.config.branch,
            "commit_hash": commit_hash,
            "file_path": relative_path,
            "language": chunk['language'],
            "node_type": combined_analysis['node_type'],
            "node_name": combined_analysis['node_name'],
            "start_line": chunk['start_line'],
            "end_line": chunk['end_line'],
            "summary_zh": combined_analysis.get('summary_zh', ''),
            "summary_en": combined_analysis.get('summary_en', ''),
            "tags": combined_analysis.get('tags', []),
            "dependencies": combined_analysis.get('dependencies', []),
            "embedding": embedding_vector
        }
