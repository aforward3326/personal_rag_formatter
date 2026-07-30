import json
import logging
from typing import Optional, Tuple, Any, Dict
from openai import AsyncOpenAI, OpenAI
from .base import BaseAIProvider

class OpenAIProvider(BaseAIProvider):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Standard synchronous client for `analyze`
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        # Asynchronous client for `analyze_batch`
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def create_context_cache(self, system_prompt: str, ttl_hours: int = 1) -> Optional[str]:
        # OpenAI API does not support server-side context caching in the same way as Vertex AI
        self.logger.debug("Context caching is not supported by OpenAIProvider.")
        return None

    def analyze(self, chunk_data: dict, system_prompt: str, model_name: str, schema_class: Any, cache_id: Optional[str] = None, is_thinking_mode: bool = False) -> Tuple[Dict[str, Any], int, int]:
        """Analyzes a single data chunk synchronously."""
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk_data['content']}
                ],
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content
            llm_analysis = json.loads(content)
            
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            return llm_analysis, input_tokens, output_tokens
        except Exception as e:
            self.logger.error(f"Error during OpenAI analysis: {e}", exc_info=True)
            return {}, 0, 0

    async def analyze_batch(self, chunk_data: dict, system_prompt: str, model_name: str, schema_class: Any, cache_id: Optional[str] = None, is_thinking_mode: bool = False) -> Tuple[Dict[str, Any], int, int]:
        """Analyzes a single data chunk asynchronously."""
        try:
            response = await self.async_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk_data['content']}
                ],
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content
            llm_analysis = json.loads(content)
            
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            return llm_analysis, input_tokens, output_tokens
        except Exception as e:
            self.logger.error(f"Error during async OpenAI analysis: {e}", exc_info=True)
            return {}, 0, 0

    def submit_batch_job(self, input_file_path: str, job_name_prefix: str, system_prompt: str, model_name: str, config: dict, cache_id: Optional[str] = None, is_thinking_mode: bool = False) -> str:
        # This provider processes requests in real-time, so it doesn't implement the batch job submission pattern.
        raise NotImplementedError("OpenAI/LM Studio provider uses real-time processing, not batch job submission.")

    def retrieve_batch_results(self, job_name: str, output_dir: str) -> str:
        raise NotImplementedError("OpenAI/LM Studio provider uses real-time processing, not batch job submission.")
