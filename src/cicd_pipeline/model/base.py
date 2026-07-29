from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any

class BaseAIProvider(ABC):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        # Allow for other provider-specific kwargs
        self.kwargs = kwargs

    @abstractmethod
    def create_context_cache(self, system_prompt: str, ttl_hours: int = 1) -> Optional[str]:
        """Creates a context cache and returns a cache ID. Returns None if not supported."""
        pass

    @abstractmethod
    def analyze(self, chunk_data: dict, system_prompt: str, model_name: str, schema_class: Any, cache_id: Optional[str] = None, is_thinking_mode: bool = False) -> Tuple:
        """Analyzes a single data chunk."""
        pass

    @abstractmethod
    async def analyze_batch(self, chunk_data: dict, system_prompt: str, model_name: str, schema_class: Any, cache_id: Optional[str] = None, is_thinking_mode: bool = False) -> Tuple:
        """Analyzes a batch of data chunks."""
        pass

    @abstractmethod
    def submit_batch_job(self, input_file_path: str, job_name_prefix: str, system_prompt: str, model_name: str, config: dict, cache_id: Optional[str] = None, is_thinking_mode: bool = False) -> str:
        """Submits a batch job and returns a job ID."""
        pass

    @abstractmethod
    def retrieve_batch_results(self, job_name: str, output_dir: str) -> str:
        """Retrieves results from a completed batch job."""
        pass

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def get_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        """Generates embeddings for a list of texts."""
        pass
