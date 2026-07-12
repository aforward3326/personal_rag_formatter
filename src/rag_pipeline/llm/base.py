from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

from src.rag_pipeline.llm.schemas import AnalysisDetail, MiniBatchAnalysisResult

class BaseLLM(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    async def get_summary(self, text: str, system_prompt: str) -> Tuple[str, int, int]:
        """
        Summarizes a single chunk of text.

        Returns:
            A tuple of (summary_text, input_tokens, output_tokens).
        """
        pass

    @abstractmethod
    async def analyze(self, chunk_data: Dict, system_prompt: str) -> Tuple[AnalysisDetail, int, int]:
        """
        Analyzes a single chunk of text.

        Returns:
            A tuple of (AnalysisDetail, input_tokens, output_tokens).
        """
        pass

    @abstractmethod
    async def analyze_batch(self, batch_payload: List[Dict], system_prompt: str) -> Tuple[MiniBatchAnalysisResult, int, int]:
        """
        Analyzes a batch of chunks.

        Returns:
            A tuple of (MiniBatchAnalysisResult, input_tokens, output_tokens).
        """
        pass

    async def submit_batch_job(self, *args, **kwargs):
        """Submits a job to a provider's Batch API."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support Batch API submission.")

    async def retrieve_batch_results(self, *args, **kwargs):
        """Retrieves results from a provider's Batch API."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support Batch API retrieval.")


class BaseEmbedding(ABC):
    """Abstract base class for all embedding providers."""

    @abstractmethod
    async def get_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        """
        Generates embeddings for a list of texts.
        """
        pass