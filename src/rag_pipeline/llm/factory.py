from typing import Optional

from src.rag_pipeline import config
from .base import BaseLLM, BaseEmbedding
from .openai import OpenAIClient
from .gemini_studio import GeminiStudioClient
from .gemini_vertex import GeminiVertexClient
from .lm_studio import LMStudioClient

class AIProviderFactory:
    """
    Factory class to create instances of LLM and Embedding clients based on
    the configuration.
    """
    @staticmethod
    def create_llm(provider: str = config.AI_PROVIDER) -> Optional[BaseLLM]:
        """
        Creates an LLM client instance based on the provider string.
        """
        provider = provider.lower()
        # if provider == "openai":
        #     return OpenAIClient()
        if provider == "gemini":
            return GeminiStudioClient(model_name=config.LLM_MODEL_NAME)
        elif provider == "vertex":
            # Use BATCH_MODEL_NAME for batch mode, otherwise use the standard LLM_MODEL_NAME
            model = config.BATCH_MODEL_NAME if config.USE_BATCH_API else config.LLM_MODEL_NAME
            return GeminiVertexClient(model_name=model)
        elif provider == "lm_studio":
            return LMStudioClient()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def create_embedding(
        provider: str = config.EMBEDDING_PROVIDER, 
        llm_client: Optional[BaseLLM] = None
    ) -> Optional[BaseEmbedding]:
        """
        Creates an Embedding client instance.
        """
        provider = provider.lower()
        # if provider == "openai":
        #     # Reuse the client if it's already an OpenAI client
        #     if isinstance(llm_client, OpenAIClient):
        #         return llm_client
        #     return OpenAIClient()
        if provider == "lm_studio":
             # Reuse the client if it's already an LM Studio client
            if isinstance(llm_client, LMStudioClient):
                return llm_client
            return LMStudioClient()
        # Add other embedding providers like Gemini here when available
        # elif provider == "gemini":
        #     return GeminiEmbeddingClient() 
        else:
            raise NotImplementedError(f"Embedding provider '{provider}' is not currently supported.")
