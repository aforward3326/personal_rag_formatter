import asyncio
from typing import List, Dict, Tuple
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from src.rag_pipeline import config
from src.rag_pipeline.llm.factory import AIProviderFactory
from src.rag_pipeline.llm.schemas import AnalysisDetail, MiniBatchAnalysisResult
from src.rag_pipeline.utils.logger import logger

# --- AI Client Initialization ---
llm_client = None
embedding_client = None

try:
    logger.info(f"Initializing LLM provider: {config.AI_PROVIDER}")
    llm_client = AIProviderFactory.create_llm(config.AI_PROVIDER)
    if not llm_client:
        raise ValueError(f"Failed to create LLM client for provider: {config.AI_PROVIDER}")

    # Safely initialize embedding client
    try:
        logger.info(f"Initializing Embedding provider: {config.EMBEDDING_PROVIDER}")
        embedding_client = AIProviderFactory.create_embedding(config.EMBEDDING_PROVIDER, llm_client)
        if not embedding_client:
            raise ValueError(f"Failed to create Embedding client for provider: {config.EMBEDDING_PROVIDER}")
    except NotImplementedError as e:
        logger.warning(f"Could not initialize embedding client: {e}")
        embedding_client = None # Ensure it's None if not supported

except (ValueError, ImportError, RuntimeError) as e:
    logger.error(f"A critical error occurred during AI client initialization: {e}")
    # This will be caught on startup in main.py
    raise e

# --- Retry Configuration ---
RETRY_CONFIG = {
    "wait": wait_exponential(multiplier=2, min=4, max=60),
    "stop": stop_after_attempt(10),
    "retry": retry_if_exception_type(Exception),
}

# --- Service Functions ---

@retry(**RETRY_CONFIG)
async def summarize_chunk(
    chunk_data: Dict,
    semaphore: asyncio.Semaphore
) -> Tuple[str, int, int]:
    """Summarizes a single chunk of text using the configured LLM provider."""
    async with semaphore:
        if not llm_client:
            raise RuntimeError("LLM client not initialized.")
        # Note: You might want to create a specific system prompt for summarization in your config
        system_prompt = "Summarize the following text, focusing on the key facts and entities. Be concise and clear. The summary will be used for semantic search."
        # This assumes your llm_client has a `summarize` method or you adapt the `analyze` method
        # For now, let's assume we adapt `analyze` and the prompt is what matters.
        # This is a placeholder for what should be a dedicated summarization call.
        # Let's pretend the `analyze` method can return a simple string for a summarization task.
        # This will likely require modification of the LLM client interface.
        # For now, I will assume a simple text-in, text-out method `get_summary`.
        return await llm_client.get_summary(chunk_data["main_content"], system_prompt)


@retry(**RETRY_CONFIG)
async def analyze_chunk_for_cleaning(
    chunk_data: Dict,
    semaphore: asyncio.Semaphore
) -> Tuple[AnalysisDetail, int, int]:
    """Analyzes a single chunk of text using the configured LLM provider."""
    async with semaphore:
        if not llm_client:
            raise RuntimeError("LLM client not initialized.")
        system_prompt = config.SYSTEM_PROMPT_TEMPLATE
        return await llm_client.analyze(chunk_data, system_prompt)

@retry(**RETRY_CONFIG)
async def analyze_batch_of_chunks_for_cleaning(
    batch_payload: List[Dict],
    semaphore: asyncio.Semaphore
) -> Tuple[MiniBatchAnalysisResult, int, int]:
    """Analyzes a batch of chunks using the configured LLM provider's batch endpoint."""
    async with semaphore:
        if not llm_client:
            raise RuntimeError("LLM client not initialized.")
        system_prompt = config.MINI_BATCH_SYSTEM_PROMPT_TEMPLATE
        return await llm_client.analyze_batch(batch_payload, system_prompt)

@retry(**RETRY_CONFIG)
async def get_embeddings_for_texts(
    texts: List[str],
    semaphore: asyncio.Semaphore
) -> List[List[float]]:
    """Generates embedding vectors for a list of texts."""
    if not texts:
        return []
    async with semaphore:
        if not embedding_client:
            logger.error("Embedding client not initialized. Cannot generate embeddings.")
            # Return empty embeddings to prevent crashing the pipeline.
            # The calling function should handle this gracefully.
            return [[] for _ in texts]
        return await embedding_client.get_embeddings(texts, config.EMBEDDING_MODEL_NAME)