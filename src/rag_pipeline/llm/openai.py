import os
from typing import List, Dict, Any, Tuple
from openai import AsyncOpenAI
import tiktoken

from src.rag_pipeline.llm.base import BaseLLM, BaseEmbedding
from src.rag_pipeline.llm.schemas import AnalysisDetail, MiniBatchAnalysisResult

# Initialize the OpenAI client
# It's good practice to initialize it once and reuse it.
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize tokenizer for token counting
try:
    TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    TIKTOKEN_ENCODING = None

def _count_tokens(text: str) -> int:
    if not TIKTOKEN_ENCODING:
        return len(text) // 4 # Rough estimate
    return len(TIKTOKEN_ENCODING.encode(text))


class OpenAIClient(BaseLLM, BaseEmbedding):
    """
    Client for interacting with OpenAI's API for both LLM and Embedding tasks.
    """
    async def get_summary(self, text: str, system_prompt: str) -> Tuple[str, int, int]:
        """Summarizes a single chunk using OpenAI's chat completion."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        response = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
            messages=messages,
            temperature=0.1,
        )

        summary_text = response.choices[0].message.content
        input_tokens = _count_tokens(system_prompt + text)
        output_tokens = _count_tokens(summary_text)

        return summary_text, input_tokens, output_tokens

    async def analyze(self, chunk_data: Dict, system_prompt: str) -> Tuple[AnalysisDetail, int, int]:
        """Analyzes a single chunk using OpenAI's chat completion."""
        user_content = f"[Overlap Context]\n{chunk_data.get('overlap_context', '')}\n\n[Main Content]\n{chunk_data.get('main_content', '')}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
            messages=messages,
            response_model=AnalysisDetail,
            temperature=0.2,
        )
        
        input_tokens = _count_tokens(system_prompt + user_content)
        output_tokens = _count_tokens(response.model_dump_json())

        return response, input_tokens, output_tokens

    async def analyze_batch(self, batch_payload: List[Dict], system_prompt: str) -> Tuple[MiniBatchAnalysisResult, int, int]:
        """
        OpenAI's standard API does not support batching in the same way as some other providers.
        This method will simulate batching by sending requests concurrently.
        However, for this reconstruction, we will raise NotImplementedError and recommend
        using the non-batch pipeline for OpenAI.
        """
        # For a true implementation, you would use asyncio.gather here.
        # For now, we signal that this is not the intended use for this client.
        raise NotImplementedError("OpenAI standard API does not support native batch analysis. Please use non-batch mode.")

    async def get_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        """Generates embeddings for a list of texts using OpenAI's embedding models."""
        if not texts:
            return []
        
        response = await client.embeddings.create(
            input=texts,
            model=model
        )
        
        return [embedding.embedding for embedding in response.data]