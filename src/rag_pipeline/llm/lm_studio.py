import os
from typing import List, Dict, Any, Tuple
from openai import AsyncOpenAI
import tiktoken

from src.rag_pipeline.llm.base import BaseLLM, BaseEmbedding
from src.rag_pipeline.llm.schemas import AnalysisDetail, MiniBatchAnalysisResult

# Configure the client to connect to the local LM Studio server
# The API key is often not required for local servers, but we pass "not-needed".
client = AsyncOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Initialize tokenizer for token counting (can be a rough estimate for local models)
try:
    TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    TIKTOKEN_ENCODING = None

def _count_tokens(text: str) -> int:
    if not TIKTOKEN_ENCODING:
        return len(text) // 4 # Rough estimate
    return len(TIKTOKEN_ENCODING.encode(text))


class LMStudioClient(BaseLLM, BaseEmbedding):
    """
    Client for interacting with a local LLM served via LM Studio.
    It uses the OpenAI-compatible API endpoint.
    """
    async def get_summary(self, text: str, system_prompt: str) -> Tuple[str, int, int]:
        """Summarizes a single chunk using the local LLM."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        response = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME", "local-model"),
            messages=messages,
            temperature=0.1,
        )

        summary_text = response.choices[0].message.content
        input_tokens = _count_tokens(system_prompt + text)
        output_tokens = _count_tokens(summary_text)

        return summary_text, input_tokens, output_tokens

    async def analyze(self, chunk_data: Dict, system_prompt: str) -> Tuple[AnalysisDetail, int, int]:
        """Analyzes a single chunk using the local LLM."""
        user_content = f"[Overlap Context]\n{chunk_data.get('overlap_context', '')}\n\n[Main Content]\n{chunk_data.get('main_content', '')}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Use the 'instructor' library's feature to get structured output
        response = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME", "local-model"), # Model name is often ignored by LM Studio but good practice to have
            messages=messages,
            response_model=AnalysisDetail,
            temperature=0.2,
        )
        
        input_tokens = _count_tokens(system_prompt + user_content)
        output_tokens = _count_tokens(response.model_dump_json())

        return response, input_tokens, output_tokens

    async def analyze_batch(self, batch_payload: List[Dict], system_prompt: str) -> Tuple[MiniBatchAnalysisResult, int, int]:
        """Local models typically don't support batching; this will raise an error."""
        raise NotImplementedError("LM Studio does not support native batch analysis. Please use non-batch mode.")

    async def get_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        """
        Generates embeddings using the local model, if it supports it.
        """
        if not texts:
            return []
        
        # The model name for embeddings might be different.
        embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "local-embedding-model")
        
        response = await client.embeddings.create(
            input=texts,
            model=embedding_model
        )
        
        return [embedding.embedding for embedding in response.data]