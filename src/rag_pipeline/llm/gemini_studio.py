import os
import json
from typing import List, Dict, Any, Tuple
import google.genai as genai
from google.genai import types

from src.rag_pipeline.llm.base import BaseLLM
from src.rag_pipeline.llm.schemas import AnalysisDetail, MiniBatchAnalysisResult, ChunkAnalysisDetail


class GeminiStudioClient(BaseLLM):
    """
    Client for interacting with Google's Gemini models via the Google AI Studio API.
    """
    def __init__(self, model_name: str = "gemini-1.5-flash-latest"):
        self.model_name = model_name
        try:
            # The new SDK initializes the client directly
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        except Exception as e:
            print(f"Warning: Gemini Studio Client could not be configured: {e}")
            self.client = None

    async def get_summary(self, text: str, system_prompt: str) -> Tuple[str, int, int]:
        """Summarizes a single chunk using Gemini."""
        generation_config = types.GenerateContentConfig(
            temperature=0.1,
        )

        full_prompt = f"{system_prompt}\n\n{text}"

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=generation_config,
        )

        summary_text = response.text.strip()

        input_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0

        return summary_text, input_tokens, output_tokens

    async def analyze(self, chunk_data: Dict, system_prompt: str) -> Tuple[AnalysisDetail, int, int]:
        """Analyzes a single chunk using Gemini."""
        user_content = f"[Overlap Context]\n{chunk_data.get('overlap_context', '')}\n\n[Main Content]\n{chunk_data.get('main_content', '')}"
        
        # Gemini uses a specific format for system instructions and JSON output
        generation_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        )
        
        full_prompt = f"{system_prompt}\n\n{user_content}"
        
        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=generation_config,
        )
        
        # Extract the JSON part from the response
        try:
            # The response text might be wrapped in markdown ```json ... ```
            json_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            analysis_data = json.loads(json_text)
            analysis_detail = AnalysisDetail.model_validate(analysis_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse JSON response from Gemini: {response.text}") from e

        # Token counting for Gemini
        input_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
        output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0

        return analysis_detail, input_tokens, output_tokens

    async def analyze_batch(self, batch_payload: List[Dict], system_prompt: str) -> Tuple[MiniBatchAnalysisResult, int, int]:
        """
        Simulates batch analysis by sending concurrent requests to the Gemini API.
        """
        # This is a simplified simulation. A robust implementation would use asyncio.gather
        # and handle individual failures.
        results = []
        total_input_tokens = 0
        total_output_tokens = 0

        for chunk in batch_payload:
            analysis, in_tokens, out_tokens = await self.analyze(chunk, system_prompt)
            results.append(ChunkAnalysisDetail(chunk_id=chunk["chunk_id"], analysis=analysis))
            total_input_tokens += in_tokens
            total_output_tokens += out_tokens
            
        return MiniBatchAnalysisResult(results=results), total_input_tokens, total_output_tokens