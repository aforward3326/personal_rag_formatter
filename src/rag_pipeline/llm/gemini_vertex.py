import os
import json
from typing import Dict, Any, Tuple, List

import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig, Content
from google.api_core.exceptions import GoogleAPICallError
from google.cloud import aiplatform
from google import genai

from src.rag_pipeline.llm.base import BaseLLM
from src.rag_pipeline.llm.schemas import AnalysisDetail, MiniBatchAnalysisResult, ChunkAnalysisDetail

# Initialize Vertex AI
try:
    vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))
except Exception as e:
    print(f"Warning: Vertex AI could not be configured: {e}")


class GeminiVertexClient(BaseLLM):
    """
    Client for interacting with Gemini models on Google Cloud Vertex AI,
    with support for both single and batch predictions.
    """
    def __init__(self, model_name: str = "gemini-1.5-flash-001"):
        self.model = GenerativeModel(model_name)

        # Capture project and location, trying env vars first, then SDK config
        self.project = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not self.project:
            try:
                self.project = vertexai.get_config().project
            except Exception:
                # If this fails, subsequent calls will raise a more specific error.
                pass

        # Use the modern GenAI client for batch operations (matches test script behavior)
        if self.project:
            self.batch_client = genai.Client(vertexai=True, project=self.project, location=self.location)

    async def get_summary(self, text: str, system_prompt: str) -> Tuple[str, int, int]:
        """Summarizes a single chunk of text using the Vertex AI backend."""
        generation_config = GenerationConfig(
            temperature=0.1,  # Summarization can be less creative
        )

        system_instruction = Content(
            parts=[Part.from_text(system_prompt)],
            role="system"
        )

        try:
            response = await self.model.generate_content_async(
                [text],
                generation_config=generation_config,
                system_instruction=system_instruction,
            )

            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            summary_text = response.text.strip()
        except (GoogleAPICallError, ValueError) as e:
            raise ValueError(f"Failed to get summary from Vertex AI: {e}") from e

        return summary_text, input_tokens, output_tokens

    async def analyze(self, chunk_data: Dict, system_prompt: str) -> Tuple[AnalysisDetail, int, int]:
        """Analyzes a single chunk using the Vertex AI backend."""
        user_content = f"[Main Content]\n{chunk_data.get('main_content', '')}"
        if chunk_data.get('overlap_context'):
            user_content = f"[Overlap Context]\n{chunk_data.get('overlap_context')}\n\n" + user_content
        
        generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2,
        )
        
        system_instruction = Content(
            parts=[Part.from_text(system_prompt)],
            role="system"
        )
        
        try:
            response = await self.model.generate_content_async(
                [user_content],
                generation_config=generation_config,
                system_instruction=system_instruction,
            )
            
            # Extract usage metadata for token counts
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count

            json_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            analysis_data = json.loads(json_text)
            analysis_detail = AnalysisDetail.model_validate(analysis_data)

        except (GoogleAPICallError, json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to get or parse valid JSON response from Vertex AI: {e}") from e

        return analysis_detail, input_tokens, output_tokens

    async def analyze_batch(self, batch_payload: List[Dict], system_prompt: str) -> Tuple[MiniBatchAnalysisResult, int, int]:
        """Analyzes a batch of chunks by calling the single analyze method for each."""
        results = []
        total_input_tokens = 0
        total_output_tokens = 0

        for chunk in batch_payload:
            try:
                analysis_detail, input_tokens, output_tokens = await self.analyze(chunk, system_prompt)
                results.append(ChunkAnalysisDetail(chunk_id=chunk['chunk_id'], analysis=analysis_detail))
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
            except ValueError as e:
                # In a real-world scenario, you might want to handle this more gracefully
                # For example, by logging the error and continuing with the next chunk
                print(f"Error analyzing chunk {chunk.get('chunk_id')}: {e}")


        return MiniBatchAnalysisResult(results=results), total_input_tokens, total_output_tokens

    def submit_batch_job(
        self,
        model_name: str,
        input_gcs_uri: str,
        output_gcs_uri_prefix: str
    ) -> Any:
        """
        Submits a batch prediction job to Vertex AI.

        Args:
            model_name: The model to use for the batch job (e.g., "gemini-1.5-flash-001").
            input_gcs_uri: The GCS URI of the input JSONL file.
            output_gcs_uri_prefix: The GCS URI prefix for the output results.

        Returns:
            The created BatchPredictionJob object.
        """
        if not self.project:
            raise ValueError(
                "Google Cloud Project ID could not be determined. "
                "Please set the GOOGLE_CLOUD_PROJECT environment variable or run 'gcloud auth application-default login'."
            )
        
        try:
            # Use the high-level genai client to submit the batch job
            job = self.batch_client.batches.create(
                model=model_name,
                src=input_gcs_uri,
                config={"dest": output_gcs_uri_prefix}
            )
            return job
        except Exception as e:
            raise RuntimeError(f"Failed to submit batch job to Vertex AI: {e}") from e

    def get_batch_job_status(self, name: str) -> Any:
        """
        Retrieves the status of a specific batch job.

        Args:
            name: The full resource name of the batch job.

        Returns:
            The BatchPredictionJob object with the latest status.
        """
        try:
            # Use the high-level genai client to retrieve job status
            return self.batch_client.batches.get(name=name)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve batch job status for '{name}': {e}") from e