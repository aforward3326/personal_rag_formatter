import json
import logging
from typing import Dict, Any
from openai import OpenAI
import vertexai
from vertexai.generative_models import GenerativeModel
from google import genai
from google.genai import types

class LLMAnalyzer:
    """Uses a Large Language Model for high-level analysis."""
    def __init__(
        self, provider: str, model_name: str, api_key: str = None, 
        base_url: str = None, google_cloud_project: str = None, 
        google_cloud_location: str = "us-central1"
    ):
        self.provider = provider
        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if self.provider == "gemini":
            if not api_key: raise ValueError("Gemini API key is required.")
            self.gemini_client = genai.Client(api_key=api_key)
        elif self.provider in ("gemini_vertex", "vertex"):
            self.project = google_cloud_project
            self.location = google_cloud_location
            vertexai.init(project=self.project, location=self.location)
            self.model = GenerativeModel(model_name)
            self.batch_client = genai.Client(vertexai=True, project=self.project, location=self.location)
        elif self.provider == "lm_studio":
            self.client = OpenAI(base_url=base_url, api_key=api_key or "lm-studio")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        self.logger.info(f"LLM Analyzer initialized with provider: {self.provider}")

    def analyze(self, code_chunk: str) -> Dict[str, Any]:
        """Analyzes a code chunk for summaries and tags."""
        prompt_gemini = f'Analyze the code and return a JSON with "summary_en", "summary_zh", and "tags" (string array).\nCode:\n```{code_chunk}```'
        prompt_local = f'Analyze the code. Provide a concise English summary, a Traditional Chinese summary, and comma-separated tags.\nFormat: summary_en|||summary_zh|||tag1,tag2,tag3\nCode:\n```{code_chunk}```'

        try:
            if self.provider == "gemini":
                response = self.gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt_gemini,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        max_output_tokens=200,
                        temperature=0.1
                    )
                )
                return json.loads(response.text)
            
            elif self.provider in ("gemini_vertex", "vertex"):
                response = self.model.generate_content(
                    prompt_gemini,
                    generation_config={"response_mime_type": "application/json", "temperature": 0.1}
                )
                clean_json = response.text.strip().replace("```json", "").replace("```", "").strip()
                return json.loads(clean_json)
            
            elif self.provider == "lm_studio":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an assistant that provides analysis in the format: summary_en|||summary_zh|||tag1,tag2,tag3"},
                        {"role": "user", "content": prompt_local}
                    ],
                    temperature=0.0, max_tokens=150
                )
                parts = response.choices[0].message.content.strip().split('|||')
                if len(parts) != 3:
                    self.logger.warning(f"LLM output format error. Got {len(parts)} parts.")
                    return {"summary_en": "Analysis failed.", "summary_zh": "分析失敗。", "tags": []}
                return {
                    "summary_en": parts[0].strip(),
                    "summary_zh": parts[1].strip(),
                    "tags": [tag.strip() for tag in parts[2].split(',') if tag.strip()]
                }
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            raise

    def submit_batch_job(self, model_name: str, input_gcs_uri: str, output_gcs_uri_prefix: str) -> Any:
        """Submits a batch prediction job to Vertex AI."""
        if self.provider not in ("gemini_vertex", "vertex"):
            raise NotImplementedError("Batch processing is only supported for 'gemini_vertex' or 'vertex' provider.")
        return self.batch_client.batches.create(
            model=model_name, src=input_gcs_uri, config={"dest": output_gcs_uri_prefix}
        )

    def get_batch_job_status(self, name: str) -> Any:
        """Retrieves the status of a specific batch job."""
        if self.provider not in ("gemini_vertex", "vertex"):
            raise NotImplementedError("Batch processing is only supported for 'gemini_vertex' or 'vertex' provider.")
        return self.batch_client.batches.get(name=name)
