import logging
from typing import List
from openai import OpenAI
from sentence_transformers import SentenceTransformer

class Embedder:
    """Converts text into numerical vector embeddings."""
    def __init__(self, provider: str, model_name: str, base_url: str = None, api_key: str = None):
        self.provider = provider
        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if self.provider == "sentence_transformers":
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        elif self.provider == "lm_studio":
            self.client = OpenAI(base_url=base_url, api_key=api_key or "lm-studio")
        else:
            raise ValueError(f"Unsupported Embedding provider: {self.provider}")
        self.logger.info(f"Embedder initialized with provider: {self.provider}")

    def embed(self, text: str) -> List[float]:
        """Generates an embedding for the given text."""
        self.logger.debug(f"Generating embedding for text: '{text[:50]}...'")
        try:
            if self.provider == "sentence_transformers":
                return self.model.encode(text).tolist()
            elif self.provider == "lm_studio":
                response = self.client.embeddings.create(input=[text], model=self.model_name)
                return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise
