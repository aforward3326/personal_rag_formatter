import logging
from typing import List, Union
from openai import OpenAI
from sentence_transformers import SentenceTransformer

class Embedder:
    """Converts text into numerical vector embeddings."""
    def __init__(self, provider: str, model_name: str, base_url: str = None, api_key: str = None):
        self.provider = provider
        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if self.provider == "sentence_transformers":
            # trust_remote_code=True is required for some models like nomic-embed-text
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        elif self.provider == "lm_studio":
            self.client = OpenAI(base_url=base_url, api_key=api_key or "lm-studio")
        else:
            raise ValueError(f"Unsupported Embedding provider: {self.provider}")
        self.logger.info(f"Embedder initialized with provider: {self.provider}")

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generates embeddings for a single text or a list of texts.
        Always returns a list of embedding vectors.
        """
        if isinstance(texts, str):
            texts = [texts] # Ensure input is always a list

        self.logger.debug(f"Generating embeddings for {len(texts)} text(s)...")
        try:
            if self.provider == "sentence_transformers":
                # SentenceTransformer returns a numpy array, so we convert it to a list of lists
                return self.model.encode(texts).tolist()
            elif self.provider == "lm_studio":
                # The 'input' parameter expects a list of strings
                response = self.client.embeddings.create(input=texts, model=self.model_name)
                # The response contains a list of embedding objects, extract the embedding from each
                return [item.embedding for item in response.data]
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}", exc_info=True)
            raise
