import re
import math
import numpy as np
from typing import List

# Assuming the embedder has an `embed` method that returns a list of floats (vector)
class HybridRouter:
    def __init__(self, embedder):
        """
        Initializes the HybridRouter with a pre-instantiated embedder.
        """
        self.embedder = embedder
        self.SIMILARITY_THRESHOLD = 0.82
        self.SCORE_THRESHOLD = 10

        # Define anchor points for complex logic and pre-embed them
        self.anchor_texts = [
            "Complex CI/CD pipeline orchestration, Jenkins automation, Docker container lifecycle management, and Kubernetes deployments.",
            "Advanced state management, reactive programming paradigms, and complex API data flows.",
            "Core algorithm implementation involving OpenCV real-time processing, TensorFlow mathematical operations, or cryptography."
        ]
        self.anchor_vectors = self.embedder.embed(self.anchor_texts)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculates the cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return dot_product / (norm_vec1 * norm_vec2)

    def requires_deep_thinking(self, file_name: str, content: str) -> bool:
        """
        Determines if a file requires a deep-thinking model using a hybrid
        approach of regex scoring and semantic similarity.
        """
        # --- Stage 1: Regex Filtering and Scoring ---
        
        # Basic defense: too few lines
        if len(content.splitlines()) < 80:
            return False

        # Filter by core programming extensions
        core_extensions = ['.py', '.dart', '.cpp', '.java', '.js', '.ts', '.go', '.rs']
        is_core_file = any(file_name.endswith(ext) for ext in core_extensions)
        
        # Filter out static config/doc files early if they are not core files
        static_extensions = ['.json', '.yaml', '.yml', '.md', '.xml', '.toml', '.ini']
        if not is_core_file and any(file_name.endswith(ext) for ext in static_extensions):
            return False

        content_lower = content.lower()
        total_score = 0

        # Scoring patterns (same as before)
        high_impact_patterns = [r'@springbootapplication', r'@enablediscoveryclient', r'fastapi\(', r'celery\(', r'kubernetes', r'jenkinsfile', r'oauth2']
        medium_impact_patterns = [r'@restcontroller', r'@transactional', r'@repository', r'completablefuture', r'redis', r'asyncio', r'sqlalchemy', r'pydantic', r'dockerfile', r'opencv', r'tensorflow']
        low_impact_patterns = [r'\bfactory\b', r'\bsingleton\b', r'\bstrategy\b', r'axios']
        trace_impact_patterns = [r'\butils?\b', r'\bhelpers?\b', r'logger']

        for pattern in high_impact_patterns:
            if re.search(pattern, content_lower): total_score += 10
        for pattern in medium_impact_patterns:
            if re.search(pattern, content_lower): total_score += 5
        for pattern in low_impact_patterns:
            if re.search(pattern, content_lower): total_score += 3
        for pattern in trace_impact_patterns:
            if re.search(pattern, content_lower): total_score += 1

        # Immediate decision based on high score
        if total_score >= self.SCORE_THRESHOLD:
            return True
        
        # If the file is not a core programming file and score is low, reject it
        if not is_core_file and total_score < 5:
            return False

        # --- Stage 2: Semantic Routing ---
        
        # Only perform semantic check if the regex score is in the ambiguous range (1-9)
        # and it's a core file type that could contain complex logic.
        if 1 <= total_score < self.SCORE_THRESHOLD and is_core_file:
            content_vector = self.embedder.embed([content])[0]
            
            max_similarity = 0.0
            for anchor_vector in self.anchor_vectors:
                similarity = self._cosine_similarity(content_vector, anchor_vector)
                if similarity > max_similarity:
                    max_similarity = similarity
            
            if max_similarity >= self.SIMILARITY_THRESHOLD:
                return True

        return False
