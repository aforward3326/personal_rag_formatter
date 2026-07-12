from .base import AnonymizeStrategy
import json
from typing import Dict

class SensitiveWordsAnonymizer(AnonymizeStrategy):
    """
    Anonymizes text by replacing a custom-defined set of sensitive words
    (e.g., project names, company names) with specific tokens.
    """
    
    def __init__(self, sensitive_words_map_json: str):
        """
        Initializes the strategy with a map of sensitive words.

        :param sensitive_words_map_json: A JSON string mapping words to tokens.
                                         e.g., '{"ProjectX": "[REDACTED_PROJECT]"}'
        """
        try:
            self.sensitive_words_map: Dict[str, str] = json.loads(sensitive_words_map_json)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse sensitive words map JSON. Using empty map. JSON: {sensitive_words_map_json}")
            self.sensitive_words_map = {}

    def anonymize(self, text: str, **kwargs) -> str:
        if not isinstance(text, str) or not self.sensitive_words_map:
            return text
            
        for word, mask_token in self.sensitive_words_map.items():
            text = text.replace(word, mask_token)
            
        return text
