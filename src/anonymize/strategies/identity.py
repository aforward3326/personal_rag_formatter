from .base import AnonymizeStrategy
import json
from typing import Dict

class IdentityAnonymizer(AnonymizeStrategy):
    """
    Anonymizes text by replacing known author identities 
    (e.g., name, email) with a specific token like '[AUTHOR_ME]'.
    """
    
    def __init__(self, identity_map_json: str):
        """
        Initializes the strategy with a map of identities.

        :param identity_map_json: A JSON string mapping identities to tokens.
                                  e.g., '{"Your Name": "[AUTHOR_ME]"}'
        """
        try:
            self.identity_map: Dict[str, str] = json.loads(identity_map_json)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse identity map JSON. Using empty map. JSON: {identity_map_json}")
            self.identity_map = {}

    def anonymize(self, text: str, **kwargs) -> str:
        if not isinstance(text, str) or not self.identity_map:
            return text
            
        for identity, mask_token in self.identity_map.items():
            # Use a simple and fast string replacement
            text = text.replace(identity, mask_token)
            
        return text
