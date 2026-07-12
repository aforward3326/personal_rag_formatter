from typing import List, Dict, Any, Set
from ..strategies.base import AnonymizeStrategy

class AnonymizationProcessor:
    """
    Orchestrates the entire anonymization process for a document.
    It manages a list of anonymization strategies and applies them
    recursively to a dictionary-like object (a document).
    It also handles speaker tracking within the scope of a single document.
    """

    def __init__(
        self,
        strategies: List[AnonymizeStrategy],
        excluded_keys: Set[str],
        speaker_tracking_keys: Set[str],
        identity_map: Dict[str, str]
    ):
        """
        Initializes the processor.

        :param strategies: A list of anonymization strategy instances to apply.
        :param excluded_keys: A set of keys that should never be masked.
        :param speaker_tracking_keys: A set of keys used to identify speakers.
        :param identity_map: A map to identify the author ('[AUTHOR_ME]').
        """
        self.strategies = strategies
        self.excluded_keys = excluded_keys
        self.speaker_tracking_keys = speaker_tracking_keys
        self.identity_map = identity_map
        self.author_identities = set(self.identity_map.keys())

    def process_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymizes a full document, tracking speakers throughout.

        :param doc: The document (as a dictionary) to anonymize.
        :return: The anonymized document.
        """
        # Speaker map is reset for each new document to ensure speaker IDs
        # (e.g., [SPEAKER_1]) are consistent only within that document.
        speaker_map: Dict[str, str] = {}
        return self._mask_recursive_helper(doc, speaker_map)

    def _apply_strategies(self, text: str) -> str:
        """Applies all registered strategies sequentially to a string."""
        for strategy in self.strategies:
            text = strategy.anonymize(text)
        return text

    def _mask_recursive_helper(self, data: Any, speaker_map: Dict[str, str], key_name: str = None) -> Any:
        """The core recursive function that traverses the data structure."""
        if isinstance(data, str):
            # 1. Check if the key is structurally excluded (e.g., 'uuid')
            if key_name in self.excluded_keys:
                return data

            # 2. Check if it's a speaker tracking key
            if key_name in self.speaker_tracking_keys:
                original_name = data.strip()
                
                # Is it the author?
                if original_name in self.author_identities:
                    return self.identity_map.get(original_name, "[AUTHOR_ME]")
                
                # Has this person spoken before in this document?
                if original_name in speaker_map:
                    return speaker_map[original_name]
                
                # New speaker, assign a new ID
                new_speaker_id = f"[SPEAKER_{len(speaker_map) + 1}]"
                speaker_map[original_name] = new_speaker_id
                return new_speaker_id

            # 3. Otherwise, it's general content text. Apply all strategies.
            return self._apply_strategies(data)
        
        elif isinstance(data, dict):
            return {k: self._mask_recursive_helper(v, speaker_map, key_name=k) for k, v in data.items()}
        
        elif isinstance(data, list):
            # Pass the key_name down for context (e.g., if it's a list under 'participants')
            return [self._mask_recursive_helper(item, speaker_map, key_name=key_name) for item in data]
        
        else:
            # Return numbers, booleans, nulls, etc. as is
            return data
