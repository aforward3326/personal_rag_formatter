from typing import List
import json

from . import config
from .core.processor import AnonymizationProcessor
from .strategies.base import AnonymizeStrategy
from .strategies.identity import IdentityAnonymizer
from .strategies.sensitive_words import SensitiveWordsAnonymizer
from .strategies.regex import RegexAnonymizer
from .strategies.presidio import PresidioAnonymizer, HAS_PRESIDIO

class AnonymizerFactory:
    """
    Factory class to construct and configure the main AnonymizationProcessor.
    
    This factory reads from the central config file to decide which
    anonymization strategies to enable and how to configure them.
    This decouples the main application logic from the specific
    implementation details of the anonymization strategies.
    """

    @staticmethod
    def create_processor() -> AnonymizationProcessor:
        """
        Builds an AnonymizationProcessor with a list of configured strategies.
        The order of strategies is important.
        """
        strategies: List[AnonymizeStrategy] = []

        # 1. Identity Anonymizer (Highest Priority)
        # This should run first to correctly identify the author before any
        # other masking might obscure their identity.
        strategies.append(IdentityAnonymizer(config.MY_IDENTITIES_MATRIX_JSON))

        # 2. Custom Sensitive Words Anonymizer
        strategies.append(SensitiveWordsAnonymizer(config.SENSITIVE_WORDS_MAP_JSON))

        # 3. Regex-based Anonymizer
        # Good for well-defined patterns like IPs, emails, etc.
        strategies.append(RegexAnonymizer())

        # 4. Presidio NLP Anonymizer (If available)
        # This is the most computationally expensive and should run last.
        if HAS_PRESIDIO:
            strategies.append(PresidioAnonymizer(
                nlp_config=config.PRESIDIO_NLP_CONFIG,
                operators=config.PRESIDIO_OPERATORS
            ))
        
        # Load the identity map to pass to the processor for speaker tracking
        try:
            identity_map = json.loads(config.MY_IDENTITIES_MATRIX_JSON)
        except json.JSONDecodeError:
            identity_map = {}

        # Assemble the final processor
        processor = AnonymizationProcessor(
            strategies=strategies,
            excluded_keys=config.EXCLUDED_KEYS,
            speaker_tracking_keys=config.SPEAKER_TRACKING_KEYS,
            identity_map=identity_map
        )
        
        return processor
