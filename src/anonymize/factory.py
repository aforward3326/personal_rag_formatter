from typing import List
import json
import logging

from . import config
from .core.processor import AnonymizationProcessor
from .strategies.base import AnonymizeStrategy
from .strategies.identity import IdentityAnonymizer
from .strategies.sensitive_words import SensitiveWordsAnonymizer
from .strategies.regex import RegexAnonymizer
from .strategies.presidio import PresidioAnonymizer, HAS_PRESIDIO

# Only import Presidio components if the library is installed
if HAS_PRESIDIO:
    from presidio_analyzer.nlp_engine import NerModelConfiguration

logger = logging.getLogger(__name__)

class AnonymizerFactory:
    """
    Factory class to construct and configure the main AnonymizationProcessor.
    
    This factory reads from the central config file to decide which
    anonymization strategies to enable and how to configure them.
    This decouples the main application logic from the specific
    implementation details of the anonymization strategies.
    """

    # Define PII-unrelated labels to ignore, preventing false positives and log spam.
    LABELS_TO_IGNORE = [
        "PRODUCT",
        "EVENT",
        "FAC",        # Facility
        "WORK_OF_ART",
        "LAW",
        "LANGUAGE"
    ]

    @staticmethod
    def create_processor() -> AnonymizationProcessor:
        """
        Builds an AnonymizationProcessor with a list of configured strategies.
        The order of strategies is important.
        """
        strategies: List[AnonymizeStrategy] = []

        # 1. Identity Anonymizer (Highest Priority)
        strategies.append(IdentityAnonymizer(config.MY_IDENTITIES_MATRIX_JSON))

        # 2. Custom Sensitive Words Anonymizer
        strategies.append(SensitiveWordsAnonymizer(config.SENSITIVE_WORDS_MAP_JSON))

        # 3. Regex-based Anonymizer
        strategies.append(RegexAnonymizer())

        # 4. Presidio NLP Anonymizer (If available and configured)
        if HAS_PRESIDIO:
            nlp_config = AnonymizerFactory._get_presidio_nlp_config()
            strategies.append(PresidioAnonymizer(
                nlp_config=nlp_config,
                operators=config.PRESIDIO_OPERATORS
            ))
            logger.info("PresidioAnonymizer strategy enabled.")

        # Load the identity map to pass to the processor for speaker tracking
        try:
            identity_map = json.loads(config.MY_IDENTITIES_MATRIX_JSON)
        except (json.JSONDecodeError, TypeError):
            identity_map = {}
            logger.warning("Could not load or parse MY_IDENTITIES_MATRIX_JSON. Speaker tracking might be incomplete.")

        # Assemble the final processor
        processor = AnonymizationProcessor(
            strategies=strategies,
            excluded_keys=config.EXCLUDED_KEYS,
            speaker_tracking_keys=config.SPEAKER_TRACKING_KEYS,
            identity_map=identity_map
        )
        
        return processor

    @staticmethod
    def _get_presidio_nlp_config() -> dict:
        """
        Constructs the NLP engine configuration for Presidio, injecting the
        list of labels to ignore.
        """
        # Start with the base configuration from the config file
        nlp_config = config.PRESIDIO_NLP_CONFIG

        # Define the NER model configuration to ignore specific labels
        ner_model_config = NerModelConfiguration(
            labels_to_ignore=AnonymizerFactory.LABELS_TO_IGNORE
        )

        # Find the SpaCy model configuration for the target language (e.g., 'zh_core_web_lg')
        # and inject the ner_model_configuration.
        # This assumes the config structure contains a list of models.
        if "nlp_engine_name" in nlp_config and nlp_config["nlp_engine_name"] == "spacy":
            models = nlp_config.get("models")
            if models:
                for model in models:
                    # Assuming the model for Chinese is present
                    if "zh" in model.get("lang_code", ""):
                        model["ner_model_configuration"] = ner_model_config
                        logger.info(f"Ignoring Presidio labels for '{model['lang_code']}': {AnonymizerFactory.LABELS_TO_IGNORE}")
                        break

        return nlp_config
