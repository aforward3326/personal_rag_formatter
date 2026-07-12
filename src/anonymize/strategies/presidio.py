from .base import AnonymizeStrategy
from typing import Dict, Any, Optional

# Attempt to import Presidio packages
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    HAS_PRESIDIO = True
except ImportError:
    HAS_PRESIDIO = False
    AnalyzerEngine = None
    AnonymizerEngine = None
    NlpEngineProvider = None
    OperatorConfig = None

class PresidioAnonymizer(AnonymizeStrategy):
    """
    Anonymizes text using Presidio's NLP capabilities to detect and mask
    entities like PERSON, LOCATION, and ORGANIZATION.
    """
    
    def __init__(self, nlp_config: Dict[str, Any], operators: Dict[str, Any]):
        """
        Initializes the Presidio strategy.

        :param nlp_config: Configuration dictionary for the Presidio NLP engine.
        :param operators: Dictionary defining how to handle found entities.
        """
        self.analyzer: Optional[AnalyzerEngine] = None
        self.anonymizer: Optional[AnonymizerEngine] = None
        
        if not HAS_PRESIDIO:
            print("Warning: 'presidio-analyzer' or 'presidio-anonymizer' not installed. PresidioAnonymizer will be disabled.")
            return

        try:
            print("Initializing Presidio NLP engine (spaCy)...")
            provider = NlpEngineProvider(nlp_configuration=nlp_config)
            nlp_engine = provider.create_engine()
            
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["zh", "en"])
            self.anonymizer = AnonymizerEngine()
            
            # Convert operator dict from config to OperatorConfig objects
            self.operators = {
                entity: OperatorConfig(operator["type"], {"new_value": operator["new_value"]})
                for entity, operator in operators.items()
            }
            print("Presidio NLP engine initialized successfully.")
        except Exception as e:
            print(f"Error: Presidio initialization failed. Please ensure spaCy models are downloaded. Error: {e}")
            self.analyzer = None
            self.anonymizer = None

    def anonymize(self, text: str, **kwargs) -> str:
        if not all([isinstance(text, str), self.analyzer, self.anonymizer]):
            return text

        try:
            # Analyze the text to find PII entities
            analyzer_results = self.analyzer.analyze(text=text, language='zh')
            
            # Anonymize the text based on the results and defined operators
            anonymized_result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=self.operators
            )
            return anonymized_result.text
        except Exception as e:
            # Log or print a debug message if something goes wrong during analysis
            print(f"Debug: Presidio processing failed for a text snippet. Error: {e}")
            return text
