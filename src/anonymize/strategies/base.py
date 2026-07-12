from abc import ABC, abstractmethod

class AnonymizeStrategy(ABC):
    """
    Abstract base class for an anonymization strategy.
    Each strategy is responsible for one type of text masking.
    """
    
    @abstractmethod
    def anonymize(self, text: str, **kwargs) -> str:
        """
        Applies the anonymization logic to the given text.

        :param text: The input string to anonymize.
        :param kwargs: Optional additional parameters for context.
        :return: The anonymized string.
        """
        pass
