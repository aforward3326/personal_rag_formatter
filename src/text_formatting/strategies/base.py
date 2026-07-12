from abc import ABC, abstractmethod
import logging
from pathlib import Path

class TextFormatterStrategy(ABC):
    """
    Abstract base class for a text formatting strategy.
    Each strategy is responsible for processing data from a specific
    source (e.g., Gmail, Meta, HTML chats) and converting it into a
    standardized RAG-ready JSON format.
    """
    
    def __init__(self, input_dir: Path, output_file: Path, **kwargs):
        """
        Initializes the strategy with input and output paths.

        :param input_dir: The directory where source data is located.
        :param output_file: The path to the target output JSON file.
        :param kwargs: Additional configuration specific to the strategy.
        """
        self.input_dir = input_dir
        self.output_file = output_file
        self.output_dir = output_file.parent
        self.logger = logging.getLogger(self.__class__.__name__)
        self.progress_file = kwargs.get("progress_file", None)

    @abstractmethod
    def process(self):
        """
        The main method that executes the formatting process.
        This should handle file discovery, parsing, transformation,
        and saving the final output.
        """
        pass
