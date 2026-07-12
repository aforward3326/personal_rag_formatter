from abc import ABC, abstractmethod
import logging
from typing import Dict, Any

class PipelineStep(ABC):
    """
    Abstract base class for a single step in a CI/CD pipeline.
    Each step is designed to be a modular, reusable component.
    """
    
    def __init__(self, config, **kwargs):
        """
        Initializes the step with configuration and a logger.

        :param config: The central project configuration object.
        :param kwargs: Can contain shared objects like a logger.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the logic for this pipeline step.

        :param context: A dictionary containing data passed from previous steps.
                        This can include file paths, commit hashes, processed data, etc.
        :return: An updated context dictionary with the results of this step.
                 This allows chaining steps together.
        """
        pass

    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Allows the step to be called like a function."""
        self.logger.info(f"--- Running Step: {self.__class__.__name__} ---")
        updated_context = self.run(context)
        self.logger.info(f"--- Finished Step: {self.__class__.__name__} ---")
        return updated_context
