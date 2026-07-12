from abc import ABC, abstractmethod
from typing import List
from ..steps.base import PipelineStep

class BasePipeline(ABC):
    """
    Abstract base class for defining a pipeline.
    A pipeline is essentially a sequence of steps.
    """
    
    @abstractmethod
    def get_steps(self) -> List[PipelineStep]:
        """
        Should return an ordered list of pipeline step instances.
        
        :return: A list of initialized PipelineStep objects.
        """
        pass
