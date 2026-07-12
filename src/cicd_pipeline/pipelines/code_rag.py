from typing import List
from .base import BasePipeline
from ..steps.base import PipelineStep
from ..steps.git_sync import GitSyncStep
from ..steps.db_setup import DBSetupStep
from ..steps.code_processing import CodeProcessingStep
from ..steps.batch_code_processing import BatchCodeProcessingStep
from ..steps.db_write import DBWriteStep

class CodeRAGPipeline(BasePipeline):
    """
    Defines the specific sequence of steps for the Code RAG ingestion pipeline.
    """
    
    def __init__(self, config):
        """
        Initializes the pipeline and all its steps with the given config.
        
        :param config: The main project configuration object.
        """
        self.config = config

    def get_steps(self) -> List[PipelineStep]:
        """
        Constructs and returns the ordered list of steps for this pipeline.
        """
        if self.config.use_batch_api:
            processing_step = BatchCodeProcessingStep(self.config)
        else:
            processing_step = CodeProcessingStep(self.config)
            
        return [
            GitSyncStep(self.config),
            DBSetupStep(self.config),
            processing_step,
            DBWriteStep(self.config),
        ]
