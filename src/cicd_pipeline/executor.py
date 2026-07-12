import logging
import os
import pickle
from typing import Dict, Any
from .pipelines.base import BasePipeline

class PipelineExecutor:
    """
    Executes a given pipeline by running its steps sequentially.
    Manages the shared context between steps.
    """
    
    def __init__(self, pipeline: BasePipeline, resume: bool = False):
        """
        Initializes the executor with a specific pipeline definition.
        
        :param pipeline: An instance of a class that inherits from BasePipeline.
        :param resume: If True, attempts to resume from the last saved checkpoint.
        """
        self.pipeline = pipeline
        self.resume = resume
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Determine a safe place to store the checkpoint
        workspace = getattr(self.pipeline, 'config', None).base_workspace_dir if hasattr(self.pipeline, 'config') else '/tmp'
        self.checkpoint_file = os.path.join(workspace, f"{self.pipeline.__class__.__name__}_checkpoint.pkl")

    def run(self):
        """
        Executes the entire pipeline from start to finish.
        """
        self.logger.info(f"Starting execution for pipeline: {self.pipeline.__class__.__name__}")
        
        # The context is a dictionary that gets passed and modified by each step.
        context: Dict[str, Any] = {}
        completed_steps = set()

        # Load checkpoint if resuming
        if self.resume and os.path.exists(self.checkpoint_file):
            self.logger.info(f"Resuming from checkpoint: {self.checkpoint_file}")
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    context = checkpoint_data.get('context', {})
                    completed_steps = checkpoint_data.get('completed_steps', set())
                self.logger.info(f"Loaded context and skipping already completed steps: {completed_steps}")
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
        
        steps = self.pipeline.get_steps()
        self.logger.info(f"Pipeline has {len(steps)} steps.")
        
        for step in steps:
            step_name = step.__class__.__name__
            
            if step_name in completed_steps:
                self.logger.info(f"--- Skipping Step: {step_name} (found in checkpoint) ---")
                continue

            try:
                # Each step receives the context and returns an updated version of it.
                context = step(context)
                
                # Optional: Filter out heavy transient objects before pickling 
                # if you decide to save vectors directly to disk in the EmbeddingStep
                checkpoint_context = {k: v for k, v in context.items() if not k.startswith('_transient_')}
                
                # Save checkpoint after a successful step
                completed_steps.add(step_name)
                # Use the highest pickle protocol for faster serialization of large lists/arrays
                with open(self.checkpoint_file, 'wb') as f:
                    pickle.dump({'context': checkpoint_context, 'completed_steps': completed_steps}, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
            except Exception as e:
                self.logger.critical(
                    f"Pipeline failed on step {step_name}. Error: {e}\n"
                    f"You can resume from this point by running the script with --resume",
                    exc_info=True
                )
                # Re-raise the exception to halt the entire process
                raise
        
        # Clean up checkpoint on successful completion
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            self.logger.info("Pipeline completed successfully. Checkpoint cleared.")
        else:
            self.logger.info(f"Pipeline {self.pipeline.__class__.__name__} executed successfully.")
