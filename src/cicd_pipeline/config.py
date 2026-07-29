# config.py
from datetime import datetime
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class ProjectConfig(BaseSettings):
    """
    Configuration object for the Code RAG Data Ingestion Pipeline.
    In a CI/CD environment, all settings are typically injected via environment variables.
    """
    # --- Core Project Settings ---
    project_name: str
    program_type: str
    git_url: str
    branch: str = "main"
    base_workspace_dir: str = "/tmp/rag_workspace"
    
    # --- CI/CD Environment Settings ---
    local_repo_path: Optional[str] = None

    # --- Log Level ---
    log_level: str = "INFO"
    log_dir: str = "/tmp/rag_workspace/logs"
    run_time_str: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))

    # --- Database Settings ---
    db_host: str
    db_port: str
    db_user: str
    db_password: str
    db_name: str
    db_table_name: Optional[str] = None

    # --- Dual-track LLM Provider Settings ---
    routing_strategy: str = "auto"  # auto, always_standard, always_thinking

    # --- Standard Track Settings ---
    standard_ai_provider: str = "openai"
    standard_model_name: str = "gpt-4o-mini"
    standard_api_key: Optional[str] = None
    standard_base_url: Optional[str] = None # For LM Studio, AnythingLLM, etc.

    # --- Thinking Track Settings ---
    thinking_ai_provider: str = "vertex"
    thinking_model_name: str = "gemini-2.0-flash-thinking-exp"
    thinking_api_key: Optional[str] = None
    thinking_base_url: Optional[str] = None # For LM Studio, AnythingLLM, etc.

    # --- Embedding Settings ---
    embedding_provider: str = "sentence_transformers"
    embedding_model_name: str = "text-embedding-nomic-embed-code"
    embedding_dim: int = 768
    embedding_api_key: Optional[str] = None
    embedding_base_url: Optional[str] = None # For LM Studio hosted embeddings

    # --- Batch & Google Cloud Settings ---
    use_batch_api: bool = False
    batch_model_name: str = "gemini-1.5-flash" # This will be used by BatchCodeProcessingStep
    google_cloud_project: Optional[str] = None
    google_cloud_location: str = "us-central1"
    gcs_bucket_name: Optional[str] = None

    # --- Resume Settings ---
    resume_batch_id: Optional[str] = None

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    @property
    def db_dsn(self) -> str:
        """Constructs the database connection string from individual components."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"