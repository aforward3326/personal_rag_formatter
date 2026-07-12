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
    # These required fields are automatically populated from environment variables
    # of the same name (e.g., PROJECT_NAME).
    project_name: str
    program_type: str
    git_url: str
    branch: str = "main"
    base_workspace_dir: str = "/tmp/rag_workspace"
    
    # --- CI/CD Environment Settings ---
    # Can be injected directly via environment variables like $WORKSPACE from Jenkins. If specified, this directory will be used as the Git root.
    local_repo_path: Optional[str] = None

    # --- Log Level ---
    log_level: str = "INFO"
    log_dir: str = "/tmp/rag_workspace/logs"
    run_time_str: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))

    # --- Database Settings ---
    # Also injected from environment variables like DB_HOST, DB_PASSWORD, etc.
    db_host: str
    db_port: str
    db_user: str
    db_password: str
    db_name: str
    db_table_name: Optional[str] = None

    # --- LLM Settings ---
    llm_provider: str = "lm-studio"
    llm_model_name: str = "google/gemma-4-31b-qat"
    gemini_api_key: Optional[str] = None

    # --- Embedding Settings ---
    embedding_provider: str = "sentence_transformers"
    embedding_model_name: str = "text-embedding-nomic-embed-code"
    embedding_dim: int = 768

    # --- LM Studio Shared Settings ---
    lm_studio_base_url: Optional[str] = "http://localhost:1234/v1"
    lm_studio_api_key: Optional[str] = "lm-studio"

    # --- Batch & Google Cloud Settings ---
    use_batch_api: bool = False
    batch_model_name: str = "gemini-1.5-flash"
    google_cloud_project: Optional[str] = None
    google_cloud_location: str = "us-central1"
    gcs_bucket_name: Optional[str] = None

    # --- Resume Settings ---
    resume_batch_id: Optional[str] = None

    # Allows Pydantic to read variables from a .env file (Pydantic V2 format)
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    @property
    def db_dsn(self) -> str:
        """Constructs the database connection string from individual components."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
