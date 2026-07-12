import psycopg2
from typing import Dict, Any
from .base import PipelineStep

class DBSetupStep(PipelineStep):
    """
    A pipeline step to connect to the database and ensure the schema is ready.
    """
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Connects to the database, creates the 'vector' extension, and
        creates the necessary table for storing code chunks.

        :param context: The pipeline context.
        :return: Updated context with 'db_dsn'.
        """
        db_dsn = self.config.db_dsn
        
        # Prioritize 'db_table_name' from environment variables if specified; otherwise, default to the repository name
        # Perform a simple character replacement to prevent SQL syntax errors if the repository name contains '-'
        table_name = self.config.db_table_name or self.config.project_name.replace('-', '_')
        embedding_dim = self.config.embedding_dim

        try:
            self.logger.info("Connecting to the database to set up schema...")
            with psycopg2.connect(db_dsn) as conn:
                with conn.cursor() as cur:
                    self.logger.info("Ensuring 'vector' extension is enabled.")
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    
                    self.logger.info(f"Checking if table '{table_name}' exists...")
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = %s
                        );
                    """, (table_name,))
                    table_exists = cur.fetchone()[0]

                    if table_exists:
                        self.logger.info(f"Table '{table_name}' already exists. Skipping creation process.")
                    else:
                        self.logger.info(f"Table '{table_name}' not found. Creating it now...")
                        cur.execute(f"""
                            CREATE TABLE {table_name} (
                                id VARCHAR(32) PRIMARY KEY,
                                content TEXT,
                                repository TEXT,
                                branch TEXT,
                                commit_hash TEXT,
                                file_path TEXT,
                                language TEXT,
                                node_type TEXT,
                                node_name TEXT,
                                start_line INTEGER,
                                end_line INTEGER,
                                summary_zh TEXT,
                                summary_en TEXT,
                                tags JSONB,
                                dependencies JSONB,
                                embedding vector({embedding_dim})
                            );
                        """)
                    
                    self.logger.info(f"Ensuring 'branch' column exists in '{table_name}' for migrations.")
                    cur.execute(f"""
                        ALTER TABLE {table_name} 
                        ADD COLUMN IF NOT EXISTS branch TEXT;
                    """)
            
            self.logger.info("Database setup complete.")
            context['db_dsn'] = db_dsn
            context['db_table_name'] = table_name
            return context

        except psycopg2.Error as e:
            self.logger.error(f"Database setup failed: {e}")
            raise
