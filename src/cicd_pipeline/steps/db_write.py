import psycopg2
import json
from psycopg2.extras import execute_batch
from typing import Dict, Any
from .base import PipelineStep

class DBWriteStep(PipelineStep):
    """
    A pipeline step to write a batch of processed records to the database.
    """
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upserts the processed records from the context into the database.

        :param context: Must contain 'processed_records'.
        :return: The original context, unmodified.
        """
        records = context.get('processed_records')
        if not records:
            self.logger.warning("No records to write. Skipping DBWriteStep.")
            return context

        # Extract config from the step itself, not from context
        db_dsn = self.config.db_dsn
        table_name = self.config.db_table_name
        
        if not db_dsn or not table_name:
            raise ValueError("Database configuration (db_dsn, db_table_name) is missing.")

        data_to_insert = []
        for r in records:
            # Defensive check for essential keys
            if 'id' not in r or 'content' not in r:
                self.logger.warning(f"Skipping record due to missing 'id' or 'content': {str(r)[:100]}")
                continue
            
            # Prepare a tuple with safe defaults for all fields
            data_to_insert.append((
                r.get('id'),
                r.get('content'),
                r.get('repository', self.config.git_url),
                r.get('branch', self.config.branch),
                r.get('commit_hash'),
                r.get('file_path'),
                r.get('language'),
                r.get('node_type', 'unknown'),
                r.get('node_name'),
                r.get('start_line'),
                r.get('end_line'),
                r.get('summary_zh', ''),
                r.get('summary_en', ''),
                json.dumps(r.get('tags', [])),  # Serialize to JSON string
                json.dumps(r.get('dependencies', [])), # Serialize to JSON string
                r.get('embedding')
            ))

        if not data_to_insert:
            self.logger.warning("All records were filtered out. Nothing to write to DB.")
            return context

        sql = f"""
            INSERT INTO {table_name}
                (id, content, repository, branch, commit_hash, file_path, language, node_type, node_name, start_line, end_line, summary_zh, summary_en, tags, dependencies, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (id) DO UPDATE SET
                content=EXCLUDED.content, branch=EXCLUDED.branch, commit_hash=EXCLUDED.commit_hash, start_line=EXCLUDED.start_line,
                end_line=EXCLUDED.end_line, summary_zh=EXCLUDED.summary_zh, summary_en=EXCLUDED.summary_en,
                tags=EXCLUDED.tags, dependencies=EXCLUDED.dependencies, embedding=EXCLUDED.embedding;
        """
        
        try:
            with psycopg2.connect(db_dsn) as conn:
                with conn.cursor() as cur:
                    self.logger.info(f"Upserting {len(data_to_insert)} records into '{table_name}'...")
                    execute_batch(cur, sql, data_to_insert, page_size=100)
            self.logger.info("Database write operation complete.")
        except psycopg2.Error as e:
            self.logger.error(f"Database write failed: {e}", exc_info=True)
            raise
            
        return context
