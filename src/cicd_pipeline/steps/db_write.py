import psycopg2
from psycopg2.extras import execute_batch, Json
from typing import Dict, Any
from .base import PipelineStep

class DBWriteStep(PipelineStep):
    """
    A pipeline step to write a batch of processed records to the database.
    """
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upserts the processed records from the context into the database.

        :param context: Must contain 'processed_records', 'db_dsn', and 'db_table_name'.
        :return: The original context, unmodified.
        """
        records = context.get('processed_records')
        db_dsn = context.get('db_dsn')
        table_name = context.get('db_table_name')
        merged_branches = context.get('merged_branches', [])
        
        if not db_dsn or not table_name:
            raise ValueError("Context must contain 'db_dsn' and 'db_table_name'.")
            
        if not records and not merged_branches:
            self.logger.warning("No records to write and no branches to clean up. Skipping.")
            return context

        data_to_insert = []
        sql = ""
        if records:
            # Prepare records for psycopg2, converting lists to JSON
            data_to_insert = [
                (
                    r['id'], r['content'], r['repository'], r['branch'], r['commit_hash'],
                    r['file_path'], r['language'], r['node_type'], r['node_name'],
                    r['start_line'], r['end_line'], r['summary_zh'], r['summary_en'],
                    Json(r['tags']), Json(r['dependencies']), r['embedding']
                ) for r in records
            ]

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
                    if merged_branches:
                        self.logger.info(f"Cleaning up merged branches from DB: {merged_branches}")
                        format_strings = ','.join(['%s'] * len(merged_branches))
                        cur.execute(f"DELETE FROM {table_name} WHERE branch IN ({format_strings})", tuple(merged_branches))
                        self.logger.info("Deleted records for merged branches.")
                    
                    if records:
                        self.logger.info(f"Upserting {len(data_to_insert)} records into '{table_name}'...")
                        execute_batch(cur, sql, data_to_insert, page_size=100)
            self.logger.info("Database write and consolidation operation complete.")
        except psycopg2.Error as e:
            self.logger.error(f"Database write failed: {e}")
            raise
            
        return context
