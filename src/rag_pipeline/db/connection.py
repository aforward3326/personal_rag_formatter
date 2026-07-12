import asyncpg
import chromadb
from pgvector.asyncpg import register_vector
from typing import Optional, Union
from src.rag_pipeline import config
from src.rag_pipeline.utils.logger import logger

class DatabaseManager:
    """
    Manages connections to vector databases (Postgres with pgvector or ChromaDB).
    """
    def __init__(self, db_type: str):
        self.db_type = db_type
        self.pg_conn: Optional[asyncpg.Connection] = None
        self.chroma_client: Optional[chromadb.PersistentClient] = None
        self.chroma_collection: Optional[chromadb.api.models.Collection.Collection] = None

    async def connect(self):
        """Establishes a connection to the configured database."""
        if self.db_type == "postgres":
            if not self.pg_conn or self.pg_conn.is_closed():
                try:
                    self.pg_conn = await asyncpg.connect(
                        host=config.PG_HOST,
                        port=config.PG_PORT,
                        user=config.PG_USER,
                        password=config.PG_PASSWORD,
                        database=config.PG_DB_NAME
                    )
                    await register_vector(self.pg_conn)
                    logger.info("Successfully connected to PostgreSQL.")
                except Exception as e:
                    logger.critical(f"Failed to connect to PostgreSQL: {e}")
                    raise
        elif self.db_type == "chromadb":
            try:
                self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_DIR)
                self.chroma_collection = self.chroma_client.get_or_create_collection(
                    name=config.COLLECTION_NAME
                )
                logger.info(f"Successfully connected to ChromaDB and got collection '{config.COLLECTION_NAME}'.")
            except Exception as e:
                logger.critical(f"Failed to connect to ChromaDB: {e}")
                raise
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    async def close(self):
        """Closes the active database connection."""
        if self.db_type == "postgres" and self.pg_conn and not self.pg_conn.is_closed():
            await self.pg_conn.close()
            logger.info("PostgreSQL connection closed.")
        # ChromaDB client does not have an explicit close method.
        
    def get_connection(self) -> Union[asyncpg.Connection, chromadb.api.models.Collection.Collection, None]:
        """
        Returns the active database connection/collection object.
        """
        if self.db_type == "postgres":
            return self.pg_conn
        elif self.db_type == "chromadb":
            return self.chroma_collection
        return None

# Global instance
db_manager = DatabaseManager(config.VECTOR_DB_TYPE)
