import uuid
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncpg
import chromadb

from src.rag_pipeline import config
from src.rag_pipeline.db.schemas import DBChunk

def _validate_iso8601(ts_str: Optional[str]) -> Optional[str]:
    """Checks if a string is a valid ISO 8601 timestamp, returns it or None."""
    if not ts_str:
        return None
    try:
        # Attempt to parse the timestamp to validate its format
        datetime.fromisoformat(str(ts_str).replace('Z', '+00:00'))
        return str(ts_str)
    except (ValueError, TypeError):
        return None

async def get_existing_chunk_hashes(
    file_name: str,
    chunk_indices: List[int],
    db_connection: Any
) -> Dict[int, str]:
    """
    Retrieves the content hashes for existing chunks from the database
    to avoid reprocessing identical content.
    """
    existing_hashes = {}
    if config.VECTOR_DB_TYPE == "postgres":
        if not isinstance(db_connection, asyncpg.Connection):
            return {}
        query = f"""
            SELECT chunk_index, content_hash FROM {config.COLLECTION_NAME}
            WHERE file_name = $1 AND chunk_index = ANY($2::int[])
        """
        records = await db_connection.fetch(query, file_name, chunk_indices)
        for record in records:
            existing_hashes[record["chunk_index"]] = record["content_hash"]
            
    elif config.VECTOR_DB_TYPE == "chromadb":
        if not isinstance(db_connection, chromadb.api.models.Collection.Collection):
            return {}
        results = db_connection.get(
            where={"file_name": file_name},
            include=['metadatas']
        )
        if results and results['ids']:
            for metadata in results['metadatas']:
                chunk_idx = metadata.get('chunk_index')
                if chunk_idx in chunk_indices:
                    existing_hashes[chunk_idx] = metadata.get('content_hash')
                    
    return existing_hashes

async def insert_chunks_batch(
    chunks_batch: List[DBChunk],
    embeddings: List[List[float]],
    db_connection: Any
):
    """
    Inserts a batch of processed chunks and their embeddings into the database.
    """
    if not chunks_batch:
        return

    if config.VECTOR_DB_TYPE == "postgres":
        await _insert_chunks_to_postgres(chunks_batch, embeddings, db_connection)
    elif config.VECTOR_DB_TYPE == "chromadb":
        await _insert_chunks_to_chroma(chunks_batch, embeddings, db_connection)

async def _insert_chunks_to_postgres(
    chunks_batch: List[DBChunk],
    embeddings: List[List[float]],
    db_conn: asyncpg.Connection
):
    """Helper function for inserting chunks into PostgreSQL."""
    query = f"""
        INSERT INTO {config.COLLECTION_NAME} (
            file_name, group_title, chunk_index, raw_content, content_hash,
            ai_summary, content_category, style_tags, information_weight,
            needs_human_review, human_review_reason, embedding,
            original_time, speaker, data_source
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
            $13::text::timestamptz, $14, $15
        )
    """
    records_to_insert = [
        (
            chunk.file_name,
            chunk.group_title,
            chunk.chunk_index,
            chunk.raw_content,
            chunk.content_hash,
            chunk.ai_summary,
            chunk.content_category,
            chunk.style_tags,
            chunk.information_weight,
            chunk.needs_human_review,
            chunk.human_review_reason,
            embeddings[i],
            _validate_iso8601(chunk.original_time),
            chunk.speaker,
            chunk.data_source
        ) for i, chunk in enumerate(chunks_batch)
    ]
    await db_conn.executemany(query, records_to_insert)

async def _insert_chunks_to_chroma(
    chunks_batch: List[DBChunk],
    embeddings: List[List[float]],
    collection: chromadb.api.models.Collection.Collection
):
    """Helper function for inserting chunks into ChromaDB."""
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks_batch:
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.file_name}-{chunk.chunk_index}-{chunk.content_hash}"))
        ids.append(chunk_id)
        documents.append(chunk.raw_content)
        
        metadata = {
            "file_name": chunk.file_name,
            "group_title": chunk.group_title or "",
            "chunk_index": chunk.chunk_index,
            "content_hash": chunk.content_hash,
            "ai_summary": chunk.ai_summary or "",
            "content_category": chunk.content_category or "",
            "style_tags": ",".join(chunk.style_tags) if chunk.style_tags else "",
            "information_weight": chunk.information_weight or 0.0,
            "needs_human_review": chunk.needs_human_review or False,
            "human_review_reason": chunk.human_review_reason or "",
            "original_time": _validate_iso8601(chunk.original_time) or "",
            "speaker": chunk.speaker or "",
            "data_source": chunk.data_source or ""
        }
        metadatas.append(metadata)

    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )