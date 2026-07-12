def get_insert_query(table_name: str) -> str:
    """Returns the parameterized SQL query for inserting a chunk."""
    return f"""
        INSERT INTO {table_name} (
            file_name, group_title, chunk_index, raw_content, content_hash,
            ai_summary, content_category, style_tags, information_weight,
            needs_human_review, human_review_reason, embedding,
            original_time, speaker, data_source
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
            $13::text::timestamptz, $14, $15
        )
    """

def get_select_hashes_query(table_name: str) -> str:
    """Returns the parameterized SQL query for selecting existing chunk hashes."""
    return f"""
        SELECT chunk_index, content_hash FROM {table_name}
        WHERE file_name = $1 AND chunk_index = ANY($2::int[])
    """
