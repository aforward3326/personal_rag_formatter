from typing import List, Dict

try:
    import tiktoken
    # Cache encoding globally to avoid overhead of loading it for every chunk
    TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    tiktoken = None
    TIKTOKEN_ENCODING = None

def create_sliding_window_chunks(
    text: str, 
    group_id: str, 
    max_tokens: int = 2500, 
    overlap_tokens: int = 150
) -> List[Dict]:
    """
    Splits a long text into smaller chunks using a sliding window approach,
    based on token counts if tiktoken is available, otherwise falling back to
    character counts.

    Args:
        text: The full text to be chunked.
        group_id: An identifier for the group this text belongs to.
        max_tokens: The target maximum size of each chunk in tokens.
        overlap_tokens: The number of tokens from the previous chunk to include
                      as overlap in the current chunk.

    Returns:
        A list of dictionaries, where each dictionary represents a chunk
        containing the main content, overlap context, and token count.
    """
    chunks = []
    
    if not tiktoken or not TIKTOKEN_ENCODING:
        # Fallback to character-based chunking if tiktoken is not installed
        max_chars = max_tokens * 4  # Rough estimation
        overlap_chars = overlap_tokens * 4
        
        start_idx = 0
        while start_idx < len(text):
            end_idx = min(start_idx + max_chars, len(text))
            main_content = text[start_idx:end_idx]
            
            overlap_content = ""
            if start_idx > 0:
                overlap_start = max(0, start_idx - overlap_chars)
                overlap_content = text[overlap_start:start_idx]
                
            chunks.append({
                "group_id": group_id,
                "main_content": main_content,
                "overlap_context": overlap_content,
                "tokens_count": len(main_content) // 4  # Rough estimate
            })
            start_idx += max_chars
        return chunks

    # Token-based chunking using tiktoken
    tokens = TIKTOKEN_ENCODING.encode(text)
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        main_tokens = tokens[start_idx:end_idx]
        main_content = TIKTOKEN_ENCODING.decode(main_tokens)
        
        overlap_content = ""
        if start_idx > 0:
            overlap_start = max(0, start_idx - overlap_tokens)
            overlap_tokens_list = tokens[overlap_start:start_idx]
            overlap_content = TIKTOKEN_ENCODING.decode(overlap_tokens_list)
            
        chunks.append({
            "group_id": group_id,
            "main_content": main_content,
            "overlap_context": overlap_content,
            "tokens_count": len(main_tokens)
        })
        start_idx += max_tokens
        
    return chunks
