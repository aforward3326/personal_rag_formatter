import uuid
from typing import List, Dict, Any
from src.rag_pipeline.utils.logger import logger

def _standardize_metadata(raw_metadata: Dict[str, Any], default_source: str = "unknown") -> Dict[str, Any]:
    """
    Normalizes metadata keys and ensures a 'data_source' field exists.
    """
    standardized = {}
    # Rename 'timestamp' to 'original_time' for consistency
    if "timestamp" in raw_metadata:
        standardized["original_time"] = raw_metadata.pop("timestamp")
        
    # Ensure 'data_source' is present
    if "source" in raw_metadata:
        standardized["data_source"] = raw_metadata.pop("source")
    elif "data_source" not in standardized:
        standardized["data_source"] = default_source
        
    # Bring over any remaining metadata fields
    standardized.update(raw_metadata)
    return standardized

def parse_rag_ready_json(json_data: Any, file_name: str) -> List[Dict[str, Any]]:
    """
    Parses various known JSON structures from input files and extracts a
    standardized list of documents.

    Each document is a dictionary with 'page_content' and 'metadata'.

    Args:
        json_data: The loaded JSON data from a file.
        file_name: The name of the file being processed, for logging.

    Returns:
        A list of document dictionaries.
    """
    documents = []

    # Structure: {"corpus": [{"page_content": ..., "metadata": ...}]}
    if isinstance(json_data, dict) and "corpus" in json_data and isinstance(json_data["corpus"], list):
        for item in json_data["corpus"]:
            if isinstance(item, dict) and "page_content" in item:
                documents.append({
                    "page_content": item["page_content"],
                    "metadata": _standardize_metadata(item.get("metadata", {}), default_source="corpus_file")
                })
        return documents

    # Structure: {"messages": [{"conversation": [...]}]} for chat logs
    if isinstance(json_data, dict) and "messages" in json_data and isinstance(json_data["messages"], list):
        for thread in json_data["messages"]:
            if not isinstance(thread, dict): continue
            thread_id = thread.get("thread_id", str(uuid.uuid4()))
            thread_metadata = _standardize_metadata(thread.get("metadata", {}), default_source="chat_html")
            
            if "conversation" in thread and isinstance(thread["conversation"], list):
                for message in thread["conversation"]:
                    if not isinstance(message, dict): continue
                    msg_page_content = message.get("page_content")
                    if not msg_page_content: continue
                    
                    msg_metadata = _standardize_metadata(message.get("metadata", {}), default_source=thread_metadata.get("data_source", "chat_html"))
                    final_metadata = {**thread_metadata, **msg_metadata, "thread_id": thread_id, "message_id": message.get("message_id", str(uuid.uuid4()))}
                    documents.append({"page_content": msg_page_content, "metadata": final_metadata})
        return documents

    # Structure: {"email_threads": [{"conversation": [...]}]}
    elif isinstance(json_data, dict) and "email_threads" in json_data and isinstance(json_data["email_threads"], list):
        for thread in json_data["email_threads"]:
            if not isinstance(thread, dict): continue
            thread_id = thread.get("thread_id", str(uuid.uuid4()))
            subject = thread.get("subject", "No Subject")
            thread_metadata = _standardize_metadata(thread.get("metadata", {}), default_source="gmail")

            if "conversation" in thread and isinstance(thread["conversation"], list):
                for message in thread["conversation"]:
                    if not isinstance(message, dict): continue
                    msg_page_content = message.get("page_content")
                    if not msg_page_content: continue

                    msg_metadata = _standardize_metadata(message.get("metadata", {}), default_source=thread_metadata.get("data_source", "gmail"))
                    final_metadata = {**thread_metadata, **msg_metadata, "thread_id": thread_id, "subject": subject, "message_id": message.get("message_id", str(uuid.uuid4()))}
                    documents.append({"page_content": msg_page_content, "metadata": final_metadata})
        return documents

    # Structure: {"posts": [{"page_content": ..., "comments": [...]}]}
    elif isinstance(json_data, dict) and "posts" in json_data and isinstance(json_data["posts"], list):
        for post in json_data["posts"]:
            if not isinstance(post, dict): continue
            post_id = post.get("post_id", str(uuid.uuid4()))
            post_page_content = post.get("page_content")
            post_metadata = _standardize_metadata(post.get("metadata", {}), default_source="meta_post")

            if post_page_content:
                final_metadata = {**post_metadata, "post_id": post_id}
                documents.append({"page_content": post_page_content, "metadata": final_metadata})
            
            if "comments" in post and isinstance(post["comments"], list):
                for comment in post["comments"]:
                    if not isinstance(comment, dict): continue
                    comment_page_content = comment.get("page_content")
                    if not comment_page_content: continue
                    
                    comment_metadata = _standardize_metadata(comment.get("metadata", {}), default_source=post_metadata.get("data_source", "meta_comment"))
                    final_metadata = {**post_metadata, **comment_metadata, "post_id": post_id, "comment_id": comment.get("comment_id", str(uuid.uuid4()))}
                    documents.append({"page_content": comment_page_content, "metadata": final_metadata})
        return documents
    
    # Fallback for simple list of documents or single document object
    elif isinstance(json_data, list):
        for item in json_data:
            if isinstance(item, dict) and "page_content" in item:
                documents.append({"page_content": item["page_content"], "metadata": _standardize_metadata(item.get("metadata", {}), default_source="unknown_list")})
        return documents
    elif isinstance(json_data, dict) and "page_content" in json_data:
        documents.append({"page_content": json_data["page_content"], "metadata": _standardize_metadata(json_data.get("metadata", {}), default_source="unknown_single")})
        return documents
        
    else:
        logger.warning(f"File {file_name} has an unrecognized JSON structure. No documents extracted.")

    return documents