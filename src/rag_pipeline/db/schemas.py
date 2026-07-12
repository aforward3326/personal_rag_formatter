from typing import List, Optional
from ..llm.schemas import AnalysisDetail

class DBChunk(AnalysisDetail):
    """
    Represents the final, enriched data structure for a single chunk,
    ready for insertion into the vector database.

    It inherits all the analysis fields from the AI model's output (`AnalysisDetail`)
    and adds database-specific fields.
    """
    # --- Fields specific to the database record ---
    file_name: str
    group_title: Optional[str]
    chunk_index: int
    raw_content: str
    content_hash: str
    embedding: Optional[List[float]] = None

    class Config:
        # This allows the model to be created from arbitrary class instances,
        # which is useful when constructing it from other objects.
        from_attributes = True