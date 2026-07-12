from pydantic import BaseModel, Field
from typing import List, Optional

class AnalysisDetail(BaseModel):
    """
    Defines the detailed analysis structure returned by the AI model for a single chunk.
    This serves as the standard contract for what the LLM is expected to produce.
    """
    is_ad: bool = Field(description="True if the text is an advertisement.")
    ad_reason: Optional[str] = Field(default=None, description="Reason for marking as ad. Omit if not an ad.")
    is_noise: bool = Field(description="True if the text is meaningless noise or completely irrelevant.")
    noise_reason: Optional[str] = Field(default=None, description="Reason for marking as noise. Omit if not noise.")
    ai_summary: str = Field(description="An ultra-concise 5-10 word summary of the valid content, or empty string if noise/ad.")
    content_category: str = Field(description="The category of the text (e.g., 'casual_chat', 'professional_email', 'article').")
    style_tags: List[str] = Field(description="Exactly 3-5 relevant topic/style keywords for vector search.")
    information_weight: float = Field(default=0.5, description="Information density and value (0.0 to 1.0).")
    original_time: Optional[str] = Field(default=None, description="The primary timestamp in ISO 8601 format, or null.")
    speaker: Optional[str] = Field(default=None, description="The primary speaker or sender, or null.")
    data_source: Optional[str] = Field(default=None, description="Inferred source of the data (e.g., 'messenger', 'email').")
    needs_human_review: bool = Field(default=False, description="True if AI is uncertain about the analysis.")
    human_review_reason: Optional[str] = Field(default=None, description="Brief explanation of why human review is needed. Omit if not needed.")

class ChunkAnalysisDetail(BaseModel):
    """Wrapper for associating an analysis with its original chunk ID in a batch request."""
    chunk_id: str = Field(description="The unique identifier of the chunk from the input array.")
    analysis: AnalysisDetail = Field(description="The analysis result for this specific chunk.")

class MiniBatchAnalysisResult(BaseModel):
    """Represents the full output of a batch analysis request."""
    results: List[ChunkAnalysisDetail] = Field(description="List of analysis results matching the input chunks.")
