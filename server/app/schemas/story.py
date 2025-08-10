"""
Pydantic schemas for story generation API
Shared between local app and Modal endpoint
"""

from pydantic import BaseModel
from typing import Optional


class StoryRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7


class StoryResponse(BaseModel):
    success: bool
    generated_text: str
    tokens_generated: Optional[int] = None
    error: Optional[str] = None
