from __future__ import annotations
from typing import Any, Generic, TypeVar
from pydantic import BaseModel, Field
from models.schemas import MatchResult, SkippedItem
T = TypeVar("T")
class BaseResponse(BaseModel, Generic[T]):
    success: bool
    data: T | None = None
    error: str | None = None
class IngestionData(BaseModel):
    processed_count: int = 0
    skipped_count: int = 0
    ingested_source_ids: list[str] = Field(default_factory=list)
    skipped_items: list[SkippedItem] = Field(default_factory=list)
class IngestionResponse(BaseResponse[IngestionData]):
    data: IngestionData | None = None
class MatchData(BaseModel):
    job_id: str
    top_k: int
    results: list[MatchResult] = Field(default_factory=list)
class MatchResponse(BaseResponse[MatchData]):
    data: MatchData | None = None
class GenericResponse(BaseResponse[Any]):
    data: Any | None = None
