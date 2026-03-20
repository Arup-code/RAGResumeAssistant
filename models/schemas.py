from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SkippedReason(StrEnum):
    PARSE_ERROR_PDF = "parse_error_pdf"
    PARSE_ERROR_DOCX = "parse_error_docx"
    PARSE_ERROR_TXT = "parse_error_txt"
    PARSE_ERROR_UNKNOWN = "parse_error_unknown"


class ResumeDocument(BaseModel):
    source_id: str
    file_path: str
    text: str = Field(min_length=1)


class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    text: str = Field(min_length=1)
    section_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Metadata(BaseModel):
    name: str = ""
    skills: list[str] = Field(default_factory=list)
    experience_years: float = 0.0
    education: str = ""

    @field_validator("skills")
    @classmethod
    def normalize_skills(cls, value: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for skill in value:
            normalized = skill.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(normalized)
        return deduped


class EmbeddingVector(BaseModel):
    item_id: str
    values: list[float] = Field(min_length=1)


class SearchResult(BaseModel):
    source_id: str
    chunk_id: str
    score: float
    chunk_text: str
    section_type: str = "general"
    metadata: Metadata


class JobDescription(BaseModel):
    job_id: str
    title: str
    text: str
    required_skills: list[str] = Field(default_factory=list)
    min_experience_years: float = 0.0

    @field_validator("required_skills")
    @classmethod
    def normalize_required_skills(cls, value: list[str]) -> list[str]:
        return [item.strip().lower() for item in value if item.strip()]


class MatchResult(BaseModel):
    source_id: str
    candidate_name: str = ""
    final_score: float = Field(ge=0.0, le=100.0)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    relevant_sections: list[str] = Field(default_factory=list)
    relevant_excerpts: list[str] = Field(default_factory=list)


class SkippedItem(BaseModel):
    source_id: str
    skipped_reason: SkippedReason


