from __future__ import annotations

from models.schemas import JobDescription, MatchResult, SearchResult


def build_reasoning(candidate: SearchResult, job: JobDescription, final_score: float) -> MatchResult:
    required = set(job.required_skills)
    candidate_skills = set(candidate.metadata.skills)

    matched_skills = sorted(required & candidate_skills)
    missing_skills = sorted(required - candidate_skills)
    excerpts = [candidate.chunk_text[:240]] if candidate.chunk_text else []
    relevant_sections = [candidate.section_type] if candidate.section_type else ["general"]

    return MatchResult(
        source_id=candidate.source_id,
        candidate_name=candidate.metadata.name or candidate.source_id,
        final_score=final_score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        relevant_sections=relevant_sections,
        relevant_excerpts=[item for item in excerpts if item],
    )


