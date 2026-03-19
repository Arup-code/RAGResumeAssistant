from __future__ import annotations

from models.schemas import JobDescription, SearchResult


def filter_candidates(results: list[SearchResult], job: JobDescription) -> list[SearchResult]:
    required = set(skill.lower() for skill in job.required_skills)
    filtered: list[SearchResult] = []
    for item in results:
        if item.metadata.experience_years < job.min_experience_years:
            continue
        candidate_skills = set(skill.lower() for skill in item.metadata.skills)
        if required and not required.issubset(candidate_skills):
            continue
        filtered.append(item)
    return filtered

