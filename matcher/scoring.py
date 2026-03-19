from __future__ import annotations

from models.schemas import JobDescription, SearchResult


class ScoringService:
    def __init__(
        self,
        skills_weight: float = 0.40,
        experience_weight: float = 0.30,
        semantic_weight: float = 0.30,
    ) -> None:
        self.skills_weight = skills_weight
        self.experience_weight = experience_weight
        self.semantic_weight = semantic_weight

    def score(self, candidate: SearchResult, job: JobDescription) -> float:
        required_skills = set(job.required_skills)
        candidate_skills = set(candidate.metadata.skills)

        if not required_skills:
            skills_ratio = 1.0
        else:
            skills_ratio = len(required_skills & candidate_skills) / len(required_skills)

        if job.min_experience_years <= 0:
            exp_ratio = 1.0
        else:
            exp_ratio = min(candidate.metadata.experience_years / job.min_experience_years, 1.0)

        semantic = max(0.0, min(candidate.score, 1.0))

        final = (
            skills_ratio * self.skills_weight
            + exp_ratio * self.experience_weight
            + semantic * self.semantic_weight
        ) * 100.0
        return round(final, 2)

