from __future__ import annotations

from models.schemas import MatchResult


def rank_and_deduplicate(results: list[MatchResult], top_k: int = 10) -> list[MatchResult]:
    best_by_source: dict[str, MatchResult] = {}
    for result in results:
        existing = best_by_source.get(result.source_id)
        if not existing or result.final_score > existing.final_score:
            best_by_source[result.source_id] = result

    ranked = sorted(best_by_source.values(), key=lambda item: item.final_score, reverse=True)
    return ranked[:top_k]

