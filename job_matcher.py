from __future__ import annotations

import json
from pathlib import Path

from infra.embedding_service import EmbeddingService
from infra.vector_store import ChromaVectorStore
from matcher.filtering import filter_candidates
from matcher.ranking import rank_and_deduplicate
from matcher.reasoning import build_reasoning
from matcher.retrieval import retrieve_candidates
from matcher.scoring import ScoringService
from models.responses import MatchData, MatchResponse
from models.schemas import JobDescription
from utils.exceptions import AppException


def _load_job_description(file_path: str) -> JobDescription:
    path = Path(file_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return JobDescription.model_validate(payload)


def match_job(
    job_file_path: str,
    top_k: int = 10,
    persist_directory: str = "storage/chroma",
    collection_name: str = "global_resume_collection",
) -> MatchResponse:
    try:
        job = _load_job_description(job_file_path)
        embedding_service = EmbeddingService()
        vector_store = ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

        retrieved = retrieve_candidates(job, embedding_service, vector_store, top_k=top_k)
        filtered = filter_candidates(retrieved, job)

        scorer = ScoringService()
        matched = [build_reasoning(item, job, scorer.score(item, job)) for item in filtered]
        ranked = rank_and_deduplicate(matched, top_k=top_k)

        return MatchResponse(
            success=True,
            data=MatchData(job_id=job.job_id, top_k=top_k, results=ranked),
        )
    except AppException as exc:
        return MatchResponse(success=False, error=f"{exc.code}: {exc.message}")
    except Exception as exc:  # pragma: no cover - defensive fallback
        return MatchResponse(success=False, error=f"unhandled_error: {exc}")


if __name__ == "__main__":
    jd_files = sorted(Path("dataset/jds").glob("*.json"))
    if not jd_files:
        raise SystemExit("No JD JSON files found in dataset/jds")

    response = match_job(str(jd_files[0]))
    print(response.model_dump_json(indent=2))


