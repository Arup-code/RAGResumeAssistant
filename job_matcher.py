from __future__ import annotations

import json
from pathlib import Path

from infra.embedding_service import EmbeddingService
from infra.vector_store import ChromaVectorStore
from matcher.jd_extractor import JDExtractor
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


def _match_with_job_description(
    job: JobDescription,
    top_k: int,
    persist_directory: str,
    collection_name: str,
) -> MatchResponse:
    embedding_service = EmbeddingService()
    vector_store = ChromaVectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    retrieved = retrieve_candidates(job, embedding_service, vector_store, top_k=top_k)
    filtered = filter_candidates(retrieved, job)

    warnings: str | None = None
    candidates = filtered
    if retrieved and not filtered:
        # Metadata can be sparse when extraction is unavailable; keep semantic ranking as fallback.
        candidates = retrieved
        warnings = (
            "No candidates passed strict skill/experience filtering; "
            "falling back to semantic-only ranking."
        )

    scorer = ScoringService()
    matched = [build_reasoning(item, job, scorer.score(item, job)) for item in candidates]
    ranked = rank_and_deduplicate(matched, top_k=top_k)

    return MatchResponse(
        success=True,
        data=MatchData(job_id=job.job_id, top_k=top_k, results=ranked),
        warnings=warnings,
    )


def match_job(
    job_file_path: str,
    top_k: int = 10,
    persist_directory: str = "storage/chroma",
    collection_name: str = "global_resume_collection",
) -> MatchResponse:
    try:
        job = _load_job_description(job_file_path)
        return _match_with_job_description(
            job=job,
            top_k=top_k,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    except AppException as exc:
        return MatchResponse(success=False, error=f"{exc.code}: {exc.message}")
    except Exception as exc:  # pragma: no cover - defensive fallback
        return MatchResponse(success=False, error=f"unhandled_error: {exc}")


def match_job_from_text(
    jd_text: str,
    top_k: int = 10,
    persist_directory: str = "storage/chroma",
    collection_name: str = "global_resume_collection",
) -> MatchResponse:
    try:
        extractor = JDExtractor()
        job = extractor.extract(jd_text)
        return _match_with_job_description(
            job=job,
            top_k=top_k,
            persist_directory=persist_directory,
            collection_name=collection_name,
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


