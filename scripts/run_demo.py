from __future__ import annotations

from pathlib import Path

from job_matcher import match_job
from resume_rag import ingest_resumes


if __name__ == "__main__":
    ingest_response = ingest_resumes()
    print("Ingestion response:")
    print(ingest_response.model_dump_json(indent=2))

    jd_files = sorted(Path("dataset/jds").glob("*.json"))
    if not jd_files:
        raise SystemExit("No JDs found in dataset/jds")

    match_response = match_job(str(jd_files[0]))
    print("\nMatch response:")
    print(match_response.model_dump_json(indent=2))

