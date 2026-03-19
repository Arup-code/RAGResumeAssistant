from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_+#.-]+")


@dataclass(slots=True)
class ResumeRecord:
    source_id: str
    text: str
    skills: list[str]
    experience_years: float


@dataclass(slots=True)
class JDRecord:
    job_id: str
    title: str
    text: str
    required_skills: list[str]
    min_experience_years: float


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text or "")}


def parse_resume_text(source_id: str, text: str) -> ResumeRecord:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    skills: list[str] = []
    experience_years = 0.0

    for idx, line in enumerate(lines):
        lower = line.lower()
        if lower == "skills" and idx + 1 < len(lines):
            raw_skills = lines[idx + 1]
            skills = [item.strip().lower() for item in raw_skills.split(",") if item.strip()]

        match = re.search(r"(\d+(?:\.\d+)?)\s+years?", lower)
        if match:
            experience_years = max(experience_years, float(match.group(1)))

    if not skills:
        # Fallback skill extraction from whole text when explicit section is missing.
        skills = sorted(tokenize(text))[:15]

    return ResumeRecord(
        source_id=source_id,
        text=text,
        skills=skills,
        experience_years=experience_years,
    )


def load_resumes(resume_dir: str = "dataset/resumes") -> list[ResumeRecord]:
    records: list[ResumeRecord] = []
    for path in sorted(Path(resume_dir).glob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".pdf", ".docx"}:
            continue
        if path.suffix.lower() != ".txt":
            # Offline notebook baseline focuses on text files.
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        records.append(parse_resume_text(path.stem, text))
    return records


def load_jds(jd_dir: str = "dataset/jds") -> list[JDRecord]:
    jobs: list[JDRecord] = []
    for path in sorted(Path(jd_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        jobs.append(
            JDRecord(
                job_id=str(payload["job_id"]),
                title=str(payload.get("title", "")),
                text=str(payload.get("text", "")),
                required_skills=[str(skill).strip().lower() for skill in payload.get("required_skills", []) if str(skill).strip()],
                min_experience_years=float(payload.get("min_experience_years", 0.0)),
            )
        )
    return jobs


def relevance_label(resume: ResumeRecord, job: JDRecord, mode: str = "all_required") -> bool:
    required = set(job.required_skills)
    resume_skills = set(resume.skills)

    if mode == "any_required":
        skill_ok = not required or bool(required & resume_skills)
    elif mode == "half_required":
        needed = max(1, len(required) // 2) if required else 0
        skill_ok = not required or len(required & resume_skills) >= needed
    else:
        skill_ok = required.issubset(resume_skills)

    exp_ok = resume.experience_years >= job.min_experience_years
    return skill_ok and exp_ok


def score_resume_for_jd(resume: ResumeRecord, job: JDRecord) -> float:
    resume_skills = set(resume.skills)
    required = set(job.required_skills)

    if required:
        skill_overlap = len(required & resume_skills) / len(required)
    else:
        skill_overlap = 1.0

    jd_tokens = tokenize(job.text)
    resume_tokens = tokenize(resume.text)
    lexical_overlap = len(jd_tokens & resume_tokens) / max(1, len(jd_tokens))

    exp_component = 1.0 if job.min_experience_years <= 0 else min(resume.experience_years / job.min_experience_years, 1.0)

    return 0.6 * skill_overlap + 0.25 * lexical_overlap + 0.15 * exp_component


def rank_resumes(job: JDRecord, resumes: list[ResumeRecord]) -> list[tuple[str, float]]:
    scored = [(resume.source_id, score_resume_for_jd(resume, job)) for resume in resumes]
    return sorted(scored, key=lambda item: item[1], reverse=True)


def evaluate_retrieval(
    resumes: list[ResumeRecord],
    jobs: list[JDRecord],
    k_values: list[int],
    relevance_mode: str = "all_required",
    latency_repeats: int = 5,
) -> dict[str, object]:
    if not resumes or not jobs:
        return {
            "metrics": [],
            "latency_rows": [],
            "summary": {
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
            },
        }

    k_values = sorted(set(k_values))
    metrics_by_k = {k: {"hits": 0, "total_jds": 0, "recall_sum": 0.0} for k in k_values}
    latency_rows: list[dict[str, float | str]] = []
    all_latencies_ms: list[float] = []

    resume_map = {resume.source_id: resume for resume in resumes}

    for job in jobs:
        relevant_ids = {
            resume.source_id
            for resume in resumes
            if relevance_label(resume, job, mode=relevance_mode)
        }

        elapsed_ms_samples: list[float] = []
        ranked_ids: list[str] = []
        for _ in range(max(1, latency_repeats)):
            start = perf_counter()
            ranked = rank_resumes(job, resumes)
            elapsed_ms = (perf_counter() - start) * 1000.0
            elapsed_ms_samples.append(elapsed_ms)
            ranked_ids = [item[0] for item in ranked]

        mean_latency_ms = sum(elapsed_ms_samples) / len(elapsed_ms_samples)
        latency_rows.append({"job_id": job.job_id, "latency_ms": mean_latency_ms})
        all_latencies_ms.append(mean_latency_ms)

        for k in k_values:
            top_k_ids = set(ranked_ids[:k])
            relevant_in_top_k = relevant_ids & top_k_ids

            metrics = metrics_by_k[k]
            metrics["total_jds"] += 1
            if relevant_in_top_k:
                metrics["hits"] += 1

            if relevant_ids:
                metrics["recall_sum"] += len(relevant_in_top_k) / len(relevant_ids)

    metric_rows: list[dict[str, float | int]] = []
    for k in k_values:
        values = metrics_by_k[k]
        total = max(1, values["total_jds"])
        metric_rows.append(
            {
                "k": k,
                "hit_rate_at_k": values["hits"] / total,
                "recall_at_k": values["recall_sum"] / total,
                "total_jds": values["total_jds"],
            }
        )

    sorted_latency = sorted(all_latencies_ms)
    p50_idx = int(0.50 * (len(sorted_latency) - 1))
    p95_idx = int(0.95 * (len(sorted_latency) - 1))

    summary = {
        "avg_latency_ms": sum(sorted_latency) / len(sorted_latency),
        "p50_latency_ms": sorted_latency[p50_idx],
        "p95_latency_ms": sorted_latency[p95_idx],
        "jobs_evaluated": len(jobs),
        "resumes_evaluated": len(resumes),
    }

    return {
        "metrics": metric_rows,
        "latency_rows": latency_rows,
        "summary": summary,
        "resume_map": resume_map,
    }

