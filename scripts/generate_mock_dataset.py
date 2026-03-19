from __future__ import annotations

import json
from pathlib import Path

RESUME_TEMPLATE = """{name}\n\nExperience\n{years} years building backend services with Python and APIs.\n\nSkills\n{skills}\n\nEducation\nB.Tech in Computer Science\n"""

JD_TEMPLATE = {
    "job_id": "",
    "title": "",
    "text": "",
    "required_skills": [],
    "min_experience_years": 0,
}


def generate_dataset(base_dir: str = "dataset", resumes: int = 30, jds: int = 5) -> None:
    base = Path(base_dir)
    resumes_dir = base / "resumes"
    jds_dir = base / "jds"
    resumes_dir.mkdir(parents=True, exist_ok=True)
    jds_dir.mkdir(parents=True, exist_ok=True)

    all_skills = ["python", "fastapi", "sql", "docker", "aws", "ml", "pandas"]

    for idx in range(1, resumes + 1):
        years = 1 + (idx % 8)
        skills = ", ".join(all_skills[idx % len(all_skills):] + all_skills[: idx % len(all_skills)])
        content = RESUME_TEMPLATE.format(name=f"Candidate {idx}", years=years, skills=skills)
        (resumes_dir / f"resume_{idx:03d}.txt").write_text(content, encoding="utf-8")

    for idx in range(1, jds + 1):
        required = [all_skills[idx % len(all_skills)], all_skills[(idx + 1) % len(all_skills)]]
        jd = dict(JD_TEMPLATE)
        jd["job_id"] = f"jd_{idx:03d}"
        jd["title"] = f"Role {idx}"
        jd["text"] = f"Need engineer with {', '.join(required)} experience"
        jd["required_skills"] = required
        jd["min_experience_years"] = 2 + (idx % 4)
        (jds_dir / f"jd_{idx:03d}.json").write_text(json.dumps(jd, indent=2), encoding="utf-8")


if __name__ == "__main__":
    generate_dataset()
    print("Generated dataset/resumes and dataset/jds")

