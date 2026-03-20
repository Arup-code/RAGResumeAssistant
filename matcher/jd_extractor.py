from __future__ import annotations

import json
import os
import re
from uuid import uuid4

import httpx
from pydantic import ValidationError

from models.schemas import JobDescription
from utils.exceptions import ValidationException


class JDExtractor:
    """Extract structured JD fields from raw text using OpenRouter JSON mode."""

    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-4o-mini") -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValidationException("OPENROUTER_API_KEY is required for JD extraction")

    def extract(self, jd_text: str) -> JobDescription:
        cleaned_text = jd_text.strip()
        if not cleaned_text:
            raise ValidationException("job description text cannot be empty")

        schema = {
            "name": "JobDescription",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "title": {"type": "string"},
                    "text": {"type": "string"},
                    "required_skills": {"type": "array", "items": {"type": "string"}},
                    "min_experience_years": {"type": "number"},
                },
                "required": ["title", "text", "required_skills", "min_experience_years"],
                "additionalProperties": False,
            },
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Extract job details into strict JSON. "
                        "If a field is not present in the JD, use: required_skills=[], min_experience_years=0. "
                        "Keep text as concise normalized JD text."
                    ),
                },
                {"role": "user", "content": cleaned_text[:12000]},
            ],
            "response_format": {"type": "json_schema", "json_schema": schema},
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

        raw_content = response.json()["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise ValidationException("JD extraction returned invalid JSON") from exc

        parsed.setdefault("job_id", f"jd_{uuid4().hex[:8]}")
        parsed.setdefault("title", self._fallback_title(cleaned_text))
        parsed.setdefault("text", cleaned_text)
        parsed.setdefault("required_skills", [])
        parsed.setdefault("min_experience_years", 0)

        if not parsed.get("required_skills"):
            parsed["required_skills"] = self._infer_required_skills(cleaned_text)
        if float(parsed.get("min_experience_years", 0) or 0) <= 0:
            parsed["min_experience_years"] = self._infer_min_experience_years(cleaned_text)

        try:
            return JobDescription.model_validate(parsed)
        except ValidationError as exc:
            raise ValidationException("JD extraction schema validation failed") from exc

    def _heuristic_extract(self, jd_text: str) -> JobDescription:
        return JobDescription(
            job_id=f"jd_{uuid4().hex[:8]}",
            title=self._fallback_title(jd_text),
            text=jd_text,
            required_skills=self._infer_required_skills(jd_text),
            min_experience_years=self._infer_min_experience_years(jd_text),
        )

    @staticmethod
    def _fallback_title(jd_text: str) -> str:
        first_line = next((line.strip() for line in jd_text.splitlines() if line.strip()), "Untitled role")
        return first_line[:80]

    @staticmethod
    def _infer_required_skills(jd_text: str) -> list[str]:
        text = jd_text.lower()
        known_skills = [
            "python",
            "java",
            "sql",
            "fastapi",
            "django",
            "flask",
            "docker",
            "kubernetes",
            "aws",
            "azure",
            "gcp",
            "react",
            "node",
            "pandas",
            "spark",
            "machine learning",
            "ml",
        ]
        found = [skill for skill in known_skills if skill in text]
        deduped: list[str] = []
        seen: set[str] = set()
        for skill in found:
            normalized = skill.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                deduped.append(normalized)
        return deduped

    @staticmethod
    def _infer_min_experience_years(jd_text: str) -> float:
        text = jd_text.lower()
        patterns = [
            r"(\d+(?:\.\d+)?)\s*\+?\s*years",
            r"minimum\s*(\d+(?:\.\d+)?)\s*years",
            r"at least\s*(\d+(?:\.\d+)?)\s*years",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return 0.0

