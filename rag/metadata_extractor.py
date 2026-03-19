from __future__ import annotations

import json
import os

import httpx
from pydantic import ValidationError

from models.schemas import Metadata
from utils.exceptions import ValidationException


class MetadataExtractor:
    """Extract structured resume metadata via OpenRouter with strict JSON schema."""

    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-4o-mini") -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValidationException("OPENROUTER_API_KEY is required for metadata extraction")

    def extract(self, resume_text: str) -> Metadata:
        schema = {
            "name": "ResumeMetadata",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "skills": {"type": "array", "items": {"type": "string"}},
                    "experience_years": {"type": "number"},
                    "education": {"type": "string"},
                },
                "required": ["name", "skills", "experience_years", "education"],
                "additionalProperties": False,
            },
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Extract resume metadata and return strict JSON.",
                },
                {
                    "role": "user",
                    "content": resume_text[:12000],
                },
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
            return Metadata.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValidationException("OpenRouter metadata extraction violated strict schema") from exc

