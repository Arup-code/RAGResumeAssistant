from __future__ import annotations

import hashlib
import math
import os
import time
from dotenv import load_dotenv

import httpx

from utils.exceptions import EmbeddingException, ValidationException
load_dotenv()

class EmbeddingService:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        local_dimension: int = 1536,
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.local_dimension = local_dimension
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        if not self.api_key:
            raise ValidationException("OPENROUTER_API_KEY is required for embedding generation")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []


        payload = {"model": self.model, "input": texts}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(
                        "https://openrouter.ai/api/v1/embeddings",
                        headers=headers,
                        json=payload,
                    )
                    if response.status_code in {429, 500, 502, 503, 504}:
                        raise EmbeddingException(f"transient embedding API error: {response.status_code}")
                    response.raise_for_status()
                    data = response.json()["data"]
                    return [item["embedding"] for item in data]
            except Exception as exc:  # pragma: no cover - network failure path
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * attempt)

        raise EmbeddingException(f"embedding generation failed after retries: {last_error}")

    def _embed_texts_local(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text_local(text) for text in texts]

    def _embed_text_local(self, text: str) -> list[float]:
        # Hash-based bag-of-tokens embedding for offline/dev mode.
        vector = [0.0] * self.local_dimension
        for raw_token in text.lower().split():
            token = raw_token.strip(".,;:!?()[]{}\"'")
            if not token:
                continue
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.local_dimension
            sign = -1.0 if digest[8] & 1 else 1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def embed_text(self, text: str) -> list[float]:
        vectors = self.embed_texts([text])
        if not vectors:
            raise EmbeddingException("empty embedding response")
        return vectors[0]

