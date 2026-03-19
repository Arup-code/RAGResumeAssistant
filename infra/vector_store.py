from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models.schemas import Chunk, Metadata, SearchResult
from utils.exceptions import RetrievalException


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: str = "storage/chroma",
        collection_name: str = "global_resume_collection",
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RetrievalException("chromadb is required for vector persistence") from exc

        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_directory)
        self.collection = client.get_or_create_collection(name=collection_name)

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        metadata: Metadata,
    ) -> None:
        if len(chunks) != len(vectors):
            raise RetrievalException("chunk/vector length mismatch")

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(
                {
                    "source_id": chunk.source_id,
                    "section_type": chunk.section_type,
                    "name": metadata.name,
                    "skills": json.dumps(metadata.skills),
                    "experience_years": float(metadata.experience_years),
                    "education": metadata.education,
                }
            )

        self.collection.upsert(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)

    def query(self, query_vector: list[float], top_k: int = 10) -> list[SearchResult]:
        response = self.collection.query(query_embeddings=[query_vector], n_results=top_k)

        ids = response.get("ids", [[]])[0]
        docs = response.get("documents", [[]])[0]
        distances = response.get("distances", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]

        results: list[SearchResult] = []
        for chunk_id, doc, distance, raw_meta in zip(ids, docs, distances, metadatas, strict=False):
            normalized_score = max(0.0, 1.0 - float(distance))
            skills = []
            if isinstance(raw_meta.get("skills"), str):
                try:
                    skills = json.loads(raw_meta["skills"])
                except json.JSONDecodeError:
                    skills = []
            metadata = Metadata(
                name=str(raw_meta.get("name", "")),
                skills=skills,
                experience_years=float(raw_meta.get("experience_years", 0.0)),
                education=str(raw_meta.get("education", "")),
            )
            results.append(
                SearchResult(
                    source_id=str(raw_meta.get("source_id", "")),
                    chunk_id=chunk_id,
                    score=normalized_score,
                    chunk_text=doc,
                    section_type=str(raw_meta.get("section_type", "general")),
                    metadata=metadata,
                )
            )

        return results


