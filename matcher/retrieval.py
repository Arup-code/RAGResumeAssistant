from __future__ import annotations

from infra.embedding_service import EmbeddingService
from infra.vector_store import ChromaVectorStore
from models.schemas import JobDescription, SearchResult


def retrieve_candidates(
    job: JobDescription,
    embedding_service: EmbeddingService,
    vector_store: ChromaVectorStore,
    top_k: int = 10,
) -> list[SearchResult]:
    query_vector = embedding_service.embed_text(job.text)
    return vector_store.query(query_vector=query_vector, top_k=top_k)

