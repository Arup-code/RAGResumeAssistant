from __future__ import annotations

from pathlib import Path

from infra.embedding_service import EmbeddingService
from infra.vector_store import ChromaVectorStore
from models.responses import IngestionData, IngestionResponse
from models.schemas import SkippedItem, SkippedReason
from rag.chunker import chunk_resume
from rag.loader import load_resume
from rag.metadata_extractor import MetadataExtractor
from utils.exceptions import AppException, ParseException
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def _list_resume_files(resume_dir: str) -> list[str]:
    supported = {".pdf", ".docx", ".txt"}
    return [
        str(path)
        for path in sorted(Path(resume_dir).glob("*"))
        if path.is_file() and path.suffix.lower() in supported
    ]


def ingest_resumes(
    resume_dir: str = "dataset/resumes",
    persist_directory: str = "storage/chroma",
    collection_name: str = "global_resume_collection",
) -> IngestionResponse:
    try:
        vector_store = ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        embedding_service = EmbeddingService()
        metadata_extractor = MetadataExtractor()
    except AppException as exc:
        return IngestionResponse(success=False, error=f"{exc.code}: {exc.message}")

    ingested_ids: list[str] = []
    skipped_items: list[SkippedItem] = []

    for file_path in _list_resume_files(resume_dir):
        try:
            document = load_resume(file_path)
            chunks = chunk_resume(document)
            if not chunks:
                raise ParseException(
                    "chunking produced no content",
                    skipped_reason=SkippedReason.PARSE_ERROR_UNKNOWN,
                )
            metadata = metadata_extractor.extract(document.text)
            vectors = embedding_service.embed_texts([chunk.text for chunk in chunks])
            vector_store.upsert_chunks(chunks=chunks, vectors=vectors, metadata=metadata)
            ingested_ids.append(document.source_id)
            logger.info("Ingested resume: %s", document.source_id)
        except ParseException as exc:
            logger.warning("Skipping resume due to parse failure: %s", file_path)
            source_id = Path(file_path).stem
            skipped_items.append(
                SkippedItem(source_id=source_id, skipped_reason=exc.skipped_reason)
            )
        except AppException as exc:
            logger.error("Ingestion failed for %s due to %s", file_path, exc.code)
            return IngestionResponse(success=False, error=f"{exc.code}: {exc.message}")
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("Unhandled ingestion failure for %s", file_path)
            return IngestionResponse(success=False, error=f"unhandled_error: {exc}")

    data = IngestionData(
        processed_count=len(ingested_ids),
        skipped_count=len(skipped_items),
        ingested_source_ids=ingested_ids,
        skipped_items=skipped_items,
    )
    return IngestionResponse(success=True, data=data)


if __name__ == "__main__":
    response = ingest_resumes()
    print(response.model_dump_json(indent=2))


