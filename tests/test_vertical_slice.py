from __future__ import annotations

import unittest
from unittest.mock import patch

import job_matcher
import resume_rag
from models.schemas import Chunk, JobDescription, Metadata, ResumeDocument, SearchResult, SkippedReason
from utils.exceptions import ParseException
from utils.exceptions import ValidationException


class FakeEmbeddingService:
    def __init__(self, *args, **kwargs):
        pass

    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorStore:
    def __init__(self, *args, **kwargs):
        self.rows = []

    def upsert_chunks(self, chunks, vectors, metadata):
        self.rows.append((chunks, vectors, metadata))


class FakeMetadataExtractor:
    def __init__(self, *args, **kwargs):
        pass

    def extract(self, resume_text):
        return Metadata(name="Test User", skills=["python", "sql"], experience_years=3, education="B.Tech")


class InspectableVectorStore:
    last_metadata: Metadata | None = None

    def __init__(self, *args, **kwargs):
        pass

    def upsert_chunks(self, chunks, vectors, metadata):
        InspectableVectorStore.last_metadata = metadata


class VerticalSliceTests(unittest.TestCase):
    def test_ingestion_infers_candidate_name_when_metadata_extractor_unavailable(self):
        InspectableVectorStore.last_metadata = None
        chunk = Chunk(
            chunk_id="resume_1::chunk::0",
            source_id="resume_1",
            text="Python developer",
            section_type="general",
        )

        with (
            patch("resume_rag._list_resume_files", return_value=["resume_1.txt"]),
            patch(
                "resume_rag.load_resume",
                return_value=ResumeDocument(
                    source_id="resume_1",
                    file_path="resume_1.txt",
                    text="Jane Doe\nPython engineer with APIs",
                ),
            ),
            patch("resume_rag.chunk_resume", return_value=[chunk]),
            patch("resume_rag.ChromaVectorStore", InspectableVectorStore),
            patch("resume_rag.EmbeddingService", FakeEmbeddingService),
            patch("resume_rag.MetadataExtractor", side_effect=ValidationException("missing key")),
        ):
            response = resume_rag.ingest_resumes()

        self.assertTrue(response.success)
        self.assertIsNotNone(InspectableVectorStore.last_metadata)
        self.assertEqual(InspectableVectorStore.last_metadata.name, "Jane Doe")

    def test_ingestion_skips_parse_failures_and_tracks_count(self):
        chunk = Chunk(
            chunk_id="ok::chunk::0",
            source_id="ok",
            text="Experience in Python APIs",
            section_type="experience",
        )

        with (
            patch("resume_rag._list_resume_files", return_value=["ok.txt", "bad.txt"]),
            patch("resume_rag.load_resume") as mock_load,
            patch("resume_rag.chunk_resume", return_value=[chunk]),
            patch("resume_rag.ChromaVectorStore", FakeVectorStore),
            patch("resume_rag.EmbeddingService", FakeEmbeddingService),
            patch("resume_rag.MetadataExtractor", FakeMetadataExtractor),
        ):
            mock_load.side_effect = [
                ResumeDocument(source_id="ok", file_path="ok.txt", text="Experience\nPython"),
                ParseException("bad parse", skipped_reason=SkippedReason.PARSE_ERROR_TXT),
            ]
            response = resume_rag.ingest_resumes()

        self.assertTrue(response.success)
        self.assertIsNotNone(response.data)
        self.assertEqual(response.data.processed_count, 1)
        self.assertEqual(response.data.skipped_count, len(response.data.skipped_items))
        self.assertEqual(response.data.skipped_count, 1)
        self.assertEqual(response.data.skipped_items[0].skipped_reason, SkippedReason.PARSE_ERROR_TXT)

    def test_job_matcher_returns_ranked_results(self):
        fake_job = JobDescription(
            job_id="jd_1",
            title="Backend",
            text="Need python and sql",
            required_skills=["python", "sql"],
            min_experience_years=2,
        )

        candidates = [
            SearchResult(
                source_id="cand_a",
                chunk_id="cand_a::chunk::1",
                score=0.9,
                chunk_text="Built APIs with Python and SQL",
                section_type="experience",
                metadata=Metadata(name="A", skills=["python", "sql"], experience_years=4, education="B.Tech"),
            ),
            SearchResult(
                source_id="cand_a",
                chunk_id="cand_a::chunk::2",
                score=0.6,
                chunk_text="Additional details",
                section_type="skills",
                metadata=Metadata(name="A", skills=["python"], experience_years=4, education="B.Tech"),
            ),
        ]

        with (
            patch("job_matcher._load_job_description", return_value=fake_job),
            patch("job_matcher.EmbeddingService", FakeEmbeddingService),
            patch("job_matcher.ChromaVectorStore", FakeVectorStore),
            patch("job_matcher.retrieve_candidates", return_value=candidates),
        ):
            response = job_matcher.match_job("dataset/jds/sample_jd.json", top_k=5)

        self.assertTrue(response.success)
        self.assertIsNotNone(response.data)
        self.assertEqual(response.data.job_id, "jd_1")
        self.assertEqual(len(response.data.results), 1)
        self.assertEqual(response.data.results[0].source_id, "cand_a")
        self.assertEqual(response.data.results[0].candidate_name, "A")

    def test_job_matcher_falls_back_to_semantic_results_with_warning(self):
        fake_job = JobDescription(
            job_id="jd_2",
            title="Platform",
            text="Need kubernetes",
            required_skills=["kubernetes"],
            min_experience_years=5,
        )

        candidates = [
            SearchResult(
                source_id="cand_sparse",
                chunk_id="cand_sparse::chunk::1",
                score=0.88,
                chunk_text="Worked on cloud platforms and deployments",
                section_type="experience",
                metadata=Metadata(name="Sparse", skills=[], experience_years=0, education=""),
            )
        ]

        with (
            patch("job_matcher._load_job_description", return_value=fake_job),
            patch("job_matcher.EmbeddingService", FakeEmbeddingService),
            patch("job_matcher.ChromaVectorStore", FakeVectorStore),
            patch("job_matcher.retrieve_candidates", return_value=candidates),
        ):
            response = job_matcher.match_job("dataset/jds/sample_jd.json", top_k=5)

        self.assertTrue(response.success)
        self.assertIsNotNone(response.data)
        self.assertEqual(len(response.data.results), 1)
        self.assertEqual(response.data.results[0].candidate_name, "Sparse")
        self.assertIsNotNone(response.warnings)
        self.assertIn("semantic-only", response.warnings.lower())


if __name__ == "__main__":
    unittest.main()

