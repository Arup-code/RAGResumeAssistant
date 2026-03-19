from __future__ import annotations

import unittest

from experiments.retrieval_metrics import JDRecord, ResumeRecord, evaluate_retrieval, relevance_label


class RetrievalMetricsTests(unittest.TestCase):
    def test_relevance_label_all_required(self) -> None:
        resume = ResumeRecord(
            source_id="r1",
            text="Python SQL",
            skills=["python", "sql"],
            experience_years=3,
        )
        jd = JDRecord(
            job_id="j1",
            title="Backend",
            text="Need python and sql",
            required_skills=["python", "sql"],
            min_experience_years=2,
        )
        self.assertTrue(relevance_label(resume, jd, mode="all_required"))

    def test_metrics_and_latency_shape(self) -> None:
        resumes = [
            ResumeRecord("r1", "Python SQL", ["python", "sql"], 4),
            ResumeRecord("r2", "Java Spring", ["java", "spring"], 5),
        ]
        jobs = [
            JDRecord("j1", "Backend", "Need python", ["python"], 2),
            JDRecord("j2", "Java", "Need java", ["java"], 3),
        ]

        result = evaluate_retrieval(
            resumes=resumes,
            jobs=jobs,
            k_values=[1, 2],
            relevance_mode="all_required",
            latency_repeats=3,
        )

        self.assertEqual(len(result["metrics"]), 2)
        self.assertEqual(len(result["latency_rows"]), 2)
        self.assertIn("avg_latency_ms", result["summary"])
        self.assertGreaterEqual(result["summary"]["avg_latency_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()

