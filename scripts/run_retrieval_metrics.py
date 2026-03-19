from __future__ import annotations

from pprint import pprint
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.retrieval_metrics import evaluate_retrieval, load_jds, load_resumes


if __name__ == "__main__":
    resumes = load_resumes("dataset/resumes")
    jobs = load_jds("dataset/jds")

    result = evaluate_retrieval(
        resumes=resumes,
        jobs=jobs,
        k_values=[1, 3, 5, 10],
        relevance_mode="all_required",
        latency_repeats=25,
    )

    print("=== Retrieval Accuracy ===")
    pprint(result["metrics"])
    print("\n=== Latency Summary (ms) ===")
    pprint(result["summary"])

