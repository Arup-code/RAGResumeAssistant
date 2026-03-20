from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from models.responses import IngestionData, IngestionResponse, MatchData, MatchResponse


class MainCliTests(unittest.TestCase):
    def test_default_flow_accepts_pasted_jd(self):
        ingest_response = IngestionResponse(success=True, data=IngestionData(processed_count=1))
        match_response = MatchResponse(success=True, data=MatchData(job_id="jd_x", top_k=3, results=[]))

        with (
            patch("main.ingest_resumes", return_value=ingest_response),
            patch("main.match_job_from_text", return_value=match_response) as mock_match,
            patch("builtins.input", side_effect=["1", "3", "Backend Engineer", "Need Python and SQL", "END"]),
            patch("builtins.print"),
        ):
            exit_code = main._run_default_interactive_flow()

        self.assertEqual(exit_code, 0)
        mock_match.assert_called_once()
        called_kwargs = mock_match.call_args.kwargs
        self.assertEqual(called_kwargs["top_k"], 3)
        self.assertIn("Backend Engineer", called_kwargs["jd_text"])

    def test_default_flow_accepts_local_json_path(self):
        ingest_response = IngestionResponse(success=True, data=IngestionData(processed_count=1))
        match_response = MatchResponse(success=True, data=MatchData(job_id="jd_file", top_k=10, results=[]))

        with tempfile.TemporaryDirectory() as temp_dir:
            jd_path = Path(temp_dir) / "jd.json"
            jd_path.write_text(
                '{"job_id":"jd_file","title":"Role","text":"Need python","required_skills":[],"min_experience_years":0}',
                encoding="utf-8",
            )

            with (
                patch("main._default_jd_path", return_value=str(jd_path)),
                patch("main.ingest_resumes", return_value=ingest_response),
                patch("main.match_job", return_value=match_response) as mock_match,
                patch("builtins.input", side_effect=["2", "", ""]),
                patch("builtins.print"),
            ):
                exit_code = main._run_default_interactive_flow()

        self.assertEqual(exit_code, 0)
        mock_match.assert_called_once_with(job_file_path=str(jd_path), top_k=10)

    def test_main_without_args_runs_interactive_default_flow(self):
        with (
            patch.object(main.sys, "argv", ["main.py"]),
            patch("main._run_default_interactive_flow", return_value=0) as mock_default,
        ):
            exit_code = main.main()

        self.assertEqual(exit_code, 0)
        mock_default.assert_called_once()


if __name__ == "__main__":
    unittest.main()

