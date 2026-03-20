from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from job_matcher import match_job, match_job_from_text
from resume_rag import ingest_resumes


def _default_jd_path(jd_dir: str) -> str:
    jd_files = sorted(Path(jd_dir).glob("*.json"))
    if not jd_files:
        raise SystemExit(f"No JD JSON files found in {jd_dir}")
    return str(jd_files[0])


def _print_model_response(response: object, pretty: bool) -> None:
    indent = 2 if pretty else None
    print(response.model_dump_json(indent=indent))


def _print_payload(payload: dict[str, object], pretty: bool) -> None:
    indent = 2 if pretty else None
    print(json.dumps(payload, indent=indent))


def _add_output_mode_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--json",
        dest="pretty",
        action="store_false",
        help="Print compact JSON output",
    )
    group.add_argument(
        "--pretty",
        dest="pretty",
        action="store_true",
        help="Print indented JSON output (default)",
    )
    parser.set_defaults(pretty=True)


def _add_match_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--job-file-path", default=None)
    parser.add_argument("--jd-dir", default="dataset/jds")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--persist-directory", default="storage/chroma")
    parser.add_argument("--collection-name", default="global_resume_collection")


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resume-dir", default="dataset/resumes")
    parser.add_argument("--job-file-path", default=None)
    parser.add_argument("--jd-dir", default="dataset/jds")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--persist-directory", default="storage/chroma")
    parser.add_argument("--collection-name", default="global_resume_collection")


def _run_ingest(args: argparse.Namespace) -> int:
    response = ingest_resumes(
        resume_dir=args.resume_dir,
        persist_directory=args.persist_directory,
        collection_name=args.collection_name,
    )
    _print_model_response(response, pretty=args.pretty)
    return 0 if response.success else 1


def _run_match(args: argparse.Namespace) -> int:
    jd_path = args.job_file_path or _default_jd_path(args.jd_dir)
    response = match_job(
        job_file_path=jd_path,
        top_k=args.top_k,
        persist_directory=args.persist_directory,
        collection_name=args.collection_name,
    )
    _print_model_response(response, pretty=args.pretty)
    return 0 if response.success else 1


def _run_pipeline(args: argparse.Namespace) -> int:
    ingest_response = ingest_resumes(
        resume_dir=args.resume_dir,
        persist_directory=args.persist_directory,
        collection_name=args.collection_name,
    )
    payload: dict[str, object] = {"ingestion": ingest_response.model_dump()}

    if not ingest_response.success:
        payload["matching"] = None
        _print_payload(payload, pretty=args.pretty)
        return 1

    jd_path = args.job_file_path or _default_jd_path(args.jd_dir)
    match_response = match_job(
        job_file_path=jd_path,
        top_k=args.top_k,
        persist_directory=args.persist_directory,
        collection_name=args.collection_name,
    )
    payload["matching"] = match_response.model_dump()

    _print_payload(payload, pretty=args.pretty)
    return 0 if ingest_response.success and match_response.success else 1


def _read_multiline_jd() -> str:
    print("Paste the job description. Enter a single line with END when done.")
    lines: list[str] = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _load_jd_text_from_path(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _parse_top_k(raw_value: str, default: int = 10) -> int:
    value = raw_value.strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        print(f"Invalid top-k '{value}', using default {default}.")
        return default
    return parsed if parsed > 0 else default


def _run_default_interactive_flow() -> int:
    print("RAG Resume Assistant - default interactive mode")
    print("For advanced options, run: python main.py --help")

    default_jd_path: str | None = None
    try:
        default_jd_path = _default_jd_path("dataset/jds")
    except SystemExit:
        default_jd_path = None

    print("How do you want to provide the job description?")
    print("  1) Paste JD text")
    print("  2) Local file path (JSON JD or plain text)")
    choice = input("Select option [2]: ").strip().lower() or "2"
    top_k = _parse_top_k(input("Top-k matches [10]: "), default=10)

    ingest_response = ingest_resumes()
    payload: dict[str, object] = {"ingestion": ingest_response.model_dump()}
    if not ingest_response.success:
        payload["matching"] = None
        _print_payload(payload, pretty=True)
        return 1

    if choice in {"1", "paste", "p"}:
        jd_text = _read_multiline_jd()
        match_response = match_job_from_text(jd_text=jd_text, top_k=top_k)
    else:
        if default_jd_path:
            prompt = f"Enter local JD path [{default_jd_path}]: "
            jd_path = input(prompt).strip() or default_jd_path
        else:
            jd_path = input("Enter local JD path: ").strip()

        if not jd_path:
            payload["matching"] = {"success": False, "error": "validation_error: JD path is required"}
            _print_payload(payload, pretty=True)
            return 1

        path = Path(jd_path)
        if not path.exists() or not path.is_file():
            payload["matching"] = {
                "success": False,
                "error": f"validation_error: JD file not found: {jd_path}",
            }
            _print_payload(payload, pretty=True)
            return 1

        if path.suffix.lower() == ".json":
            match_response = match_job(job_file_path=str(path), top_k=top_k)
        else:
            jd_text = _load_jd_text_from_path(str(path))
            match_response = match_job_from_text(jd_text=jd_text, top_k=top_k)

    payload["matching"] = match_response.model_dump()
    _print_payload(payload, pretty=True)
    return 0 if match_response.success else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI for resume ingestion and job matching",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest resumes into vector store")
    ingest_parser.add_argument("--resume-dir", default="dataset/resumes")
    ingest_parser.add_argument("--persist-directory", default="storage/chroma")
    ingest_parser.add_argument("--collection-name", default="global_resume_collection")
    _add_output_mode_args(ingest_parser)
    ingest_parser.set_defaults(func=_run_ingest)

    match_parser = subparsers.add_parser("match", help="Match a job description against resumes")
    _add_match_args(match_parser)
    _add_output_mode_args(match_parser)
    match_parser.set_defaults(func=_run_match)

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run ingestion and then matching",
    )
    _add_pipeline_args(pipeline_parser)
    _add_output_mode_args(pipeline_parser)
    pipeline_parser.set_defaults(func=_run_pipeline)

    run_parser = subparsers.add_parser(
        "run",
        help="Alias for pipeline",
    )
    _add_pipeline_args(run_parser)
    _add_output_mode_args(run_parser)
    run_parser.set_defaults(func=_run_pipeline)

    return parser


def main() -> int:
    if len(sys.argv) == 1:
        return _run_default_interactive_flow()

    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())



