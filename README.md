# RAG Resume Assistant

Production-oriented starter for resume ingestion + job matching with strict Pydantic contracts, local persistent ChromaDB, and controlled parse-skip reporting.

## Implemented contracts

- Modular folders: `rag/`, `matcher/`, `models/`, `infra/`, `utils/`
- Strict response envelopes via `BaseResponse`
- Ingestion includes:
  - `skipped_items` with controlled enum reasons
  - `skipped_count` enforced as `len(skipped_items)`
- Matching output excludes skip-report fields by design

## Dataset layout

- `dataset/resumes/` (flat files: `.pdf`, `.docx`, `.txt`)
- `dataset/jds/` (flat files: `.json`)

## Install

```bash
python -m pip install -r requirements.txt
```

## Generate 30 resumes + 5 JDs quickly

```bash
python scripts/generate_mock_dataset.py
```

## Import Kaggle resumes into `dataset/resumes`

```bash
python scripts/import_kaggle_resumes.py
```

This downloads `palaksood97/resume-dataset`, then copies `.txt/.pdf/.docx` files and extracts resume text from CSV rows into `dataset/resumes/`.

## Run ingestion

```bash
python resume_rag.py
```

## Run matching (first JD in dataset)

```bash
python job_matcher.py
```

## Run demo orchestrator

```bash
python scripts/run_demo.py
```

## Unified CLI (`main.py`)

Default (interactive) mode:

```bash
python main.py
```

The interactive flow uses default ingestion settings and then asks you to provide a JD via:
- multiline paste (type `END` on its own line to finish), or
- a local file path (`.json` JobDescription file or plain-text JD file)

For advanced parameters and subcommands:

```bash
python main.py --help
```

```bash
python main.py ingest
python main.py match --job-file-path dataset/jds/jd_001.json --top-k 10
python main.py pipeline --job-file-path dataset/jds/jd_001.json --top-k 10
python main.py run --job-file-path dataset/jds/jd_001.json --json
```

If `--job-file-path` is omitted, the CLI uses the first JSON file in `dataset/jds`.

Output mode flags are available on each subcommand:
- `--pretty` (default): indented JSON
- `--json`: compact JSON

`run` is an alias for `pipeline`.

## Retrieval experimentation notebook (accuracy + latency)

```bash
jupyter notebook notebooks/retrieval_experimentation.ipynb
```

The notebook reports:
- `Recall@K` and `HitRate@K` for retrieval accuracy
- mean/p50/p95 retrieval latency in milliseconds

## CLI metrics runner (same two metrics)

```bash
python scripts/run_retrieval_metrics.py
```

## Run tests

```bash
python -m unittest -v tests/test_vertical_slice.py
python -m unittest -v tests/test_retrieval_metrics.py
```

## Notes

- `OPENROUTER_API_KEY` is mandatory for metadata extraction, JD extraction, and embeddings.
- Chroma persistence defaults to `storage/chroma` using collection `global_resume_collection`.
- Parse failures are skipped and reported in ingestion response only.

