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

## Quick start (clone + venv + install)

```bash
git clone https://github.com/Arup-code/RAGResumeAssistant.git
cd RAGResumeAssistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configure OpenRouter API key (mandatory)

Set `OPENROUTER_API_KEY` before running ingestion or matching.

Option 1: export in your shell

```bash
export OPENROUTER_API_KEY="your_openrouter_key"
```

Option 2: add it to a `.env` file in project root

```bash
cat > .env <<'EOF'
OPENROUTER_API_KEY=your_openrouter_key
EOF
```

If you use the shell export option, verify it is available:

```bash
python -c "import os; print(bool(os.getenv('OPENROUTER_API_KEY')))"
```

## Generate local mock dataset (30 resumes + 5 JDs)

```bash
python scripts/generate_mock_dataset.py
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

## Latest benchmark snapshot (strict relevance)

Command used:

```bash
python scripts/run_retrieval_metrics.py
```

Configuration:
- `relevance_mode="all_required"`
- `k_values=[1,3,5,10]`
- `latency_repeats=25`
- Dataset: 30 resumes, 6 JDs

Results:

| K  | HitRate@K | Recall@K |
|----|-----------|----------|
| 1  | 1.0000    | 0.0467   |
| 3  | 1.0000    | 0.1402   |
| 5  | 1.0000    | 0.2336   |
| 10 | 1.0000    | 0.4672   |

Latency summary (ms):
- Average: `1.3996`
- P50: `1.2323`
- P95: `1.5626`

## Retrieval experimentation notebook (accuracy + latency analysis)

Notebook files/links:
- Project notebook path: `notebooks/retrieval_experimentation.ipynb`
- GitHub view: https://github.com/Arup-code/RAGResumeAssistant/blob/main/notebooks/retrieval_experimentation.ipynb
- nbviewer view: https://nbviewer.org/github/Arup-code/RAGResumeAssistant/blob/main/notebooks/retrieval_experimentation.ipynb

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

