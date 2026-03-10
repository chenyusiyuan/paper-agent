# Repository Guidelines

## Project Structure & Module Organization
`app/` contains the runtime code, split by responsibility: `parsing/` for GROBID and TEI ingestion, `indexing/` for FAISS and BM25 indexes, `retrieval/` for ranking and fusion, and `core/` for shared config, schemas, and logging. `scripts/` holds CLI entrypoints such as `ingest_papers.py` and `build_indexes.py`. `tests/` mirrors the package layout with `test_parsing/`, `test_indexing/`, and `test_retrieval/`. Runtime data lives under `data/`: place PDFs in `data/raw_pdfs/`, parsed JSON in `data/parsed/`, and generated indexes in `data/indexes/`.

## Build, Test, and Development Commands
Create an environment and install dependencies with `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. Start the local GROBID service with `docker compose up -d grobid`; the app expects `http://localhost:8070`. Parse PDFs with `python scripts/ingest_papers.py --pdf-dir data/raw_pdfs`. Rebuild retrieval indexes with `python scripts/build_indexes.py`. Run the test suite with `pytest`, or target one area with `pytest tests/test_indexing`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, type hints on public functions, `from __future__ import annotations`, and small focused modules. Use `snake_case` for functions, variables, and filenames; use `PascalCase` for classes and Pydantic models. Keep new code aligned with the current standard-library-first import style and concise logging via `app.core.logger`. No formatter or linter is configured here, so match the surrounding file style exactly before introducing new tooling.

## Testing Guidelines
Tests use `pytest` with straightforward unit-style fixtures and stubs. Name files `test_<module>.py` and keep test functions behavior-focused, for example `test_paper_retriever_runs_full_pipeline`. Add tests alongside the matching package whenever you change parsing, indexing, or retrieval behavior. Prefer deterministic fixtures over live model or network calls.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects such as `Add phase 2 indexing layer`. Keep commits focused and readable, ideally one logical change per commit. Pull requests should describe the user-visible or pipeline impact, note any config or data-directory changes, link the relevant plan or issue, and include sample commands or test results when behavior changes.

## Configuration & Data Tips
Settings are loaded from `.env` through `app.core.config.Settings`. Keep secrets like `llm_api_key` out of source control, and avoid committing generated files under `data/indexes/` unless the change explicitly requires checked-in artifacts.
