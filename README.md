# GT-Transformer workbench

This repository contains the FastAPI web workbench for GT-Transformer recombination breakpoint detection. It includes the inference code, bundled model checkpoints, sample alignments, and an API-backed interface for loading a FASTA alignment, running inference, viewing results, and exporting reports.

## Live workbench

- Current production site: https://gt-transformer-prod-061ed.containers.snapdeploy.app/
- Permanent bookmark: https://math12345678.github.io/gt-transformer-web/

The GitHub Pages address is a redirect page. It points to the current production site so that old links from the manuscript keep working.

## Requirements

- Python 3.10 or newer
- `pip`
- Optional: Docker, if you want to run the app in a container
- Optional: a Supabase/Postgres connection string, if you want shared persistence instead of the local SQLite file
- Optional: an API key, if you want the AI results assistant in the web interface

## Local setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Configuration is optional. To change settings, copy the example file and edit it:

```bash
cp .env.example .env
```

Never commit `.env`. It is ignored by Git and Docker.

The main settings are:

- `SUPABASE_DB_DSN`: Postgres connection string for shared runs and accounts. If omitted, the app creates a local `recomb_app.db` SQLite file.
- `AI_PROVIDER` and `ANTHROPIC_API_KEY`: enable the AI assistant. `ANTHROPIC_MODEL` sets the model name.
- `ADMIN_USERNAME` and `ADMIN_PASSWORD`: admin login for the deployed workbench.
- `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`: optional Google sign-in.

You can run the app without any of these. The assistant will simply report that it is not configured.

## Run locally

```bash
./run_server.sh
```

Then open http://127.0.0.1:8050.

The health check is available at http://127.0.0.1:8050/healthz.

To run on a different port:

```bash
PORT=9000 ./run_server.sh
```

You can also start the server directly:

```bash
python3 -m uvicorn app:app --host 127.0.0.1 --port 8050
```

## Using the workbench

1. Click **Open** and choose an aligned FASTA file. You can also paste a FASTA alignment.
   - The sequences must already be aligned to the same length.
   - Inputs need 5 to 500 sequences and up to 50,000 columns.
   - The upload limit is 20 MB.
2. Use **Options** to choose the checkpoint and scoring mode if needed.
3. Click **Run**. The job runs in the background and the interface polls for results.
4. Use the results tabs to inspect breakpoints, parents, diagnostics, timings, and annotations.
5. Use the export menu to download CSV, JSON, TSV, FASTA, HTML, or a complete ZIP bundle.

Small example alignments are in `sample_data/`. Validation fixtures used by the tests are in `tests/fixtures/`.

## Tests

Install the dependencies first, then run:

```bash
python3 -m unittest discover -s tests -v
```

The tests use the bundled checkpoints and local fixtures. They do not need API keys, a network connection, or a Supabase database.

The benchmark scripts are in `scripts/`:

```bash
python3 scripts/benchmark_timings.py
python3 scripts/build_and_run_martin_suite.py
```

These write updated timing/results files into the repository root, so only run them when you intend to regenerate those files.

## Docker

```bash
docker build -t gt-transformer .
docker run --rm -p 8050:8000 gt-transformer
```

Then open http://127.0.0.1:8050. Pass environment variables with `--env` if you need Postgres or the AI assistant.

## Repository layout

- `app.py` — FastAPI application and route registration
- `routers/` — HTTP routes for runs, authentication, and chat
- `services/` — inference adapter, database, timing, diagnostics, and chat logic
- `gt_core/` — model, data handling, pipeline, decoding, and training code
- `checkpoints/` — bundled model weights
- `sample_data/` — small alignments for manual testing
- `tests/` — unit and integration tests, plus `fixtures/`
- `static/` and `templates/` — web interface assets
- `index.html` — GitHub Pages redirect only
- `scripts/` — timing and benchmark helpers
- `Dockerfile`, `requirements.txt`, `run_server.sh` — build and run files
