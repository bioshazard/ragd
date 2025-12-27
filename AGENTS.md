## Usage (Immutable)

- This file is used to provide instruction to the agent
- The code gen agent must update this file to capture DX preferences and codebase description
- Put your long term memory stuff about the project and how we like to operate below here as you see fit!

## Amendments (Mutable)

- Project: ragd service implementing the spec in `docs/design-init.md` using Python + FastAPI + PostgreSQL (pgvector) + OpenAI-compatible Ollama embeddings.
- DX: keep the stack minimal (FastAPI, psycopg, openai client); favor explicit SQL and small modules over heavy frameworks or ORMs.
- Config: environment-driven (`DATABASE_URL`, `OLLAMA_BASE_URL`, `EMBED_MODEL_DEFAULT`, chunking/search knobs); prefer sane defaults but fail fast when required env is missing.
- DB: initialize via SQL schema file; app may auto-apply schema on startup for dev; avoid migrations tooling unless requested.
