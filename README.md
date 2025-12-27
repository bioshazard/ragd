# ragd

Minimal RAG service backed by PostgreSQL + pgvector and OpenAI-compatible embeddings (Ollama).

## Quick start

1. Ensure Postgres has pgvector installed.
2. Set env vars:

```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/ragd"
export OLLAMA_BASE_URL="http://ollama.lab1.bios.dev/v1"
export OLLAMA_API_KEY="ollama"
export EMBED_MODEL_DEFAULT="nomic-embed-text:latest"
export LLM_BASE_URL="http://ollama.lab1.bios.dev/v1"
export LLM_API_KEY="ollama"
export RAGD_AUTO_MIGRATE="true"
```

3. Initialize schema (optional if `RAGD_AUTO_MIGRATE=true`):

```bash
psql "$DATABASE_URL" -f scripts/init_db.sql
```

4. Create a venv and install deps with `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

5. Run:

```bash
uvicorn ragd.main:app --reload
```

## Notes

- API auth uses `Authorization: Bearer <secret>`.
- Create an API key via `POST /v1/api-keys`.
- The service uses the OpenAI Python client against your OAI-compatible endpoint.
