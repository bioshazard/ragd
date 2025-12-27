# ragd

Minimal RAG service backed by PostgreSQL + pgvector and OpenAI-compatible embeddings.

## Quick start

1. Ensure Postgres has pgvector installed.
2. Configure env vars (copy `.env.dist` to `.env` and edit as needed).
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
- The service uses the OpenAI Python client against your OpenAI-compatible endpoint.
