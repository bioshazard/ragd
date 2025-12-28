# ragd

Minimal RAG service backed by PostgreSQL + pgvector and OpenAI-compatible embeddings.

## Quick start

1. Ensure Postgres has pgvector installed.
2. Configure env vars (copy `.env.dist` to `.env` and edit as needed). `LLM_*` overrides are optional.
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
- HNSW vector indexing requires a fixed vector dimension; see `scripts/init_db.sql` for a manual index example.

## Usage guide

### 1) Create an API key

```bash
curl -sS -X POST http://localhost:8000/v1/api-keys \
  -H 'Content-Type: application/json' \
  -d '{"label":"local"}'
```

Export the returned `secret_once` as `RAGD_API_KEY`.

### 2) Create a collection

```bash
curl -sS -X POST http://localhost:8000/v1/collections \
  -H "Authorization: Bearer $RAGD_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"name":"podcast","embed_model":"nomic-embed-text:latest","hybrid_enabled":true}'
```

### 3) Ingest transcripts (100 .txt files)

Chunking notes:

- ragd splits text into word-based chunks with overlap (defaults: `CHUNK_TARGET_TOKENS=700`, `CHUNK_OVERLAP_TOKENS=120`).
- If you have timestamps, send segment arrays (`{text,t_start,t_end}`) so chunks carry time metadata for citations.

Use the helper script to ingest a folder of `.txt` files (each file becomes a document, idempotent by doc_id):

```bash
export RAGD_URL="http://localhost:8000"
export RAGD_API_KEY="..."

python scripts/ragd_client.py ingest \
  --collection podcast \
  --paths "./transcripts/*.txt" \
  --tags "podcast" \
  --ingest-mode replace
```

### 4) Retrieve top-k chunks

```bash
python scripts/ragd_client.py search \
  --collection podcast \
  --query "how we onboard new hosts" \
  --k 8 \
  --mode vector
```

Use `--mode hybrid` if the collection has `hybrid_enabled=true`.

### 5) Basic RAG completion (optional)

```bash
python scripts/ragd_client.py ask \
  --collection podcast \
  --query "What do we recommend for guest audio setups?" \
  --k 6 \
  --mode vector \
  --llm-model gpt-4o-mini
```

If `--llm-model` is omitted, the server uses `LLM_MODEL_DEFAULT`.
