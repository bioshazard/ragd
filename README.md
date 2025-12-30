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
- Chunking defaults are env-driven (`CHUNK_TARGET_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `CHUNK_MAX_CHARS`).
- Reset all data (keep schema): `scripts/reset_db.sh` (or `psql "$DATABASE_URL" -f scripts/reset_db.sql`)

## Canonical library

The retrieval blocks live in `ragd.canon` and are the source of truth for REST/MCP.

- Spec: `docs/canon-spec.md`
- Conformance: `python3 scripts/conformance.py`

Library usage (Postgres-backed example):

```python
from ragd import canon
from ragd.config import load_settings
from ragd.db import init_pool, get_pool
from ragd.embeddings import build_client, embed_texts_batched
from ragd.store import PostgresStore

settings = load_settings()
init_pool(settings.database_url)
store = PostgresStore(get_pool())

collection = store.get_collection("docs")
if not collection:
    collection = store.create_collection(
        "docs",
        embed_model=settings.embed_model,
        embed_dims=settings.embed_dims,
        hybrid_enabled=True,
    )

client = build_client(settings.openai_base_url, settings.openai_api_key)
embedder = lambda texts: embed_texts_batched(
    client, settings.embed_model, texts, settings.embed_batch_size
)

policy = canon.ChunkPolicy(
    target_tokens=settings.chunk_target_tokens,
    overlap_tokens=settings.chunk_overlap_tokens,
    max_chars=settings.chunk_max_chars,
)
chunks = canon.chunk("Cats are small mammals.", policy)
records = canon.prepare_chunks("doc-1", chunks, tags=["docs"], metadata={"source": "demo"})
canon.index(
    records,
    store,
    collection_id=collection.id,
    embedder=embedder,
    embed_dims=settings.embed_dims,
    ingest_mode="upsert",
)

plan = canon.RetrievePlan(mode="vector", k=5)
candidates = canon.retrieve(
    "small mammals",
    plan,
    store,
    collection_id=collection.id,
    embedder=embedder,
    embed_dims=settings.embed_dims,
    hybrid_enabled=collection.hybrid_enabled,
)
```

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
  -d '{"name":"docs","hybrid_enabled":true}'
```

### 3) Ingest transcripts (100 .txt files)

Chunking notes:

- ragd splits text into word-based chunks with overlap (defaults: `CHUNK_TARGET_TOKENS=600`, `CHUNK_OVERLAP_TOKENS=120` or `CHUNK_OVERLAP_PERCENT=15`).
- If you have timestamps or speakers, send segment arrays (`{text,t_start,t_end,speaker}`) so chunks carry time metadata for citations.

Use the helper script to ingest a folder of `.txt` files (each file becomes a document, idempotent by doc_id):

```bash
export RAGD_URL="http://localhost:8000"
export RAGD_API_KEY="..."

python scripts/ragd_client.py ingest \
  --collection docs \
  --paths "./transcripts/*.txt" \
  --tags "docs" \
  --ingest-mode replace
```

### 4) Retrieve top-k chunks

```bash
python scripts/ragd_client.py search \
  --collection docs \
  --query "how we onboard new hosts" \
  --k 8 \
  --mode vector
```

Use `--mode hybrid` if the collection has `hybrid_enabled=true`.

### 5) Basic RAG completion (optional)

```bash
python scripts/ragd_client.py ask \
  --collection docs \
  --query "What do we recommend for guest audio setups?" \
  --k 6 \
  --mode vector \
  --llm-model gpt-4o-mini
```

If `--llm-model` is omitted, the server uses `LLM_MODEL_DEFAULT`.
