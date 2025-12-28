Below is a **self-contained “ragd” spec** your dev can implement against an **OpenAI-compatible embeddings API** + **Postgres + pgvector** with optional **Postgres full-text hybrid**. It’s intentionally small, single-tenant, free-form tags, and “collections = namespaces”.

---

## ragd v0.1 spec

### Purpose

A single service that provides:

1. **Ingest**: transcript → chunks → embeddings → store in Postgres (pgvector)
2. **Retrieve**: query → embedding → top-k chunks (optionally filtered by tags/metadata)
3. **Ask** (optional): retrieve → LLM answer with citations

### Dependencies

* **OpenAI-compatible** embeddings endpoint: `POST /v1/embeddings`
* **PostgreSQL** with **pgvector extension** ([Neon][2])
* Optional hybrid search uses **Postgres Full Text Search + GIN index** ([PostgreSQL][3])

---

## Data model (in DB)

### 1) `collections`

Represents a namespace/corpus.

Fields:

* `id` (pk)
* `name` (unique, e.g. `"podcast"`)
* `embed_model` (text; default `"nomic-embed-text"`)
* `embed_dims` (int; set once at collection creation by probing Ollama; do not change)
* `created_at`

Rules:

* All chunks in a collection MUST use the same embedding model + dims.

### 2) `api_keys` (single-tenant auth)

Fields:

* `id`
* `label`
* `key_hash` (hash of secret; store bytes)
* `created_at`
* `revoked_at` (nullable)

Rules:

* Auth is global (single tenant). No per-collection ACLs.

### 3) `documents`

Logical doc record (episode transcript).

Fields:

* `collection_id` (fk)
* `doc_id` (string; caller-provided stable ID, e.g. `"ep-0130"`)
* `title` (optional)
* `tags` (text array; free-form)
* `metadata` (jsonb; free-form)
* `source` (optional: URL/path)
* `created_at`, `updated_at`

Uniq:

* `(collection_id, doc_id)`

### 4) `chunks`

The retrieval unit.

Fields:

* `id` (pk)
* `collection_id` (fk)
* `doc_id` (string)
* `chunk_index` (int; 0..n, deterministic)
* `content` (text; the chunk)
* `tags` (text array; default empty)
* `metadata` (jsonb; include at least episode/time anchors if present)
* `embedding` (vector; dims must equal `collections.embed_dims`) ([GitHub][4])
* Optional for hybrid: `fts` (tsvector) computed from content

Uniq:

* `(collection_id, doc_id, chunk_index)` (idempotent upsert)

Indexes:

* Vector: **HNSW** on `embedding` using cosine ops (recommended) ([GitHub][4])
* Tags: GIN on `tags` (for `ANY`/`ALL` tag filters)
* Hybrid: GIN on `fts` (tsvector) ([PostgreSQL][3])

> Note: pgvector recommends creating an index per distance operator class, and shows HNSW + `vector_cosine_ops`. HNSW requires a fixed vector dimension on the column. ([GitHub][4])

---

## Chunking rules (podcast transcripts)

Goal: stable chunk IDs + good semantic recall.

Defaults (configurable per collection):

* `chunk_target_tokens`: 450–900
* `chunk_overlap_tokens`: 80–150
* Prefer splitting on:

  1. Whisper segment boundaries / speaker turns (if you have them)
  2. Paragraph breaks
  3. Sentence boundaries
  4. Hard fallback: token/window split

Required metadata per chunk (strongly recommended):

* `episode_id` (same as doc_id)
* `segment_index` or `chunk_index`
* If available: `t_start`, `t_end` (seconds) for citations

---

## Embeddings contract (OpenAI-compatible)

ragd MUST support:

* `POST {OPENAI_BASE}/v1/embeddings` with `{ model, input }`
* Input may be string or array of strings.

Collection creation MUST:

* probe dims by embedding a short string once and taking vector length
* persist as `collections.embed_dims`

---

## Retrieval modes

### Mode A: Vector-only (default)

* Embed the query with the collection’s `embed_model`
* Similarity search using cosine distance operator and `ORDER BY … LIMIT k` (to use the index pattern) ([GitHub][5])

### Mode B: Hybrid (optional, “nearly free”)

Use:

* Vector search (pgvector)
* Lexical search (Postgres FTS)
* Fuse rankings with **Reciprocal Rank Fusion (RRF)** (simple, stable) ([Jonathan Katz][5])

Notes:

* Postgres docs show standard FTS query + index usage patterns. ([PostgreSQL][3])
* RRF explanation/reference for hybrid fusion. ([Jonathan Katz][5])

Defaults:

* `candidate_pool`: 50 per modality
* `rrf_k`: 60
* final `k`: 8–20

---

## Filtering

Supported filters (applied before ranking where possible):

* `tags_any`: match chunks where `tags && provided_tags`
* `tags_all`: match chunks where `tags @> provided_tags`
* `metadata`: simple equality filters on top-level keys (optional for v0.1)

---

## REST API (single-tenant)

### Auth

All endpoints require:

* `Authorization: Bearer <secret>`

Server behavior:

* Hash secret, constant-time compare against `api_keys.key_hash`
* Reject revoked keys

### Endpoints

#### 1) Collections

**POST `/v1/collections`**
Request:

* `name` (string)
* `embed_model` (string; default `nomic-embed-text`)
* `hybrid_enabled` (bool; default false)
  Response:
* `collection`: `{ name, embed_model, embed_dims, hybrid_enabled }`

**GET `/v1/collections`**
Response: list of collections + settings.

#### 2) Ingest (document-level)

**PUT `/v1/collections/{collection}/documents/{doc_id}`**
Request:

* `title` (optional)
* `content` (string transcript OR array of segment objects)
* `tags` (string[])
* `metadata` (object)
* `ingest_mode`: `"replace"` | `"upsert"` (default replace)

Behavior:

* Replace: delete existing chunks for `(collection, doc_id)` then insert
* Upsert: upsert chunks by `(collection, doc_id, chunk_index)` deterministically
  Response:
* counts: `{ chunks_written, tokens_estimated }`

Segment input (optional):

* `content` may be `[{text, t_start?, t_end?, speaker?}, ...]`
* ragd converts to chunks, carrying timestamps forward

#### 3) Search (retrieval-only)

**POST `/v1/collections/{collection}/search`**
Request:

* `query` (string)
* `k` (int; default 12)
* `mode`: `"vector"` | `"hybrid"` (default vector)
* `tags_any` (optional string[])
* `tags_all` (optional string[])
  Response:
* `results`: array of

  * `doc_id`
  * `chunk_index`
  * `content`
  * `score` (distance or fused score)
  * `metadata`
  * `tags`

#### 4) Ask (optional synthesis)

**POST `/v1/collections/{collection}/ask`**
Request:

* same as `/search` plus:

  * `llm_model` (string; optional, defaults to server `LLM_MODEL_DEFAULT`)
  * `system_prompt` (optional)
    Response:
* `answer` (string)
* `sources`: list of `{ doc_id, chunk_index, t_start?, t_end?, excerpt }`

Synthesis rule:

* The LLM MUST be instructed to cite `doc_id + time` when available.
* By default, use the same OpenAI-compatible base URL as embeddings; allow an optional override for the LLM client.

#### 5) Admin

**POST `/v1/api-keys`**
Request: `{ label }`
Response: `{ secret_once }` (only returned on create)

**GET `/v1/health`**

* verifies DB + Ollama reachable

---

## Operational rules

* **Idempotency**: deterministic chunking + `(collection, doc_id, chunk_index)` unique key.
* **Safety**: enforce `embed_dims` match on write (reject if mismatch).
* **Performance**:

  * batch embedding calls (send `input: [..]`) when possible
  * create vector index after bulk ingest if initial load is large (pgvector notes typical index build tradeoffs) ([GitHub][4])
* **Backups**: Postgres backups cover everything.

---

## v0.2 ideas (non-required)

* Metadata query language (range filters on timestamps/dates)
* Per-document re-chunk without deleting doc record
* Pagination for `/search`
* Hybrid weight knobs
* Multi-tenant via separate DB (your stated plan)

---

If you want, I can also provide an **OpenAPI 3.1 YAML** for this contract (still “no-code” but copy/paste into tooling).

[1]: https://platform.openai.com/docs/api-reference/embeddings "OpenAI embeddings API"
[2]: https://neon.com/docs/extensions/pgvector?utm_source=chatgpt.com "The pgvector extension - Neon Docs"
[3]: https://www.postgresql.org/docs/current/textsearch-tables.html "PostgreSQL: Documentation: 18: 12.2. Tables and Indexes"
[4]: https://github.com/pgvector/pgvector "GitHub - pgvector/pgvector: Open-source vector similarity search for Postgres"
[5]: https://jkatz.github.io/post/postgres/hybrid-search-postgres-pgvector/ "Hybrid search with PostgreSQL and pgvector |
Jonathan Katz
"
