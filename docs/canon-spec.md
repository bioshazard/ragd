# ragd canonical RAG spec (draft v0.1)

This document defines a minimal, canonical decomposition of retrieval-augmented generation (RAG) that is:

- minimal: few dependencies, explicit SQL
- canonical: well-defined behavior and determinism
- composable: stable interfaces between blocks
- operational: library-first with REST/MCP adapters
- reproducible: deterministic ingest + conformance tests

This spec stops before agentic or multi-step LLM pipelines. Optional blocks like reranking
and compression are included as pure retrieval components.

---

## 0. Scope and non-goals

### In scope

- Chunking, embedding, storage, retrieval, and fusion
- Deterministic ingest and retrieval semantics
- Reference Postgres schema and query patterns
- Conformance tests and golden vectors
- Minimal Python API with stable signatures

### Out of scope

- Agent orchestration or tool-calling pipelines
- Multi-tenant auth/ACLs beyond a single-tenant API key
- Vector DB abstractions beyond Postgres + pgvector

---

## 1. Canonical decomposition

The system decomposes into the following blocks. Each block has explicit inputs,
outputs, and determinism rules.

### 1.1 Ingest blocks

1) Normalize
- Input: raw document content or segments
- Output: normalized text + metadata
- Rules: stable normalization (no locale-specific behavior)

2) Chunk
- Input: normalized text or segments, chunk policy
- Output: ordered chunks with deterministic chunk_index
- Rules: stable split order; no randomness

3) Embed
- Input: chunk text
- Output: fixed-dim embeddings
- Rules: model + dims are fixed per collection

4) Store
- Input: chunks + embeddings + metadata
- Output: persisted chunks, idempotent upserts
- Rules: uniqueness on (collection_id, doc_id, chunk_index)

### 1.2 Retrieval blocks

1) Query embed
- Input: query text
- Output: query embedding

2) Candidate generation
- Vector: top-k by cosine similarity
- Optional lexical: Postgres FTS top-k

3) Filtering
- tags_any, tags_all, metadata equality (top-level)

4) Fuse (if multiple candidate sets)
- Default: Reciprocal Rank Fusion (RRF)

5) Rerank (optional)
- Re-score candidates with a model, stable ordering

6) Compress (optional)
- Extract spans or sentence windows from top candidates

7) Package
- Return ordered results with scores and metadata

---

## 2. Minimal Python API (library, stable)

The library is the source of truth; services are thin adapters.
Functions are pure (deterministic for a given input + config) where possible.

### 2.1 Types (conceptual)

- Chunk: { doc_id, chunk_index, content, tags, metadata }
- Embed: list[float] length = embed_dims
- Candidate: { doc_id, chunk_index, content, score, tags, metadata }

### 2.2 Functions

```python
def chunk(text_or_segments, policy) -> list[Chunk]:
    """Deterministic chunking with stable chunk_index ordering."""

def index(chunks, store) -> dict:
    """Upsert chunks + embeddings into the store. Returns counts/handles."""

def retrieve(query, plan, store) -> list[Candidate]:
    """Generate candidates (vector/lexical), apply filters, return ordered list."""

def fuse(candidate_sets, method="rrf", params=None) -> list[Candidate]:
    """Fuse multiple ranked lists into one ranked list."""

def rerank(query, candidates, model=None) -> list[Candidate]:
    """Optional reranking block. Must be deterministic for a given model/config."""

def compress(query, candidates, policy=None) -> list[dict]:
    """Optional extraction of spans/segments from candidates."""
```

### 2.3 Determinism rules

- Stable ordering: when scores tie, sort by (doc_id, chunk_index)
- Chunking is deterministic for a given policy + input
- Indexing is idempotent for a given chunk list
- Retrieval results are deterministic for a given store state

---

## 3. Reference Postgres schema

The schema encodes the canonical contract:

- collections: fixed embed_model + embed_dims
- documents: logical doc metadata
- chunks: retrieval unit, deterministic chunk_index
- embeddings: vector column on chunks with fixed dims

Vector index: HNSW on embedding with cosine ops.
Optional: FTS tsvector + GIN for lexical search.

Schema details remain aligned with docs/design-init.md.

---

## 4. Retrieval semantics (canonical)

### 4.1 Vector search

- Use cosine distance (vector_cosine_ops)
- Query: ORDER BY embedding <=> query_embedding LIMIT k
- Candidate pool size default: 50

### 4.2 Lexical search (optional)

- Use Postgres FTS with GIN index
- Rank by ts_rank_cd or similar deterministic ranking
- Candidate pool size default: 50

### 4.3 Fusion (RRF)

- Score = sum(1 / (rrf_k + rank_i))
- Default rrf_k: 60
- Final k default: 12

### 4.4 Filtering

- tags_any: tags && provided
- tags_all: tags @> provided
- metadata: equality on top-level keys

---

## 5. Conformance suite

### 5.1 Golden vectors

- Fixed corpus + queries
- Expected top-k IDs with score ordering constraints
- Store and compare deterministic outputs

### 5.2 Deterministic ingest

- Same input yields identical chunks, chunk_index, embeddings, and storage keys
- Chunking rules must be stable across platforms

### 5.3 Versioning

- Spec versions are semver
- Breaking changes require new major spec version
- Golden vectors are pinned per spec version

---

## 6. Service adapter rules

- REST/MCP endpoints are thin wrappers over the library
- No behavioral drift between service and library
- Canonical behavior is defined by the library + spec

---

## 7. Implementation notes (non-normative)

- Keep dependencies minimal (FastAPI, psycopg, openai client)
- Prefer explicit SQL over ORMs
- Batch embedding calls
- Optional auto-apply schema in dev only
