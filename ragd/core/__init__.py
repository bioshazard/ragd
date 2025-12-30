from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Protocol, Sequence

from ragd import chunking


@dataclass(frozen=True)
class ChunkPolicy:
    target_tokens: int
    overlap_tokens: int
    max_chars: int = 4000
    max_unit_tokens: int = 240


@dataclass(frozen=True)
class Chunk:
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ChunkRecord:
    doc_id: str
    chunk_index: int
    content: str
    tags: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    chunk_index: int
    content: str
    score: float
    metadata: dict[str, Any]
    tags: tuple[str, ...]


@dataclass(frozen=True)
class RetrievePlan:
    mode: Literal["vector", "hybrid"] = "vector"
    k: int = 12
    candidate_pool: int = 50
    rrf_k: int = 60
    tags_any: list[str] | None = None
    tags_all: list[str] | None = None


@dataclass(frozen=True)
class IndexResult:
    chunks_written: int
    tokens_estimated: int


@dataclass(frozen=True)
class CompressionPolicy:
    max_chars: int | None = None
    max_chunks: int | None = None


Embedder = Callable[[list[str]], list[list[float]]]
Reranker = Callable[[str, Sequence[Candidate]], Sequence[float]]


class Store(Protocol):
    def delete_chunks(self, collection_id: int, doc_id: str) -> None: ...
    def write_chunks(
        self,
        collection_id: int,
        chunks: Sequence[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> None: ...

    def vector_search(
        self,
        collection_id: int,
        query_vector: Sequence[float],
        k: int,
        tags_any: list[str] | None,
        tags_all: list[str] | None,
    ) -> list[Candidate]: ...

    def fts_search(
        self,
        collection_id: int,
        query_text: str,
        candidate_pool: int,
        tags_any: list[str] | None,
        tags_all: list[str] | None,
    ) -> list[Candidate]: ...


def ensure_collection_embeddings(
    *,
    collection_embed_model: str,
    collection_embed_dims: int,
    expected_embed_model: str,
    expected_embed_dims: int,
) -> None:
    if (
        collection_embed_model != expected_embed_model
        or collection_embed_dims != expected_embed_dims
    ):
        raise ValueError("Collection embedding config does not match server settings")


def chunk(text_or_segments: str | list[dict], policy: ChunkPolicy) -> list[Chunk]:
    if policy.target_tokens <= 0:
        raise ValueError("chunk target_tokens must be positive")
    if policy.overlap_tokens < 0:
        raise ValueError("chunk overlap_tokens must be non-negative")
    if policy.max_chars <= 0:
        raise ValueError("chunk max_chars must be positive")

    units = chunking.build_units(text_or_segments, max_unit_tokens=policy.max_unit_tokens)
    raw_chunks = chunking.chunk_units(
        units,
        target_tokens=policy.target_tokens,
        overlap_tokens=policy.overlap_tokens,
        max_chars=policy.max_chars,
    )
    return [Chunk(content=raw.text, metadata=raw.metadata) for raw in raw_chunks]


def prepare_chunks(
    doc_id: str,
    chunks: Sequence[Chunk],
    *,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[ChunkRecord]:
    if not doc_id:
        raise ValueError("doc_id is required")
    tags_tuple = tuple(tags or [])
    base_meta = dict(metadata or {})

    prepared: list[ChunkRecord] = []
    for idx, item in enumerate(chunks):
        merged = dict(base_meta)
        merged.update(item.metadata)
        merged["chunk_index"] = idx
        prepared.append(
            ChunkRecord(
                doc_id=doc_id,
                chunk_index=idx,
                content=item.content,
                tags=tags_tuple,
                metadata=merged,
            )
        )
    return prepared


def index(
    chunks: Sequence[ChunkRecord],
    store: Store,
    *,
    collection_id: int,
    embedder: Embedder,
    embed_dims: int,
    ingest_mode: Literal["replace", "upsert"] = "upsert",
) -> IndexResult:
    if ingest_mode not in {"replace", "upsert"}:
        raise ValueError("ingest_mode must be replace or upsert")
    if not chunks:
        raise ValueError("No chunks to index")

    doc_ids = {chunk.doc_id for chunk in chunks}
    if len(doc_ids) != 1:
        raise ValueError("index expects chunks for a single doc_id")
    doc_id = next(iter(doc_ids))

    if ingest_mode == "replace":
        store.delete_chunks(collection_id, doc_id)

    texts = [chunk.content for chunk in chunks]
    vectors = embedder(texts)
    _ensure_dims(vectors, embed_dims)

    store.write_chunks(collection_id, chunks, vectors)

    tokens_estimated = sum(_word_count(chunk.content) for chunk in chunks)
    return IndexResult(chunks_written=len(chunks), tokens_estimated=tokens_estimated)


def retrieve(
    query: str,
    plan: RetrievePlan,
    store: Store,
    *,
    collection_id: int,
    embedder: Embedder,
    embed_dims: int,
    hybrid_enabled: bool = False,
) -> list[Candidate]:
    if plan.mode == "hybrid" and not hybrid_enabled:
        raise ValueError("Hybrid search disabled for collection")

    vectors = embedder([query])
    _ensure_dims(vectors, embed_dims)
    query_vector = vectors[0]

    if plan.mode == "vector":
        return store.vector_search(
            collection_id,
            query_vector,
            plan.k,
            plan.tags_any,
            plan.tags_all,
        )

    vector_candidates = store.vector_search(
        collection_id,
        query_vector,
        plan.candidate_pool,
        plan.tags_any,
        plan.tags_all,
    )
    fts_candidates = store.fts_search(
        collection_id,
        query,
        plan.candidate_pool,
        plan.tags_any,
        plan.tags_all,
    )
    fused = fuse([vector_candidates, fts_candidates], method="rrf", params={"k": plan.rrf_k})
    return fused[: plan.k]


def fuse(
    candidate_sets: Iterable[Sequence[Candidate]],
    *,
    method: Literal["rrf"] = "rrf",
    params: dict[str, Any] | None = None,
) -> list[Candidate]:
    if method != "rrf":
        raise ValueError("Unsupported fusion method")
    params = params or {}
    rrf_k = int(params.get("k", 60))

    scored: dict[tuple[str, int], dict[str, Any]] = {}
    for candidates in candidate_sets:
        for rank, item in enumerate(candidates, start=1):
            key = (item.doc_id, item.chunk_index)
            scored.setdefault(
                key,
                {
                    "doc_id": item.doc_id,
                    "chunk_index": item.chunk_index,
                    "content": item.content,
                    "metadata": item.metadata,
                    "tags": item.tags,
                    "score": 0.0,
                },
            )
            scored[key]["score"] += 1.0 / (rrf_k + rank)

    merged = list(scored.values())
    merged.sort(key=lambda item: (-item["score"], item["doc_id"], item["chunk_index"]))
    return [
        Candidate(
            doc_id=item["doc_id"],
            chunk_index=item["chunk_index"],
            content=item["content"],
            score=float(item["score"]),
            metadata=item["metadata"],
            tags=tuple(item["tags"]),
        )
        for item in merged
    ]


def rerank(
    query: str,
    candidates: Sequence[Candidate],
    model: Reranker | None = None,
) -> list[Candidate]:
    if model is None:
        return list(candidates)

    scores = list(model(query, candidates))
    if len(scores) != len(candidates):
        raise ValueError("Reranker returned mismatched score count")

    reranked = [
        Candidate(
            doc_id=item.doc_id,
            chunk_index=item.chunk_index,
            content=item.content,
            score=float(score),
            metadata=item.metadata,
            tags=item.tags,
        )
        for item, score in zip(candidates, scores, strict=True)
    ]
    reranked.sort(key=lambda item: (-item.score, item.doc_id, item.chunk_index))
    return reranked


def compress(
    query: str,
    candidates: Sequence[Candidate],
    policy: CompressionPolicy | None = None,
) -> list[dict[str, Any]]:
    if policy is None:
        return [_candidate_payload(item) for item in candidates]

    items = list(candidates)
    if policy.max_chunks is not None:
        items = items[: policy.max_chunks]

    payloads = []
    for item in items:
        content = item.content
        if policy.max_chars is not None and len(content) > policy.max_chars:
            content = content[: policy.max_chars]
        payloads.append({**_candidate_payload(item), "content": content})
    return payloads


def _candidate_payload(candidate: Candidate) -> dict[str, Any]:
    return {
        "doc_id": candidate.doc_id,
        "chunk_index": candidate.chunk_index,
        "content": candidate.content,
        "score": candidate.score,
        "metadata": candidate.metadata,
        "tags": list(candidate.tags),
    }


def _ensure_dims(vectors: Sequence[Sequence[float]], dims: int) -> None:
    if any(len(vector) != dims for vector in vectors):
        raise ValueError("Embedding dimension mismatch")


def _word_count(text: str) -> int:
    return len(text.split())
