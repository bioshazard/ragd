from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from ragd.core import (  # noqa: E402
    Candidate,
    ChunkPolicy,
    ChunkRecord,
    RetrievePlan,
    chunk,
    index,
    prepare_chunks,
    retrieve,
)


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


class ToyEmbedder:
    def __init__(self, vocab: list[str]) -> None:
        self._index = {term: idx for idx, term in enumerate(vocab)}
        self.dims = len(vocab)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        for token in tokenize(text):
            idx = self._index.get(token)
            if idx is not None:
                vector[idx] += 1.0
        return vector


class InMemoryStore:
    def __init__(self) -> None:
        self._records: list[dict[str, object]] = []

    def delete_chunks(self, collection_id: int, doc_id: str) -> None:
        self._records = [
            record
            for record in self._records
            if not (
                record["collection_id"] == collection_id
                and record["doc_id"] == doc_id
            )
        ]

    def write_chunks(
        self,
        collection_id: int,
        chunks: list[ChunkRecord],
        embeddings: list[list[float]],
    ) -> None:
        for chunk, vector in zip(chunks, embeddings, strict=True):
            self._records.append(
                {
                    "collection_id": collection_id,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "tags": set(chunk.tags),
                    "metadata": chunk.metadata,
                    "embedding": vector,
                }
            )

    def vector_search(
        self,
        collection_id: int,
        query_vector: list[float],
        k: int,
        tags_any: list[str] | None,
        tags_all: list[str] | None,
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        for record in self._records:
            if record["collection_id"] != collection_id:
                continue
            tags = record["tags"]
            if tags_any and not tags.intersection(tags_any):
                continue
            if tags_all and not set(tags_all).issubset(tags):
                continue
            distance = cosine_distance(record["embedding"], query_vector)
            candidates.append(
                Candidate(
                    doc_id=record["doc_id"],
                    chunk_index=record["chunk_index"],
                    content=record["content"],
                    score=distance,
                    metadata=record["metadata"],
                    tags=tuple(sorted(tags)),
                )
            )
        candidates.sort(key=lambda item: (item.score, item.doc_id, item.chunk_index))
        return candidates[:k]

    def fts_search(
        self,
        collection_id: int,
        query_text: str,
        candidate_pool: int,
        tags_any: list[str] | None,
        tags_all: list[str] | None,
    ) -> list[Candidate]:
        query_tokens = tokenize(query_text)
        candidates: list[Candidate] = []
        for record in self._records:
            if record["collection_id"] != collection_id:
                continue
            tags = record["tags"]
            if tags_any and not tags.intersection(tags_any):
                continue
            if tags_all and not set(tags_all).issubset(tags):
                continue
            content_tokens = tokenize(record["content"])
            rank = sum(content_tokens.count(token) for token in query_tokens)
            if rank <= 0:
                continue
            candidates.append(
                Candidate(
                    doc_id=record["doc_id"],
                    chunk_index=record["chunk_index"],
                    content=record["content"],
                    score=float(rank),
                    metadata=record["metadata"],
                    tags=tuple(sorted(tags)),
                )
            )
        candidates.sort(key=lambda item: (-item.score, item.doc_id, item.chunk_index))
        return candidates[:candidate_pool]


def load_cases(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_chunking_cases(cases: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    results: dict[str, list[dict[str, object]]] = {}
    for case in cases["chunking"]:
        policy = ChunkPolicy(**case["policy"])
        if "segments" in case:
            content = case["segments"]
        else:
            content = case["text"]
        chunks = chunk(content, policy)
        results[case["name"]] = [
            {"content": chunk.content, "metadata": chunk.metadata} for chunk in chunks
        ]
    return results


def build_store(cases: dict[str, object], embedder: ToyEmbedder) -> InMemoryStore:
    store = InMemoryStore()
    policy = ChunkPolicy(**cases["ingest_policy"])
    for doc in cases["documents"]:
        chunks = chunk(doc["content"], policy)
        prepared = prepare_chunks(
            doc["doc_id"],
            chunks,
            tags=doc.get("tags") or [],
            metadata=doc.get("metadata") or {},
        )
        index(
            prepared,
            store,
            collection_id=1,
            embedder=embedder,
            embed_dims=embedder.dims,
            ingest_mode="upsert",
        )
    return store


def run_retrieval_cases(
    cases: dict[str, object],
    store: InMemoryStore,
    embedder: ToyEmbedder,
) -> dict[str, list[dict[str, object]]]:
    results: dict[str, list[dict[str, object]]] = {}
    for case in cases["retrieval"]:
        plan = RetrievePlan(
            mode=case["mode"],
            k=case["k"],
            candidate_pool=case.get("candidate_pool", 50),
            rrf_k=case.get("rrf_k", 60),
            tags_any=case.get("tags_any"),
            tags_all=case.get("tags_all"),
        )
        candidates = retrieve(
            case["query"],
            plan,
            store,
            collection_id=1,
            embedder=embedder,
            embed_dims=embedder.dims,
            hybrid_enabled=case["mode"] == "hybrid",
        )
        results[case["name"]] = [
            {"doc_id": item.doc_id, "chunk_index": item.chunk_index} for item in candidates
        ]
    return results


def generate_golden(cases_path: Path, golden_path: Path) -> None:
    cases = load_cases(cases_path)
    embedder = ToyEmbedder(cases["embedder"]["vocab"])
    store = build_store(cases, embedder)
    payload = {
        "version": cases["version"],
        "chunking": run_chunking_cases(cases),
        "retrieval": run_retrieval_cases(cases, store, embedder),
    }
    golden_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def compare_results(label: str, expected: object, actual: object) -> list[str]:
    if expected == actual:
        return []
    return [f"{label} mismatch: expected {expected}, got {actual}"]


def check_conformance(cases_path: Path, golden_path: Path) -> int:
    cases = load_cases(cases_path)
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    if cases.get("version") != golden.get("version"):
        print("Version mismatch between cases and golden.", file=sys.stderr)
        return 1

    embedder = ToyEmbedder(cases["embedder"]["vocab"])
    store = build_store(cases, embedder)

    errors: list[str] = []
    errors.extend(compare_results("chunking", golden["chunking"], run_chunking_cases(cases)))
    errors.extend(compare_results("retrieval", golden["retrieval"], run_retrieval_cases(cases, store, embedder)))

    if errors:
        for line in errors:
            print(line, file=sys.stderr)
        return 1

    print("Conformance OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ragd conformance runner")
    parser.add_argument(
        "--cases",
        default=str(ROOT / "conformance" / "cases.json"),
        help="Path to conformance cases JSON",
    )
    parser.add_argument(
        "--golden",
        default=str(ROOT / "conformance" / "golden.json"),
        help="Path to golden results JSON",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate golden results from cases",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases)
    golden_path = Path(args.golden)
    if args.generate:
        generate_golden(cases_path, golden_path)
        print(f"Wrote {golden_path}")
        return 0
    return check_conformance(cases_path, golden_path)


if __name__ == "__main__":
    raise SystemExit(main())
