from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from psycopg.types.json import Json
from psycopg_pool import ConnectionPool

from ragd.canon import Candidate, ChunkRecord


@dataclass(frozen=True)
class Collection:
    id: int
    name: str
    embed_model: str
    embed_dims: int
    hybrid_enabled: bool


class PostgresStore:
    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool

    def get_collection(self, name: str) -> Collection | None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, embed_model, embed_dims, hybrid_enabled
                    FROM collections
                    WHERE name = %s
                    """,
                    (name,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return Collection(
            id=row[0],
            name=row[1],
            embed_model=row[2],
            embed_dims=row[3],
            hybrid_enabled=row[4],
        )

    def create_collection(
        self,
        name: str,
        *,
        embed_model: str,
        embed_dims: int,
        hybrid_enabled: bool,
    ) -> Collection:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO collections (name, embed_model, embed_dims, hybrid_enabled)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, name, embed_model, embed_dims, hybrid_enabled
                    """,
                    (name, embed_model, embed_dims, hybrid_enabled),
                )
                row = cur.fetchone()
            conn.commit()

        if not row:
            raise RuntimeError("Failed to create collection")
        return Collection(
            id=row[0],
            name=row[1],
            embed_model=row[2],
            embed_dims=row[3],
            hybrid_enabled=row[4],
        )

    def list_collections(self) -> list[Collection]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, embed_model, embed_dims, hybrid_enabled FROM collections ORDER BY name"
                )
                rows = cur.fetchall()
        return [
            Collection(
                id=row[0],
                name=row[1],
                embed_model=row[2],
                embed_dims=row[3],
                hybrid_enabled=row[4],
            )
            for row in rows
        ]

    def upsert_document(
        self,
        *,
        collection_id: int,
        doc_id: str,
        title: str | None,
        tags: Sequence[str],
        metadata: dict[str, Any],
        source: str | None,
    ) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (collection_id, doc_id, title, tags, metadata, source)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (collection_id, doc_id)
                    DO UPDATE SET
                        title = EXCLUDED.title,
                        tags = EXCLUDED.tags,
                        metadata = EXCLUDED.metadata,
                        source = EXCLUDED.source,
                        updated_at = NOW()
                    """,
                    (
                        collection_id,
                        doc_id,
                        title,
                        list(tags),
                        Json(metadata),
                        source,
                    ),
                )
            conn.commit()

    def delete_chunks(self, collection_id: int, doc_id: str) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chunks WHERE collection_id = %s AND doc_id = %s",
                    (collection_id, doc_id),
                )
            conn.commit()

    def write_chunks(
        self,
        collection_id: int,
        chunks: Sequence[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks (
                        collection_id, doc_id, chunk_index, content, tags, metadata, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (collection_id, doc_id, chunk_index)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        tags = EXCLUDED.tags,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                    [
                        (
                            collection_id,
                            chunk.doc_id,
                            chunk.chunk_index,
                            chunk.content,
                            list(chunk.tags),
                            Json(chunk.metadata),
                            _vector_literal(vector),
                        )
                        for chunk, vector in zip(chunks, embeddings, strict=True)
                    ],
                )
            conn.commit()

    def vector_search(
        self,
        collection_id: int,
        query_vector: Sequence[float],
        k: int,
        tags_any: list[str] | None,
        tags_all: list[str] | None,
    ) -> list[Candidate]:
        clauses = ["collection_id = %s"]
        where_params: list[Any] = [collection_id]
        if tags_any:
            clauses.append("tags && %s")
            where_params.append(tags_any)
        if tags_all:
            clauses.append("tags @> %s")
            where_params.append(tags_all)

        where = " AND ".join(clauses)
        vector_param = _vector_literal(query_vector)
        params: list[Any] = [vector_param]
        params.extend(where_params)
        params.append(vector_param)
        params.append(k)

        sql = f"""
            SELECT doc_id, chunk_index, content, embedding <=> %s::vector AS distance, metadata, tags
            FROM chunks
            WHERE {where}
            ORDER BY embedding <=> %s::vector, doc_id, chunk_index
            LIMIT %s
        """

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [
            Candidate(
                doc_id=row[0],
                chunk_index=row[1],
                content=row[2],
                score=float(row[3]),
                metadata=row[4] or {},
                tags=tuple(row[5] or []),
            )
            for row in rows
        ]

    def fts_search(
        self,
        collection_id: int,
        query_text: str,
        candidate_pool: int,
        tags_any: list[str] | None,
        tags_all: list[str] | None,
    ) -> list[Candidate]:
        clauses = ["collection_id = %s", "fts @@ plainto_tsquery('english', %s)"]
        where_params: list[Any] = [collection_id, query_text]
        if tags_any:
            clauses.append("tags && %s")
            where_params.append(tags_any)
        if tags_all:
            clauses.append("tags @> %s")
            where_params.append(tags_all)

        where = " AND ".join(clauses)
        params: list[Any] = [query_text]
        params.extend(where_params)
        params.append(candidate_pool)

        sql = f"""
            SELECT doc_id, chunk_index, ts_rank_cd(fts, plainto_tsquery('english', %s)) AS rank, content, metadata, tags
            FROM chunks
            WHERE {where}
            ORDER BY rank DESC, doc_id, chunk_index
            LIMIT %s
        """

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [
            Candidate(
                doc_id=row[0],
                chunk_index=row[1],
                content=row[3],
                score=float(row[2]),
                metadata=row[4] or {},
                tags=tuple(row[5] or []),
            )
            for row in rows
        ]

    def ensure_embedding_index(self, embed_dims: int) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT CASE WHEN a.atttypmod > 0 THEN a.atttypmod - 4 ELSE NULL END
                    FROM pg_attribute a
                    WHERE a.attrelid = 'chunks'::regclass
                      AND a.attname = 'embedding'
                      AND a.attnum > 0
                      AND NOT a.attisdropped
                    """
                )
                row = cur.fetchone()
                current_dim = row[0] if row else None
                if current_dim is None or current_dim != embed_dims:
                    cur.execute(
                        f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({embed_dims});"
                    )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
                )
            conn.commit()


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(repr(value) for value in vector) + "]"
