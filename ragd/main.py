from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from openai import OpenAI
from psycopg.types.json import Json
from pydantic import ValidationError

from ragd import auth
from ragd.chunking import build_units, chunk_units
from ragd.config import Settings, load_settings
from ragd.db import init_pool, get_pool
from ragd.embeddings import build_client, embed_texts, probe_embed_dims
from ragd.schemas import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    AskRequest,
    AskResponse,
    AskSource,
    CollectionCreateRequest,
    CollectionResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

app = FastAPI(title="ragd")

MCP_PROTOCOL_VERSION = "2025-06-18"
MCP_TOOL_NAME = "collections-search"


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(value) for value in vector) + "]"


def _get_collection(name: str) -> dict[str, Any]:
    pool = get_pool()
    with pool.connection() as conn:
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
        raise HTTPException(status_code=404, detail="Collection not found")
    return {
        "id": row[0],
        "name": row[1],
        "embed_model": row[2],
        "embed_dims": row[3],
        "hybrid_enabled": row[4],
    }


def _embed_texts(settings: Settings, client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    batch_size = settings.embed_batch_size
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(embed_texts(client, model, batch))
    return vectors


def _ensure_dims(vectors: list[list[float]], dims: int) -> None:
    if any(len(vector) != dims for vector in vectors):
        raise HTTPException(status_code=400, detail="Embedding dimension mismatch")


def _upsert_document(
    collection_id: int,
    doc_id: str,
    request: DocumentIngestRequest,
) -> None:
    pool = get_pool()
    with pool.connection() as conn:
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
                    request.title,
                    request.tags,
                    Json(request.metadata),
                    None,
                ),
            )
        conn.commit()


def _delete_chunks(collection_id: int, doc_id: str) -> None:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE collection_id = %s AND doc_id = %s",
                (collection_id, doc_id),
            )
        conn.commit()


def _write_chunks(
    collection_id: int,
    doc_id: str,
    chunks: list[dict[str, Any]],
) -> None:
    pool = get_pool()
    with pool.connection() as conn:
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
                        doc_id,
                        chunk["chunk_index"],
                        chunk["content"],
                        chunk["tags"],
                        Json(chunk["metadata"]),
                        chunk["embedding"],
                    )
                    for chunk in chunks
                ],
            )
        conn.commit()


def _build_chunk_payloads(
    doc_id: str,
    request: DocumentIngestRequest,
    settings: Settings,
) -> tuple[list[dict[str, Any]], int]:
    content = request.content
    if isinstance(content, list):
        segment_dicts = [segment.model_dump() for segment in content]
        units = build_units(segment_dicts)
    else:
        units = build_units(content)

    raw_chunks = chunk_units(
        units,
        target_tokens=settings.chunk_target_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )

    chunks: list[dict[str, Any]] = []
    tokens_estimated = 0
    for idx, chunk in enumerate(raw_chunks):
        metadata = dict(request.metadata)
        metadata.update(chunk.metadata)
        metadata["episode_id"] = doc_id
        metadata["chunk_index"] = idx
        tokens_estimated += len(chunk.text.split())

        chunks.append(
            {
                "chunk_index": idx,
                "content": chunk.text,
                "tags": request.tags,
                "metadata": metadata,
            }
        )

    return chunks, tokens_estimated


def _vector_search(
    collection_id: int,
    query_vector: list[float],
    k: int,
    tags_any: list[str] | None,
    tags_all: list[str] | None,
) -> list[SearchResult]:
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
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    results = []
    for row in rows:
        results.append(
            SearchResult(
                doc_id=row[0],
                chunk_index=row[1],
                content=row[2],
                score=float(row[3]),
                metadata=row[4] or {},
                tags=row[5] or [],
            )
        )
    return results


def _fts_search(
    collection_id: int,
    query: str,
    candidate_pool: int,
    tags_any: list[str] | None,
    tags_all: list[str] | None,
) -> list[tuple[str, int, float, str, dict[str, Any], list[str]]]:
    clauses = ["collection_id = %s", "fts @@ plainto_tsquery('english', %s)"]
    where_params: list[Any] = [collection_id, query]
    if tags_any:
        clauses.append("tags && %s")
        where_params.append(tags_any)
    if tags_all:
        clauses.append("tags @> %s")
        where_params.append(tags_all)

    where = " AND ".join(clauses)
    params: list[Any] = [query]
    params.extend(where_params)
    params.append(candidate_pool)

    sql = f"""
        SELECT doc_id, chunk_index, ts_rank_cd(fts, plainto_tsquery('english', %s)) AS rank, content, metadata, tags
        FROM chunks
        WHERE {where}
        ORDER BY rank DESC
        LIMIT %s
    """

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return rows


def _hybrid_search(
    settings: Settings,
    collection_id: int,
    query_vector: list[float],
    query_text: str,
    k: int,
    tags_any: list[str] | None,
    tags_all: list[str] | None,
) -> list[SearchResult]:
    vector_candidates = _vector_search(
        collection_id,
        query_vector,
        settings.search_candidate_pool,
        tags_any,
        tags_all,
    )
    fts_candidates = _fts_search(
        collection_id,
        query_text,
        settings.search_candidate_pool,
        tags_any,
        tags_all,
    )

    scored: dict[tuple[str, int], dict[str, Any]] = {}
    rrf_k = settings.search_rrf_k

    for idx, item in enumerate(vector_candidates, start=1):
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
        scored[key]["score"] += 1.0 / (rrf_k + idx)

    for idx, row in enumerate(fts_candidates, start=1):
        key = (row[0], row[1])
        scored.setdefault(
            key,
            {
                "doc_id": row[0],
                "chunk_index": row[1],
                "content": row[3],
                "metadata": row[4] or {},
                "tags": row[5] or [],
                "score": 0.0,
            },
        )
        scored[key]["score"] += 1.0 / (rrf_k + idx)

    merged = sorted(scored.values(), key=lambda item: item["score"], reverse=True)
    return [
        SearchResult(
            doc_id=item["doc_id"],
            chunk_index=item["chunk_index"],
            content=item["content"],
            score=float(item["score"]),
            metadata=item["metadata"],
            tags=item["tags"],
        )
        for item in merged[:k]
    ]


def _default_system_prompt() -> str:
    return (
        "You answer questions using the provided transcript chunks. "
        "Cite sources inline like [doc_id @ t_start-t_end] or [doc_id] if time is missing. "
        "If the answer is not in the sources, say you don't know."
    )


@app.on_event("startup")
def startup() -> None:
    settings = load_settings()
    init_pool(settings.database_url)
    app.state.settings = settings
    app.state.embed_client = build_client(settings.openai_base_url, settings.openai_api_key)
    app.state.llm_client = build_client(settings.llm_base_url, settings.llm_api_key)

    if settings.auto_migrate:
        schema_path = Path(__file__).parent / "sql" / "schema.sql"
        schema_sql = schema_path.read_text(encoding="utf-8")
        pool = get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
            conn.commit()


@app.get("/v1/health", operation_id="health")
def health() -> dict[str, str]:
    settings: Settings = app.state.settings
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()

    client: OpenAI = app.state.embed_client
    try:
        embed_texts(client, settings.embed_model_default, ["ping"])
    except Exception as exc:  # pragma: no cover - depends on external service
        raise HTTPException(status_code=503, detail=f"Embedding backend error: {exc}") from exc

    return {"status": "ok"}


@app.post(
    "/v1/collections",
    response_model=CollectionResponse,
    dependencies=[Depends(auth.require_api_key)],
    operation_id="collections-create",
)
def create_collection(request: CollectionCreateRequest) -> CollectionResponse:
    settings: Settings = app.state.settings
    embed_model = request.embed_model or settings.embed_model_default
    client: OpenAI = app.state.embed_client
    embed_dims = probe_embed_dims(client, embed_model)

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO collections (name, embed_model, embed_dims, hybrid_enabled)
                VALUES (%s, %s, %s, %s)
                RETURNING name, embed_model, embed_dims, hybrid_enabled
                """,
                (request.name, embed_model, embed_dims, request.hybrid_enabled),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=500, detail="Failed to create collection")

    return CollectionResponse(
        name=row[0],
        embed_model=row[1],
        embed_dims=row[2],
        hybrid_enabled=row[3],
    )


@app.get(
    "/v1/collections",
    response_model=list[CollectionResponse],
    dependencies=[Depends(auth.require_api_key)],
    operation_id="collections-list",
)
def list_collections() -> list[CollectionResponse]:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name, embed_model, embed_dims, hybrid_enabled FROM collections ORDER BY name"
            )
            rows = cur.fetchall()

    return [
        CollectionResponse(
            name=row[0],
            embed_model=row[1],
            embed_dims=row[2],
            hybrid_enabled=row[3],
        )
        for row in rows
    ]


@app.put(
    "/v1/collections/{collection}/documents/{doc_id}",
    response_model=DocumentIngestResponse,
    dependencies=[Depends(auth.require_api_key)],
    operation_id="documents-ingest",
)
def ingest_document(
    collection: str, doc_id: str, request: DocumentIngestRequest
) -> DocumentIngestResponse:
    collection_row = _get_collection(collection)
    settings: Settings = app.state.settings
    embed_model = collection_row["embed_model"]
    embed_dims = collection_row["embed_dims"]

    _upsert_document(collection_row["id"], doc_id, request)

    if request.ingest_mode == "replace":
        _delete_chunks(collection_row["id"], doc_id)

    chunks, tokens_estimated = _build_chunk_payloads(doc_id, request, settings)
    if not chunks:
        raise HTTPException(status_code=400, detail="No content to ingest")

    texts = [chunk["content"] for chunk in chunks]
    vectors = _embed_texts(settings, app.state.embed_client, embed_model, texts)
    _ensure_dims(vectors, embed_dims)

    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk["embedding"] = _vector_literal(vector)

    _write_chunks(collection_row["id"], doc_id, chunks)

    return DocumentIngestResponse(
        chunks_written=len(chunks),
        tokens_estimated=tokens_estimated,
    )


@app.post(
    "/v1/collections/{collection}/search",
    response_model=SearchResponse,
    dependencies=[Depends(auth.require_api_key)],
    operation_id="collections-search",
)
def search(collection: str, request: SearchRequest) -> SearchResponse:
    collection_row = _get_collection(collection)
    if request.mode == "hybrid" and not collection_row["hybrid_enabled"]:
        raise HTTPException(status_code=400, detail="Hybrid search disabled for collection")

    settings: Settings = app.state.settings
    vectors = _embed_texts(settings, app.state.embed_client, collection_row["embed_model"], [request.query])
    _ensure_dims(vectors, collection_row["embed_dims"])

    if request.mode == "vector":
        results = _vector_search(
            collection_row["id"],
            vectors[0],
            request.k,
            request.tags_any,
            request.tags_all,
        )
    else:
        results = _hybrid_search(
            settings,
            collection_row["id"],
            vectors[0],
            request.query,
            request.k,
            request.tags_any,
            request.tags_all,
        )

    return SearchResponse(results=results)


@app.post(
    "/v1/collections/{collection}/ask",
    response_model=AskResponse,
    dependencies=[Depends(auth.require_api_key)],
    operation_id="collections-ask",
)
def ask(collection: str, request: AskRequest) -> AskResponse:
    search_request = SearchRequest(
        query=request.query,
        k=request.k,
        mode=request.mode,
        tags_any=request.tags_any,
        tags_all=request.tags_all,
    )
    search_response = search(collection, search_request)

    if not search_response.results:
        return AskResponse(answer="No sources found.", sources=[])

    context_lines = []
    sources = []
    for result in search_response.results:
        meta = result.metadata or {}
        t_start = meta.get("t_start")
        t_end = meta.get("t_end")
        time_fragment = ""
        if t_start is not None or t_end is not None:
            t_start_text = f"{t_start:.2f}" if isinstance(t_start, (int, float)) else str(t_start)
            t_end_text = f"{t_end:.2f}" if isinstance(t_end, (int, float)) else str(t_end)
            time_fragment = f" @ {t_start_text}-{t_end_text}"

        label = f"{result.doc_id}{time_fragment}"
        context_lines.append(f"[{label}] {result.content}")
        sources.append(
            AskSource(
                doc_id=result.doc_id,
                chunk_index=result.chunk_index,
                t_start=t_start if isinstance(t_start, (int, float)) else None,
                t_end=t_end if isinstance(t_end, (int, float)) else None,
                excerpt=result.content,
            )
        )

    system_prompt = request.system_prompt or _default_system_prompt()
    user_prompt = "\n\n".join(
        [
            "Sources:",
            "\n".join(context_lines),
            f"Question: {request.query}",
        ]
    )

    settings: Settings = app.state.settings
    llm_model = request.llm_model or settings.llm_model_default
    if not llm_model:
        raise HTTPException(status_code=400, detail="Missing llm_model")

    client: OpenAI = app.state.llm_client
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = response.choices[0].message.content or ""

    return AskResponse(answer=answer, sources=sources)


@app.post("/v1/api-keys", response_model=ApiKeyCreateResponse, operation_id="api-keys-create")
def create_api_key(request: ApiKeyCreateRequest, authorized: bool = Depends(auth.optional_api_key)) -> ApiKeyCreateResponse:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM api_keys WHERE revoked_at IS NULL")
            total = cur.fetchone()[0]

    if total > 0 and not authorized:
        raise HTTPException(status_code=403, detail="Admin auth required")

    secret, _ = auth.create_api_key(request.label)
    return ApiKeyCreateResponse(secret_once=secret)


def run() -> None:
    import uvicorn

    uvicorn.run("ragd.main:app", host="0.0.0.0", port=8000, reload=False)


def _mcp_result(request_id: Any, result: dict[str, Any]) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})


def _mcp_error(request_id: Any, code: int, message: str, data: Any | None = None) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "error": error})


def _mcp_tool_schema() -> dict[str, Any]:
    return {
        "name": MCP_TOOL_NAME,
        "title": "Collection Search",
        "description": "Run a vector or hybrid search against a collection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "collection": {
                    "type": "string",
                    "description": "Collection name to search.",
                },
                "query": {
                    "type": "string",
                    "description": "Search query text.",
                },
                "k": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 12,
                    "description": "Number of results to return.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["vector", "hybrid"],
                    "default": "vector",
                    "description": "Search mode to use.",
                },
                "tags_any": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Match any of these tags.",
                },
                "tags_all": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Require all of these tags.",
                },
            },
            "required": ["collection", "query"],
        },
    }


def _mcp_tools_list(request_id: Any) -> JSONResponse:
    result = {
        "tools": [_mcp_tool_schema()],
        "nextCursor": None,
    }
    return _mcp_result(request_id, result)


def _mcp_tools_call(request_id: Any, params: Any) -> JSONResponse:
    if not isinstance(params, dict):
        return _mcp_error(request_id, -32602, "Invalid params")

    tool_name = params.get("name")
    if tool_name != MCP_TOOL_NAME:
        return _mcp_error(request_id, -32601, "Tool not found")

    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        return _mcp_error(request_id, -32602, "Invalid params")

    collection = arguments.get("collection")
    query = arguments.get("query")
    if not collection or not query:
        return _mcp_error(request_id, -32602, "Missing required arguments")

    try:
        search_request = SearchRequest(
            query=query,
            k=arguments.get("k", 12),
            mode=arguments.get("mode", "vector"),
            tags_any=arguments.get("tags_any"),
            tags_all=arguments.get("tags_all"),
        )
    except ValidationError as exc:
        return _mcp_error(request_id, -32602, "Invalid params", exc.errors())

    try:
        search_response = search(collection, search_request)
    except HTTPException as exc:
        return _mcp_error(request_id, -32000, str(exc.detail))
    except Exception as exc:  # pragma: no cover - defensive
        return _mcp_error(request_id, -32603, "Internal error", str(exc))

    payload = json.dumps(search_response.model_dump(), ensure_ascii=True)
    return _mcp_result(
        request_id,
        {
            "content": [{"type": "text", "text": payload}],
            "isError": False,
        },
    )


@app.get(
    "/mcp",
    dependencies=[Depends(auth.require_api_key)],
    include_in_schema=False,
)
def mcp_get() -> Response:
    return Response(status_code=200, media_type="text/event-stream")


@app.post(
    "/mcp",
    dependencies=[Depends(auth.require_api_key)],
    operation_id="mcp",
)
async def mcp_post(request: Request) -> Response:
    try:
        payload = await request.json()
    except Exception:
        return _mcp_error(None, -32700, "Parse error")

    if isinstance(payload, list):
        return _mcp_error(None, -32600, "Batch requests not supported")
    if not isinstance(payload, dict):
        return _mcp_error(None, -32600, "Invalid request")

    request_id = payload.get("id")
    if payload.get("jsonrpc") != "2.0":
        return _mcp_error(request_id, -32600, "Invalid JSON-RPC version")

    method = payload.get("method")
    if method == "initialize":
        result = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "ragd", "version": "0.1.0"},
        }
        if request_id is None:
            return Response(status_code=204)
        return _mcp_result(request_id, result)

    if method == "notifications/initialized":
        return Response(status_code=204)

    if method == "ping":
        if request_id is None:
            return Response(status_code=204)
        return _mcp_result(request_id, {})

    if method == "tools/list":
        if request_id is None:
            return Response(status_code=204)
        return _mcp_tools_list(request_id)

    if method == "tools/call":
        if request_id is None:
            return Response(status_code=204)
        return _mcp_tools_call(request_id, payload.get("params"))

    return _mcp_error(request_id, -32601, "Method not found")
