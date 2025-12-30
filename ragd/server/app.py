from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import ValidationError

from ragd import auth, core
from ragd.config import Settings, load_settings
from ragd.db import init_pool, get_pool
from ragd.embeddings import build_client, embed_texts_batched
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
from ragd.core.store import PostgresStore

app = FastAPI(title="ragd")

MCP_PROTOCOL_VERSION = "2025-06-18"
MCP_TOOL_SEARCH_NAME = "collections-search"
MCP_TOOL_LIST_NAME = "collections-list"


def _embedder(settings: Settings, client: OpenAI) -> Callable[[list[str]], list[list[float]]]:
    def _call(texts: list[str]) -> list[list[float]]:
        return embed_texts_batched(
            client,
            settings.embed_model,
            texts,
            settings.embed_batch_size,
        )

    return _call


def _chunk_policy(settings: Settings, request: DocumentIngestRequest) -> core.ChunkPolicy:
    target_tokens = (
        settings.chunk_target_tokens
        if request.chunk_target_tokens is None
        else request.chunk_target_tokens
    )
    overlap_tokens = (
        settings.chunk_overlap_tokens
        if request.chunk_overlap_tokens is None
        else request.chunk_overlap_tokens
    )
    max_chars = settings.chunk_max_chars if request.chunk_max_chars is None else request.chunk_max_chars
    return core.ChunkPolicy(
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        max_chars=max_chars,
    )


def _default_system_prompt() -> str:
    return (
        "You answer questions using the provided document chunks. "
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
    app.state.store = PostgresStore(get_pool())
    app.state.embedder = _embedder(settings, app.state.embed_client)

    if settings.auto_migrate:
        schema_path = Path(__file__).parent / "sql" / "schema.sql"
        schema_sql = schema_path.read_text(encoding="utf-8")
        pool = get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
            conn.commit()
        app.state.store.ensure_embedding_index(settings.embed_dims)


@app.get("/v1/health", operation_id="health")
def health() -> dict[str, str]:
    settings: Settings = app.state.settings
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()

    embedder = app.state.embedder
    try:
        vectors = embedder(["ping"])
        if not vectors or len(vectors[0]) != settings.embed_dims:
            raise RuntimeError("Embedding dimension mismatch")
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
    store: PostgresStore = app.state.store
    collection = store.create_collection(
        request.name,
        embed_model=settings.embed_model,
        embed_dims=settings.embed_dims,
        hybrid_enabled=request.hybrid_enabled,
    )
    return CollectionResponse(
        name=collection.name,
        embed_model=collection.embed_model,
        embed_dims=collection.embed_dims,
        hybrid_enabled=collection.hybrid_enabled,
    )


@app.get(
    "/v1/collections",
    response_model=list[CollectionResponse],
    dependencies=[Depends(auth.require_api_key)],
    operation_id="collections-list",
)
def list_collections() -> list[CollectionResponse]:
    store: PostgresStore = app.state.store
    collections = store.list_collections()
    return [
        CollectionResponse(
            name=collection.name,
            embed_model=collection.embed_model,
            embed_dims=collection.embed_dims,
            hybrid_enabled=collection.hybrid_enabled,
        )
        for collection in collections
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
    store: PostgresStore = app.state.store
    collection_row = store.get_collection(collection)
    if not collection_row:
        raise HTTPException(status_code=404, detail="Collection not found")
    settings: Settings = app.state.settings
    try:
        core.ensure_collection_embeddings(
            collection_embed_model=collection_row.embed_model,
            collection_embed_dims=collection_row.embed_dims,
            expected_embed_model=settings.embed_model,
            expected_embed_dims=settings.embed_dims,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store.upsert_document(
        collection_id=collection_row.id,
        doc_id=doc_id,
        title=request.title,
        tags=request.tags,
        metadata=request.metadata,
        source=None,
    )

    policy = _chunk_policy(settings, request)
    chunks = core.chunk(request.content, policy)
    if not chunks:
        raise HTTPException(status_code=400, detail="No content to ingest")

    prepared = core.prepare_chunks(
        doc_id,
        chunks,
        tags=request.tags,
        metadata=request.metadata,
    )
    try:
        result = core.index(
            prepared,
            store,
            collection_id=collection_row.id,
            embedder=app.state.embedder,
            embed_dims=settings.embed_dims,
            ingest_mode=request.ingest_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DocumentIngestResponse(
        chunks_written=result.chunks_written,
        tokens_estimated=result.tokens_estimated,
    )


@app.post(
    "/v1/collections/{collection}/search",
    response_model=SearchResponse,
    dependencies=[Depends(auth.require_api_key)],
    operation_id="collections-search",
)
def search(collection: str, request: SearchRequest) -> SearchResponse:
    store: PostgresStore = app.state.store
    collection_row = store.get_collection(collection)
    if not collection_row:
        raise HTTPException(status_code=404, detail="Collection not found")
    if request.mode == "hybrid" and not collection_row.hybrid_enabled:
        raise HTTPException(status_code=400, detail="Hybrid search disabled for collection")

    settings: Settings = app.state.settings
    try:
        core.ensure_collection_embeddings(
            collection_embed_model=collection_row.embed_model,
            collection_embed_dims=collection_row.embed_dims,
            expected_embed_model=settings.embed_model,
            expected_embed_dims=settings.embed_dims,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    plan = core.RetrievePlan(
        mode=request.mode,
        k=request.k,
        candidate_pool=settings.search_candidate_pool,
        rrf_k=settings.search_rrf_k,
        tags_any=request.tags_any,
        tags_all=request.tags_all,
    )
    try:
        candidates = core.retrieve(
            request.query,
            plan,
            store,
            collection_id=collection_row.id,
            embedder=app.state.embedder,
            embed_dims=settings.embed_dims,
            hybrid_enabled=collection_row.hybrid_enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    results = [
        SearchResult(
            doc_id=item.doc_id,
            chunk_index=item.chunk_index,
            content=item.content,
            score=item.score,
            metadata=item.metadata,
            tags=list(item.tags),
        )
        for item in candidates
    ]
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

    uvicorn.run("ragd.server.app:app", host="0.0.0.0", port=8000, reload=False)


def _mcp_result(request_id: Any, result: dict[str, Any]) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})


def _mcp_error(request_id: Any, code: int, message: str, data: Any | None = None) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "error": error})


def _mcp_tool_schema_search() -> dict[str, Any]:
    return {
        "name": MCP_TOOL_SEARCH_NAME,
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


def _mcp_tool_schema_list() -> dict[str, Any]:
    return {
        "name": MCP_TOOL_LIST_NAME,
        "title": "Collections List",
        "description": "List available collections.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    }


def _mcp_tools_list(request_id: Any) -> JSONResponse:
    result = {
        "tools": [_mcp_tool_schema_search(), _mcp_tool_schema_list()],
        "nextCursor": None,
    }
    return _mcp_result(request_id, result)


def _mcp_tools_call(request_id: Any, params: Any) -> JSONResponse:
    if not isinstance(params, dict):
        return _mcp_error(request_id, -32602, "Invalid params")

    tool_name = params.get("name")
    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        return _mcp_error(request_id, -32602, "Invalid params")

    if tool_name == MCP_TOOL_SEARCH_NAME:
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

    if tool_name == MCP_TOOL_LIST_NAME:
        try:
            collections = list_collections()
        except Exception as exc:  # pragma: no cover - defensive
            return _mcp_error(request_id, -32603, "Internal error", str(exc))

        payload = json.dumps([item.model_dump() for item in collections], ensure_ascii=True)
        return _mcp_result(
            request_id,
            {
                "content": [{"type": "text", "text": payload}],
                "isError": False,
            },
        )

    return _mcp_error(request_id, -32601, "Tool not found")


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
