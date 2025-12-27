from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CollectionCreateRequest(BaseModel):
    name: str
    embed_model: str | None = None
    hybrid_enabled: bool = False


class CollectionResponse(BaseModel):
    name: str
    embed_model: str
    embed_dims: int
    hybrid_enabled: bool


class SegmentInput(BaseModel):
    text: str
    t_start: float | None = None
    t_end: float | None = None
    speaker: str | None = None


class DocumentIngestRequest(BaseModel):
    title: str | None = None
    content: str | list[SegmentInput]
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingest_mode: Literal["replace", "upsert"] = "replace"


class DocumentIngestResponse(BaseModel):
    chunks_written: int
    tokens_estimated: int


class SearchRequest(BaseModel):
    query: str
    k: int = 12
    mode: Literal["vector", "hybrid"] = "vector"
    tags_any: list[str] | None = None
    tags_all: list[str] | None = None


class SearchResult(BaseModel):
    doc_id: str
    chunk_index: int
    content: str
    score: float
    metadata: dict[str, Any]
    tags: list[str]


class SearchResponse(BaseModel):
    results: list[SearchResult]


class AskRequest(SearchRequest):
    llm_model: str | None = None
    system_prompt: str | None = None


class AskSource(BaseModel):
    doc_id: str
    chunk_index: int
    t_start: float | None = None
    t_end: float | None = None
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[AskSource]


class ApiKeyCreateRequest(BaseModel):
    label: str


class ApiKeyCreateResponse(BaseModel):
    secret_once: str
