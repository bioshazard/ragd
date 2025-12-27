import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    openai_base_url: str
    openai_api_key: str
    embed_model_default: str
    llm_base_url: str
    llm_api_key: str
    chunk_target_tokens: int
    chunk_overlap_tokens: int
    embed_batch_size: int
    search_candidate_pool: int
    search_rrf_k: int
    auto_migrate: bool


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value or ""


def _get_optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _normalize_base_url(url: str) -> str:
    if url and not url.startswith(("http://", "https://")):
        return f"http://{url}"
    return url


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid int for {name}: {value}") from exc


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    database_url = _get_env("DATABASE_URL", required=True)
    openai_base_url = _normalize_base_url(_get_env("OPENAI_BASE_URL", required=True))
    openai_api_key = _get_env("OPENAI_API_KEY", "openai")
    embed_model_default = _get_env("EMBED_MODEL_DEFAULT", "nomic-embed-text:latest")
    llm_base_override = _get_optional_env("LLM_BASE_URL")
    llm_base_url = _normalize_base_url(llm_base_override or openai_base_url)
    llm_api_key = _get_optional_env("LLM_API_KEY") or openai_api_key

    return Settings(
        database_url=database_url,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        embed_model_default=embed_model_default,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        chunk_target_tokens=_get_int("CHUNK_TARGET_TOKENS", 700),
        chunk_overlap_tokens=_get_int("CHUNK_OVERLAP_TOKENS", 120),
        embed_batch_size=_get_int("EMBED_BATCH_SIZE", 64),
        search_candidate_pool=_get_int("SEARCH_CANDIDATE_POOL", 50),
        search_rrf_k=_get_int("SEARCH_RRF_K", 60),
        auto_migrate=_get_bool("RAGD_AUTO_MIGRATE", False),
    )
