import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    openai_base_url: str
    openai_api_key: str
    embed_model: str
    embed_dims: int
    llm_model_default: str
    llm_base_url: str
    llm_api_key: str
    chunk_target_tokens: int
    chunk_overlap_tokens: int
    chunk_max_chars: int
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


def _get_required_int(name: str) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Missing required env var: {name}")
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid int for {name}: {value}") from exc


def _get_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid int for {name}: {value}") from exc


def _get_optional_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid float for {name}: {value}") from exc


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    database_url = _get_env("DATABASE_URL", required=True)
    openai_base_url = _normalize_base_url(_get_env("OPENAI_BASE_URL", required=True))
    openai_api_key = _get_env("OPENAI_API_KEY", "openai")
    embed_model = _get_env("EMBED_MODEL", required=True)
    embed_dims = _get_required_int("EMBED_DIMS")
    llm_model_default = _get_env("LLM_MODEL_DEFAULT", "gpt-4o-mini")
    llm_base_override = _get_optional_env("LLM_BASE_URL")
    llm_base_url = _normalize_base_url(llm_base_override or openai_base_url)
    llm_api_key = _get_optional_env("LLM_API_KEY") or openai_api_key
    chunk_target_tokens = _get_int("CHUNK_TARGET_TOKENS", 600)
    chunk_overlap_tokens = _get_optional_int("CHUNK_OVERLAP_TOKENS")
    overlap_percent = _get_optional_float("CHUNK_OVERLAP_PERCENT")
    overlap_ratio = _get_optional_float("CHUNK_OVERLAP_RATIO")
    if chunk_overlap_tokens is None:
        if overlap_percent is not None:
            if not 0 <= overlap_percent <= 100:
                raise RuntimeError("CHUNK_OVERLAP_PERCENT must be between 0 and 100")
            chunk_overlap_tokens = int(round(chunk_target_tokens * (overlap_percent / 100)))
        elif overlap_ratio is not None:
            if not 0 <= overlap_ratio <= 1:
                raise RuntimeError("CHUNK_OVERLAP_RATIO must be between 0 and 1")
            chunk_overlap_tokens = int(round(chunk_target_tokens * overlap_ratio))
        else:
            chunk_overlap_tokens = 120
    chunk_max_chars = _get_int("CHUNK_MAX_CHARS", 4000)
    if chunk_max_chars <= 0:
        raise RuntimeError("CHUNK_MAX_CHARS must be positive")

    return Settings(
        database_url=database_url,
        openai_base_url=openai_base_url,
        openai_api_key=openai_api_key,
        embed_model=embed_model,
        embed_dims=embed_dims,
        llm_model_default=llm_model_default,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        chunk_target_tokens=chunk_target_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        chunk_max_chars=chunk_max_chars,
        embed_batch_size=_get_int("EMBED_BATCH_SIZE", 64),
        search_candidate_pool=_get_int("SEARCH_CANDIDATE_POOL", 50),
        search_rrf_k=_get_int("SEARCH_RRF_K", 60),
        auto_migrate=_get_bool("RAGD_AUTO_MIGRATE", False),
    )
