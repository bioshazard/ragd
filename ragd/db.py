from collections.abc import Generator
from psycopg_pool import ConnectionPool


_pool: ConnectionPool | None = None


def init_pool(database_url: str) -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(database_url, min_size=1, max_size=10)
    return _pool


def get_pool() -> ConnectionPool:
    if _pool is None:
        raise RuntimeError("Database pool not initialized")
    return _pool


def get_conn() -> Generator:
    pool = get_pool()
    with pool.connection() as conn:
        yield conn
