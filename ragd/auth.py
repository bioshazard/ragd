import hashlib
import hmac
import secrets
from typing import Optional

from fastapi import HTTPException, Request
from psycopg import Binary

from ragd.db import get_pool


def hash_key(secret: str) -> bytes:
    return hashlib.sha256(secret.encode("utf-8")).digest()


def verify_secret(secret: str, stored_hashes: list[bytes]) -> bool:
    candidate = hash_key(secret)
    return any(hmac.compare_digest(candidate, stored) for stored in stored_hashes)


def generate_api_key() -> tuple[str, bytes]:
    secret = f"ragd_{secrets.token_urlsafe(32)}"
    return secret, hash_key(secret)


def extract_bearer_token(request: Request) -> Optional[str]:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    return auth.split("Bearer ", 1)[1].strip()


def require_api_key(request: Request) -> None:
    token = extract_bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT key_hash FROM api_keys WHERE revoked_at IS NULL")
            rows = cur.fetchall()

    stored_hashes = [row[0] for row in rows]
    if not stored_hashes or not verify_secret(token, stored_hashes):
        raise HTTPException(status_code=403, detail="Invalid API key")


def optional_api_key(request: Request) -> bool:
    token = extract_bearer_token(request)
    if not token:
        return False

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT key_hash FROM api_keys WHERE revoked_at IS NULL")
            rows = cur.fetchall()

    stored_hashes = [row[0] for row in rows]
    return bool(stored_hashes and verify_secret(token, stored_hashes))


def create_api_key(label: str) -> tuple[str, bytes]:
    secret, key_hash = generate_api_key()
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO api_keys (label, key_hash) VALUES (%s, %s)",
                (label, Binary(key_hash)),
            )
        conn.commit()
    return secret, key_hash
