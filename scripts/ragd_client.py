#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _request_json(method: str, url: str, api_key: str | None, payload: dict[str, Any], timeout: int) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc


def _base_url(value: str) -> str:
    base = value.rstrip("/")
    if not base.startswith(("http://", "https://")):
        base = f"http://{base}"
    return base


def _parse_tags(value: str | None) -> list[str]:
    if not value:
        return []
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def _parse_metadata(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError("metadata must be valid JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError("metadata must be a JSON object")
    return data


def _ingest(args: argparse.Namespace) -> int:
    base_url = _base_url(args.base_url)
    api_key = args.api_key
    tags = _parse_tags(args.tags)
    metadata = _parse_metadata(args.metadata)

    paths = sorted(glob.glob(args.paths))
    if not paths:
        raise RuntimeError(f"No files matched: {args.paths}")

    for path in paths:
        doc_id = Path(path).stem
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()

        payload = {
            "title": Path(path).name,
            "content": content,
            "tags": tags,
            "metadata": {**metadata, "source": path},
            "ingest_mode": args.ingest_mode,
        }
        url = f"{base_url}/v1/collections/{args.collection}/documents/{doc_id}"
        response = _request_json("PUT", url, api_key, payload, args.timeout)
        print(json.dumps({"doc_id": doc_id, "result": response}, indent=2))

    return 0


def _search(args: argparse.Namespace) -> int:
    base_url = _base_url(args.base_url)
    api_key = args.api_key
    payload: dict[str, Any] = {
        "query": args.query,
        "k": args.k,
        "mode": args.mode,
    }
    if args.tags_any:
        payload["tags_any"] = _parse_tags(args.tags_any)
    if args.tags_all:
        payload["tags_all"] = _parse_tags(args.tags_all)

    url = f"{base_url}/v1/collections/{args.collection}/search"
    response = _request_json("POST", url, api_key, payload, args.timeout)
    print(json.dumps(response, indent=2))
    return 0


def _ask(args: argparse.Namespace) -> int:
    base_url = _base_url(args.base_url)
    api_key = args.api_key
    payload: dict[str, Any] = {
        "query": args.query,
        "k": args.k,
        "mode": args.mode,
        "llm_model": args.llm_model,
    }
    if args.tags_any:
        payload["tags_any"] = _parse_tags(args.tags_any)
    if args.tags_all:
        payload["tags_all"] = _parse_tags(args.tags_all)

    url = f"{base_url}/v1/collections/{args.collection}/ask"
    response = _request_json("POST", url, api_key, payload, args.timeout)
    print(json.dumps(response, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ragd CLI helper")
    parser.add_argument(
        "--base-url",
        default=os.getenv("RAGD_URL", "http://localhost:8000"),
        help="ragd base URL (env: RAGD_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("RAGD_API_KEY"),
        help="ragd API key (env: RAGD_API_KEY)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="request timeout in seconds",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="ingest transcripts")
    ingest.add_argument("--collection", required=True)
    ingest.add_argument("--paths", required=True, help="glob like ./transcripts/*.txt")
    ingest.add_argument("--tags", default=None, help="comma-separated tags")
    ingest.add_argument("--metadata", default=None, help="JSON object")
    ingest.add_argument("--ingest-mode", default="replace", choices=["replace", "upsert"])
    ingest.set_defaults(func=_ingest)

    search = subparsers.add_parser("search", help="search for top-k chunks")
    search.add_argument("--collection", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--k", type=int, default=12)
    search.add_argument("--mode", default="vector", choices=["vector", "hybrid"])
    search.add_argument("--tags-any", default=None, help="comma-separated tags")
    search.add_argument("--tags-all", default=None, help="comma-separated tags")
    search.set_defaults(func=_search)

    ask = subparsers.add_parser("ask", help="RAG answer with citations")
    ask.add_argument("--collection", required=True)
    ask.add_argument("--query", required=True)
    ask.add_argument("--k", type=int, default=6)
    ask.add_argument("--mode", default="vector", choices=["vector", "hybrid"])
    ask.add_argument("--llm-model", required=True)
    ask.add_argument("--tags-any", default=None, help="comma-separated tags")
    ask.add_argument("--tags-all", default=None, help="comma-separated tags")
    ask.set_defaults(func=_ask)

    args = parser.parse_args()
    if not args.api_key:
        raise RuntimeError("Missing API key: set --api-key or RAGD_API_KEY")

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
