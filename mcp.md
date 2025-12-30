# MCP usage (restish)

This guide shows how to call the `/mcp` endpoint with `restish` using JSON-RPC.

## Prereqs

- `ragd` running at `http://127.0.0.1:8000`.
- `restish local --help` shows the `local` API.
- Set `RAGD_API_KEY` and pass `Authorization: Bearer $RAGD_API_KEY`.

## 1) Initialize

```bash
cat <<'JSON' | restish local mcp \
  -H "Authorization: Bearer $RAGD_API_KEY" \
  --rsh-raw --rsh-verbose
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-06-18",
    "capabilities": {},
    "clientInfo": { "name": "restish", "version": "1.0" }
  }
}
JSON
```

## 2) List tools

```bash
cat <<'JSON' | restish local mcp \
  -H "Authorization: Bearer $RAGD_API_KEY" \
  --rsh-raw --rsh-verbose
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list"
}
JSON
```

## 3) Call collections-search

```bash
cat <<'JSON' | restish local mcp \
  -H "Authorization: Bearer $RAGD_API_KEY" \
  --rsh-raw --rsh-verbose
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "collections-search",
    "arguments": {
      "collection": "docs",
      "query": "how we onboard new hosts",
      "k": 8,
      "mode": "vector"
    }
  }
}
JSON
```

## 4) Call collections-list

```bash
cat <<'JSON' | restish local mcp \
  -H "Authorization: Bearer $RAGD_API_KEY" \
  --rsh-raw --rsh-verbose
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/call",
  "params": {
    "name": "collections-list",
    "arguments": {}
  }
}
JSON
```
