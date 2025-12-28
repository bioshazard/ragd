# restish usage

This guide mirrors the README flow (ingest → search → ask) using `restish`.

## Prereqs

- `ragd` running at `http://127.0.0.1:8000`.
- `restish local --help` shows the `local` API.
- If your server is not on 127.0.0.1:8000, add `--rsh-server http://host:port` to each command.

## 1) Create an API key

If no API keys exist yet, this endpoint is open:

```bash
restish local api-keys-create label:"local"
```

Export the secret:

```bash
export RAGD_API_KEY="<secret_once>"
```

## 2) Create a collection

```bash
restish local collections-create \
  -H "Authorization: Bearer $RAGD_API_KEY" \
  name:podcast \
  embed_model:"nomic-embed-text:latest" \
  hybrid_enabled:true
```

## 3) Ingest transcripts (folder of .txt)

Chunking notes:

- ragd splits text into word-based chunks with overlap (defaults: `CHUNK_TARGET_TOKENS=700`, `CHUNK_OVERLAP_TOKENS=120`).
- If you have timestamps, send segment arrays (`{text,t_start,t_end}`) so chunks carry time metadata for citations.

This loop JSON-escapes each file so multiline content is safe:

```bash
for path in ./transcripts/*.txt; do
  doc_id=$(basename "$path" .txt)
  content_json=$(python - <<'PY' "$path"
import json,sys
text=open(sys.argv[1], encoding="utf-8").read()
print(json.dumps(text))
PY
)
  restish local documents-ingest podcast "$doc_id" \
    -H "Authorization: Bearer $RAGD_API_KEY" \
    "content:$content_json" \
    ingest_mode:replace \
    tags:[podcast] \
    metadata.source:"$path" \
    title:"$doc_id"
done
```

## 4) Retrieve top-k chunks

```bash
restish local collections-search podcast \
  -H "Authorization: Bearer $RAGD_API_KEY" \
  query:"how we onboard new hosts" \
  k:8 \
  mode:vector
```

Hybrid search (RRF) if the collection has `hybrid_enabled:true`:

```bash
restish local collections-search podcast \
  -H "Authorization: Bearer $RAGD_API_KEY" \
  query:"audio normalization workflow" \
  k:8 \
  mode:hybrid
```

## 5) Basic RAG completion (optional)

```bash
restish local collections-ask podcast \
  -H "Authorization: Bearer $RAGD_API_KEY" \
  query:"What do we recommend for guest audio setups?" \
  k:6 \
  mode:vector \
  llm_model:"gpt-4o-mini"
```

If you omit `llm_model`, the server uses `LLM_MODEL_DEFAULT`.
