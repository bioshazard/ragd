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

And set the header:

`restish api configure ragd URI`

Then update the profile > header:

* Header `Authorization`
* Value: `$RAGD_API_KEY` (use actual key)

## 2) Create a collection

```bash
restish local collections-create \
  -H "Authorization: Bearer $RAGD_API_KEY" \
  name:docs, \
  hybrid_enabled:true
```

## 3) Ingest transcripts (folder of .txt)

Chunking notes:

- ragd splits text into word-based chunks with overlap (defaults: `CHUNK_TARGET_TOKENS=600`, `CHUNK_OVERLAP_TOKENS=120` or `CHUNK_OVERLAP_PERCENT=15`).
- If you have timestamps, send segment arrays (`{text,t_start,t_end}`) so chunks carry time metadata for citations.

This loop builds a full JSON payload and pipes it to `restish` (avoids shell parsing issues):

```bash
set -euo pipefail

# expand while globbing is still on
files=( ./data/cache/source/docs/*/*.txt )

set -o noglob

for path in "${files[@]}"; do
  [[ -e "$path" ]] || continue  # handles no matches

  doc_id=$(basename "$path" .txt)
  echo "Ingesting $doc_id..."

  python3 - "$path" "$doc_id" <<'PY' | time \
    restish ragd documents-ingest docs "$doc_id" \
      --rsh-ignore-status-code --rsh-raw --rsh-verbose
import json, sys
path = sys.argv[1]
doc_id = sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    text = f.read()
print(json.dumps({
  "content": text,
  "ingest_mode": "replace",
  "tags": ["documents"],
  "metadata": {"source": path},
  "title": doc_id,
}))
PY

  # break
done
```

## 4) Retrieve top-k chunks

```bash
restish local collections-search docs \
  -H "Authorization: Bearer $RAGD_API_KEY" \
  query:"how we onboard new hosts", \
  k:8, \
  mode:vector
```

Hybrid search (RRF) if the collection has `hybrid_enabled:true`:

```bash
restish local collections-search docs \
  -H "Authorization: Bearer $RAGD_API_KEY" \
  query:"audio normalization workflow", \
  k:8, \
  mode:hybrid
```

## 5) Basic RAG completion (optional)

```bash
restish local collections-ask docs \
  -H "Authorization: Bearer $RAGD_API_KEY" \
  query:"What do we recommend for guest audio setups?", \
  k:6, \
  mode:vector, \
  llm_model:"gpt-4o-mini"
```

If you omit `llm_model`, the server uses `LLM_MODEL_DEFAULT`.
