#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <password>" >&2
  echo "Reads from stdin if <password> is '-'" >&2
  exit 1
fi

password="$1"
if [[ "$password" == "-" ]]; then
  IFS= read -r password
fi

python3 - <<'PY' "$password"
import sys
from urllib.parse import quote

pw = sys.argv[1]
print(quote(pw, safe=""))
PY
