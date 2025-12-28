set -a

source .env

set +a

uv run uvicorn ragd.main:app $@