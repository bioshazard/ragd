FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY ragd ./ragd
COPY scripts/init_db.sql ./scripts/init_db.sql

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "ragd.main:app", "--host", "0.0.0.0", "--port", "8000"]
