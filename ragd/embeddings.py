from __future__ import annotations

from openai import OpenAI


def build_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def embed_texts(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def embed_texts_batched(
    client: OpenAI,
    model: str,
    texts: list[str],
    batch_size: int,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(embed_texts(client, model, batch))
    return vectors


def probe_embed_dims(client: OpenAI, model: str) -> int:
    vectors = embed_texts(client, model, ["dimension probe"])
    if not vectors or not vectors[0]:
        raise RuntimeError("Embedding probe returned empty vector")
    return len(vectors[0])
