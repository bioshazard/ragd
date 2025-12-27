from __future__ import annotations

from openai import OpenAI


def build_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def embed_texts(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def probe_embed_dims(client: OpenAI, model: str) -> int:
    vectors = embed_texts(client, model, ["dimension probe"])
    if not vectors or not vectors[0]:
        raise RuntimeError("Embedding probe returned empty vector")
    return len(vectors[0])
