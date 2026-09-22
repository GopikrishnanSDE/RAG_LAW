"""Text -> vector, via OpenAI embeddings. Batched to stay under request limits."""

from openai import OpenAI

from rag_core import config

_BATCH_SIZE = 96

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = _get_client()
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        embeddings.extend(item.embedding for item in resp.data)
    return embeddings


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
