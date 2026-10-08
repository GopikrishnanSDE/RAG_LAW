"""Text -> vector. Two backends, picked by EMBEDDING_MODEL:

- `text-embedding-*` -> OpenAI API (paid, batched to stay under request limits)
- anything else      -> local sentence-transformers model (free, no key),
  e.g. `BAAI/bge-small-en-v1.5` (384-dim)

EMBEDDING_DIM must match the chosen model, since it sizes the pgvector column.
"""

from rag_core import config

_BATCH_SIZE = 96

# BGE models are trained with an instruction prefix on the *query* side only;
# passages are embedded as-is. Skipping it measurably hurts retrieval.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_client = None
_local_model = None


def _use_openai() -> bool:
    return config.EMBEDDING_MODEL.startswith("text-embedding-")


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _local_model


def _embed_openai(texts: list[str]) -> list[list[float]]:
    client = _get_client()
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        embeddings.extend(item.embedding for item in resp.data)
    return embeddings


def _embed_local(texts: list[str], show_progress: bool = False) -> list[list[float]]:
    # Normalized so pgvector's cosine distance behaves the same as with OpenAI.
    vectors = _get_local_model().encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return vectors.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    if _use_openai():
        return _embed_openai(texts)
    return _embed_local(texts, show_progress=len(texts) > 100)


def embed_query(text: str) -> list[float]:
    if _use_openai():
        return _embed_openai([text])[0]
    if "bge" in config.EMBEDDING_MODEL.lower():
        text = _BGE_QUERY_PREFIX + text
    return _embed_local([text])[0]
