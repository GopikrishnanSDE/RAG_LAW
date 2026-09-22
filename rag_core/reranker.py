"""Cross-encoder reranking via Cohere Rerank — the precision pass RRF can't do.

RRF fusion ranks by *agreement between two cheap signals*, not by actually
reading the query against each candidate. Cohere's reranker scores every
(query, candidate) pair directly, which is what catches the case where a
chunk ranks well on both vector and keyword signals but is about the wrong
sub-section of a similar-sounding deduction.
"""

import cohere

from rag_core import config

_client = None


def _get_client() -> "cohere.Client":
    global _client
    if _client is None:
        _client = cohere.Client(api_key=config.COHERE_API_KEY)
    return _client


def rerank(query: str, candidates: list[dict], top_n: int = config.FINAL_CONTEXT_N) -> list[dict]:
    if not candidates:
        return []

    if not config.COHERE_API_KEY:
        # Graceful degradation: keep RRF order rather than hard-failing when
        # the optional reranker isn't configured (e.g. local dev, free tier
        # exhausted).
        return candidates[:top_n]

    client = _get_client()
    response = client.rerank(
        model=config.RERANK_MODEL,
        query=query,
        documents=[c["text"] for c in candidates],
        top_n=min(top_n, len(candidates)),
    )

    reranked = []
    for result in response.results:
        rec = dict(candidates[result.index])
        rec["rerank_score"] = result.relevance_score
        reranked.append(rec)
    return reranked
