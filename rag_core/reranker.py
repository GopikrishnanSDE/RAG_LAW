"""Cross-encoder reranking — the precision pass RRF can't do.

RRF fusion ranks by *agreement between two cheap signals*, not by actually
reading the query against each candidate. A cross-encoder scores every
(query, candidate) pair directly, which is what catches the case where a
chunk ranks well on both vector and keyword signals but is about the wrong
sub-section of a similar-sounding deduction.

Backend is picked by RERANK_MODEL:
- `rerank-*`  -> Cohere Rerank API (paid, needs COHERE_API_KEY)
- otherwise   -> local sentence-transformers CrossEncoder (free), e.g.
  `BAAI/bge-reranker-base`
"""

from rag_core import config

_client = None
_local_model = None


def _use_cohere() -> bool:
    return config.RERANK_MODEL.startswith("rerank-")


def _get_client():
    global _client
    if _client is None:
        import cohere

        _client = cohere.Client(api_key=config.COHERE_API_KEY)
    return _client


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import CrossEncoder

        _local_model = CrossEncoder(config.RERANK_MODEL)
    return _local_model


def _rerank_cohere(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    response = _get_client().rerank(
        model=config.RERANK_MODEL,
        query=query,
        documents=[c["text"] for c in candidates],
        top_n=top_n,
    )
    reranked = []
    for result in response.results:
        rec = dict(candidates[result.index])
        rec["rerank_score"] = result.relevance_score
        reranked.append(rec)
    return reranked


def _rerank_local(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    scores = _get_local_model().predict([(query, c["text"]) for c in candidates])
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    reranked = []
    for i in order[:top_n]:
        rec = dict(candidates[i])
        rec["rerank_score"] = float(scores[i])
        reranked.append(rec)
    return reranked


def _diversify(ranked: list[dict], top_n: int, per_section: int) -> list[dict]:
    """Keep rank order, but at most `per_section` chunks from any one section."""
    picked, counts = [], {}
    for rec in ranked:
        sec = rec.get("metadata", {}).get("section_number")
        if counts.get(sec, 0) >= per_section:
            continue
        counts[sec] = counts.get(sec, 0) + 1
        picked.append(rec)
        if len(picked) == top_n:
            break
    return picked


def rerank(
    query: str,
    candidates: list[dict],
    top_n: int = config.FINAL_CONTEXT_N,
    per_section: int = config.MAX_CHUNKS_PER_SECTION,
) -> list[dict]:
    if not candidates:
        return []

    if _use_cohere():
        if not config.COHERE_API_KEY:
            # Graceful degradation: keep RRF order rather than hard-failing
            # when the Cohere key isn't configured.
            ranked = candidates
        else:
            ranked = _rerank_cohere(query, candidates, len(candidates))
    else:
        ranked = _rerank_local(query, candidates, len(candidates))

    return _diversify(ranked, top_n, per_section)
