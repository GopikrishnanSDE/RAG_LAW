"""Hybrid retrieval: vector search + keyword (FTS) search, fused with RRF.

Why fuse instead of picking one: vector search finds semantically similar
text but is blind to exact statutory terms (a query for "Section 80C" or a
rupee figure embeds close to lots of unrelated deduction sections). Keyword
search nails exact terms and numbers but misses paraphrases ("tax break for
LIC premium" vs. "life insurance premia"). Reciprocal Rank Fusion combines
the two *rankings* (not raw scores, which live on incomparable scales) so a
chunk that's merely decent on both signals can outrank one that's great on
only one.
"""

from rag_core import config, db, embeddings


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], k: int = config.RRF_K
) -> dict[int, float]:
    """ranked_lists: each a list of chunk ids, best first. Returns id -> RRF score."""
    scores: dict[int, float] = {}
    for ranked_ids in ranked_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_search(
    conn,
    query: str,
    vector_k: int = config.VECTOR_K,
    keyword_k: int = config.KEYWORD_K,
    rrf_k: int = config.RRF_K,
    top_n: int = config.RERANK_TOP_N,
) -> list[dict]:
    query_embedding = embeddings.embed_query(query)

    vector_hits = db.vector_search(conn, query_embedding, vector_k)
    keyword_hits = db.keyword_search(conn, query, keyword_k)

    by_id = {h["id"]: h for h in vector_hits}
    for h in keyword_hits:
        by_id.setdefault(h["id"], h)

    fused = reciprocal_rank_fusion(
        [
            [h["id"] for h in vector_hits],
            [h["id"] for h in keyword_hits],
        ],
        k=rrf_k,
    )

    ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:top_n]

    results = []
    for cid in ranked_ids:
        rec = dict(by_id[cid])
        rec["rrf_score"] = fused[cid]
        rec["in_vector_results"] = any(h["id"] == cid for h in vector_hits)
        rec["in_keyword_results"] = any(h["id"] == cid for h in keyword_hits)
        results.append(rec)
    return results
