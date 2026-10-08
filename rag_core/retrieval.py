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


def add_governing_sections(conn, contexts: list[dict]) -> list[dict]:
    """Append the section each retrieved Schedule chunk belongs to.

    A Schedule is only half the rule: Schedule XV lists what qualifies, but
    the cap (₹1,50,000) and who may claim (individual/HUF) live in Section
    123. Retrieval tends to surface the long, keyword-dense Schedule and
    miss the short section that governs it, so pull that section in
    explicitly whenever one of its Schedule chunks made the cut.
    """
    present = {c["metadata"]["section_number"] for c in contexts}
    extra = []
    for c in contexts:
        part = c["metadata"].get("part") or ""
        if not part.startswith("See section "):
            continue
        # "[See section 123]" -> "123"; skip multi-section refs like "123 and 124"
        ref = part.removeprefix("See section ").strip()
        if not ref.isalnum() or ref in present:
            continue
        for rec in db.get_section_chunks(conn, ref, limit=1):
            rec["governs"] = c["metadata"]["section_number"]
            extra.append(rec)
        present.add(ref)
    return contexts + extra
