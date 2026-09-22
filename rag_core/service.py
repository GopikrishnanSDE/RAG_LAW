"""FastAPI service exposing the RAG pipeline. The Node API layer is a thin
proxy in front of this — this is where retrieval/generation actually happen.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_core import config, db, generation, reranker, retrieval

app = FastAPI(title="RAG-Law core service")


class QueryRequest(BaseModel):
    question: str


class Citation(BaseModel):
    section_number: str | None
    section_title: str | None
    chapter: str | None


class QueryResponse(BaseModel):
    answer: str
    model: str | None
    citations: list[Citation]
    contexts: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    with db.get_connection() as conn:
        candidates = retrieval.hybrid_search(conn, question)

    reranked = reranker.rerank(question, candidates)
    result = generation.generate_answer(question, reranked)

    citations = [
        {
            "section_number": c["metadata"].get("section_number"),
            "section_title": c["metadata"].get("section_title"),
            "chapter": c["metadata"].get("chapter"),
        }
        for c in reranked
    ]

    contexts = [
        {
            "text": c["text"],
            "metadata": c["metadata"],
            "rrf_score": c.get("rrf_score"),
            "rerank_score": c.get("rerank_score"),
        }
        for c in reranked
    ]

    return {
        "answer": result["answer"],
        "model": result["model"],
        "citations": citations,
        "contexts": contexts,
    }
