"""FastAPI service exposing the RAG pipeline. The Node API layer is a thin
proxy in front of this — this is where retrieval/generation actually happen.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_core import config, db, generation, query_analysis, reranker, retrieval, tax_calculator

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
    calculation: dict | None = None
    search_queries: list[str] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    # 1. Rewrite into the Act's vocabulary + extract facts for a computation.
    analysis = query_analysis.analyze(question)
    rewrites = analysis["search_queries"]

    # 2. Compute tax in code, if the question asks for an amount.
    calc = tax_calculator.calculate(analysis["calculation"]) if analysis["calculation"] else None

    with db.get_connection() as conn:
        if calc and calc.supported:
            # Computed answers are rendered in code, never by the LLM, and
            # shown with exactly the provisions the calculator applied — so
            # every figure is traceable. No retrieval needed.
            reranked = retrieval.calculator_sources(conn, calc.sections, [])
            result = {
                "answer": tax_calculator.format_answer(calc),
                "model": "tax calculator (s.202, s.156, s.19, s.58)",
            }
        else:
            candidates = retrieval.multi_query_search(conn, [question, *rewrites])
            candidates = retrieval.add_siblings(conn, candidates)
            reranked = [
                c
                for c in reranker.rerank(question, candidates, extra_queries=rewrites[:1])
                # Cohere-less fallback has no score; keep RRF order as-is then.
                if c.get("rerank_score") is None or c["rerank_score"] >= config.MIN_RERANK_SCORE
            ]
            reranked = retrieval.add_governing_sections(conn, reranked)
            result = None

    if result is None:
        # No chunk relevant enough -> generate_answer refuses without an LLM call.
        # A calculation the calculator couldn't do (company, old regime, over
        # the presumptive limit) is passed along so the answer explains why.
        result = generation.generate_answer(
            question,
            reranked,
            calculation_text=tax_calculator.format_for_prompt(calc) if calc else None,
        )

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
            "calculator_source": c.get("calculator_source", False),
        }
        for c in reranked
    ]

    return {
        "answer": result["answer"],
        "model": result["model"],
        "citations": citations,
        "contexts": contexts,
        "calculation": calc.to_dict() if calc else None,
        "search_queries": rewrites,
    }
