# RAG-Law — Ask My Tax Docs

A production-shaped RAG pipeline over the Income-tax Act, built to prove
**retrieval quality engineering**, not just "call an LLM."

## The problem this solves

Naive vector-only RAG on legal text confidently retrieves the wrong or
superseded section — the answer still *reads* fluently, because retrieval
failures don't show up in fluency. A layperson asking "can I claim this
deduction?" and getting a confidently wrong answer has real financial
consequences. This project is built specifically to catch that failure
mode, and to prove it does via an automated, CI-gated eval — not a demo
that only works on the happy path.

## Domain

Chapter VIII ("Deductions to be made in computing total income") of the
Income-tax Act, 2025, as amended by the Finance Act, 2026 — Sections
122–154, sourced from the corpus PDF in `data/`. Deliberately bounded to one
chapter: breadth wasn't the point, retrieval correctness was.

## Architecture

```
                 ┌─────────────┐      POST /api/query       ┌──────────────────┐
   client  ───▶  │  Node API    │ ─────────────────────────▶ │  Python RAG core  │
                 │  (Express)   │ ◀───────────────────────── │  (FastAPI)        │
                 │  thin layer: │        JSON response        │  hybrid retrieval│
                 │  validation, │                              │  + rerank        │
                 │  rate limit  │                              │  + generation    │
                 └─────────────┘                              └─────────┬────────┘
                                                                         │
                                                          ┌──────────────┴──────────────┐
                                                          │        Postgres              │
                                                          │  pgvector  (cosine ANN)       │
                                                          │  tsvector  (keyword / FTS)    │
                                                          └───────────────────────────────┘
```

- **`rag_core/`** (Python) — chunking, embeddings, hybrid retrieval, reranking,
  grounded generation. Kept end-to-end in Python since the ML/retrieval
  ecosystem is Python-first.
- **`api/`** (Node.js + Express) — thin proxy between the frontend and the
  Python core: request validation, rate limiting, error shaping. No
  retrieval or generation logic lives here on purpose.
- **`eval/`** — Ragas eval, run as a CI gate against a *deployed* instance of
  the API, not against internal functions.

### Why hybrid retrieval, not just vector search

Vector search finds semantically similar text but is blind to exact
statutory anchors — a query for "Section 80C" or a rupee figure embeds close
to a dozen unrelated deduction sections. Keyword search nails exact terms
and numbers but misses paraphrase ("tax break for LIC premium" vs. "life
insurance premia"). [`rag_core/retrieval.py`](rag_core/retrieval.py) runs
both (Postgres `pgvector` cosine search + native `tsvector`/`ts_rank_cd`)
and fuses the two *rankings* with Reciprocal Rank Fusion (k≈60) rather than
their raw scores, which live on incomparable scales.
[`rag_core/reranker.py`](rag_core/reranker.py) then runs Cohere's
cross-encoder reranker over the fused candidates as a precision pass — RRF
only knows the two signals agree, not that a chunk is actually about the
right sub-section.

### Why section-aware chunking

[`rag_core/chunking.py`](rag_core/chunking.py) splits on the Act's own
structure (section → sub-section → clause) instead of a fixed token window,
which will happily cut "...the deduction under this section shall not
exceed" away from the number that follows it — a chunk that embeds fine and
answers the wrong question. Verified against the 686-page corpus: 2,857
chunks, avg 156 tokens, max under the 1,200-token hard cap.

### Why the eval is the actual deliverable

[`eval/ragas_eval.py`](eval/ragas_eval.py) hits a running instance of the
API with a held-out question set ([`eval/testset.jsonl`](eval/testset.jsonl),
grounded in the actual corpus text) and scores **faithfulness, context
precision, context recall, and answer relevancy** via
[Ragas](https://github.com/explodinggradients/ragas). It exits non-zero if
any metric drops below threshold — see
[`.github/workflows/eval.yml`](.github/workflows/eval.yml) — so a retrieval
regression fails CI the same way a broken unit test would, instead of
shipping silently.

## Running it locally

```bash
cp .env.example .env   # fill in OPENAI_API_KEY, COHERE_API_KEY, ANTHROPIC_API_KEY
docker compose up -d db

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/ingest.py --reset          # PDF -> chunks -> embeddings -> Postgres

uvicorn rag_core.service:app --reload --port 8000   # RAG core
cd api && npm install && npm run dev                # Express layer, in another shell
```

```bash
curl -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the maximum deduction for life insurance premium under the new Act?"}'
```

Run the eval gate against a running instance:

```bash
python eval/ragas_eval.py --api-url http://localhost:3000/api/query
```

Or bring up the whole stack (Postgres + both services) with Docker:

```bash
docker compose up --build
```

## Known limitations

- The section-boundary detector uses a monotonic section-number heuristic;
  it assumes sections are scanned in increasing order and will misfire on a
  reordered corpus (documented in `chunking.py`).
- Table-heavy content (rate schedules) is chunked as ordinary text — a
  dedicated table extractor would be needed if the eval set grows to include
  table lookups.
- Generation is prompted to refuse when context is insufficient, but that's
  a prompt-level guardrail, not a formal groundedness classifier.
