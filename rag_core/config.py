"""Central config — every tunable knob for the pipeline, read once from env."""

import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag_law"
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1536"))

RERANK_MODEL = os.environ.get("RERANK_MODEL", "rerank-english-v3.0")

GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "claude-sonnet-5")

# Retrieval fan-out before fusion/reranking
VECTOR_K = int(os.environ.get("VECTOR_K", "20"))
KEYWORD_K = int(os.environ.get("KEYWORD_K", "20"))
RRF_K = int(os.environ.get("RRF_K", "60"))  # RRF's k constant, not a result count
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "8"))
FINAL_CONTEXT_N = int(os.environ.get("FINAL_CONTEXT_N", "4"))
