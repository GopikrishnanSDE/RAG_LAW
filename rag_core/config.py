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

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))

RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")

# `claude-*` -> Anthropic API (paid); anything else -> local model via Ollama (free)
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "qwen2.5:7b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Retrieval fan-out before fusion/reranking
VECTOR_K = int(os.environ.get("VECTOR_K", "20"))
KEYWORD_K = int(os.environ.get("KEYWORD_K", "20"))
IVFFLAT_PROBES = int(os.environ.get("IVFFLAT_PROBES", "20"))  # of 100 lists
RRF_K = int(os.environ.get("RRF_K", "60"))  # RRF's k constant, not a result count
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "8"))
FINAL_CONTEXT_N = int(os.environ.get("FINAL_CONTEXT_N", "4"))
