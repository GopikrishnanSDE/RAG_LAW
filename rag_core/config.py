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
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "20"))
# With query rewriting, candidates from every query are fused and the top
# MULTI_QUERY_CANDIDATES go to the reranker.
MULTI_QUERY_CANDIDATES = int(os.environ.get("MULTI_QUERY_CANDIDATES", "30"))
FINAL_CONTEXT_N = int(os.environ.get("FINAL_CONTEXT_N", "6"))
# Cap per section in the final context, so one long section/Schedule
# (Schedule XV is 47 chunks) can't take every slot.
MAX_CHUNKS_PER_SECTION = int(os.environ.get("MAX_CHUNKS_PER_SECTION", "3"))
# Sibling expansion: the reranker also sees every chunk (up to a cap) of the
# top few sections retrieval landed on, so a section's main rule competes
# even when only its side provisions matched the query.
SIBLING_SECTIONS = int(os.environ.get("SIBLING_SECTIONS", "3"))
SIBLING_MAX_CHUNKS = int(os.environ.get("SIBLING_MAX_CHUNKS", "20"))
# Reranked chunks below this relevance are dropped; if none remain, the
# service refuses without calling the LLM. Calibrated on bge-reranker-base:
# in-scope questions score >= ~0.4, nonsense/off-topic ~0.0-0.02.
MIN_RERANK_SCORE = float(os.environ.get("MIN_RERANK_SCORE", "0.1"))
