"""Grounded generation: answer only from retrieved context, cite sections, refuse otherwise.

The whole point of this project is that a naive RAG answer *sounds* right
even when retrieval failed. This prompt is the second line of defense (the
first is retrieval quality itself, measured in eval/): it instructs the
model to say so when the retrieved chunks don't actually cover the question,
instead of filling the gap from parametric knowledge of tax law that may be
outdated or for the wrong jurisdiction/act version.
"""

import anthropic

from rag_core import config

_client = None

SYSTEM_PROMPT = """You are a tax-law assistant answering questions about the Income-tax Act \
using ONLY the retrieved sections provided below. You are not a licensed tax advisor.

Rules:
1. Base your answer strictly on the provided context. Do not use outside knowledge of tax law.
2. Every claim must cite the section it comes from, like "(Section 123)".
3. If the retrieved context does not contain enough information to answer the question, \
say exactly: "I don't have enough information in the retrieved sections to answer that." \
Do not guess or fill gaps from general knowledge.
4. Be concise. This is informational, not personalized financial or legal advice."""


def _get_client() -> "anthropic.Anthropic":
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def _format_context(contexts: list[dict]) -> str:
    blocks = []
    for c in contexts:
        m = c["metadata"]
        header = f"Section {m.get('section_number')}"
        if m.get("section_title"):
            header += f" — {m['section_title']}"
        blocks.append(f"[{header}]\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(query: str, contexts: list[dict]) -> dict:
    if not contexts:
        return {
            "answer": "I don't have enough information in the retrieved sections to answer that.",
            "model": None,
        }

    context_block = _format_context(contexts)
    user_message = (
        f"Retrieved context:\n\n{context_block}\n\n---\n\nQuestion: {query}"
    )

    client = _get_client()
    response = client.messages.create(
        model=config.GENERATION_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    answer_text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    return {"answer": answer_text, "model": config.GENERATION_MODEL}
