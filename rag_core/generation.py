"""Grounded generation: answer only from retrieved context, cite sections, refuse otherwise.

The whole point of this project is that a naive RAG answer *sounds* right
even when retrieval failed. This prompt is the second line of defense (the
first is retrieval quality itself, measured in eval/): it instructs the
model to say so when the retrieved chunks don't actually cover the question,
instead of filling the gap from parametric knowledge of tax law that may be
outdated or for the wrong jurisdiction/act version.
"""

import requests

from rag_core import config

_client = None

SYSTEM_PROMPT = """You are a tax-law assistant answering questions about the Income-tax Act \
using ONLY the retrieved sections provided below. You are not a licensed tax advisor.

Rules:
1. Base your answer strictly on the provided context. Do not use outside knowledge of tax law.
2. Every claim must cite the section it comes from, like "(Section 123)".
3. If the context is relevant but only partly answers the question, answer the part it \
supports and say plainly what the retrieved sections don't cover. Provisions often say \
"this Chapter" or "this section" — read them in light of the section header above them.
4. Only if the context contains nothing relevant to the question, reply with exactly: \
"I don't have enough information in the retrieved sections to answer that." \
Do not guess or fill gaps from general knowledge.
5. Be concise. This is informational, not personalized financial or legal advice."""


def _get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


def _generate_anthropic(user_message: str) -> str:
    response = _get_client().messages.create(
        model=config.GENERATION_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _generate_ollama(user_message: str) -> str:
    # Local model served by Ollama (free, no key). Low temperature: this is
    # extraction from the given context, not creative writing.
    resp = requests.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": config.GENERATION_MODEL,
            "stream": False,
            "options": {"temperature": 0.1},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


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

    if config.GENERATION_MODEL.startswith("claude-"):
        answer_text = _generate_anthropic(user_message)
    else:
        answer_text = _generate_ollama(user_message)
    return {"answer": answer_text, "model": config.GENERATION_MODEL}
