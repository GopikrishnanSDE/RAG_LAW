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

SYSTEM_PROMPT = """You are an assistant for India's Income-tax Act, 2025, answering questions \
from individuals and businesses using ONLY the retrieved sections provided below. You are not \
a licensed tax advisor.

Rules:
1. Base your answer strictly on the provided context. Do not use outside knowledge of tax law.
2. Every claim must cite the provision it comes from, like "(Section 123)" or "(Schedule XV)".
3. Never compute a tax amount yourself, and never apply a rate from a section that doesn't \
match the user's situation. If a CALCULATOR line says a computation wasn't possible, explain \
its reason and point to the relevant provisions instead.
4. If the answer depends on facts the user didn't give (for example individual or company, \
turnover or profit, resident or not), say which assumption the answer rests on or ask for the \
missing detail.
5. If the context is relevant but only partly answers the question, answer the part it \
supports and say plainly what the retrieved sections don't cover. Provisions often say \
"this Chapter" or "this section" — read them in light of the section header above them.
6. Only if the context contains nothing relevant to the question, reply with exactly: \
"I don't have enough information in the retrieved sections to answer that." \
Do not guess or fill gaps from general knowledge.
7. Be concise. This is informational, not personalized financial or legal advice."""


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
            # num_predict caps runaway answers (a 7B model given six sections
            # will otherwise summarise all of them).
            "options": {"temperature": 0.1, "num_predict": 600},
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
        num = str(m.get("section_number"))
        header = num if num.startswith("Schedule") else f"Section {num}"
        if m.get("section_title"):
            header += f" — {m['section_title']}"
        blocks.append(f"[{header}]\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(query: str, contexts: list[dict], calculation_text: str | None = None) -> dict:
    if not contexts:
        return {
            "answer": "I don't have enough information in the retrieved sections to answer that.",
            "model": None,
        }

    context_block = _format_context(contexts)
    if calculation_text:
        context_block = f"{calculation_text}\n\n---\n\n{context_block}"
    user_message = (
        f"Retrieved context:\n\n{context_block}\n\n---\n\nQuestion: {query}"
    )

    if config.GENERATION_MODEL.startswith("claude-"):
        answer_text = _generate_anthropic(user_message)
    else:
        answer_text = _generate_ollama(user_message)
    return {"answer": answer_text, "model": config.GENERATION_MODEL}
