"""One LLM pass before retrieval: rewrite the question into the Act's
vocabulary, and pull out the facts a tax computation needs.

Why rewrite: users say "my business earns 20 lakh, how much tax"; the Act
says "rate of income-tax", "total income", "profits and gains of business or
profession", "new tax regime". Neither embeddings nor keyword search bridge
that gap reliably — that question retrieved the TDS and rebate sections
(rerank score ~0.1) and never the s.202 rate table. Searching with a few
statute-worded rewrites alongside the original fixes the vocabulary gap
without losing what the user literally asked.

Why extract: the numbers go to tax_calculator.py, which does the arithmetic
in code. The LLM never computes tax.

Runs on the local Ollama model. On any failure (Ollama down, malformed
JSON, a non-Ollama generation backend) it falls back to "no rewrites, no
calculation", so retrieval degrades to the plain original question.
"""

from __future__ import annotations

import json
import logging
import re

import requests

from rag_core import config
from rag_core.tax_calculator import TaxInputs

log = logging.getLogger(__name__)

PROMPT = """You help search India's Income-tax Act, 2025. Analyse the user's question and reply \
with JSON only.

1. "search_queries": 1-3 short search queries that restate the question in the Act's own \
wording. Useful terms: "rate of income-tax", "total income", "new tax regime" (s.202), \
"rebate" (s.156), "profits and gains of business or profession", "presumptive basis" \
(s.58), "turnover or gross receipts", "Salaries", "standard deduction" (s.19), "Income \
from house property", "Capital gains", "short-term / long-term capital asset", "tax \
year", "resident", "return of income", "deduction at source", "assessee", "Hindu \
undivided family". Only use terms that match the question's topic — e.g. a property \
sale is about "Capital gains", not "Income from house property". Use plain words; never \
invent section numbers.

2. "calculation": null unless the question gives an amount of income, salary, profit, \
turnover or fees and asks how much tax is due on it (even tersely, e.g. "40 lakh fees, tax?"). Otherwise an object with these keys (null when not stated; amounts in rupees as \
plain numbers — 1 lakh = 100000, 1 crore = 10000000):
  "person_type": "individual" | "huf" | "firm" | "company" | "other" | null
  "resident": true | false | null   (null unless the user says so)
  "regime": "new" | "old" | null
  "salary": number | null
  "business_profit": number | null   (amount the business "earns", "makes", "profit", "income")
  "business_turnover": number | null (only if they say turnover, sales, revenue or receipts)
  "professional_receipts": number | null (fees or receipts of a "specified profession" under \
s.62(4): legal, medical, engineering, architectural, accountancy, technical consultancy, \
interior decoration, information technology or company secretary — e.g. doctor, lawyer, \
CA, architect, engineer, IT consultant, software freelancer)
  "digital_share": number between 0 and 1 | null (share received by bank transfer, UPI, \
card or online)
  "other_income": number | null

Question: {question}"""


_NUM = {"type": ["number", "null"]}

# Ollama structured output: constrains the reply to this shape, so the
# small model can't skip search_queries or drift into prose.
SCHEMA = {
    "type": "object",
    "properties": {
        "search_queries": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "calculation": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "person_type": {"enum": ["individual", "huf", "firm", "company", "other", None]},
                        "resident": {"type": ["boolean", "null"]},
                        "regime": {"enum": ["new", "old", None]},
                        "salary": _NUM,
                        "business_profit": _NUM,
                        "business_turnover": _NUM,
                        "professional_receipts": _NUM,
                        "digital_share": _NUM,
                        "other_income": _NUM,
                    },
                    "required": [
                        "person_type", "resident", "regime", "salary", "business_profit",
                        "business_turnover", "professional_receipts", "digital_share", "other_income",
                    ],
                },
            ]
        },
    },
    "required": ["search_queries", "calculation"],
}


_MULTIPLIERS = {
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "l": 1e5,
    "crore": 1e7, "crores": 1e7, "cr": 1e7,
    "k": 1e3, "thousand": 1e3, "million": 1e6, "mn": 1e6,
}
_AMOUNT_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(" + "|".join(sorted(_MULTIPLIERS, key=len, reverse=True)) + r")?\b",
    re.IGNORECASE,
)


def amounts_in(text: str) -> list[float]:
    """Every rupee amount literally stated in the text ("20 lakh", "1.5 crore", "15,00,000")."""
    out = []
    for num, unit in _AMOUNT_RE.findall(text):
        n = float(num.replace(",", ""))
        out.append(n * _MULTIPLIERS[unit.lower()] if unit else n)
    return out


def _grounded(inputs: "TaxInputs", question: str) -> bool:
    """True if every amount the LLM extracted is actually stated in the question.

    Without this, "I sold my flat after 3 years, how is the gain taxed?" came
    back with an invented ₹20 lakh of income and a confident tax figure.
    """
    stated = amounts_in(question)
    extracted = [
        inputs.salary, inputs.business_profit, inputs.business_turnover,
        inputs.professional_receipts, inputs.other_income,
    ]
    return all(
        any(abs(a - s) <= 0.01 * s for s in stated) for a in extracted if a is not None
    )


_FIELDS = ["salary", "professional_receipts", "business_turnover", "business_profit", "other_income"]
_SALARY_RE = re.compile(r"\b(salary|salaried|ctc|pay package|employ)", re.IGNORECASE)
_PROFESSION_RE = re.compile(
    r"\b(doctor|physician|surgeon|dentist|lawyer|advocate|chartered accountant|accountant|"
    r"architect|engineer|interior|company secretary|consultan|freelanc|software|developer|"
    r"professional fees|my fees)"
    # Abbreviations are case-sensitive, or "is it taxed?" would read as IT.
    r"|(?-i:\bCA\b|\bIT\b)",
    re.IGNORECASE,
)
_TURNOVER_RE = re.compile(r"\b(turnover|sales|revenue|gross receipts|receipts)\b", re.IGNORECASE)


def _resolve_fields(inputs: "TaxInputs", question: str) -> "TaxInputs":
    """Assign each stated amount to exactly one income field.

    The 7B model often copies one amount into two fields — "IT consultant
    with 30 lakh of gross receipts" came back as both turnover and profit, so
    the same ₹30 lakh was taxed twice. The field is decided here from the
    question's own words instead.
    """
    by_value: dict[float, list[str]] = {}
    for f in _FIELDS:
        v = getattr(inputs, f)
        if v is not None:
            by_value.setdefault(v, []).append(f)

    for value, fields in by_value.items():
        if _SALARY_RE.search(question) and "salary" in fields:
            keep = "salary"
        elif _PROFESSION_RE.search(question) and not _SALARY_RE.search(question):
            keep = "professional_receipts"
        elif _TURNOVER_RE.search(question):
            keep = "business_turnover"
        else:
            keep = fields[0]  # _FIELDS order = priority
        for f in fields:
            setattr(inputs, f, None)
        setattr(inputs, keep, value)
    return inputs


def _empty() -> dict:
    return {"search_queries": [], "calculation": None}


def analyze(question: str) -> dict:
    """Return {"search_queries": [str], "calculation": TaxInputs | None}."""
    if config.GENERATION_MODEL.startswith("claude-"):
        return _empty()  # rewrite/extraction is implemented for the local backend only

    try:
        resp = requests.post(
            f"{config.OLLAMA_URL}/api/chat",
            json={
                "model": config.GENERATION_MODEL,
                "stream": False,
                "format": SCHEMA,
                "options": {"temperature": 0},
                "messages": [{"role": "user", "content": PROMPT.format(question=question)}],
            },
            timeout=120,
        )
        resp.raise_for_status()
        out = json.loads(resp.json()["message"]["content"])
    except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
        log.warning("query analysis failed, using the original question only: %s", exc)
        return _empty()

    queries = [
        q.strip()
        for q in out.get("search_queries") or []
        if isinstance(q, str) and q.strip() and q.strip().lower() != question.strip().lower()
    ][:3]

    calculation = _to_inputs(out.get("calculation"))
    if calculation:
        calculation = _resolve_fields(calculation, question)
    if calculation and not _grounded(calculation, question):
        log.warning("dropping calculation with amounts not stated in the question: %s", calculation)
        calculation = None
    return {"search_queries": queries, "calculation": calculation}


def _num(v) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _to_inputs(calc) -> TaxInputs | None:
    if not isinstance(calc, dict):
        return None
    person = calc.get("person_type")
    regime = calc.get("regime")
    share = _num(calc.get("digital_share"))
    inputs = TaxInputs(
        person_type=person if person in {"individual", "huf", "firm", "company", "other"} else None,
        resident=calc.get("resident") if isinstance(calc.get("resident"), bool) else None,
        regime=regime if regime in {"new", "old"} else None,
        salary=_num(calc.get("salary")),
        business_profit=_num(calc.get("business_profit")),
        business_turnover=_num(calc.get("business_turnover")),
        professional_receipts=_num(calc.get("professional_receipts")),
        digital_share=min(share, 1.0) if share is not None else None,
        other_income=_num(calc.get("other_income")),
    )
    amounts = [
        inputs.salary, inputs.business_profit, inputs.business_turnover,
        inputs.professional_receipts, inputs.other_income,
    ]
    return inputs if any(amounts) else None
