"""CI-gated retrieval/generation quality eval, run against a *running* API.

This is the actual point of the project: a vector-only RAG system will pass
every unit test and still confidently cite the wrong section, because
retrieval failures don't show up in output fluency. So the gate scores the
pipeline end to end, through the public API, on a held-out question set:

Retrieval (deterministic — no LLM involved, so no judge noise):
  - hit_rate         share of in-scope questions where an expected section
                     appears in the retrieved context. A Schedule counts for
                     the section that points to it (Schedule XV -> s.123).
  - mrr              mean reciprocal rank of the first expected section.

Generation (LLM-as-judge, run locally via Ollama — free, no API key):
  - faithfulness     every claim in the answer is supported by the retrieved
                     context (1 / 0.5 / 0).
  - correctness      the answer agrees with the ground truth (1 / 0.5 / 0).

Tax computation (deterministic):
  - calc_accuracy    share of "how much tax on X" questions whose answer
                     states exactly the hand-computed tax. These answers are
                     built in code (tax_calculator.py), so they're checked
                     against expected_tax, not judged by an LLM.

Refusal behaviour (deterministic):
  - refusal_rate     out-of-scope questions answered with the refusal line.
  - false_refusals   in-scope questions wrongly refused (reported, not gated:
                     it's already penalised through correctness).

Why not Ragas: its current release imports LangChain modules that no longer
exist in the versions the chunker needs, and its multi-step judge prompts
are brittle with small local models (malformed JSON). Two single-prompt
judges with a JSON-constrained response are enough for this test set, and
the retrieval half doesn't need a judge at all.

Known limitation: the judge defaults to the same local model that writes
the answers, so it may be lenient toward its own phrasing. Point
JUDGE_MODEL at a different/larger Ollama model to reduce that bias.

Usage:
    python eval/run_eval.py [--api-url http://localhost:3000/api/query]

Exits non-zero if any gated metric falls below its threshold.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests

THRESHOLDS = {
    "hit_rate": 0.80,
    "mrr": 0.60,
    "faithfulness": 0.80,
    "correctness": 0.60,
    "calc_accuracy": 0.80,
    "refusal_rate": 0.75,
}

REFUSAL = "I don't have enough information in the retrieved sections"
TAX_LINE_RE = re.compile(r"Estimated income-tax: ₹([\d,]+)")
TESTSET_PATH = Path(__file__).parent / "testset.jsonl"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", os.environ.get("GENERATION_MODEL", "qwen2.5:7b"))

FAITHFULNESS_PROMPT = """You are grading a legal Q&A system. Decide whether EVERY factual claim \
in the ANSWER is supported by the CONTEXT. Ignore style. Do not use outside knowledge.

CONTEXT:
{context}

ANSWER:
{answer}

Reply with JSON only: {{"verdict": "supported" | "partially_supported" | "unsupported", "reason": "<one sentence>"}}"""

CORRECTNESS_PROMPT = """You are grading a legal Q&A system. Compare the ANSWER to the REFERENCE \
answer for the QUESTION. "correct" = same substance (amounts, limits, conditions, section) even if \
worded differently; "partially_correct" = right direction but missing or wrong on a key detail; \
"incorrect" = contradicts the reference or doesn't answer.

QUESTION: {question}

REFERENCE: {reference}

ANSWER: {answer}

Reply with JSON only: {{"verdict": "correct" | "partially_correct" | "incorrect", "reason": "<one sentence>"}}"""

SCORES = {
    "supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0,
    "correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0,
}


def load_testset() -> list[dict]:
    with open(TESTSET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def is_refusal(answer: str) -> bool:
    return answer.strip().startswith(REFUSAL)


def matches(context: dict, section: str) -> bool:
    m = context["metadata"]
    return m.get("section_number") == section or m.get("part") == f"See section {section}"


def first_hit_rank(contexts: list[dict], expected: list[str]) -> int | None:
    for rank, c in enumerate(contexts, start=1):
        if any(matches(c, s) for s in expected):
            return rank
    return None


def judge(prompt: str) -> tuple[float, str]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": JUDGE_MODEL,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=300,
    )
    resp.raise_for_status()
    try:
        out = json.loads(resp.json()["message"]["content"])
        return SCORES.get(out.get("verdict"), 0.0), out.get("reason", "")
    except (json.JSONDecodeError, AttributeError):
        return 0.0, "judge returned unparseable output"


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-url",
        default="http://localhost:3000/api/query",
        help="Endpoint that accepts {question} and returns {answer, contexts}",
    )
    args = parser.parse_args()

    testset = load_testset()
    print(f"Running {len(testset)} eval questions against {args.api_url} (judge: {JUDGE_MODEL})\n")

    hits, rr, faith, correct, calc_ok, refused_oos, false_refusals = [], [], [], [], [], [], 0
    for case in testset:
        q = case["question"]
        resp = requests.post(args.api_url, json={"question": q}, timeout=300)
        resp.raise_for_status()
        payload = resp.json()
        answer, contexts = payload["answer"], payload["contexts"]
        refused = is_refusal(answer)

        if case.get("expect_refusal"):
            refused_oos.append(1.0 if refused else 0.0)
            print(f"[{'PASS' if refused else 'FAIL'}] out-of-scope  {q}")
            continue

        rank = first_hit_rank(contexts, case["expected_sections"])
        hits.append(1.0 if rank else 0.0)
        rr.append(1.0 / rank if rank else 0.0)

        if "expected_tax" in case:
            m = TAX_LINE_RE.search(answer)
            got = int(m.group(1).replace(",", "")) if m else None
            ok = got == case["expected_tax"]
            calc_ok.append(1.0 if ok else 0.0)
            print(
                f"[{'PASS' if ok else 'FAIL'}] calculation   {q}\n"
                f"           expected ₹{case['expected_tax']:,}, got "
                + (f"₹{got:,}" if got is not None else "no computed figure")
            )
            continue

        if refused:
            # Refusing an in-scope question claims nothing (faithful) but
            # doesn't answer it (incorrect).
            false_refusals += 1
            f_score, c_score, why = 1.0, 0.0, "refused an in-scope question"
        else:
            context_text = "\n\n---\n\n".join(c["text"] for c in contexts)
            f_score, f_why = judge(FAITHFULNESS_PROMPT.format(context=context_text, answer=answer))
            c_score, c_why = judge(
                CORRECTNESS_PROMPT.format(question=q, reference=case["ground_truth"], answer=answer)
            )
            why = c_why if c_score < 1 else f_why
        faith.append(f_score)
        correct.append(c_score)

        retrieved = ", ".join(str(c["metadata"]["section_number"]) for c in contexts) or "none"
        print(
            f"[rank {rank or '-'}] faith={f_score:.1f} correct={c_score:.1f}  {q}\n"
            f"           expected {case['expected_sections']}, retrieved [{retrieved}]"
            + (f"\n           judge: {why}" if why and (f_score < 1 or c_score < 1) else "")
        )

    scores = {
        "hit_rate": mean(hits),
        "mrr": mean(rr),
        "faithfulness": mean(faith),
        "correctness": mean(correct),
        "calc_accuracy": mean(calc_ok),
        "refusal_rate": mean(refused_oos),
    }

    print("\n--- Scores ---")
    failed = []
    for metric, threshold in THRESHOLDS.items():
        status = "PASS" if scores[metric] >= threshold else "FAIL"
        if status == "FAIL":
            failed.append(metric)
        print(f"{metric:<15} {scores[metric]:.3f}  (threshold {threshold})  {status}")
    print(f"{'false_refusals':<15} {false_refusals}/{len(hits)}  (in-scope questions refused)")

    if failed:
        print(f"\nEval gate FAILED: {', '.join(failed)} below threshold.")
        sys.exit(1)
    print("\nEval gate PASSED.")


if __name__ == "__main__":
    main()
