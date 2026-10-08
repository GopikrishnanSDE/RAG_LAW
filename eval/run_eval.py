"""CI-gated retrieval/generation quality eval, run against a *deployed* API.

This is the actual point of the project: a vector-only RAG system will pass
every unit test and still confidently cite the wrong section, because
retrieval failures don't show up in output fluency. Ragas scores the
retrieved contexts against a held-out ground truth (context precision/
recall) and the generated answer against both the question and its own
context (answer relevancy, faithfulness) — so a regression in retrieval
quality fails the build even though nothing "crashed".

Usage:
    python eval/ragas_eval.py [--api-url http://localhost:3000/api/query]

Exits non-zero if any metric falls below its threshold, for use as a CI gate.
"""

import argparse
import json
import sys
from pathlib import Path

import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

THRESHOLDS = {
    "faithfulness": 0.80,
    "context_precision": 0.70,
    "context_recall": 0.70,
    "answer_relevancy": 0.70,
}

TESTSET_PATH = Path(__file__).parent / "testset.jsonl"


def load_testset() -> list[dict]:
    with open(TESTSET_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_queries(api_url: str, testset: list[dict]) -> dict:
    questions, answers, contexts, ground_truths = [], [], [], []
    for case in testset:
        resp = requests.post(api_url, json={"question": case["question"]}, timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        questions.append(case["question"])
        ground_truths.append(case["ground_truth"])
        answers.append(payload["answer"])
        contexts.append([c["text"] for c in payload["contexts"]])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api-url",
        default="http://localhost:3000/api/query",
        help="Endpoint that accepts {question} and returns {answer, contexts}",
    )
    args = parser.parse_args()

    testset = load_testset()
    print(f"Running {len(testset)} eval questions against {args.api_url}")
    dataset_dict = run_queries(args.api_url, testset)

    dataset = Dataset.from_dict(dataset_dict)
    result = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
    )
    scores = result.to_pandas().mean(numeric_only=True).to_dict()

    print("\n--- Ragas scores (mean across test set) ---")
    failed = []
    for metric, threshold in THRESHOLDS.items():
        score = scores.get(metric)
        status = "PASS" if score is not None and score >= threshold else "FAIL"
        if status == "FAIL":
            failed.append(metric)
        print(f"{metric:<20} {score:.3f}  (threshold {threshold})  {status}")

    if failed:
        print(f"\nEval gate FAILED: {', '.join(failed)} below threshold.")
        sys.exit(1)

    print("\nEval gate PASSED.")


if __name__ == "__main__":
    main()
