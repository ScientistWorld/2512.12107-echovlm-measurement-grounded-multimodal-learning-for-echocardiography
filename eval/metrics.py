"""Artifact-only evaluator for EchoVLM-style outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def binary_auc(labels: list[int], scores: list[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pr_metrics(labels: list[int], scores: list[float]) -> tuple[float, float]:
    preds = [int(s >= 0.0) for s in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return precision, recall


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def recall_at_k(image_z: list[list[float]], text_z: list[list[float]], k: int) -> float:
    hits = 0
    for i, image_vec in enumerate(image_z):
        scores = [(dot(image_vec, text_vec), j) for j, text_vec in enumerate(text_z)]
        topk = sorted(scores, reverse=True)[: min(k, len(scores))]
        hits += int(any(j == i for _, j in topk))
    return hits / max(1, len(image_z))


def evaluate_embeddings(path: Path) -> dict[str, float]:
    rows = load_jsonl(path)
    image_z = [r["image_embedding"] for r in rows]
    text_z = [r["text_embedding"] for r in rows]
    neg_z = [r["neg_text_embedding"] for r in rows]
    labels = [int(r["label"]) for r in rows]

    pos_sim = [dot(image_vec, text_vec) for image_vec, text_vec in zip(image_z, text_z)]
    neg_sim = [dot(image_vec, neg_vec) for image_vec, neg_vec in zip(image_z, neg_z)]
    scores = [pos - neg for pos, neg in zip(pos_sim, neg_sim)]
    precision, recall = pr_metrics(labels, scores)
    neg_margin = sum(dot(text_vec, neg_vec) for text_vec, neg_vec in zip(text_z, neg_z)) / max(1, len(text_z))
    align_margin = sum(scores) / max(1, len(scores))
    return {
        "auc": 100.0 * binary_auc(labels, scores),
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "retrieval_recall_at_5": 100.0 * recall_at_k(image_z, text_z, 5),
        "retrieval_recall_at_10": 100.0 * recall_at_k(image_z, text_z, 10),
        "negation_cosine": neg_margin,
        "positive_vs_negated_margin": align_margin,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="name=path/to/test_embeddings.jsonl")
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()
    results = {}
    for item in args.inputs:
        if "=" not in item:
            raise ValueError(f"Expected name=path input, got {item}")
        name, path = item.split("=", 1)
        metrics = evaluate_embeddings(Path(path))
        results[name] = metrics

    # The toy retrieval pool is useful for debugging, but it is not on the
    # same scale as the paper's full-test retrieval benchmark. The negation
    # cosine is retained as a scored constraint because EchoVLM's second
    # paper-specific objective explicitly separates original and negated
    # captions; reference.json defines the metric but intentionally has no
    # paper value for it because no table reports this direct diagnostic.
    scores = {
        "experiments": {
            "toy_disease_zero_shot": {
                "results": {
                    name: {k: v for k, v in metrics.items() if k in {"auc", "precision", "recall", "negation_cosine"}}
                    for name, metrics in results.items()
                }
            },
        }
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
