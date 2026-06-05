"""Visible train-slice artifact evaluator.

This evaluator intentionally scores precomputed embedding artifacts only. It
does not import from method/ or run model inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SLICE = "train"
EXPERIMENT_NAME = "toy_disease_zero_shot"
TRAIN_IDS = {
    "echo_toy_0004",
    "echo_toy_0009",
    "echo_toy_0012",
    "echo_toy_0019",
    "echo_toy_0027",
    "echo_toy_0041",
    "echo_toy_0074",
}
SLICE_IDS = TRAIN_IDS


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_slice(rows: list[dict]) -> list[dict]:
    matched = [row for row in rows if row.get("id") in SLICE_IDS]
    if matched:
        missing = sorted(SLICE_IDS - {row.get("id") for row in matched})
        if missing:
            raise ValueError(f"{SLICE} slice input is missing expected ids: {missing}")
        return matched
    return [row for i, row in enumerate(rows) if i % 2 == 0]


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


def threshold_metrics(labels: list[int], scores: list[float]) -> tuple[float, float, float]:
    preds = [int(score >= 0.0) for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    balanced_accuracy = 0.5 * (recall + specificity)
    return precision, recall, balanced_accuracy


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def retrieval_mrr(image_z: list[list[float]], text_z: list[list[float]]) -> float:
    reciprocal_ranks = []
    for i, image_vec in enumerate(image_z):
        ranked = sorted(
            ((dot(image_vec, text_vec), j) for j, text_vec in enumerate(text_z)),
            reverse=True,
        )
        rank = next(rank for rank, (_, j) in enumerate(ranked, start=1) if j == i)
        reciprocal_ranks.append(1.0 / rank)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def evaluate_embeddings(path: Path) -> dict[str, float]:
    rows = select_slice(load_jsonl(path))
    if not rows:
        raise ValueError(f"{path} has no rows for the {SLICE} slice")
    image_z = [row["image_embedding"] for row in rows]
    text_z = [row["text_embedding"] for row in rows]
    neg_z = [row["neg_text_embedding"] for row in rows]
    labels = [int(row["label"]) for row in rows]

    pos_sim = [dot(image_vec, text_vec) for image_vec, text_vec in zip(image_z, text_z)]
    neg_sim = [dot(image_vec, neg_vec) for image_vec, neg_vec in zip(image_z, neg_z)]
    scores = [pos - neg for pos, neg in zip(pos_sim, neg_sim)]
    precision, recall, balanced_accuracy = threshold_metrics(labels, scores)
    negation_cosine = sum(dot(text_vec, neg_vec) for text_vec, neg_vec in zip(text_z, neg_z)) / len(rows)
    hard_negative_accuracy = sum(1 for score in scores if score > 0.0) / len(rows)
    margin = sum(scores) / len(rows)
    return {
        "auc": 100.0 * binary_auc(labels, scores),
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "balanced_accuracy": 100.0 * balanced_accuracy,
        "caption_retrieval_mrr": 100.0 * retrieval_mrr(image_z, text_z),
        "hard_negative_accuracy": 100.0 * hard_negative_accuracy,
        "negation_cosine": negation_cosine,
        "positive_vs_negated_margin": margin,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="name=path/to/test_embeddings.jsonl")
    parser.add_argument("--output", default="scoring/scores_train.json")
    args = parser.parse_args()

    results = {}
    input_paths = {}
    for item in args.inputs:
        if "=" not in item:
            raise ValueError(f"Expected name=path input, got {item}")
        name, path_text = item.split("=", 1)
        path = Path(path_text)
        results[name] = evaluate_embeddings(path)
        input_paths[name] = str(path)

    scores = {
        "experiments": {
            EXPERIMENT_NAME: {
                "slice": SLICE,
                "split": {
                    "style": "fixed_id_half_split",
                    "ids": sorted(SLICE_IDS),
                    "input_paths": input_paths,
                },
                "results": results,
            }
        }
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
