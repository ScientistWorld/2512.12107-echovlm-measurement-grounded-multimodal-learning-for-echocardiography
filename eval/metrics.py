"""Artifact-only evaluator for EchoVLM-style outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


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


def recall_at_k(image_z: torch.Tensor, text_z: torch.Tensor, k: int) -> float:
    sims = image_z @ text_z.T
    topk = sims.topk(min(k, sims.shape[1]), dim=1).indices
    target = torch.arange(sims.shape[0], device=sims.device)[:, None]
    return topk.eq(target).any(dim=1).float().mean().item()


def evaluate_embeddings(path: Path) -> dict[str, float]:
    rows = load_jsonl(path)
    image_z = torch.tensor([r["image_embedding"] for r in rows], dtype=torch.float32)
    text_z = torch.tensor([r["text_embedding"] for r in rows], dtype=torch.float32)
    neg_z = torch.tensor([r["neg_text_embedding"] for r in rows], dtype=torch.float32)
    labels = [int(r["label"]) for r in rows]

    pos_sim = (image_z * text_z).sum(dim=1)
    neg_sim = (image_z * neg_z).sum(dim=1)
    scores = (pos_sim - neg_sim).tolist()
    precision, recall = pr_metrics(labels, scores)
    neg_margin = (text_z * neg_z).sum(dim=1).mean().item()
    align_margin = (pos_sim - neg_sim).mean().item()
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

    # The toy retrieval pool and embedding-margin diagnostics are useful for
    # debugging, but they are not on the same scale as the paper's reported
    # full-test retrieval and no paper table reports a direct negation margin.
    # Keep the automated score contract limited to paper-comparable disease
    # metrics from Table 1/Table 6.
    scores = {
        "experiments": {
            "toy_disease_zero_shot": {
                "results": {
                    name: {k: v for k, v in metrics.items() if k in {"auc", "precision", "recall"}}
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
