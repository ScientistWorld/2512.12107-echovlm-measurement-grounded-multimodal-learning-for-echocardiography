"""Hidden-test evaluator (evaluator-owned; never imports method code).

Every method is a directory under the scoring root (default: checkpoints/)
that contains an ``export/`` folder:

    export/image_encoder.pt   TorchScript: float[B,3,112,112] in [0,1] -> float[B,D]
    export/text_encoder.pt    TorchScript: (long[B,L] input_ids, long[B,L] attention_mask) -> float[B,D]
    export/tokenizer.json     HuggingFace `tokenizers` file used to tokenize text
    export/export.json        {"max_length": L, "pad_id": int}

The evaluator runs the encoders itself, one input at a time, on hidden test
images (eval/test/test_images.npz) and on evaluator-owned text (hidden
zero-shot prompts and hidden retrieval captions). Labels, captions, prompts,
pools, thresholds and the test size are fixed here. Nothing a method writes
besides its encoders is read.

A method whose export is missing, does not load, crashes, returns a
non-[1,D] / non-finite / zero / dimension-mismatched embedding, or whose
disease discrimination is not significantly above chance (patient-level
permutation test) is scored as failed: worst value (0) in every metric of
every experiment.

Usage:
    python eval/test/evaluate.py [--root checkpoints] [--output scoring/scores_test.json] [--only NAME ...]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
IMAGE_SIZE = 112
DISEASES = ("lv_systolic_dysfunction", "lv_dilation", "la_enlargement")
VIEW_CLASSES = ("A2C", "A4C")
# Hidden zero-shot prompts (paraphrases differ from the visible ones in eval/train).
PROMPTS = {
    "lv_systolic_dysfunction": {
        "pos": ["reduced left ventricular ejection fraction", "left ventricular systolic dysfunction"],
        "neg": ["normal left ventricular ejection fraction", "no left ventricular systolic dysfunction"],
    },
    "lv_dilation": {
        "pos": ["left ventricular dilation", "enlarged left ventricular cavity"],
        "neg": ["normal left ventricular size", "no left ventricular dilation"],
    },
    "la_enlargement": {
        "pos": ["left atrial enlargement", "enlarged left atrium"],
        "neg": ["normal left atrial size", "no left atrial enlargement"],
    },
}
KNN_K = 20
KNN_TEMPERATURE = 0.07
N_PERMUTATIONS = 2000
GATE_ALPHA = 0.01
PERMUTATION_SEED = 20251212

EXPERIMENTS = {
    "disease_zero_shot": (
        "auc",
        "precision",
        "recall",
        "auc_lv_systolic_dysfunction",
        "auc_lv_dilation",
        "auc_la_enlargement",
    ),
    "image_text_retrieval": ("recall_at_5", "recall_at_10"),
    "view_knn": ("f1", "accuracy"),
}
WORST = 0.0  # every metric is a percentage in [0, 100], higher is better


class MethodFailure(Exception):
    pass


# ----------------------------------------------------------------------------- data


def load_test() -> tuple[np.ndarray, list[dict]]:
    images_path = HERE / "test_images.npz"
    meta_path = HERE / "test_meta.json"
    if not images_path.is_file() or not meta_path.is_file():
        raise SystemExit("hidden test set missing: run `python eval/test/prepare_test.py` (needs raw test patients)")
    with np.load(images_path) as z:
        images = z["images"]
    records = json.loads(meta_path.read_text())["records"]
    if images.shape != (len(records), IMAGE_SIZE, IMAGE_SIZE) or images.dtype != np.uint8:
        raise SystemExit(f"corrupt hidden test set: {images.shape} {images.dtype} vs {len(records)} records")
    return images, records


# ----------------------------------------------------------------------------- method loading / embedding


def load_method(method_dir: Path, device: str):
    export = method_dir / "export"
    for name in ("image_encoder.pt", "text_encoder.pt", "tokenizer.json", "export.json"):
        if not (export / name).is_file():
            raise MethodFailure(f"missing export/{name}")
    cfg = json.loads((export / "export.json").read_text())
    max_length = int(cfg["max_length"])
    pad_id = int(cfg.get("pad_id", 0))
    if not 1 <= max_length <= 2048:
        raise MethodFailure(f"invalid max_length {max_length}")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(export / "tokenizer.json"))
    tok.enable_truncation(max_length=max_length)
    tok.enable_padding(pad_id=pad_id, length=max_length)
    image_encoder = torch.jit.load(str(export / "image_encoder.pt"), map_location=device).eval()
    text_encoder = torch.jit.load(str(export / "text_encoder.pt"), map_location=device).eval()
    return tok, image_encoder, text_encoder


def _row(out, what: str) -> np.ndarray:
    if not isinstance(out, torch.Tensor) or out.dim() != 2 or out.shape[0] != 1 or out.shape[1] < 1:
        raise MethodFailure(f"{what} must return a float tensor of shape [1, D]")
    row = out.detach().to("cpu", torch.float64)[0].numpy()
    if not np.all(np.isfinite(row)):
        raise MethodFailure(f"{what} returned non-finite values")
    return row


def embed_images(encoder, images: np.ndarray, device: str) -> np.ndarray:
    rows = []
    with torch.inference_mode():
        for img in images:
            x = torch.from_numpy(img).to(device=device, dtype=torch.float32).div_(255.0)
            x = x.view(1, 1, IMAGE_SIZE, IMAGE_SIZE).expand(1, 3, IMAGE_SIZE, IMAGE_SIZE).contiguous()
            rows.append(_row(encoder(x), "image_encoder"))
    return _stack(rows, "image_encoder")


def embed_texts(tok, encoder, texts: list[str], device: str) -> np.ndarray:
    rows = []
    with torch.inference_mode():
        for text in texts:
            enc = tok.encode(text)
            ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
            mask = torch.tensor([enc.attention_mask], dtype=torch.long, device=device)
            rows.append(_row(encoder(ids, mask), "text_encoder"))
    return _stack(rows, "text_encoder")


def _stack(rows: list[np.ndarray], what: str) -> np.ndarray:
    dims = {r.shape[0] for r in rows}
    if len(dims) != 1:
        raise MethodFailure(f"{what} returned inconsistent embedding sizes {sorted(dims)}")
    return np.stack(rows)


def l2_normalize(x: np.ndarray, what: str) -> np.ndarray:
    scale = np.max(np.abs(x), axis=1, keepdims=True)
    if np.any(scale <= 0):
        raise MethodFailure(f"{what} returned an all-zero embedding")
    x = x / scale
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ----------------------------------------------------------------------------- metrics


def average_ranks(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    bounds = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    sizes = np.diff(bounds)
    avg = (bounds[:-1] + 1 + bounds[1:]) / 2.0
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.repeat(avg, sizes)
    return ranks


def auc_from_ranks(ranks: np.ndarray, y: np.ndarray) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def zero_shot(img: np.ndarray, protos: dict, records: list[dict]) -> tuple[dict, dict]:
    patients = sorted({r["patient"] for r in records})
    pidx = np.array([patients.index(r["patient"]) for r in records])
    n_img = np.bincount(pidx, minlength=len(patients)).astype(np.float64)
    out, per_disease_ranks, patient_labels = {}, [], []
    aucs, precs, recs = [], [], []
    for d in DISEASES:
        y = np.array([int(r["labels"][d]) for r in records])
        score = img @ protos[d]["pos"] - img @ protos[d]["neg"]
        pred = score > 0
        ranks = average_ranks(score)
        auc = auc_from_ranks(ranks, y)
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        precs.append(tp / (tp + fp) if tp + fp else 0.0)
        recs.append(tp / (tp + fn) if tp + fn else 0.0)
        aucs.append(auc)
        out[f"auc_{d}"] = 100.0 * auc
        per_disease_ranks.append(np.bincount(pidx, weights=ranks, minlength=len(patients)))
        yp = np.zeros(len(patients))
        yp[pidx] = y
        patient_labels.append(yp)
    out["auc"] = 100.0 * float(np.mean(aucs))
    out["precision"] = 100.0 * float(np.mean(precs))
    out["recall"] = 100.0 * float(np.mean(recs))

    # Patient-level permutation test of the macro AUC (labels move with their patient).
    rng = np.random.default_rng(PERMUTATION_SEED)
    perms = np.stack([rng.permutation(len(patients)) for _ in range(N_PERMUTATIONS)])
    n_total = float(len(records))
    null = np.zeros(N_PERMUTATIONS)
    for rank_sum, yp in zip(per_disease_ranks, patient_labels):
        yperm = yp[perms]  # [n_perm, n_patients]
        n1 = yperm @ n_img
        n0 = n_total - n1
        with np.errstate(divide="ignore", invalid="ignore"):
            auc_perm = (yperm @ rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)
        null += np.where((n1 > 0) & (n0 > 0), auc_perm, 0.5)
    null /= len(DISEASES)
    observed = float(np.mean(aucs))
    p_value = float((1 + np.sum(null >= observed - 1e-12)) / (1 + N_PERMUTATIONS))
    return out, {"macro_auc": observed, "permutation_p_value": p_value}


def retrieval(img: np.ndarray, cap: np.ndarray, pools: np.ndarray) -> dict:
    r5, r10 = [], []
    for pool in np.unique(pools):
        idx = np.flatnonzero(pools == pool)
        sims = img[idx] @ cap[idx].T
        diag = np.diag(sims)[:, None]
        # Pessimistic ties: any other caption scoring >= the true one ranks ahead.
        rank = 1 + np.sum(sims >= diag, axis=1) - 1
        r5.append(np.mean(rank <= 5))
        r10.append(np.mean(rank <= 10))
    return {"recall_at_5": 100.0 * float(np.mean(r5)), "recall_at_10": 100.0 * float(np.mean(r10))}


def view_knn(img: np.ndarray, records: list[dict], pools: np.ndarray) -> dict:
    views = np.array([VIEW_CLASSES.index(r["view"]) for r in records])
    patients = np.array([r["patient"] for r in records])
    accs, f1s = [], []
    for pool in np.unique(pools):
        idx = np.flatnonzero(pools == pool)
        sims = img[idx] @ img[idx].T
        same = patients[idx][:, None] == patients[idx][None, :]
        sims = np.where(same, -np.inf, sims)
        order = np.argsort(-sims, axis=1, kind="mergesort")[:, :KNN_K]
        top = np.take_along_axis(sims, order, axis=1)
        w = np.exp((top - 1.0) / KNN_TEMPERATURE)
        votes = np.zeros((len(idx), len(VIEW_CLASSES)))
        for c in range(len(VIEW_CLASSES)):
            votes[:, c] = np.sum(w * (views[idx][order] == c), axis=1)
        pred = np.argmax(votes, axis=1)
        true = views[idx]
        accs.append(np.mean(pred == true))
        f1 = []
        for c in range(len(VIEW_CLASSES)):
            tp = np.sum((pred == c) & (true == c))
            fp = np.sum((pred == c) & (true != c))
            fn = np.sum((pred != c) & (true == c))
            f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
        f1s.append(np.mean(f1))
    return {"f1": 100.0 * float(np.mean(f1s)), "accuracy": 100.0 * float(np.mean(accs))}


# ----------------------------------------------------------------------------- scoring


def worst_scores() -> dict:
    return {exp: {m: WORST for m in metrics} for exp, metrics in EXPERIMENTS.items()}


def score_method(method_dir: Path, images: np.ndarray, records: list[dict], device: str) -> tuple[dict, dict]:
    tok, image_encoder, text_encoder = load_method(method_dir, device)
    img = l2_normalize(embed_images(image_encoder, images, device), "image_encoder")
    prompt_texts = [t for d in DISEASES for side in ("pos", "neg") for t in PROMPTS[d][side]]
    captions = [r["caption"] for r in records]
    txt = embed_texts(tok, text_encoder, prompt_texts + captions, device)
    if txt.shape[1] != img.shape[1]:
        raise MethodFailure(f"image dim {img.shape[1]} != text dim {txt.shape[1]}")
    txt = l2_normalize(txt, "text_encoder")
    prompt_emb, cap = txt[: len(prompt_texts)], txt[len(prompt_texts) :]
    protos, k = {}, 0
    for d in DISEASES:
        protos[d] = {}
        for side in ("pos", "neg"):
            n = len(PROMPTS[d][side])
            v = prompt_emb[k : k + n].mean(axis=0)
            k += n
            norm = np.linalg.norm(v)
            if norm <= 0:
                raise MethodFailure("degenerate prompt embedding")
            protos[d][side] = v / norm
    pools = np.array([int(r["pool"]) for r in records])
    zs, gate = zero_shot(img, protos, records)
    scores = {
        "disease_zero_shot": {m: zs[m] for m in EXPERIMENTS["disease_zero_shot"]},
        "image_text_retrieval": retrieval(img, cap, pools),
        "view_knn": view_knn(img, records, pools),
    }
    status = {"status": "ok", **gate, "embed_dim": int(img.shape[1])}
    if gate["permutation_p_value"] > GATE_ALPHA:
        status["status"] = "failed_task_gate"
        status["reason"] = (
            f"macro AUC {100 * gate['macro_auc']:.2f} not significantly above chance "
            f"(p={gate['permutation_p_value']:.4f} > {GATE_ALPHA})"
        )
        scores = worst_scores()
    return scores, status


def discover(root: Path, only: list[str] | None) -> list[Path]:
    if not root.is_dir():
        return []
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))
    if only:
        dirs = [p for p in dirs if p.name in set(only)]
    return dirs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="checkpoints", help="directory whose sub-directories are methods")
    ap.add_argument("--output", default="scoring/scores_test.json")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    images, records = load_test()
    methods = discover(Path(args.root), args.only)
    results = {exp: {"results": {}} for exp in EXPERIMENTS}
    statuses = {}
    for method_dir in methods:
        name = method_dir.name
        t0 = time.time()
        tried = [device] if device == "cpu" else [device, "cpu"]
        scores, status = None, None
        for dev in tried:
            try:
                scores, status = score_method(method_dir, images, records, dev)
                break
            except MethodFailure as exc:
                scores, status = worst_scores(), {"status": "failed", "reason": str(exc)}
                break  # invalid output is not a device problem
            except Exception as exc:  # crash: retry once on CPU, then fail
                scores = worst_scores()
                status = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:500]}
                traceback.print_exc()
        for exp in EXPERIMENTS:
            results[exp]["results"][name] = scores[exp]
        status["seconds"] = round(time.time() - t0, 1)
        statuses[name] = status
        print(json.dumps({"method": name, **status, **{e: scores[e] for e in EXPERIMENTS}}), flush=True)

    out = {
        "experiments": results,
        "_meta": {
            "split": "hidden_test",
            "n_images": len(records),
            "n_patients": len({r["patient"] for r in records}),
            "root": str(args.root),
            "method_status": statuses,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
