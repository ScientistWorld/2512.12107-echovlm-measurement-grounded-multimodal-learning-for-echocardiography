"""Visible-split evaluator (same protocol as the hidden test, visible data).

Every method is a directory under the scoring root (default: checkpoints/)
that contains an ``export/`` folder:

    export/image_encoder.pt   TorchScript: float[B,3,112,112] in [0,1] -> float[B,D]
    export/text_encoder.pt    TorchScript: (long[B,L] input_ids, long[B,L] attention_mask) -> float[B,D]
    export/tokenizer.json     HuggingFace `tokenizers` file used to tokenize text
    export/export.json        {"max_length": L, "pad_id": int}

The evaluator runs the encoders itself, one input at a time, on the visible
validation patients (data/downloads/camus_vlm/val_*) with the visible
zero-shot prompts below. The hidden test uses different patients (150 vs 50),
different prompt paraphrases and different caption wordings, but the same
metrics, pool size (50 patients) and failure rules.

Encoders must be pure functions of their input:
  * their TorchScript graphs may not read files, keep Python state between
    calls (prim::SetAttr), fork, or call opaque functions;
  * inputs are presented in a fixed shuffled order and a probe subset is
    embedded again at the end; any embedding that changed is a failure.

A method whose export is missing, does not load, fails these checks, crashes,
returns a non-[1,D] / non-finite / zero / dimension-mismatched embedding, or
whose disease discrimination is not significantly above chance (patient-level
permutation test) is scored as failed: worst value (0) in every metric of
every experiment.

Usage:
    python eval/train/evaluate.py [--root checkpoints] [--output scoring/scores_train.json] [--only NAME ...]
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

DATA_DIR = Path("data/downloads/camus_vlm")
IMAGE_SIZE = 112
DISEASES = ("lv_systolic_dysfunction", "lv_dilation", "la_enlargement")
VIEW_CLASSES = ("A2C", "A4C")
# Visible zero-shot prompts.
PROMPTS = {
    "lv_systolic_dysfunction": {
        "pos": ["reduced LV systolic function", "LV systolic dysfunction"],
        "neg": ["normal LV systolic function", "no LV systolic dysfunction"],
    },
    "lv_dilation": {
        "pos": ["dilated left ventricle", "LV dilation"],
        "neg": ["normal LV size", "no LV dilation"],
    },
    "la_enlargement": {
        "pos": ["dilated left atrium", "LA enlargement"],
        "neg": ["normal LA size", "no LA enlargement"],
    },
}
KNN_K = 20
KNN_TEMPERATURE = 0.07
N_PERMUTATIONS = 2000
# 50 visible patients instead of 150 hidden ones: same test, looser level.
GATE_ALPHA = 0.05
PERMUTATION_SEED = 7
ORDER_SEED = 11
N_PROBES = 32
PROBE_MIN_COSINE = 0.999
# TorchScript node kinds an encoder may not contain (after inlining).
FORBIDDEN_KINDS = {
    "prim::SetAttr",  # Python-level state carried between calls
    "prim::CallFunction",  # opaque calls that survive inlining
    "prim::CallMethod",
    "prim::PythonOp",
    "prim::fork",
    "aten::wait",
}
FORBIDDEN_SUBSTRINGS = ("from_file", "pickle", "rpc")  # file / remote access

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


def load_split(data_dir: Path) -> tuple[np.ndarray, list[dict]]:
    images_path = data_dir / "val_images.npy"
    meta_path = data_dir / "val.jsonl"
    if not images_path.is_file() or not meta_path.is_file():
        raise SystemExit("visible split missing: run `python -m data.prepare_camus` first")
    images = np.load(images_path)
    records = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
    if images.shape != (len(records), IMAGE_SIZE, IMAGE_SIZE) or images.dtype != np.uint8:
        raise SystemExit(f"corrupt visible split: {images.shape} {images.dtype} vs {len(records)} records")
    for r in records:
        r["pool"] = 0  # all 50 visible patients form one pool
    return images, records


# ----------------------------------------------------------------------------- method loading / checks


def _graph_kinds(graph) -> set[str]:
    kinds = set()
    stack = [graph]
    while stack:
        g = stack.pop()
        for node in g.nodes():
            kinds.add(node.kind())
            for block in node.blocks():
                stack.append(block)
    return kinds


def check_pure(module, what: str) -> None:
    """Reject encoders whose (inlined) forward graph can read files or keep state."""
    try:
        graph = module.inlined_graph
        kinds = _graph_kinds(graph)
    except Exception as exc:  # cannot inspect -> cannot trust
        raise MethodFailure(f"{what}: TorchScript graph cannot be inspected ({type(exc).__name__})")
    bad = sorted(k for k in kinds if k in FORBIDDEN_KINDS or any(s in k.lower() for s in FORBIDDEN_SUBSTRINGS))
    if bad:
        raise MethodFailure(f"{what}: forbidden TorchScript operations {bad[:5]}")


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
    check_pure(image_encoder, "image_encoder")
    check_pure(text_encoder, "text_encoder")
    return tok, image_encoder, text_encoder


def _row(out, what: str) -> np.ndarray:
    if not isinstance(out, torch.Tensor) or out.dim() != 2 or out.shape[0] != 1 or out.shape[1] < 1:
        raise MethodFailure(f"{what} must return a float tensor of shape [1, D]")
    row = out.detach().to("cpu", torch.float64)[0].numpy()
    if not np.all(np.isfinite(row)):
        raise MethodFailure(f"{what} returned non-finite values")
    return row


def _image_input(img: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(img).to(device=device, dtype=torch.float32).div_(255.0)
    return x.view(1, 1, IMAGE_SIZE, IMAGE_SIZE).expand(1, 3, IMAGE_SIZE, IMAGE_SIZE).contiguous()


def _text_input(tok, text: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    enc = tok.encode(text)
    ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
    mask = torch.tensor([enc.attention_mask], dtype=torch.long, device=device)
    return ids, mask


def _embed(call, n: int, rng: np.random.Generator, what: str) -> np.ndarray:
    """Embed n inputs one at a time in a fixed shuffled order, then re-embed probes."""
    order = rng.permutation(n)
    rows: list[np.ndarray | None] = [None] * n
    with torch.inference_mode():
        for i in order:
            rows[i] = _row(call(int(i)), what)
        dims = {r.shape[0] for r in rows}
        if len(dims) != 1:
            raise MethodFailure(f"{what} returned inconsistent embedding sizes {sorted(dims)}")
        out = np.stack(rows)
        k = min(n, N_PROBES)
        probes = np.unique(np.concatenate([order[: k // 4], rng.choice(n, size=k, replace=False)]))
        for i in probes:
            again = _row(call(int(i)), what)
            a, b = out[i], again
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na <= 0 or nb <= 0 or float(a @ b) / (na * nb) < PROBE_MIN_COSINE:
                raise MethodFailure(f"{what} is not a pure function of its input (embedding changed on re-evaluation)")
    return out


def embed_images(encoder, images: np.ndarray, device: str, rng: np.random.Generator) -> np.ndarray:
    return _embed(lambda i: encoder(_image_input(images[i], device)), len(images), rng, "image_encoder")


def embed_texts(tok, encoder, texts: list[str], device: str, rng: np.random.Generator) -> np.ndarray:
    return _embed(lambda i: encoder(*_text_input(tok, texts[i], device)), len(texts), rng, "text_encoder")


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


def disease_scores(img: np.ndarray, protos: dict) -> np.ndarray:
    """[N, n_diseases] positive-minus-negative prompt similarity."""
    return np.stack([img @ protos[d]["pos"] - img @ protos[d]["neg"] for d in DISEASES], axis=1)


def zero_shot(scores: np.ndarray, records: list[dict]) -> tuple[dict, dict]:
    patients = sorted({r["patient"] for r in records})
    index = {p: i for i, p in enumerate(patients)}
    pidx = np.array([index[r["patient"]] for r in records])
    n_img = np.bincount(pidx, minlength=len(patients)).astype(np.float64)
    out, per_disease_ranks, patient_labels = {}, [], []
    aucs, precs, recs = [], [], []
    for j, d in enumerate(DISEASES):
        y = np.array([int(r["labels"][d]) for r in records])
        score = scores[:, j]
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


def score_method(method_dir: Path, images: np.ndarray, records: list[dict], device: str, dump: Path | None):
    tok, image_encoder, text_encoder = load_method(method_dir, device)
    rng = np.random.default_rng(ORDER_SEED)
    img = l2_normalize(embed_images(image_encoder, images, device, rng), "image_encoder")
    prompt_texts = [t for d in DISEASES for side in ("pos", "neg") for t in PROMPTS[d][side]]
    captions = [r["caption"] for r in records]
    txt = embed_texts(tok, text_encoder, prompt_texts + captions, device, rng)
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
    dscores = disease_scores(img, protos)
    zs, gate = zero_shot(dscores, records)
    if dump is not None:
        dump.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dump / f"{method_dir.name}.npz",
            disease_scores=dscores.astype(np.float32),
            image=img.astype(np.float16),
            caption=cap.astype(np.float16),
        )
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
    ap.add_argument("--output", default="scoring/scores_train.json")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dump-dir", default=None, help="optional: write per-image scores/embeddings per method")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    images, records = load_split(Path(args.data_dir))
    methods = discover(Path(args.root), args.only)
    dump = Path(args.dump_dir) if args.dump_dir else None
    results = {exp: {"results": {}} for exp in EXPERIMENTS}
    statuses = {}
    for method_dir in methods:
        name = method_dir.name
        t0 = time.time()
        tried = [device] if device == "cpu" else [device, "cpu"]
        scores, status = None, None
        for dev in tried:
            try:
                scores, status = score_method(method_dir, images, records, dev, dump)
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
            "split": "visible_val",
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
