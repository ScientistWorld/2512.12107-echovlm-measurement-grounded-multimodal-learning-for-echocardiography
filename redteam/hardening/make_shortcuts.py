"""Build shortcut submissions against the hardened interface.

Each shortcut is a method directory (like checkpoints/<name>/) derived from a
genuine reference export. They re-express the red-team shortcuts in the new
interface (the evaluator now runs the encoders itself on hidden inputs):

  sc_missing_export   directory without export/ (old: missing artifact)
  sc_forged_scores    perfect score file + manifest, no encoders (old: forged scores)
  sc_nan_outputs      reference encoders returning NaN (old: null vectors)
  sc_empty_outputs    image encoder returning [1, 0] tensors (old: empty vectors)
  sc_wrong_shape      image encoder returning a 1-D vector (malformed output)
  sc_dim_mismatch     image embedding one dimension longer than text
  sc_constant         constant image and text embeddings (old: constant antipodes)
  sc_norm_scaled      reference embeddings multiplied by 1e6 (old: norm scaling)
  sc_huge_constant    constant 1e9-scaled embeddings (old: constant scaled failure)
  sc_text_only        constant image embedding + reference text encoder (old: text-only)
  sc_image_only       reference image encoder + constant text embedding

Added in the final hardening pass (attacks on the encoder interface itself):

  sc_file_reader      encoders that read the hidden ground truth from disk while
                      they run (torch.from_file on the plaintext test metadata the
                      previous evaluator kept in eval/test/) and use call counters
                      to map each input to its record -> perfect answers
  sc_stateful         reference encoder + in-place buffers that pool embeddings
                      over consecutive calls (the previous evaluator presented
                      each patient's 10 frames consecutively) -> patient-level
                      aggregation the evaluator never asked for
  sc_handcrafted      a short hand-written rule: dark-pixel fractions in fixed
                      LV / LA regions of the image (thresholds set on the visible
                      training split) + a keyword text encoder

Usage: python redteam/hardening/make_shortcuts.py --ref RUN_DIR --out ROOT
           [--leak-file eval/test/legacy_v1/test_meta.json] [--train-data data/downloads/camus_vlm]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List

import torch
from torch import nn


class Scaled(nn.Module):
    def __init__(self, inner, factor: float):
        super().__init__()
        self.inner = inner
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x) * self.factor


class ScaledText(nn.Module):
    def __init__(self, inner, factor: float):
        super().__init__()
        self.inner = inner
        self.factor = factor

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.inner(ids, mask) * self.factor


class ConstImage(nn.Module):
    def __init__(self, dim: int, value: float):
        super().__init__()
        self.dim = dim
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], self.dim), self.value, dtype=x.dtype, device=x.device)


class ConstText(nn.Module):
    def __init__(self, dim: int, value: float):
        super().__init__()
        self.dim = dim
        self.value = value

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.full((ids.shape[0], self.dim), self.value, dtype=torch.float32, device=ids.device)


class PadImage(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inner(x)
        return torch.cat([z, torch.zeros_like(z[:, :1])], dim=1)


class FlatImage(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x).reshape(-1)


class EmptyImage(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros((x.shape[0], 0))


class LeakImage(nn.Module):
    """Reads the plaintext ground truth with torch.from_file; maps call i -> record i."""

    def __init__(self, path: str, nbytes: int, label_pattern: List[int], offsets: List[int], view_pattern: List[int], dim: int):
        super().__init__()
        self.path = path
        self.nbytes = nbytes
        self.label_pattern = label_pattern
        self.offsets = offsets
        self.view_pattern = view_pattern
        self.dim = dim
        self.counter = 0
        self.loaded = False
        self.codes = torch.zeros(1, 4)

    def _positions(self, raw: torch.Tensor, pattern: List[int]) -> torch.Tensor:
        pat = torch.tensor(pattern, dtype=torch.uint8)
        win = raw.unfold(0, len(pattern), 1)
        hit = (win == pat.unsqueeze(0)).all(dim=1)
        return torch.nonzero(hit).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.loaded:
            raw = torch.from_file(self.path, False, self.nbytes, dtype=torch.uint8)
            lab = self._positions(raw, self.label_pattern)
            vpos = self._positions(raw, self.view_pattern)
            n = min(int(lab.shape[0]), int(vpos.shape[0]))
            codes = torch.zeros(n, 4)
            for j in range(3):
                codes[:, j] = raw[lab[:n] + self.offsets[j]].float() - 48.0
            codes[:, 3] = (raw[vpos[:n] + len(self.view_pattern)].float() - 50.0) / 2.0
            self.codes = codes
            self.loaded = True
        i = self.counter
        self.counter = i + 1
        out = torch.zeros(x.shape[0], self.dim, device=x.device)
        out[:, 4] = 1.0
        if i < int(self.codes.shape[0]):
            c = self.codes[i].to(x.device)
            out[:, 0:4] = 2.0 * c.unsqueeze(0) - 1.0
        return out


class LeakText(nn.Module):
    """Call counter: the evaluator embeds the 12 zero-shot prompts first (pos, pos, neg, neg per disease)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.counter = 0

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        i = self.counter
        self.counter = i + 1
        out = torch.zeros(ids.shape[0], self.dim, device=ids.device)
        if i < 12:
            sign = 1.0 if (i % 4) < 2 else -1.0
            out[:, i // 4] = sign
        else:
            out[:, 5] = 1.0
        return out


class PooledImage(nn.Module):
    """Reference image encoder + running mean over blocks of 10 consecutive calls (in-place buffers)."""

    def __init__(self, inner, dim: int, block: int):
        super().__init__()
        self.inner = inner
        self.block = block
        self.register_buffer("count", torch.zeros(1))
        self.register_buffer("acc", torch.zeros(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inner(x)
        z = z / z.norm(dim=-1, keepdim=True)
        c = int(self.count[0])
        if c % self.block == 0:
            self.acc.zero_()
        self.acc.add_(z.to(self.acc.device))
        self.count.add_(1.0)
        return (self.acc / float(c % self.block + 1)).to(z.device)


class HandImage(nn.Module):
    """Hand-written rule: dark (blood-pool) pixel fractions in fixed upper (LV) and lower (LA) regions."""

    def __init__(self, thr: float, centres: List[float], lv_rows: List[int], la_rows: List[int], cols: List[int], dim: int):
        super().__init__()
        self.thr = thr
        self.centres = centres
        self.lv_rows = lv_rows
        self.la_rows = la_rows
        self.cols = cols
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x[:, 0]
        pool = ((g > 0.004) & (g < self.thr)).float()
        f_lv = pool[:, self.lv_rows[0] : self.lv_rows[1], self.cols[0] : self.cols[1]].mean(dim=[1, 2])
        f_la = pool[:, self.la_rows[0] : self.la_rows[1], self.cols[0] : self.cols[1]].mean(dim=[1, 2])
        out = torch.zeros(x.shape[0], self.dim, device=x.device)
        out[:, 0] = 10.0 * (f_lv - self.centres[0])  # large LV blood pool ~ dilated, poorly contracting LV
        out[:, 1] = 10.0 * (f_lv - self.centres[1])
        out[:, 2] = 10.0 * (f_la - self.centres[2])
        out[:, 3] = 8.0
        return out


class KeywordText(nn.Module):
    """Keyword rule on token ids: which finding a phrase is about, and whether it is negated / normal."""

    def __init__(self, la: List[int], ef: List[int], lv: List[int], negators: List[int], dim: int):
        super().__init__()
        self.la = la
        self.ef = ef
        self.lv = lv
        self.negators = negators
        self.dim = dim

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        toks: List[int] = ids[0].tolist()
        d = 3
        for t in toks:
            if t in self.la:
                d = 2
        if d == 3:
            for t in toks:
                if t in self.ef:
                    d = 0
        if d == 3:
            for t in toks:
                if t in self.lv:
                    d = 1
        sign = 1.0
        for t in toks:
            if t in self.negators:
                sign = -1.0
        out = torch.zeros(ids.shape[0], self.dim, device=ids.device)
        out[:, d] = sign
        out[:, 4] = 0.01
        return out


def _keyword_ids(tokenizer_json: Path, words: list[str]) -> list[int]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_json))
    special = {tok.token_to_id(t) for t in ("<start_of_text>", "<end_of_text>", "<|startoftext|>", "<|endoftext|>")}
    ids = set()
    for w in words:
        for variant in (w, w.capitalize(), w.upper()):
            enc = [i for i in tok.encode(variant).ids if i not in special and i != 0]
            if len(enc) == 1:  # only words that are a single token, to avoid sub-word collisions
                ids.add(enc[0])
    return sorted(ids) or [-1]


def build_handcrafted(train_dir: Path, ref_export: Path, dim: int):
    """Fit the 3 thresholds of the hand-written rule on the visible training split."""
    import numpy as np

    images = np.load(train_dir / "train_images.npy", mmap_mode="r")
    records = [json.loads(l) for l in (train_dir / "train.jsonl").read_text().splitlines() if l.strip()]
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(records), size=min(3000, len(records)), replace=False))
    g = images[idx].astype(np.float32) / 255.0
    best = None
    for thr in (0.10, 0.14, 0.18, 0.22):
        pool = (g > 0.004) & (g < thr)
        f_lv = pool[:, 8:62, 30:82].mean(axis=(1, 2))
        f_la = pool[:, 62:100, 30:82].mean(axis=(1, 2))
        aucs = []
        for f, d in ((f_lv, "lv_systolic_dysfunction"), (f_lv, "lv_dilation"), (f_la, "la_enlargement")):
            y = np.array([records[i]["labels"][d] for i in idx])
            r = np.argsort(np.argsort(f)) + 1
            n1 = y.sum()
            aucs.append((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))
        if best is None or np.mean(aucs) > best[0]:
            best = (float(np.mean(aucs)), thr, f_lv, f_la)
    _, thr, f_lv, f_la = best
    centres = []
    for f, d in ((f_lv, "lv_systolic_dysfunction"), (f_lv, "lv_dilation"), (f_la, "la_enlargement")):
        prev = float(np.mean([records[i]["labels"][d] for i in idx]))
        centres.append(float(np.quantile(f, 1.0 - prev)))  # predict the visible prevalence as positive
    image = HandImage(thr, centres, [8, 62], [62, 100], [30, 82], dim)
    tj = ref_export / "tokenizer.json"
    text = KeywordText(
        _keyword_ids(tj, ["atrium", "atrial", "la", "atria"]),
        _keyword_ids(tj, ["ejection", "systolic", "dysfunction", "function", "fraction", "ef", "contractility"]),
        _keyword_ids(tj, ["ventricle", "ventricular", "lv", "dilated", "dilation", "enlarged", "enlargement", "size", "cavity"]),
        _keyword_ids(tj, ["normal", "no", "preserved", "not", "without"]),
        dim,
    )
    info = {"visible_train_auc_fit": round(best[0], 4), "threshold": thr, "centres": centres}
    return image, text, info


def build_leak(leak_file: Path, dim: int):
    text = leak_file.read_text()
    marker = '"labels": {"lv_systolic_dysfunction": '
    start = text.index(marker)
    block = text[start : text.index("}", start) + 1]
    offsets = [len(marker)]
    offsets.append(block.index('"lv_dilation": ') + len('"lv_dilation": '))
    offsets.append(block.index('"la_enlargement": ') + len('"la_enlargement": '))
    view_marker = '"view": "A'
    image = LeakImage(str(leak_file.resolve()), leak_file.stat().st_size, list(marker.encode()), offsets, list(view_marker.encode()), dim)
    return image, LeakText(dim)


def write(root: Path, name: str, image: nn.Module | None, text: nn.Module | None, ref_export: Path) -> None:
    out = root / name / "export"
    if out.parent.exists():
        shutil.rmtree(out.parent)
    out.mkdir(parents=True)
    for f in ("tokenizer.json", "export.json"):
        shutil.copyfile(ref_export / f, out / f)
    for module, fname in ((image, "image_encoder.pt"), (text, "text_encoder.pt")):
        if module is None:
            shutil.copyfile(ref_export / fname, out / fname)
        else:
            torch.jit.script(module).save(str(out / fname))
    (out.parent / "SHORTCUT.txt").write_text(f"{name}: see redteam/hardening/make_shortcuts.py\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--leak-file", default="eval/test/legacy_v1/test_meta.json")
    ap.add_argument("--train-data", default="data/downloads/camus_vlm")
    args = ap.parse_args()
    ref_export = Path(args.ref) / "export"
    root = Path(args.out)
    img = torch.jit.load(str(ref_export / "image_encoder.pt"), map_location="cpu")
    txt = torch.jit.load(str(ref_export / "text_encoder.pt"), map_location="cpu")
    dim = int(json.loads((ref_export / "export.json").read_text()).get("embed_dim", 512))

    # Directory-level shortcuts.
    for name in ("sc_missing_export", "sc_forged_scores"):
        d = root / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    perfect = {m: 100.0 for m in ("auc", "precision", "recall", "recall_at_5", "recall_at_10", "f1", "accuracy")}
    (root / "sc_forged_scores" / "scores_test.json").write_text(json.dumps({"experiments": {"all": perfect}}))
    (root / "sc_forged_scores" / "test_embeddings.jsonl").write_text('{"auc": 100}\n')

    write(root, "sc_nan_outputs", Scaled(img, float("nan")), None, ref_export)
    write(root, "sc_empty_outputs", EmptyImage(), None, ref_export)
    write(root, "sc_wrong_shape", FlatImage(img), None, ref_export)
    write(root, "sc_dim_mismatch", PadImage(img), None, ref_export)
    write(root, "sc_constant", ConstImage(dim, 1.0), ConstText(dim, 1.0), ref_export)
    write(root, "sc_norm_scaled", Scaled(img, 1e6), ScaledText(txt, 1e6), ref_export)
    write(root, "sc_huge_constant", ConstImage(dim, 1e9), ConstText(dim, -1e9), ref_export)
    write(root, "sc_text_only", ConstImage(dim, 1.0), None, ref_export)
    write(root, "sc_image_only", None, ConstText(dim, 1.0), ref_export)

    # Attacks on the encoder interface (each built independently).
    import traceback

    built = {}
    leak = Path(args.leak_file)
    try:
        li, lt = build_leak(leak, dim)
        write(root, "sc_file_reader", li, lt, ref_export)
        built["sc_file_reader"] = "ok"
    except Exception as exc:
        traceback.print_exc()
        built["sc_file_reader"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        write(root, "sc_stateful", PooledImage(img, dim, 10), None, ref_export)
        built["sc_stateful"] = "ok"
    except Exception as exc:
        traceback.print_exc()
        built["sc_stateful"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        hi, ht, info = build_handcrafted(Path(args.train_data), ref_export, dim)
        write(root, "sc_handcrafted", hi, ht, ref_export)
        (root / "sc_handcrafted" / "rule.json").write_text(json.dumps(info, indent=2))
        built["sc_handcrafted"] = info
    except Exception as exc:
        traceback.print_exc()
        built["sc_handcrafted"] = f"{type(exc).__name__}: {exc}"[:300]
    print("shortcuts written to", root, json.dumps(built))


if __name__ == "__main__":
    main()
