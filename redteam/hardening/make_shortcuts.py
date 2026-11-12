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

Usage: python redteam/hardening/make_shortcuts.py --ref RUN_DIR --out ROOT
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

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
    print("shortcuts written to", root)


if __name__ == "__main__":
    main()
