"""Build score-level ensembles of trained exports (a candidate "genuinely better" variant).

The ensemble's image / text encoders return the concatenation of each member's
L2-normalised embedding divided by sqrt(K). Dot products between ensemble
embeddings are therefore the mean of the members' cosine similarities, so
zero-shot scores, retrieval and k-NN all average over the members.

Usage:
    python redteam/hardening/make_variants.py --members RUN0 RUN1 RUN2 --out ROOT/NAME
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class EnsImage(nn.Module):
    def __init__(self, members: list):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.scale = 1.0 / math.sqrt(len(members))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for m in self.members:
            outs.append(F.normalize(m(x), dim=-1))
        return torch.cat(outs, dim=1) * self.scale


class EnsText(nn.Module):
    def __init__(self, members: list):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.scale = 1.0 / math.sqrt(len(members))

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        outs = []
        for m in self.members:
            outs.append(F.normalize(m(ids, mask), dim=-1))
        return torch.cat(outs, dim=1) * self.scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    exports = [Path(m) / "export" for m in args.members]
    cfgs = [json.loads((e / "export.json").read_text()) for e in exports]
    toks = {(e / "tokenizer.json").read_bytes() for e in exports}
    if len(toks) != 1 or len({(c["max_length"], c.get("pad_id", 0)) for c in cfgs}) != 1:
        raise SystemExit("ensemble members must share tokenizer and max_length")
    images = [torch.jit.load(str(e / "image_encoder.pt"), map_location="cpu") for e in exports]
    texts = [torch.jit.load(str(e / "text_encoder.pt"), map_location="cpu") for e in exports]
    out = Path(args.out) / "export"
    if out.parent.exists():
        shutil.rmtree(out.parent)
    out.mkdir(parents=True)
    torch.jit.script(EnsImage(images)).save(str(out / "image_encoder.pt"))
    torch.jit.script(EnsText(texts)).save(str(out / "text_encoder.pt"))
    shutil.copyfile(exports[0] / "tokenizer.json", out / "tokenizer.json")
    cfg = dict(cfgs[0], embed_dim=int(cfgs[0].get("embed_dim", 512)) * len(exports))
    (out / "export.json").write_text(json.dumps(cfg, indent=2))
    (out.parent / "members.json").write_text(json.dumps([str(m) for m in args.members], indent=2))
    print(json.dumps({"ensemble": str(out.parent), "members": len(exports)}))


if __name__ == "__main__":
    main()
