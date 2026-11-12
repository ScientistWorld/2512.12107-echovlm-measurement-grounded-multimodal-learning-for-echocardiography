"""CLIP-only baseline: fine-tune the same backbone with the image-text contrastive loss only.

This is the L_CLIP row of the paper's Table 6 ("Pretrain losses: L_CLIP"),
self-contained so it runs even when method/ is empty.

Paper settings (Sec. 3.2, App. Table 15): AdamW (0.9, 0.999), lr 1e-4,
weight decay 0.005 (Table 15; the main text says 0.05), global batch 512,
cosine decay with 200 linear warm-up steps, 20 epochs, 112x112 images,
same optimisation recipe as the paper's EchoVLM, objective = L_CLIP only.

Outputs (in --output-dir):
    export/                 TorchScript encoders + tokenizer (the evaluator's input)
    run_config.json, history.json

Usage:
    python -m baseline.clip_finetune --output-dir checkpoints/clip_only
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from torch.nn import functional as F

from baseline.clip_model import EchoVLM, build_pretrained_echovlm

VIEW_IDS = {"A2C": 0, "A4C": 1}


# ----------------------------------------------------------------------------- data


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, list[dict]]:
    images = np.load(data_dir / f"{split}_images.npy")
    records = [json.loads(line) for line in (data_dir / f"{split}.jsonl").read_text().splitlines() if line.strip()]
    assert len(images) == len(records), (images.shape, len(records))
    return images, records


def make_tokenizer(path: Path, max_length: int, pad_id: int):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path))
    tok.enable_truncation(max_length=max_length)
    tok.enable_padding(pad_id=pad_id, length=max_length)
    return tok


def tokenize(tok, texts: list[str]) -> torch.Tensor:
    return torch.tensor([e.ids for e in tok.encode_batch(texts)], dtype=torch.long)


# ----------------------------------------------------------------------------- export


class ImageExport(nn.Module):
    """float[B,3,112,112] in [0,1] -> L2-normalised float[B,512] (only the vision tower)."""

    def __init__(self, model: EchoVLM):
        super().__init__()
        self.visual = model.visual
        self.register_buffer("pixel_mean", model.pixel_mean.clone())
        self.register_buffer("pixel_std", model.pixel_std.clone())

    def forward(self, images01: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.visual((images01 - self.pixel_mean) / self.pixel_std), dim=-1)


class TextExport(nn.Module):
    """(input_ids, attention_mask) -> L2-normalised float[B,512] (only the text tower; causal, EOT-pooled)."""

    def __init__(self, model: EchoVLM):
        super().__init__()
        self.text = model.text

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text(input_ids), dim=-1)


def export_model(model: EchoVLM, out_dir: Path, tokenizer_json: Path, max_length: int, pad_id: int, tok) -> None:
    """Trace the encoders on CPU in float32 and write the evaluator interface."""
    export = out_dir / "export"
    export.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu", torch.float32).eval()
    model.set_grad_checkpointing(False)
    gen = torch.Generator().manual_seed(0)
    example_img = torch.rand(2, 3, 112, 112, generator=gen)
    ids = tokenize(tok, ["Apical four-chamber view.", "Normal LV size, no LA enlargement, LVEF 60%."])
    mask = (ids != pad_id).long()
    with torch.no_grad():
        img_ts = torch.jit.trace(ImageExport(model), example_img, check_trace=False)
        txt_ts = torch.jit.trace(TextExport(model), (ids, mask), check_trace=False)
        # Sanity: traced == eager for a different batch size.
        probe_img = torch.rand(3, 3, 112, 112, generator=gen)
        probe_ids = tokenize(tok, ["Severely dilated left atrium.", "A2C view, end-systolic frame.", "x"])
        probe_mask = (probe_ids != pad_id).long()
        assert torch.allclose(img_ts(probe_img), model.encode_image(probe_img), atol=1e-4)
        assert torch.allclose(txt_ts(probe_ids, probe_mask), model.encode_text(probe_ids), atol=1e-4)
        assert torch.allclose(img_ts(probe_img[:1]), model.encode_image(probe_img[:1]), atol=1e-4)
    img_ts.save(str(export / "image_encoder.pt"))
    txt_ts.save(str(export / "text_encoder.pt"))
    shutil.copyfile(tokenizer_json, export / "tokenizer.json")
    (export / "export.json").write_text(
        json.dumps({"max_length": max_length, "pad_id": pad_id, "image_size": 112, "embed_dim": 512}, indent=2)
    )


# ----------------------------------------------------------------------------- training


def clip_loss(image_z: torch.Tensor, text_z: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    logits = scale * image_z @ text_z.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def param_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "embedding" in name or "logit_scale" in name or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/downloads/camus_vlm")
    ap.add_argument("--weights-dir", default="data/downloads/models")
    ap.add_argument("--output-dir", default="checkpoints/clip_only")
    ap.add_argument("--train-splits", nargs="+", default=["train"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.005)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=77)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit-images", type=int, default=0, help="debug: subsample the training set")
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data_dir, weights_dir, out_dir = Path(args.data_dir), Path(args.weights_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json = weights_dir / "echo-clip" / "tokenizer.json"
    pad_id = 0
    tok = make_tokenizer(tokenizer_json, args.max_length, pad_id)

    images, records = [], []
    for split in args.train_splits:
        im, rec = load_split(data_dir, split)
        images.append(im)
        records.extend(rec)
    images = np.concatenate(images)
    if args.limit_images:
        keep = np.random.RandomState(args.seed).permutation(len(records))[: args.limit_images]
        images, records = images[keep], [records[i] for i in keep]
    n = len(records)
    caption_ids = tokenize(tok, [r["caption"] for r in records])
    negated_ids = tokenize(tok, [r["negated_caption"] for r in records])
    cap_len = (caption_ids != pad_id).sum(dim=1).float()
    neg_len = (negated_ids != pad_id).sum(dim=1).float()
    print(
        json.dumps(
            {
                "caption_tokens_mean": round(float(cap_len.mean()), 1),
                "caption_tokens_max": int(cap_len.max()),
                "captions_truncated": int((cap_len >= args.max_length).sum()),
                "negated_tokens_max": int(neg_len.max()),
            }
        ),
        flush=True,
    )
    views =torch.tensor([VIEW_IDS[r["view"]] for r in records], dtype=torch.long)
    images_t = torch.from_numpy(images)

    model = build_pretrained_echovlm(weights_dir).to(device)
    model.set_grad_checkpointing(not args.no_grad_checkpointing and device.type == "cuda")
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.999), eps=1e-6)
    steps_per_epoch = n // args.batch_size if n >= args.batch_size else 1
    total_steps = max(1, args.epochs * steps_per_epoch)
    batch = min(args.batch_size, n)
    use_amp = device.type == "cuda"
    images_dev = images_t.to(device)
    caption_dev, negated_dev, views_dev = caption_ids.to(device), negated_ids.to(device), views.to(device)
    gen = torch.Generator().manual_seed(args.seed)

    history, step, t0 = [], 0, time.time()
    print(json.dumps({"train_images": n, "steps_per_epoch": steps_per_epoch, "total_steps": total_steps}), flush=True)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        sums = {"loss": 0.0, "clip": 0.0, "view": 0.0, "neg": 0.0}
        for b in range(steps_per_epoch):
            idx = perm[b * batch : (b + 1) * batch].to(device)
            for group in opt.param_groups:
                group["lr"] = lr_at(step, total_steps, args.warmup_steps, args.lr)
            x = images_dev[idx].float().div_(255.0).unsqueeze(1).expand(-1, 3, -1, -1)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                zi = model.encode_image(x)
                zt = model.encode_text(caption_dev[idx])
            loss = clip_loss(zi.float(), zt.float(), model.logit_scale.exp().clamp(max=100.0))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))
            step += 1
            sums["loss"] += float(loss.detach())
            sums["clip"] += float(loss.detach())
        entry = {"epoch": epoch + 1, "step": step, "seconds": round(time.time() - t0, 1)}
        entry.update({k: round(v / max(1, steps_per_epoch), 5) for k, v in sums.items()})
        entry["logit_scale"] = round(float(model.logit_scale.exp()), 3)
        history.append(entry)
        print(json.dumps(entry), flush=True)

    export_model(model, out_dir, tokenizer_json, args.max_length, pad_id, tok)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps({"exported": str(out_dir / "export"), "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
