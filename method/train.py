"""Train the reproduced EchoVLM objective on the lightweight echo dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.make_echo_toy import main as make_data_main
from method.dataset import EchoToyDataset, load_jsonl
from method.echovlm import EchoVLM, build_vocab, echovlm_loss


def ensure_data(data_root: Path) -> None:
    if not (data_root / "train.jsonl").exists():
        import sys

        old_argv = sys.argv
        sys.argv = ["make_echo_toy", "--output", str(data_root)]
        try:
            make_data_main()
        finally:
            sys.argv = old_argv


def collate(batch: list[dict]) -> dict:
    return {
        "id": [b["id"] for b in batch],
        "image": torch.stack([b["image"] for b in batch]),
        "text": torch.stack([b["text"] for b in batch]),
        "neg_text": torch.stack([b["neg_text"] for b in batch]),
        "view": torch.stack([b["view"] for b in batch]),
        "disease": [b["disease"] for b in batch],
        "label": torch.stack([b["label"] for b in batch]),
        "caption": [b["caption"] for b in batch],
    }


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_embeddings(model: EchoVLM, loader: DataLoader, device: torch.device, out_path: Path) -> None:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            text = batch["text"].to(device)
            neg_text = batch["neg_text"].to(device)
            image_z, text_z, neg_z = model(image, text, neg_text)
            for i, sample_id in enumerate(batch["id"]):
                rows.append(
                    {
                        "id": sample_id,
                        "disease": batch["disease"][i],
                        "label": int(batch["label"][i].item()),
                        "caption": batch["caption"][i],
                        "image_embedding": image_z[i].cpu().tolist(),
                        "text_embedding": text_z[i].cpu().tolist(),
                        "neg_text_embedding": neg_z[i].cpu().tolist(),
                    }
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/echo_toy")
    parser.add_argument("--output-dir", default="checkpoints/echovlm_full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--lambda-view", type=float, default=0.5)
    parser.add_argument("--lambda-neg", type=float, default=0.1)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data_root = Path(args.data_root)
    ensure_data(data_root)
    train_rows = load_jsonl(data_root / "train.jsonl")
    vocab = build_vocab([r["caption"] for r in train_rows] + [r["negated_caption"] for r in train_rows])

    train_ds = EchoToyDataset(data_root, "train", vocab)
    test_ds = EchoToyDataset(data_root, "test", vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EchoVLM(vocab_size=len(vocab), embed_dim=args.embed_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    total_steps = max(1, args.epochs * len(train_loader))
    global_step = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        totals = {"loss": 0.0, "clip": 0.0, "view": 0.0, "neg": 0.0}
        for batch in train_loader:
            lr = cosine_lr(global_step, total_steps, args.warmup_steps, args.lr)
            for group in opt.param_groups:
                group["lr"] = lr
            image_z, text_z, neg_z = model(batch["image"].to(device), batch["text"].to(device), batch["neg_text"].to(device))
            losses = echovlm_loss(
                image_z,
                text_z,
                neg_z,
                batch["view"].to(device),
                model.logit_scale,
                args.lambda_view,
                args.lambda_neg,
            )
            opt.zero_grad(set_to_none=True)
            losses.total.backward()
            opt.step()
            totals["loss"] += float(losses.total.detach().cpu())
            totals["clip"] += float(losses.clip.detach().cpu())
            totals["view"] += float(losses.view.detach().cpu())
            totals["neg"] += float(losses.negation.detach().cpu())
            global_step += 1
        denom = max(1, len(train_loader))
        entry = {"epoch": epoch + 1, **{k: v / denom for k, v in totals.items()}}
        print(json.dumps(entry), flush=True)
        history.append(entry)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab": vocab, "args": vars(args), "history": history}, out_dir / "model.pt")
    save_embeddings(model, test_loader, device, out_dir / "test_embeddings.jsonl")
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
