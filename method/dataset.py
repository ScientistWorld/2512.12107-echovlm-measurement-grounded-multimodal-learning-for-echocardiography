"""Dataset helpers for EchoVLM training."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from method.echovlm import encode_texts


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def disease_probe(row: dict) -> tuple[str, int]:
    keys = sorted(row["disease_labels"])
    index = int(row.get("source_index", 0)) % len(keys)
    key = keys[index]
    return key, int(bool(row["disease_labels"][key]))


class EchoToyDataset(Dataset):
    def __init__(self, root: str | Path, split: str, vocab: dict[str, int], max_len: int = 48):
        self.root = Path(root)
        self.rows = load_jsonl(self.root / f"{split}.jsonl")
        self.vocab = vocab
        self.max_len = max_len
        all_rows = load_jsonl(self.root / "all.jsonl") if (self.root / "all.jsonl").exists() else self.rows
        self.view_to_id = {view: i for i, view in enumerate(sorted({row["view"] for row in all_rows}))}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image = Image.open(self.root / row["image"]).convert("RGB").resize((112, 112))
        pixels = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes())).float()
        pixels = pixels.view(image.height, image.width, 3).permute(2, 0, 1) / 255.0
        disease, label = disease_probe(row)
        return {
            "id": row["id"],
            "image": pixels,
            "text": encode_texts([row["caption"]], self.vocab, self.max_len)[0],
            "neg_text": encode_texts([row["negated_caption"]], self.vocab, self.max_len)[0],
            "view": torch.tensor(self.view_to_id[row["view"]], dtype=torch.long),
            "disease": disease,
            "label": torch.tensor(label, dtype=torch.long),
            "caption": row["caption"],
        }
