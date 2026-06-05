"""Small CLIP-style model and the EchoVLM losses from the paper."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts: list[str], min_freq: int = 1) -> dict[str, int]:
    counts = Counter(tok for text in texts for tok in tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, freq in sorted(counts.items()):
        if freq >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode_texts(texts: list[str], vocab: dict[str, int], max_len: int = 48) -> torch.Tensor:
    rows = []
    for text in texts:
        ids = [vocab.get(tok, 1) for tok in tokenize(text)[:max_len]]
        ids += [0] * (max_len - len(ids))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x).flatten(1)
        return F.normalize(self.proj(z), dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, width: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, embed_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(0).float().unsqueeze(-1)
        emb = self.embedding(tokens) * mask
        pooled = emb.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.mlp(pooled), dim=-1)


class EchoVLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(tokens)

    def forward(self, images: torch.Tensor, text: torch.Tensor, neg_text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encode_image(images), self.encode_text(text), self.encode_text(neg_text)


@dataclass
class LossParts:
    total: torch.Tensor
    clip: torch.Tensor
    view: torch.Tensor
    negation: torch.Tensor


def clip_loss(image_z: torch.Tensor, text_z: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    logits = logit_scale.exp().clamp(max=100.0) * image_z @ text_z.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def view_informed_loss(image_z: torch.Tensor, views: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    logits = logit_scale.exp().clamp(max=100.0) * image_z @ image_z.T
    logits = logits.masked_fill(torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool), -1e9)
    log_probs = F.log_softmax(logits, dim=1)
    same_view = views[:, None].eq(views[None, :])
    same_view.fill_diagonal_(False)
    positives = same_view.float()
    denom = positives.sum(dim=1)
    valid = denom > 0
    if not valid.any():
        return image_z.new_tensor(0.0)
    per_row = -(log_probs * positives).sum(dim=1)[valid] / denom[valid]
    return per_row.mean()


def negation_loss(text_z: torch.Tensor, neg_z: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    logits = logit_scale.exp().clamp(max=100.0) * (text_z * neg_z).sum(dim=1)
    return F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))


def echovlm_loss(
    image_z: torch.Tensor,
    text_z: torch.Tensor,
    neg_z: torch.Tensor,
    views: torch.Tensor,
    logit_scale: torch.Tensor,
    lambda_view: float,
    lambda_neg: float,
) -> LossParts:
    l_clip = clip_loss(image_z, text_z, logit_scale)
    l_view = view_informed_loss(image_z, views, logit_scale)
    l_neg = negation_loss(text_z, neg_z, logit_scale)
    total = l_clip + lambda_view * l_view + lambda_neg * l_neg
    return LossParts(total=total, clip=l_clip, view=l_view, negation=l_neg)
