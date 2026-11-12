"""EchoVLM pretraining objectives (paper Eq. 1-3 and Algorithm 1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class LossParts:
    total: torch.Tensor
    clip: torch.Tensor
    view: torch.Tensor
    negation: torch.Tensor


def clip_loss(image_z: torch.Tensor, text_z: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Symmetric image<->text cross-entropy against the diagonal (Alg. 1, CLIP_LOSS)."""
    logits = scale * image_z @ text_z.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def view_informed_loss(image_z: torch.Tensor, views: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Eq. 1 / Alg. 1 VIEW_CONTRASTIVE: same-view images are positives, self excluded."""
    logits = scale * image_z @ image_z.T
    eye = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(eye, float("-inf"))
    log_prob = F.log_softmax(logits, dim=1).masked_fill(eye, 0.0)
    positives = (views[:, None] == views[None, :]) & ~eye
    denom = positives.sum(dim=1).clamp_min(1).to(log_prob.dtype)
    per_row = -(log_prob * positives.to(log_prob.dtype)).sum(dim=1) / denom
    return per_row.mean()


def negation_loss(text_z: torch.Tensor, neg_z: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Eq. 2 / Alg. 1 NEG_BCE: BCEWithLogits(tau * <z_txt, z_neg>, 0)."""
    logits = scale * (text_z * neg_z).sum(dim=1)
    return F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))


def echovlm_loss(
    image_z: torch.Tensor,
    text_z: torch.Tensor,
    neg_z: torch.Tensor | None,
    views: torch.Tensor,
    logit_scale: torch.Tensor,
    lambda_view: float,
    lambda_neg: float,
) -> LossParts:
    """Eq. 3: L = L_CLIP + lambda_view * L_view + lambda_neg * L_neg (one learnable tau)."""
    scale = logit_scale.exp().clamp(max=100.0)
    zero = image_z.new_zeros(())
    l_clip = clip_loss(image_z, text_z, scale)
    l_view = view_informed_loss(image_z, views, scale) if lambda_view > 0 else zero
    l_neg = negation_loss(text_z, neg_z, scale) if (lambda_neg > 0 and neg_z is not None) else zero
    total = l_clip + lambda_view * l_view + lambda_neg * l_neg
    return LossParts(total=total, clip=l_clip, view=l_view, negation=l_neg)
