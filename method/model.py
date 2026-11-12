"""EchoVLM backbone: ViT-B/16 image encoder + CLIP text encoder (paper Sec. 3.2).

Paper: "we use ViT-B as the vision backbone and CLIP text encoder as text
backbone ... All images are resized to 112x112 ... Vision encoder is
initialized from weights pretrained on an internal echo dataset. Text encoder
is initialized from EchoCLIP."

Here the text encoder is initialised from the public EchoCLIP weights
(mkaichristensen/echo-clip, open_clip format) exactly as in the paper. The
internal echo-pretrained ViT-B weights are not public, so the ViT-B/16 vision
encoder is initialised from OpenAI CLIP ViT-B/16 (timm/vit_base_patch16_clip_224.openai)
with its position embeddings interpolated from 14x14 to 7x7 patches for 112x112 input.
The modules are written out explicitly (no open_clip dependency) so they can be
exported to TorchScript for the method-agnostic evaluator.
"""

from __future__ import annotations

import json
import math
import struct
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class Attention(nn.Module):
    """Multi-head self-attention with OpenAI-CLIP parameter names (in_proj / out_proj)."""

    def __init__(self, width: int, heads: int, causal: bool):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.in_proj_weight = nn.Parameter(torch.empty(3 * width, width))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * width))
        self.out_proj = nn.Linear(width, width)
        nn.init.xavier_uniform_(self.in_proj_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        k = k.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        v = v.unflatten(-1, (self.heads, -1)).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.out_proj(out.transpose(1, 2).flatten(2))


class ResidualBlock(nn.Module):
    def __init__(self, width: int, heads: int, act: type, causal: bool):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = Attention(width, heads, causal)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(width, 4 * width)), ("gelu", act()), ("c_proj", nn.Linear(4 * width, width))])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, act: type, causal: bool):
        super().__init__()
        self.resblocks = nn.ModuleList([ResidualBlock(width, heads, act, causal) for _ in range(layers)])
        self.grad_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            if self.grad_checkpointing and self.training and not torch.jit.is_scripting():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size=112, patch_size=16, width=768, layers=12, heads=12, output_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((image_size // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, QuickGELU, causal=False)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x).flatten(2).transpose(1, 2)
        cls = self.class_embedding.to(x.dtype).view(1, 1, -1).expand(x.shape[0], 1, x.shape[-1])
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        x = self.transformer(self.ln_pre(x))
        return self.ln_post(x[:, 0]) @ self.proj


class TextTransformer(nn.Module):
    def __init__(self, vocab_size=49408, context_length=77, width=512, layers=12, heads=8, output_dim=512):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.transformer = Transformer(width, layers, heads, nn.GELU, causal=True)
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=width**-0.5)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(ids) + self.positional_embedding[: ids.shape[1]].to(self.token_embedding.weight.dtype)
        x = self.ln_final(self.transformer(x))
        # CLIP pooling: features at the end-of-text token (highest id in each row).
        eot = ids.argmax(dim=-1)
        x = x.gather(1, eot.view(-1, 1, 1).expand(-1, 1, x.shape[-1])).squeeze(1)
        return x @ self.text_projection


class EchoVLM(nn.Module):
    def __init__(self, image_size: int = 112):
        super().__init__()
        self.visual = VisionTransformer(image_size=image_size)
        self.text = TextTransformer()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.register_buffer("pixel_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    def set_grad_checkpointing(self, enabled: bool = True) -> None:
        self.visual.transformer.grad_checkpointing = enabled
        self.text.transformer.grad_checkpointing = enabled

    def encode_image(self, images01: torch.Tensor) -> torch.Tensor:
        """images01: float [B,3,H,W] in [0,1]; returns L2-normalised embeddings."""
        x = (images01 - self.pixel_mean) / self.pixel_std
        return F.normalize(self.visual(x), dim=-1)

    def encode_text(self, ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text(ids), dim=-1)


# ----------------------------------------------------------------------------- weights


def load_safetensors(path: str | Path) -> dict[str, torch.Tensor]:
    """Minimal safetensors reader (avoids an extra dependency)."""
    dtypes = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32}
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        blob = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        arr = np.frombuffer(blob[start:end], dtype=dtypes[info["dtype"]]).reshape(info["shape"])
        out[name] = torch.from_numpy(arr.copy())
    return out


def resize_pos_embed(pos: torch.Tensor, new_tokens: int) -> torch.Tensor:
    cls, grid = pos[:1], pos[1:]
    old = int(round(math.sqrt(grid.shape[0])))
    new = int(round(math.sqrt(new_tokens - 1)))
    if old == new:
        return pos
    grid = grid.reshape(1, old, old, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(new, new), mode="bicubic", align_corners=False)
    return torch.cat([cls, grid.permute(0, 2, 3, 1).reshape(new * new, -1)], dim=0)


def build_pretrained_echovlm(weights_dir: str | Path, image_size: int = 112) -> EchoVLM:
    weights_dir = Path(weights_dir)
    model = EchoVLM(image_size=image_size)
    vision_sd = load_safetensors(weights_dir / "openai-clip-vit-b-16" / "open_clip_model.safetensors")
    visual = {k[len("visual.") :]: v for k, v in vision_sd.items() if k.startswith("visual.")}
    visual["positional_embedding"] = resize_pos_embed(
        visual["positional_embedding"], model.visual.positional_embedding.shape[0]
    )
    model.visual.load_state_dict(visual, strict=True)
    echo_sd = torch.load(weights_dir / "echo-clip" / "open_clip_pytorch_model.bin", map_location="cpu", weights_only=True)
    text_keys = ("token_embedding.", "positional_embedding", "transformer.", "ln_final.", "text_projection")
    text = {k: v for k, v in echo_sd.items() if k.startswith(text_keys)}
    model.text.load_state_dict(text, strict=True)
    return model
