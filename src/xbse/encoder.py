"""BSEEncoder — the shared architecture for every *-BSE.

base transformer -> pooling -> projection -> L2-normalized embedding.

Identical across LaBSE/LeBSE/MoBSE; only the base model id and projection dim are config. What
makes an instance is the PairSource it's trained on, not this class.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class BSEEncoder(nn.Module):
    def __init__(self, base_model: str = "BAAI/bge-m3", proj_dim: int | None = None,
                 max_len: int = 128, device: str = "cuda"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(base_model)
        # force safetensors: avoids transformers' torch.load vulnerability gate (needs torch>=2.6
        # for .bin) — BGE-M3 ships safetensors, so this loads cleanly on torch 2.4 (NRP image).
        try:
            self.backbone = AutoModel.from_pretrained(base_model, use_safetensors=True)
        except Exception:
            self.backbone = AutoModel.from_pretrained(base_model)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Identity() if proj_dim in (None, hidden) else nn.Linear(hidden, proj_dim)
        self.max_len = max_len
        self.device = device
        self.to(device)

    def forward(self, texts: list[str]) -> torch.Tensor:
        batch = self.tok(texts, padding=True, truncation=True, max_length=self.max_len,
                         return_tensors="pt").to(self.device)
        out = self.backbone(**batch).last_hidden_state
        z = self.proj(mean_pool(out, batch["attention_mask"]))
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        self.eval()
        chunks = [self.forward(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
        return torch.cat(chunks, 0)
