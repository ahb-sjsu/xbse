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
    """Encoder (BERT-family) pooling: masked mean over tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def last_token_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Decoder-embedder (LLM-based, e.g. gte-Qwen2, E5-Mistral) pooling: the last real token.

    Robust to both left- and right-padding: if the batch is left-padded the last column is the
    final token; otherwise index each row at its last non-pad position."""
    left_padded = attention_mask[:, -1].sum().item() == attention_mask.shape[0]
    if left_padded:
        return token_embeddings[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    rows = torch.arange(token_embeddings.shape[0], device=token_embeddings.device)
    return token_embeddings[rows, seq_lens]


_POOLERS = {"mean": mean_pool, "last": last_token_pool}


class BSEEncoder(nn.Module):
    def __init__(
        self,
        base_model: str = "BAAI/bge-m3",
        proj_dim: int | None = None,
        max_len: int = 128,
        device: str | None = None,
        pooling: str = "mean",
        trust_remote_code: bool = False,
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        if pooling not in _POOLERS:
            raise ValueError(f"pooling={pooling!r} must be one of {list(_POOLERS)}")
        # Default to CUDA when available, else CPU — a hardcoded "cuda" crashes CPU-only installs.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pooling = pooling
        self._pool = _POOLERS[pooling]
        self.tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        # Decoder embedders (gte-Qwen2, E5-Mistral, …) expect LEFT padding so the last token is the
        # sequence's final token; last_token_pool also handles right padding, but this matches how
        # these models were trained.
        if pooling == "last":
            self.tok.padding_side = "left"
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
        # Force safetensors: avoids transformers' torch.load vulnerability gate (needs torch>=2.6
        # for .bin). BGE-M3 ships safetensors, so this loads cleanly on torch 2.4 (NRP image). If
        # safetensors is genuinely unavailable we warn loudly rather than silently taking the
        # torch.load path the line above exists to avoid.
        try:
            self.backbone = AutoModel.from_pretrained(
                base_model, use_safetensors=True, trust_remote_code=trust_remote_code
            )
        except Exception as e:
            import warnings

            warnings.warn(
                f"[BSEEncoder] safetensors load of {base_model} failed ({e}); falling back to the "
                "default (possibly torch.load) path. Prefer a safetensors checkpoint.",
                stacklevel=2,
            )
            self.backbone = AutoModel.from_pretrained(
                base_model, trust_remote_code=trust_remote_code
            )
        # Decoder embedders (gte-Qwen2, E5-Mistral, …) are large: full fine-tuning holds params +
        # grads + Adam moments (~24GB for 1.5B) BEFORE activations, which OOMs a single 44GB GPU at
        # any useful InfoNCE batch. Gradient checkpointing trades compute for activation memory so a
        # real batch (many in-batch negatives) fits. use_cache must be off for checkpointing to work.
        if pooling == "last" and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
            self.backbone.config.use_cache = False
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Identity() if proj_dim in (None, hidden) else nn.Linear(hidden, proj_dim)
        self.max_len = max_len
        self.device = device
        self.to(device)

    def forward(self, texts: list[str]) -> torch.Tensor:
        batch = self.tok(
            texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt"
        ).to(self.device)
        out = self.backbone(**batch).last_hidden_state
        z = self.proj(self._pool(out, batch["attention_mask"]))
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        self.eval()
        chunks = [self.forward(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
        return torch.cat(chunks, 0)
