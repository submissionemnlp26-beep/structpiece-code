"""
Minimal Transformer Language Model
==================================
A small causal transformer (nanoGPT-style) for rapid experiments.

Constraints met:
- 2-4 layers
- small embedding size (e.g. 128-256)
- causal self attention
- Returns logits, loss, and attention weights
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0

        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

        self.n_heads = n_heads
        self.d_model = d_model

        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(1, 1, block_size, block_size, dtype=torch.bool))
        )
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()

        # calculate query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # causal mask
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))

        att = F.softmax(att, dim=-1)
        att_weights = att  # save for attention entropy analysis

        att = self.dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y, att_weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln_2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, att_weights = self.attn(self.ln_1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, att_weights


class NanoLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, block_size, dropout) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.head.weight = self.token_emb.weight

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass for the Language Model.

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar cross-entropy loss if targets is provided, or None
            all_att_weights: list of attention weights from each layer (B, nh, T, T)
        """
        B, T = idx.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.token_emb(idx)  # (B, T, C)
        pos_emb = self.pos_emb(pos)    # (T, C)

        x = self.drop(tok_emb + pos_emb.unsqueeze(0))

        all_att_weights: List[torch.Tensor] = []
        for block in self.blocks:
            x, att_weights = block(x)
            all_att_weights.append(att_weights)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            # Flatten to compute Cross Entropy over the vocabulary
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss, all_att_weights
