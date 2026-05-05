"""
Token-level Language-Modelling Dataset
======================================
Reads a plain-text corpus, tokenises it with any ``TokenizerWrapper``,
and serves fixed-length (input, target) chunks for causal LM training.

Usage
-----
    from lm.tokenizer_adapter import TokenizerWrapper
    from lm.dataset import TextDataset

    tok = TokenizerWrapper("indic", "models")
    ds  = TextDataset("datasets/hindi_raw.txt", tok, block_size=128)
    x, y = ds[0]   # both are LongTensors of shape (block_size,)
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from lm.tokenizer_adapter import TokenizerWrapper


class TextDataset(Dataset):
    """
    A minimal next-token-prediction dataset.

    1. Read the entire file and concatenate into one long token stream.
    2. Slice into non-overlapping windows of ``block_size + 1`` tokens.
    3. Each sample returns ``(x, y)`` where ``y = x`` shifted by one position.

    Parameters
    ----------
    file_path   : path to a UTF-8 text file (one sentence per line).
    tokenizer   : a ``TokenizerWrapper`` instance.
    block_size  : context length (number of input tokens per sample).
    max_lines   : optional cap on the number of lines to read.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerWrapper,
        block_size: int = 128,
        max_lines: int | None = None,
    ) -> None:
        self.block_size = block_size

        # ── tokenise the whole corpus into one flat ID list ──────────────
        all_ids: List[int] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_lines is not None and idx >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                all_ids.extend(tokenizer.encode(line))
                all_ids.append(0)

        # ── store as a single contiguous tensor ─────────────────────────
        # We need at least block_size + 1 tokens to form one sample.
        self.data = torch.tensor(all_ids, dtype=torch.long)
        self.n_samples = max(0, len(self.data) - block_size - 1)

    # ── Dataset interface ────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx

        x = self.data[start : start + self.block_size]
        y = self.data[start + 1 : start + self.block_size + 1]  # target tokens (shifted by 1)
        return x, y
