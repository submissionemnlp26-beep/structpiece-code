"""
Unified Tokenizer Wrapper
=========================
Provides a single interface over:
  • IndicTokenizer  (our morphology-aware BPE from python/adaptive_bpe.py)
  • SentencePiece   (pre-trained .model files in experiments/sp_models/)

Usage
-----
    from lm.tokenizer_adapter import TokenizerWrapper

    tok = TokenizerWrapper("indic", model_path="models")
    ids = tok.encode("नमस्कार")
    text = tok.decode(ids)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# ── Ensure the repo root is on sys.path so we can import python.adaptive_bpe ─
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))


class TokenizerWrapper:
    """
    Thin adapter that normalises the encode / decode API across tokenisers.

    Parameters
    ----------
    tokenizer_type : {"indic", "sp"}
        Which backend to load.
    model_path : str
        • For "indic" — directory containing ``indic_tokenizer_vocab.json``
          and ``indic_tokenizer_merges.txt`` (e.g. ``"models"``).
        • For "sp" — path to a ``.model`` file
          (e.g. ``"experiments/sp_models/sp_bpe_vs8000.model"``).
    """

    def __init__(self, tokenizer_type: str, model_path: str) -> None:
        self.tokenizer_type = tokenizer_type.lower()

        if self.tokenizer_type == "indic":
            self._init_indic(model_path)
        elif self.tokenizer_type == "sp":
            self._init_sp(model_path)
        elif self.tokenizer_type == "tiktoken":
            self._init_tiktoken(model_path)
        elif self.tokenizer_type == "llama":
            self._init_llama(model_path)
        else:
            raise ValueError(
                f"Unknown tokenizer_type={tokenizer_type!r}. "
                "Expected 'indic', 'sp', 'tiktoken', or 'llama'."
            )

    # ── backend loaders ──────────────────────────────────────────────────

    def _init_indic(self, model_dir: str) -> None:
        from adaptive_bpe import IndicTokenizer  # existing module

        self._tok = IndicTokenizer.load(model_dir)
        self._vocab_size = len(self._tok.vocab)

    def _init_sp(self, model_path: str) -> None:
        import sentencepiece as spm

        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)
        self._vocab_size = self._sp.GetPieceSize()

    def _init_tiktoken(self, model_name: str) -> None:
        import tiktoken
        self._tt = tiktoken.get_encoding(model_name)
        self._vocab_size = self._tt.n_vocab

    def _init_llama(self, model_name: str) -> None:
        from transformers import AutoTokenizer
        self._llama = AutoTokenizer.from_pretrained(model_name)
        self._vocab_size = len(self._llama)

    # ── public API ───────────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        """Tokenise *text* and return a list of integer IDs."""
        if self.tokenizer_type == "indic":
            return self._tok.encode(text)
        elif self.tokenizer_type == "sp":
            return self._sp.Encode(text, out_type=int)
        elif self.tokenizer_type == "tiktoken":
            return self._tt.encode(text)
        elif self.tokenizer_type == "llama":
            return self._llama.encode(text, add_special_tokens=False)
        return []

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to a string."""
        if self.tokenizer_type == "indic":
            return self._tok.decode(ids)
        elif self.tokenizer_type == "sp":
            return self._sp.Decode(ids)
        elif self.tokenizer_type == "tiktoken":
            return self._tt.decode(ids)
        elif self.tokenizer_type == "llama":
            return self._llama.decode(ids)
        return ""

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return self._vocab_size

    # ── repr ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"TokenizerWrapper(type={self.tokenizer_type!r}, "
            f"vocab_size={self.vocab_size})"
        )
