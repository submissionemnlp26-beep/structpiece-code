#!/usr/bin/env python3
"""
Fixed Token Budget Fairness Experiment
======================================
This script implements the user's fixed token budget protocol.
It carefully measures tokens per step, sets a token budget based on BPE,
adjusts MorphTokenizer steps, trains both on equal tokens, and evaluates.
"""

import math
import os
import sys
import time
from pathlib import Path
import numpy as np

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader

# ─── Project imports ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from lm.tokenizer_adapter import TokenizerWrapper
from lm.dataset import TextDataset
from python.adaptive_bpe import train_bpe, IndicTokenizer
from experiments.run_core_reeval import train_lm, compute_bpc, count_tokens_for_corpus

# ─── Configuration ───────────────────────────────────────────────────────────
BLOCK_SIZE = 128
BATCH_SIZE = 32
BPE_STEPS_BASELINE = 500
CORPUS_PATH = str(ROOT / "datasets" / "turkish_clean.txt")
VOCAB_SIZE = 4000
TRAIN_LINES = 2000
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

out_dir = ROOT / "experiments" / "budget_results"
out_dir.mkdir(parents=True, exist_ok=True)
models_dir = out_dir / "models"
models_dir.mkdir(exist_ok=True)

def measure_tokens_per_step(model_wrapper, corpus_path):
    # Same config as real training
    ds = TextDataset(corpus_path, model_wrapper, block_size=BLOCK_SIZE, max_lines=TRAIN_LINES)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    
    tokens_list = []
    for i, (x, y) in enumerate(dl):
        if i >= 10:
            break
        # x is a batch of input tokens: (B, T)
        tokens_list.append(x.numel())
        
    return np.mean(tokens_list) if len(tokens_list) > 0 else 0

def main():
    print("=" * 80)
    print("  FIXED TOKEN BUDGET FAIRNESS EXPERIMENT (Turkish)")
    print("=" * 80)

    # 1. Train tokenizers to ensure we have them ready
    print("\n[Prep] Training tokenizers just to be sure...")
    vocab_m, merges_m = train_bpe(CORPUS_PATH, vocab_size=VOCAB_SIZE, max_lines=TRAIN_LINES, verbose=False)
    morph_tok = IndicTokenizer(vocab_m, merges_m)
    morph_dir = str(models_dir / "turkish_morph")
    morph_tok.save(morph_dir)

    sp_prefix = str(models_dir / "turkish_sp_bpe")
    sp_model = sp_prefix + ".model"
    spm.SentencePieceTrainer.Train(
        input=CORPUS_PATH,
        model_prefix=sp_prefix,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        normalization_rule_name="nfkc"
    )

    morph_wrapper = TokenizerWrapper("indic", morph_dir)
    bpe_wrapper = TokenizerWrapper("sp", sp_model)

    # ─── Step 1: Measure tokens per step ─────────────────────────────────────
    print("\n[Step 1] Measuring tokens per step (running 10 batches)...")
    tokens_per_step_morph = measure_tokens_per_step(morph_wrapper, CORPUS_PATH)
    tokens_per_step_bpe = measure_tokens_per_step(bpe_wrapper, CORPUS_PATH)

    print(f"  BPE tokens/step:   {tokens_per_step_bpe}")
    print(f"  Morph tokens/step: {tokens_per_step_morph}")

    # ─── Step 2: Compute target token budget ─────────────────────────────────
    target_tokens = BPE_STEPS_BASELINE * tokens_per_step_bpe
    print(f"\n[Step 2] Target token budget = {BPE_STEPS_BASELINE} steps × {tokens_per_step_bpe} tokens/step = {target_tokens} tokens")

    # ─── Step 3: Adjust MorphTokenizer steps ─────────────────────────────────
    steps_morph = int(target_tokens / tokens_per_step_morph)
    steps_bpe = BPE_STEPS_BASELINE
    print(f"\n[Step 3] Adjusted training setup:")
    print(f"  BPE steps:   {steps_bpe}")
    print(f"  Morph steps: {steps_morph}")
    print(f"  Target tokens: {target_tokens}")

    # ─── PRE-COMPUTE tok/char for BPC ────────────────────────────────────────
    def get_tok_char(wrapper):
        encode_fn = wrapper.encode if wrapper.tokenizer_type == "indic" else lambda t: spm.SentencePieceProcessor(sp_model).Encode(t, out_type=int)
        total_toks, total_chars, _ = count_tokens_for_corpus(encode_fn, CORPUS_PATH, max_lines=1000)
        return total_toks / max(1, total_chars)

    tok_char_bpe = get_tok_char(bpe_wrapper)
    tok_char_morph = get_tok_char(morph_wrapper)

    # ─── Step 4: Train Models ────────────────────────────────────────────────
    print("\n[Step 4] Training LMs under strict token budget...")
    print(f"  Training SP-BPE ({steps_bpe} steps)...")
    t0 = time.time()
    res_bpe = train_lm(bpe_wrapper, CORPUS_PATH, steps_bpe, DEVICE)
    print(f"    Done in {time.time()-t0:.1f}s")

    print(f"  Training MorphTokenizer ({steps_morph} steps)...")
    t0 = time.time()
    res_morph = train_lm(morph_wrapper, CORPUS_PATH, steps_morph, DEVICE)
    print(f"    Done in {time.time()-t0:.1f}s")

    # ─── Compute formatted output ────────────────────────────────────────────
    total_tokens_bpe = steps_bpe * tokens_per_step_bpe
    total_tokens_morph = steps_morph * tokens_per_step_morph

    bpc_bpe = compute_bpc(res_bpe["avg_loss"], tok_char_bpe)
    bpc_morph = compute_bpc(res_morph["avg_loss"], tok_char_morph)

    print("\n" + "="*60)
    print("1. Tokens per step:")
    print(f"   BPE: {tokens_per_step_bpe}")
    print(f"   Morph: {tokens_per_step_morph}")
    
    print("\n2. Training setup:")
    print(f"   BPE steps: {steps_bpe}")
    print(f"   Morph steps: {steps_morph}")
    print(f"   Target tokens: {target_tokens}")
    
    print("\n3. Final results:")
    print("| Tokenizer      | Total Tokens | PPL | BPC |")
    print("| -------------- | ------------ | --- | --- |")
    print(f"| SP-BPE         | {int(total_tokens_bpe):<12} | {res_bpe['avg_ppl']:<3} | {bpc_bpe:<3} |")
    print(f"| MorphTokenizer | {int(total_tokens_morph):<12} | {res_morph['avg_ppl']:<3} | {bpc_morph:<3} |")
    print("="*60)

if __name__ == "__main__":
    main()
