#!/usr/bin/env python3
"""
Modern Tokenizer Baselines & Token Budget Experiment
====================================================
Compares MorphTokenizer, SP-BPE, and SP-Unigram against 
global pre-trained LLM tokenizers (LLaMA and tiktoken).
Enforces a strict token budget (max_steps=500 -> 2,048,000 tokens).
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import numpy as np

# ─── Project imports ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from lm.tokenizer_adapter import TokenizerWrapper
from python.adaptive_bpe import train_bpe, IndicTokenizer
from experiments.run_core_reeval import train_lm, compute_token_stats, compute_bpc

# ─── Configuration ───────────────────────────────────────────────────────────

LANGUAGES = {
    "hindi":    {"corpus": "datasets/hindi_raw.txt"},
    "arabic":   {"corpus": "datasets/arabic_clean.txt"},
    "turkish":  {"corpus": "datasets/turkish_clean.txt"},
}

VOCAB_SIZE = 4000
TRAIN_LINES = 2000
MAX_STEPS = 500
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

out_dir = ROOT / "experiments" / "modern_baselines"
out_dir.mkdir(parents=True, exist_ok=True)
models_dir = out_dir / "models"
models_dir.mkdir(exist_ok=True)

def train_or_load_baselines(lang, corpus_path):
    print(f"\n[Prep] Setting up tokenizers for {lang.upper()}...")
    
    # 1. MorphTokenizer
    morph_dir = str(models_dir / f"{lang}_morph")
    if not os.path.exists(os.path.join(morph_dir, "indic_tokenizer_vocab.json")):
        vocab_m, merges_m = train_bpe(corpus_path, vocab_size=VOCAB_SIZE, max_lines=TRAIN_LINES, verbose=False)
        morph_tok = IndicTokenizer(vocab_m, merges_m)
        morph_tok.save(morph_dir)

    # 2. SP-BPE
    sp_bpe_prefix = str(models_dir / f"{lang}_sp_bpe")
    if not os.path.exists(sp_bpe_prefix + ".model"):
        spm.SentencePieceTrainer.Train(
            input=corpus_path, model_prefix=sp_bpe_prefix, vocab_size=VOCAB_SIZE,
            model_type="bpe", character_coverage=1.0, normalization_rule_name="nfkc"
        )
        
    # 3. SP-Unigram
    sp_uni_prefix = str(models_dir / f"{lang}_sp_unigram")
    if not os.path.exists(sp_uni_prefix + ".model"):
        spm.SentencePieceTrainer.Train(
            input=corpus_path, model_prefix=sp_uni_prefix, vocab_size=VOCAB_SIZE,
            model_type="unigram", character_coverage=1.0, normalization_rule_name="nfkc"
        )
        
    return {
        "MorphTokenizer": TokenizerWrapper("indic", morph_dir),
        "SP-BPE": TokenizerWrapper("sp", sp_bpe_prefix + ".model"),
        "SP-Unigram": TokenizerWrapper("sp", sp_uni_prefix + ".model"),
        "LLaMA": TokenizerWrapper("llama", "NousResearch/Llama-2-7b-hf"),
        "tiktoken": TokenizerWrapper("tiktoken", "cl100k_base"),
    }

def main():
    print("=" * 80)
    print("  MODERN TOKENIZER BASELINES EXPERIMENT")
    print("=" * 80)

    all_results = []

    for lang, meta in LANGUAGES.items():
        corpus_path = str(ROOT / meta["corpus"])
        if not os.path.exists(corpus_path):
            print(f"SKIPPING: {corpus_path} not found")
            continue
            
        print(f"\n{'='*80}\n  LANGUAGE: {lang.upper()}\n{'='*80}")
        tokenizers = train_or_load_baselines(lang, corpus_path)

        # ─── Compute token stats ─────────────────────────────────────────
        print("\n[1/2] Computing token efficiency stats...")
        stats = {}
        for name, tok_wrapper in tokenizers.items():
            print(f"      {name}...")
            encode_fn = tok_wrapper.encode
            st = compute_token_stats(encode_fn, corpus_path, max_lines=1000)
            st["chars_per_512tokens"] = round((512 / st["tok_per_char"]) if st["tok_per_char"] > 0 else 0, 1)
            st["tokens_per_512chars"] = round(st["tok_per_char"] * 512, 1)
            stats[name] = st
            
        bpe_tok_per_word = stats["SP-BPE"]["tok_per_word"]

        # ─── Train LMs ───────────────────────────────────────────────────
        print(f"\n[2/2] Training Language Models (Strict {MAX_STEPS} steps budget)...")
        for tok_name, tok_wrapper in tokenizers.items():
            print(f"      Training LM with {tok_name} (vocab={tok_wrapper.vocab_size})...")
            t0 = time.time()
            res = train_lm(tok_wrapper, corpus_path, MAX_STEPS, DEVICE)
            elapsed = time.time() - t0
            
            avg_loss = res["avg_loss"]
            avg_ppl = res["avg_ppl"]
            bpc = compute_bpc(avg_loss, stats[tok_name]["tok_per_char"])
            inflation = round(stats[tok_name]["tok_per_word"] / bpe_tok_per_word, 2)
            
            print(f"        → avg_loss={avg_loss}, avg_ppl={avg_ppl}, BPC={bpc} ({elapsed:.1f}s)")
            
            all_results.append({
                "language": lang.capitalize(),
                "tokenizer": tok_name,
                "vocab_size": tok_wrapper.vocab_size,
                "avg_loss": avg_loss,
                "token_ppl": avg_ppl,
                "bpc": bpc,
                "tokens_per_word": stats[tok_name]["tok_per_word"],
                "tokens_per_512chars": stats[tok_name]["tokens_per_512chars"],
                "chars_per_512tokens": stats[tok_name]["chars_per_512tokens"],
                "inflation_vs_bpe": inflation
            })

    # ─── Save JSON ───────────────────────────────────────────────────────
    results_path = out_dir / "modern_baselines_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ─── Print formatted table ───────────────────────────────────────────
    print(f"\n{'='*100}")
    print("  TABLE C — GLOBAL LLM BASELINES")
    print(f"{'='*100}")
    print(f"{'Lang':<8} {'Tokenizer':<16} {'|V|':>8} {'Tok/Word':>9} {'Inflat.':>9} {'PPL':>8} {'BPC ↓':>8}")
    print("-" * 100)
    for r in all_results:
        print(f"{r['language']:<8} {r['tokenizer']:<16} {r['vocab_size']:>8} "
              f"{r['tokens_per_word']:>9.2f} {r['inflation_vs_bpe']:>8.1f}x {r['token_ppl']:>8.1f} {r['bpc']:>8.3f}")

if __name__ == "__main__":
    main()
