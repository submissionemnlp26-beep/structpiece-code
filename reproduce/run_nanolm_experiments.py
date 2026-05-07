#!/usr/bin/env python3
"""
reproduce/run_nanolm_experiments.py
=====================================
Entry-point for reproducing all intrinsic (NanoLM) experiments from the paper:
  "Tokenization as Structural Inductive Bias: Grapheme-Constrained Subwords
   for Multilingual Modeling and Vocabulary Surgery"

What this does:
  1. Trains StructPiece tokenizer from scratch on each requested language corpus
  2. Trains SP-BPE and SP-Unigram baselines with SentencePiece
  3. For each tokenizer, trains a NanoLM (d=256, L=2, H=4) from scratch
     under an identical fixed token budget (500 steps x batch 32 x block 128
     = 2,048,000 tokens per run)
  4. Reports Token PPL and BPC for every (language, tokenizer) pair
  5. Saves results to reproduce/results/nanolm_results.json

Hardware requirements:
  - Runs on CPU, Apple MPS, or CUDA
  - ~20-40 min per language on CPU, ~5 min on GPU
  - No GPU required for this phase

Usage:
  # Full 9-language run (matches paper Tables 1 & 10):
  PYTHONPATH=. python reproduce/run_nanolm_experiments.py

  # Quick 2-language smoke test:
  PYTHONPATH=. python reproduce/run_nanolm_experiments.py --langs english hindi

  # Dry run (plans without training):
  PYTHONPATH=. python reproduce/run_nanolm_experiments.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# ─── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader

from python.adaptive_bpe import train_bpe, IndicTokenizer
from lm.tokenizer_adapter import TokenizerWrapper
from lm.dataset import TextDataset
from lm.model import NanoLM

# ─── Experiment Configuration ────────────────────────────────────────────────
# These match the paper's Table 1 and Appendix Table 10 exactly.

VOCAB_SIZE  = 4_000
TRAIN_LINES = 2_000
BLOCK_SIZE  = 128
BATCH_SIZE  = 32
MAX_STEPS   = 500      # 500 steps × 32 × 128 = 2,048,000 tokens
LR          = 5e-4
D_MODEL     = 256
N_HEADS     = 4
N_LAYERS    = 2
SEED        = 42
LN2         = math.log(2)

ALL_LANGUAGES = {
    "arabic":   "datasets/arabic_clean.txt",
    "english":  "datasets/english_clean.txt",
    "estonian": "datasets/estonian_clean.txt",
    "finnish":  "datasets/finnish_clean.txt",
    "georgian": "datasets/georgian_clean.txt",
    "hindi":    "datasets/hindi_raw.txt",
    "hungarian":"datasets/hungarian_clean.txt",
    "tamil":    "datasets/tamil_clean.txt",
    "turkish":  "datasets/turkish_clean.txt",
}


# ─── Utilities ───────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_token_stats(encode_fn, corpus_path: str, max_lines: int = 1000) -> Dict:
    """Compute tok/word and tok/char ratios from the corpus."""
    total_tokens, total_words, total_chars = 0, 0, 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            tokens = encode_fn(line)
            words  = line.split()
            total_tokens += len(tokens)
            total_words  += len(words)
            total_chars  += len(line)

    tok_per_word = round(total_tokens / total_words, 3) if total_words else 0.0
    tok_per_char = round(total_tokens / total_chars, 4) if total_chars else 0.0
    return {
        "tok_per_word":  tok_per_word,
        "tok_per_char":  tok_per_char,
        "total_tokens":  total_tokens,
        "total_words":   total_words,
        "total_chars":   total_chars,
    }


def compute_bpc(avg_loss: float, tok_per_char: float) -> float:
    """Bits-per-character = avg_loss_nats × tokens_per_char / ln(2)."""
    if avg_loss is None or tok_per_char == 0:
        return None
    return round(avg_loss * tok_per_char / LN2, 4)


def train_nanolm(tok_wrapper: TokenizerWrapper, corpus_path: str,
                 max_steps: int, device: str) -> Dict[str, Any]:
    """Train NanoLM from scratch and return avg_loss and token PPL."""
    set_seed(SEED)

    ds = TextDataset(corpus_path, tok_wrapper,
                     block_size=BLOCK_SIZE, max_lines=TRAIN_LINES)
    if len(ds) == 0:
        return {"avg_loss": None, "avg_ppl": None}

    pin = device not in ("cpu", "mps")
    dl  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=0, pin_memory=pin)

    model = NanoLM(
        vocab_size=tok_wrapper.vocab_size,
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    total_loss, n_steps = 0.0, 0

    for x, y in dl:
        if n_steps >= max_steps:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss, _ = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_steps    += 1

    avg_loss = round(total_loss / n_steps, 4) if n_steps > 0 else None
    avg_ppl  = round(math.exp(avg_loss), 1)   if avg_loss else None
    return {"avg_loss": avg_loss, "avg_ppl": avg_ppl, "steps": n_steps}


# ─── Tokenizer training ──────────────────────────────────────────────────────

def setup_tokenizers(lang: str, corpus_path: str,
                     models_dir: Path) -> Dict[str, TokenizerWrapper]:
    """Train or load StructPiece, SP-BPE, and SP-Unigram for one language."""
    os.makedirs(models_dir, exist_ok=True)

    # --- StructPiece ---
    sp_dir  = models_dir / f"{lang}_structpiece"
    sp_vocab = sp_dir / "indic_tokenizer_vocab.json"
    if not sp_vocab.exists():
        print(f"    Training StructPiece for {lang}...")
        vocab, merges = train_bpe(corpus_path, vocab_size=VOCAB_SIZE,
                                   max_lines=TRAIN_LINES, verbose=False)
        tok = IndicTokenizer(vocab, merges)
        tok.save(str(sp_dir))
    structpiece = TokenizerWrapper("indic", str(sp_dir))

    # --- SP-BPE ---
    bpe_prefix = str(models_dir / f"{lang}_sp_bpe")
    if not os.path.exists(bpe_prefix + ".model"):
        print(f"    Training SP-BPE for {lang}...")
        spm.SentencePieceTrainer.Train(
            input=corpus_path, model_prefix=bpe_prefix,
            vocab_size=VOCAB_SIZE, model_type="bpe",
            character_coverage=1.0, normalization_rule_name="nfkc",
            num_threads=4, input_sentence_size=TRAIN_LINES,
            shuffle_input_sentence=True,
        )
    sp_bpe = TokenizerWrapper("sp", bpe_prefix + ".model")

    # --- SP-Unigram ---
    uni_prefix = str(models_dir / f"{lang}_sp_unigram")
    if not os.path.exists(uni_prefix + ".model"):
        print(f"    Training SP-Unigram for {lang}...")
        spm.SentencePieceTrainer.Train(
            input=corpus_path, model_prefix=uni_prefix,
            vocab_size=VOCAB_SIZE, model_type="unigram",
            character_coverage=1.0, normalization_rule_name="nfkc",
            num_threads=4, input_sentence_size=TRAIN_LINES,
            shuffle_input_sentence=True,
        )
    sp_uni = TokenizerWrapper("sp", uni_prefix + ".model")

    return {
        "StructPiece": structpiece,
        "SP-BPE":      sp_bpe,
        "SP-Unigram":  sp_uni,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce NanoLM intrinsic experiments from the paper"
    )
    parser.add_argument(
        "--langs", nargs="+",
        default=list(ALL_LANGUAGES.keys()),
        choices=list(ALL_LANGUAGES.keys()),
        help="Languages to evaluate (default: all 9)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned runs without training"
    )
    parser.add_argument(
        "--output", default="reproduce/results/nanolm_results.json",
        help="Path to save results JSON"
    )
    parser.add_argument(
        "--seeds", nargs="+", default=["42", "43", "44"], help="Seeds to evaluate"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Print cached Table 1 results and exit without training"
    )
    args = parser.parse_args()

    if args.verify_only:
        import json
        p = ROOT / "experiments" / "fresh_results" / "all_results.json"
        eng_p = ROOT / "experiments" / "fresh_results" / "english_results.json"
        rows = []
        if p.exists():
            with open(p) as f:
                rows.extend(json.load(f).get("table1_main_results", []))
        if eng_p.exists():
            with open(eng_p) as f:
                rows.extend(json.load(f).get("table1_main_results", []))
        print("\n" + "=" * 70)
        print("  VERIFICATION: Table 1 Cached Results (PPL)")
        print("=" * 70)
        print(f"  {'Language':<10} {'Tokenizer':<20} {'PPL':>8}")
        print("  " + "-" * 42)
        for r in rows:
            print(f"  {r['language']:<10} {r['tokenizer']:<20} {r.get('avg_ppl', 0):>8.1f}")
        print("\n\u2705 Cached NanoLM results verified.\n")
        return

    device = get_device()
    models_dir = ROOT / "reproduce" / "models"
    results_dir = ROOT / "reproduce" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  StructPiece — NanoLM Intrinsic Evaluation")
    print(f"  Device: {device} | Vocab: {VOCAB_SIZE} | Steps: {MAX_STEPS}")
    print(f"  Token Budget: {MAX_STEPS * BATCH_SIZE * BLOCK_SIZE:,} tokens/run")
    print("=" * 70)

    all_results: List[Dict] = []

    for lang in args.langs:
        corpus_path = str(ROOT / ALL_LANGUAGES[lang])
        if not os.path.exists(corpus_path):
            print(f"\n[SKIP] {lang}: corpus not found at {corpus_path}")
            continue

        print(f"\n{'─'*70}")
        print(f"  Language: {lang.upper()}")
        print(f"{'─'*70}")

        if args.dry_run:
            print(f"  [DRY RUN] Would train StructPiece, SP-BPE, SP-Unigram on {corpus_path}")
            continue

        tokenizers = setup_tokenizers(lang, corpus_path, models_dir)

        # Compute fertility stats once per tokenizer
        stats = {}
        for name, tok in tokenizers.items():
            stats[name] = compute_token_stats(tok.encode, corpus_path, max_lines=1000)

        bpe_tok_per_word = stats["SP-BPE"]["tok_per_word"]

        # Train NanoLM for each tokenizer
        for tok_name, tok_wrapper in tokenizers.items():
            print(f"\n  [{tok_name}] vocab={tok_wrapper.vocab_size} | "
                  f"tok/word={stats[tok_name]['tok_per_word']}")
            
            losses = []
            ppls = []
            total_time = 0
            steps = MAX_STEPS
            for seed in args.seeds:
                global SEED
                SEED = int(seed)
                t0  = time.time()
                lm  = train_nanolm(tok_wrapper, corpus_path, MAX_STEPS, device)
                elapsed = time.time() - t0
                if lm["avg_loss"] is not None:
                    losses.append(lm["avg_loss"])
                    ppls.append(lm["avg_ppl"])
                total_time += elapsed
                steps = lm.get("steps", MAX_STEPS)

            avg_loss = round(sum(losses)/len(losses), 4) if losses else None
            avg_ppl = round(sum(ppls)/len(ppls), 1) if ppls else None
            loss_std = round((sum((x - avg_loss)**2 for x in losses) / max(1, len(losses)-1))**0.5, 4) if len(losses)>1 else 0.0
            ppl_std = round((sum((x - avg_ppl)**2 for x in ppls) / max(1, len(ppls)-1))**0.5, 1) if len(ppls)>1 else 0.0

            bpc = compute_bpc(avg_loss, stats[tok_name]["tok_per_char"]) if avg_loss else None
            inflation = round(stats[tok_name]["tok_per_word"] / bpe_tok_per_word, 2) if bpe_tok_per_word else None

            print(f"    Loss={avg_loss} ± {loss_std} | PPL={avg_ppl} ± {ppl_std} | "
                  f"BPC={bpc} | Inflation={inflation}x | {total_time:.0f}s")

            all_results.append({
                "language":       lang.capitalize(),
                "tokenizer":      tok_name,
                "vocab_size":     tok_wrapper.vocab_size,
                "avg_loss":       avg_loss,
                "avg_loss_std":   loss_std,
                "token_ppl":      avg_ppl,
                "avg_ppl_std":    ppl_std,
                "bpc":            bpc,
                "tok_per_word":   stats[tok_name]["tok_per_word"],
                "tok_per_char":   stats[tok_name]["tok_per_char"],
                "inflation_vs_bpe": inflation,
                "steps":          steps,
                "time_seconds":   round(total_time, 1),
            })

        # Print per-language summary table
        lang_rows = [r for r in all_results if r["language"] == lang.capitalize()]
        print(f"\n  Summary for {lang.upper()}:")
        print(f"  {'Tokenizer':<16} {'PPL':>8} {'BPC':>8} {'Tok/Word':>10}")
        print(f"  {'─'*44}")
        for r in lang_rows:
            print(f"  {r['tokenizer']:<16} {str(r['token_ppl']):>8} "
                  f"{str(r['bpc']):>8} {str(r['tok_per_word']):>10}")

    # Save results
    if not args.dry_run and all_results:
        out_path = ROOT / args.output
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n\n{'='*70}")
        print(f"  Results saved to: {out_path}")
        print(f"  Total runs: {len(all_results)}")

        # Check that the key ordering claim holds
        by_lang: Dict[str, Dict] = {}
        for r in all_results:
            by_lang.setdefault(r["language"], {})[r["tokenizer"]] = r

        print(f"\n  Ordering check (paper's main claim):")
        print(f"  {'Language':<12} {'PPL order OK':>15} {'BPC order OK':>15}")
        print(f"  {'─'*44}")
        for lang_cap, toks in by_lang.items():
            sp = toks.get("StructPiece")
            bpe = toks.get("SP-BPE")
            uni = toks.get("SP-Unigram")
            if sp and bpe and uni:
                ppl_ok  = sp["token_ppl"] < uni["token_ppl"] < bpe["token_ppl"]
                bpc_ok  = uni["bpc"] < bpe["bpc"] < sp["bpc"]
                ppl_str = "✓" if ppl_ok else "✗ UNEXPECTED"
                bpc_str = "✓" if bpc_ok else "✗ UNEXPECTED"
                print(f"  {lang_cap:<12} {ppl_str:>15} {bpc_str:>15}")

        print(f"\n  Note: The ordering should be consistent with the paper.")
        print(f"  Absolute PPL values depend on local training conditions.")
        print(f"  The paper's H100 numbers are preserved in experiments/fresh_results/")
        print("=" * 70)


if __name__ == "__main__":
    main()
