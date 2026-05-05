#!/usr/bin/env python3
"""
Scaling Experiments Infrastructure
==================================
Runs Language Modeling scaling experiments varying the `max_steps` variable
while holding all other hyperparameters (model architecture, token budget per step) constant.
Supports dry runs, single config injections, and JSON result serialization.

Usage:
  python experiments/run_scaling_experiments.py --config configs/scaling_experiment.yaml --dry_run
"""

import argparse
import json
import os
import sys
import time
import yaml
from pathlib import Path

import torch

# ─── Project imports ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from lm.tokenizer_adapter import TokenizerWrapper
from experiments.run_core_reeval import train_lm, compute_bpc, compute_token_stats

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def get_corpus_path(lang: str) -> str:
    """Resolve the dataset path for a given language."""
    clean = ROOT / "datasets" / f"{lang.lower()}_clean.txt"
    if clean.exists():
        return str(clean)
    raw = ROOT / "datasets" / f"{lang.lower()}_raw.txt"
    if raw.exists():
        return str(raw)
    raise FileNotFoundError(f"Corpus for {lang} not found in datasets/ directory.")

def get_tokenizer_wrapper(lang: str, tok_type: str, vocab_size: int) -> TokenizerWrapper:
    """
    Locates an existing trained Tokenizer. 
    Assumes models live in experiments/modern_baselines/models/ due to prior runs.
    """
    base_models_dir = ROOT / "experiments" / "modern_baselines" / "models"
    lang_l = lang.lower()
    
    if tok_type.lower() == "morph":
        path = base_models_dir / f"{lang_l}_morph"
        return TokenizerWrapper("indic", str(path))
    elif tok_type.lower() == "bpe":
        path = base_models_dir / f"{lang_l}_sp_bpe.model"
        return TokenizerWrapper("sp", str(path))
    elif tok_type.lower() == "unigram":
        path = base_models_dir / f"{lang_l}_sp_unigram.model"
        return TokenizerWrapper("sp", str(path))
    else:
        raise ValueError(f"Unknown tokenizer type: {tok_type}")

def load_config(args):
    """Resolve config from YAML or CLI logic."""
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        languages = cfg.get("languages", [])
        tokenizers = cfg.get("tokenizers", [])
        steps_list = cfg.get("steps_list", [])
        vocab_size = cfg.get("vocab_size", 4000)
    else:
        languages = args.language if args.language else []
        tokenizers = args.tokenizer if args.tokenizer else []
        steps_list = args.steps_list if args.steps_list else []
        vocab_size = args.vocab_size

    # Validate
    if not languages or not tokenizers or not steps_list:
        raise ValueError("Must provide languages, tokenizers, and steps_list (via config or CLI flags).")
    
    return languages, tokenizers, steps_list, vocab_size

def ensure_reproducibility():
    """Fix seeds for strict comparability between runs."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    try:
        import numpy as np
        np.random.seed(42)
    except ImportError:
        pass

def main():
    parser = argparse.ArgumentParser(description="Language Model Scaling Experiments")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--language", nargs="+", help="Languages to run (e.g. turkish hindi)")
    parser.add_argument("--tokenizer", nargs="+", help="Tokenizers to run (e.g. bpe unigram morph)")
    parser.add_argument("--steps_list", nargs="+", type=int, help="List of steps (e.g. 500 1500 3000)")
    parser.add_argument("--vocab_size", type=int, default=4000, help="Vocabulary size (default 4000)")
    parser.add_argument("--dry_run", action="store_true", help="Print planned runs without training")
    parser.add_argument("--output_dir", type=str, default="experiments/scaling_results", help="Dir to save JSONs")
    
    args = parser.parse_args()
    languages, tokenizers, steps_list, vocab_size = load_config(args)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ensure_reproducibility()

    print("=" * 80)
    print("  SCALING SANITY EXPERIMENTS")
    print(f"  Dry Run: {'YES' if args.dry_run else 'NO'}")
    print("=" * 80)

    for lang in languages:
        try:
            corpus_path = get_corpus_path(lang)
        except FileNotFoundError as e:
            print(f"SKIPPING: {e}")
            continue

        print(f"\n[Language: {lang.upper()}] Corpus: {corpus_path}")

        for tok_name in tokenizers:
            print(f"\n  [Tokenizer: {tok_name.upper()}]")
            
            structured_result = {
                "language": lang,
                "tokenizer": tok_name,
                "vocab_size": vocab_size,
                "runs": []
            }
            
            try:
                tok_wrapper = get_tokenizer_wrapper(lang, tok_name, vocab_size)
            except Exception as e:
                print(f"    ERROR loading tokenizer {tok_name}: {e}")
                continue

            if not args.dry_run:
                # Pre-compute tok/char once for BPC math
                print("    Computing base token efficiency for BPC calculation...")
                st = compute_token_stats(tok_wrapper.encode, corpus_path, max_lines=1000)
                tok_per_char = st["tok_per_char"]

            for steps in steps_list:
                if args.dry_run:
                    print(f"    → (Dry Run) Planned: train '{tok_name}' on '{lang}' for EXACTLY {steps} steps.")
                    continue

                print(f"\n    → Training {tok_name} for EXACTLY {steps} steps...")
                t0 = time.time()
                
                res = train_lm(tok_wrapper, corpus_path, steps, DEVICE)
                elapsed = time.time() - t0
                
                avg_loss = res["avg_loss"]
                avg_ppl = res["avg_ppl"]
                bpc = compute_bpc(avg_loss, tok_per_char)
                
                print(f"      [RES] {steps} steps | Loss: {avg_loss:.4f} | PPL: {avg_ppl:.2f} | BPC: {bpc:.3f} | Time: {elapsed:.1f}s")
                
                structured_result["runs"].append({
                    "steps": steps,
                    "avg_loss": avg_loss,
                    "token_ppl": avg_ppl,
                    "bpc": bpc,
                    "time_seconds": elapsed
                })
                
            if not args.dry_run:
                outfile = out_dir / f"{lang.lower()}_{tok_name.lower()}.json"
                with open(outfile, "w") as f:
                    json.dump(structured_result, f, indent=2)
                print(f"    Saved results to {outfile}")

    print("\n" + "=" * 80)
    print("  SCALING EXECUTIONS COMPLETE.")
    print("=" * 80)

if __name__ == "__main__":
    main()
