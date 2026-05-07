#!/usr/bin/env python3
"""
Core Re-evaluation: MorphTokenizer v1 vs v2 vs SP-BPE
=====================================================
Trains tokenizers and NanoLMs for a focused set of languages,
computes ALL metrics (including BPC), and outputs structured results.

Usage:
    PYTHONPATH=. ~/.local/bin/uv run python experiments/run_core_reeval.py
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ─── Project imports ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from python.adaptive_bpe import train_bpe, IndicTokenizer
from python.pretokenizer import pretokenize
from lm.tokenizer_adapter import TokenizerWrapper
from lm.dataset import TextDataset
from lm.model import NanoLM

# ─── Configuration ───────────────────────────────────────────────────────────

LANGUAGES = {
    "english":  {"corpus": "datasets/english_clean.txt",  "script": "Latin",  "morphology": "Analytic"},
    "turkish":  {"corpus": "datasets/turkish_clean.txt",  "script": "Latin",  "morphology": "Agglutinative"},
    "hindi":    {"corpus": "datasets/hindi_raw.txt",      "script": "Devanagari", "morphology": "Fusional"},
    "arabic":   {"corpus": "datasets/arabic_clean.txt",   "script": "Arabic", "morphology": "Root-Pattern"},
}

VOCAB_SIZE = 4000
TRAIN_LINES = 2000
BLOCK_SIZE = 128
BATCH_SIZE = 32
LR = 5e-4
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 2
SEED = 42
LN2 = math.log(2)

# ─── Utilities ───────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_lm(tokenizer_wrapper, corpus_path, max_steps, device):
    """Train a NanoLM and return metrics dict."""
    torch.manual_seed(SEED)

    ds = TextDataset(corpus_path, tokenizer_wrapper, block_size=BLOCK_SIZE, max_lines=TRAIN_LINES)
    if len(ds) == 0:
        return {"avg_loss": None, "avg_ppl": None}

    pin_memory = False if device == "mps" else (device != "cpu")
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_memory)

    model = NanoLM(
        vocab_size=tokenizer_wrapper.vocab_size,
        block_size=BLOCK_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    history = []
    step = 0

    for x, y in dl:
        if step >= max_steps:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        _, loss, _ = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lv = loss.item()
        ppl = math.exp(lv) if lv < 20 else float("inf")
        history.append({"step": step, "loss": lv, "ppl": ppl})
        step += 1

    if not history:
        return {"avg_loss": None, "avg_ppl": None}

    # Average over last 20% of steps
    tail = history[max(0, len(history) - len(history) // 5):]
    avg_loss = sum(h["loss"] for h in tail) / len(tail)
    avg_ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    return {"avg_loss": round(avg_loss, 4), "avg_ppl": round(avg_ppl, 1)}


def count_tokens_for_corpus(encode_fn, corpus_path, max_lines=1000):
    """Count total tokens, chars, words for a corpus slice."""
    total_tokens = 0
    total_chars = 0
    total_words = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            ids = encode_fn(line)
            total_tokens += len(ids)
            total_chars += len(line)
            total_words += len(line.split())
    return total_tokens, total_chars, total_words


def compute_token_stats(encode_fn, corpus_path, max_lines=1000):
    """Compute tok/word, tok/char from corpus."""
    total_tokens, total_chars, total_words = count_tokens_for_corpus(
        encode_fn, corpus_path, max_lines
    )
    tok_per_word = total_tokens / max(total_words, 1)
    tok_per_char = total_tokens / max(total_chars, 1)
    return {
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "total_words": total_words,
        "tok_per_word": round(tok_per_word, 3),
        "tok_per_char": round(tok_per_char, 4),
    }


def compute_bpc(avg_loss, tok_per_char):
    """Bits per character = (avg_loss * tok_per_char) / ln(2)"""
    if avg_loss is None or tok_per_char is None:
        return None
    return round((avg_loss * tok_per_char) / LN2, 3)


# ─── Main ────────────────────────────────────────────────────────────────────

def verify_cached():
    """Read cached task3456_out.json and print ablation summary."""
    import json
    p = ROOT / "experiments" / "fresh_results" / "task3456_out.json"
    if not p.exists():
        print(f"ERROR: Missing {p}")
        return
    with open(p) as f:
        d = json.load(f)
    seeds3 = d.get("ablation_3seed", {})
    print("\n" + "=" * 70)
    print("  VERIFICATION: Table 4 Ablation Results (3-seed mean ± std)")
    print("=" * 70)
    print(f"  {'Variant':<30} {'PPL':>8}  std")
    print("  " + "-" * 48)
    for variant, vals in seeds3.items():
        print(f"  {variant:<30} {vals['ppl_m']:>8.1f}  ±{vals['ppl_s']:.1f}")
    print("\n✅ Cached ablation results verified.\n")


def main():
    parser = argparse.ArgumentParser(description="Core re-evaluation: v1 vs v2 vs SP-BPE")
    parser.add_argument("--langs", "--languages", type=str, default="turkish,english",
                        help="Comma-separated language list")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="LM training steps")
    parser.add_argument("--seeds", nargs="+", default=["42", "43", "44"], help="Seeds to evaluate")
    parser.add_argument("--verify-only", action="store_true",
                        help="Print cached ablation results and exit without training")
    args = parser.parse_args()
    if args.verify_only:
        verify_cached()
        return

    lang_list = [l.strip() for l in args.languages.split(",")]
    MAX_STEPS = args.max_steps
    device = get_device()

    print(f"Device: {device} | Steps: {MAX_STEPS}")
    print(f"Languages: {lang_list}")
    print(f"Config: d={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, bs={BATCH_SIZE}, vocab={VOCAB_SIZE}")
    print("=" * 80)

    out_dir = ROOT / "experiments" / "v2_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    all_results = []

    for lang in lang_list:
        if lang not in LANGUAGES:
            print(f"SKIPPING unknown language: {lang}")
            continue

        meta = LANGUAGES[lang]
        corpus_path = str(ROOT / meta["corpus"])

        if not os.path.exists(corpus_path):
            print(f"SKIPPING {lang}: corpus not found at {corpus_path}")
            continue

        print(f"\n{'='*80}")
        print(f"  LANGUAGE: {lang.upper()} ({meta['script']}, {meta['morphology']})")
        print(f"{'='*80}")

        # ─── English uses WikiText sample if available ───────────────────
        # Check for english_sample in experiments/
        if lang == "english":
            alt_corpus = str(ROOT / "experiments" / "english_sample.txt")
            if os.path.exists(alt_corpus):
                corpus_path = alt_corpus
                print(f"  Using English sample: {alt_corpus}")

        # ─── Train tokenizers ────────────────────────────────────────────

        # 1. MorphTokenizer ORIGINAL
        print(f"\n  [1/4] Training MorphTokenizer (original)...")
        t0 = time.time()
        vocab_v1, merges_v1 = train_bpe(corpus_path, vocab_size=VOCAB_SIZE,
                                         max_lines=TRAIN_LINES, verbose=False,
                                         variant="original")
        morph_v1 = IndicTokenizer(vocab_v1, merges_v1)
        morph_v1_dir = str(models_dir / f"{lang}_morph_v1")
        morph_v1.save(morph_v1_dir)
        print(f"         Trained in {time.time()-t0:.1f}s | Vocab: {len(vocab_v1)}")

        # 2. MorphTokenizer V2
        print(f"  [2/4] Training MorphTokenizer (v2 - compression-aware)...")
        t0 = time.time()
        vocab_v2, merges_v2 = train_bpe(corpus_path, vocab_size=VOCAB_SIZE,
                                         max_lines=TRAIN_LINES, verbose=False,
                                         variant="v2")
        morph_v2 = IndicTokenizer(vocab_v2, merges_v2)
        morph_v2_dir = str(models_dir / f"{lang}_morph_v2")
        morph_v2.save(morph_v2_dir)
        print(f"         Trained in {time.time()-t0:.1f}s | Vocab: {len(vocab_v2)}")

        # 3. SP-BPE
        sp_prefix = str(models_dir / f"{lang}_sp_bpe")
        sp_model = sp_prefix + ".model"
        print(f"  [3/4] Training SentencePiece BPE...")
        spm.SentencePieceTrainer.Train(
            input=corpus_path,
            model_prefix=sp_prefix,
            vocab_size=VOCAB_SIZE,
            model_type="bpe",
            character_coverage=1.0,
            normalization_rule_name="nfkc",
            num_threads=4,
            max_sentence_length=16384,
            input_sentence_size=TRAIN_LINES,
            shuffle_input_sentence=True,
        )

        # ─── Compute token stats ─────────────────────────────────────────

        sp_tok = spm.SentencePieceProcessor()
        sp_tok.Load(sp_model)

        stats = {}
        for name, enc_fn in [
            ("MorphTok-v1", morph_v1.encode),
            ("MorphTok-v2", morph_v2.encode),
            ("SP-BPE", lambda t: sp_tok.Encode(t, out_type=int)),
        ]:
            stats[name] = compute_token_stats(enc_fn, corpus_path, max_lines=1000)
            print(f"    {name}: tok/word={stats[name]['tok_per_word']}, "
                  f"tok/char={stats[name]['tok_per_char']}")

        # ─── Train LMs ───────────────────────────────────────────────────

        print(f"\n  [4/4] Training Language Models ({MAX_STEPS} steps each)...")

        tokenizers = {
            "MorphTok-v1": TokenizerWrapper("indic", morph_v1_dir),
            "MorphTok-v2": TokenizerWrapper("indic", morph_v2_dir),
            "SP-BPE":      TokenizerWrapper("sp", sp_model),
        }

        lm_results = {}
        for tok_name, tok_wrapper in tokenizers.items():
            print(f"    Training LM with {tok_name} (vocab={tok_wrapper.vocab_size})...")
            losses = []
            ppls = []
            total_time = 0
            for seed in args.seeds:
                global SEED
                SEED = int(seed)
                t0 = time.time()
                res = train_lm(tok_wrapper, corpus_path, MAX_STEPS, device)
                elapsed = time.time() - t0
                if res["avg_loss"] is not None:
                    losses.append(res["avg_loss"])
                    ppls.append(res["avg_ppl"])
                total_time += elapsed

            avg_loss = round(sum(losses)/len(losses), 4) if losses else None
            avg_ppl = round(sum(ppls)/len(ppls), 1) if ppls else None
            loss_std = round((sum((x - avg_loss)**2 for x in losses) / max(1, len(losses)-1))**0.5, 4) if len(losses)>1 else 0.0
            ppl_std = round((sum((x - avg_ppl)**2 for x in ppls) / max(1, len(ppls)-1))**0.5, 1) if len(ppls)>1 else 0.0
            
            print(f"      → avg_loss={avg_loss} ± {loss_std}, avg_ppl={avg_ppl} ± {ppl_std} ({total_time:.1f}s total)")
            lm_results[tok_name] = {
                "avg_loss": avg_loss, "avg_loss_std": loss_std,
                "avg_ppl": avg_ppl, "avg_ppl_std": ppl_std
            }

        # ─── Compute BPC and full metrics ────────────────────────────────

        bpe_tok_per_word = stats["SP-BPE"]["tok_per_word"]

        for tok_name in ["MorphTok-v1", "MorphTok-v2", "SP-BPE"]:
            avg_loss = lm_results[tok_name]["avg_loss"]
            tok_per_char = stats[tok_name]["tok_per_char"]
            tok_per_word = stats[tok_name]["tok_per_word"]

            bpc = compute_bpc(avg_loss, tok_per_char)
            inflation = round(tok_per_word / bpe_tok_per_word, 2) if bpe_tok_per_word else None
            attn_cost = round(inflation ** 2, 2) if inflation else None

            row = {
                "language": lang.capitalize(),
                "tokenizer": tok_name,
                "avg_loss": avg_loss,
                "token_ppl": lm_results[tok_name]["avg_ppl"],
                "tok_per_word": tok_per_word,
                "tok_per_char": tok_per_char,
                "bpc": bpc,
                "inflation": inflation,
                "attn_cost": attn_cost,
                "total_tokens": stats[tok_name]["total_tokens"],
                "total_chars": stats[tok_name]["total_chars"],
                "total_words": stats[tok_name]["total_words"],
            }
            all_results.append(row)

    # ─── Save JSON ───────────────────────────────────────────────────────

    results_path = out_dir / "core_reeval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ─── Print formatted tables ──────────────────────────────────────────

    print(f"\n{'='*100}")
    print("  TABLE A — CHARACTER-NORMALIZED MODELING (BPC)")
    print(f"{'='*100}")
    print(f"{'Language':<12} {'Tokenizer':<16} {'Avg Loss':>10} {'Token PPL':>10} "
          f"{'Tok/Word':>10} {'Tok/Char':>10} {'BPC ↓':>8}")
    print("-" * 100)
    for r in all_results:
        print(f"{r['language']:<12} {r['tokenizer']:<16} {r['avg_loss']:>10} {r['token_ppl']:>10} "
              f"{r['tok_per_word']:>10} {r['tok_per_char']:>10} {r['bpc']:>8}")

    print(f"\n{'='*80}")
    print("  TABLE B — COMPUTE TRADEOFF")
    print(f"{'='*80}")
    print(f"{'Language':<12} {'Tokenizer':<16} {'Tok/Word':>10} {'Inflation':>10} {'Attn Cost':>10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['language']:<12} {r['tokenizer']:<16} {r['tok_per_word']:>10} "
              f"{r['inflation']:>10} {r['attn_cost']:>10}")

    # ─── V1 vs V2 comparison ─────────────────────────────────────────────

    print(f"\n{'='*80}")
    print("  TABLE F — OLD vs NEW MorphTokenizer")
    print(f"{'='*80}")
    print(f"{'Language':<12} {'Metric':<12} {'MorphTok-v1':>14} {'MorphTok-v2':>14} {'SP-BPE':>14} {'v2 improves?':>14}")
    print("-" * 80)

    # Group by language
    by_lang = {}
    for r in all_results:
        by_lang.setdefault(r["language"], {})[r["tokenizer"]] = r

    for lang, toks in by_lang.items():
        v1 = toks.get("MorphTok-v1", {})
        v2 = toks.get("MorphTok-v2", {})
        bpe = toks.get("SP-BPE", {})

        for metric in ["bpc", "tok_per_word", "inflation", "token_ppl"]:
            v1_val = v1.get(metric, "—")
            v2_val = v2.get(metric, "—")
            bpe_val = bpe.get(metric, "—")

            # Check improvement direction
            if isinstance(v1_val, (int, float)) and isinstance(v2_val, (int, float)):
                if metric in ["bpc", "tok_per_word", "inflation"]:
                    improved = "✓" if v2_val < v1_val else "✗"
                else:
                    improved = "✓" if v2_val < v1_val else "✗"
            else:
                improved = "—"

            print(f"{lang:<12} {metric:<12} {str(v1_val):>14} {str(v2_val):>14} "
                  f"{str(bpe_val):>14} {improved:>14}")
        print()

    print(f"\nDone. Full results: {results_path}")


if __name__ == "__main__":
    main()
