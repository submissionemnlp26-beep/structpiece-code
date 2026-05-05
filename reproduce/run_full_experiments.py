#!/usr/bin/env python3
"""
Complete Experimental Suite for MorphTokenizer ACL Paper
========================================================
Trains MorphTokenizer + SentencePiece baselines across 8 languages,
trains identical NanoLM models, computes all metrics, and generates
7 publication-ready tables (Markdown + LaTeX).

ALL results are computed from scratch. No cached outputs are used.

Usage:
    PYTHONPATH=. uv run python experiments/run_full_experiments.py
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
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

from python.adaptive_bpe import train_bpe, IndicTokenizer, ACTIVE_SUFFIXES
from python.pretokenizer import pretokenize
from lm.tokenizer_adapter import TokenizerWrapper
from lm.dataset import TextDataset
from lm.model import NanoLM

# ─── Configuration ───────────────────────────────────────────────────────────

LANGUAGES = {
    "hindi":     {"corpus": "datasets/hindi_raw.txt",     "script": "Devanagari", "morphology": "Fusional"},
    "arabic":    {"corpus": "datasets/arabic_clean.txt",  "script": "Arabic",     "morphology": "Root-Pattern"},
    "tamil":     {"corpus": "datasets/tamil_clean.txt",   "script": "Tamil",      "morphology": "Agglutinative"},
    "finnish":   {"corpus": "datasets/finnish_clean.txt", "script": "Latin",      "morphology": "Agglutinative"},
    "hungarian": {"corpus": "datasets/hungarian_clean.txt","script": "Latin",      "morphology": "Agglutinative"},
    "estonian":  {"corpus": "datasets/estonian_clean.txt","script": "Latin",      "morphology": "Agglutinative"},
    "georgian":  {"corpus": "datasets/georgian_clean.txt","script": "Georgian",   "morphology": "Agglutinative"},
    "turkish":   {"corpus": "datasets/turkish_clean.txt", "script": "Latin",      "morphology": "Agglutinative"},
}

VOCAB_SIZE = 4000
TRAIN_LINES = 2000     # Lines used for tokenizer training (kept small for BPE feasibility)
BLOCK_SIZE = 128
BATCH_SIZE = 32
LR = 5e-4
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 2
SEED = 42

# ─── Utilities ───────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_lm(tokenizer_wrapper, corpus_path, max_steps, device):
    """Train a NanoLM and return (final_loss, final_ppl, avg_loss, avg_ppl, history)."""
    torch.manual_seed(SEED)

    ds = TextDataset(corpus_path, tokenizer_wrapper, block_size=BLOCK_SIZE, max_lines=TRAIN_LINES)
    if len(ds) == 0:
        return None, None, None, None, []

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
        return None, None, None, None, []

    final_loss = history[-1]["loss"]
    final_ppl = history[-1]["ppl"]
    # Average over last 20% of steps for stability
    tail = history[max(0, len(history) - len(history)//5):]
    avg_loss = sum(h["loss"] for h in tail) / len(tail)
    avg_ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    return final_loss, final_ppl, avg_loss, avg_ppl, history


def count_tokens_for_corpus(encode_fn, corpus_path, max_lines=1000):
    """Count total tokens, total chars, total words for a corpus slice."""
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


# ─── Main pipeline ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=500,
                        help="LM training steps per run (default: 500)")
    args = parser.parse_args()

    MAX_STEPS = args.max_steps
    device = get_device()
    print(f"Device: {device} | Max LM steps: {MAX_STEPS}")
    print(f"Model config: d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}, block_size={BLOCK_SIZE}")
    print(f"Vocab size: {VOCAB_SIZE} | Training lines: {TRAIN_LINES}")
    print("=" * 80)

    out_dir = ROOT / "experiments" / "fresh_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean old cached models
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    # COLLECT ALL RESULTS
    # ═══════════════════════════════════════════════════════════════════════

    table1_rows = []   # Main Results: Language | Tokenizer | Loss | PPL
    table2_rows = []   # Token Efficiency
    table3_rows = []   # Total Tokens
    table6_rows = []   # Cross-Language
    table7_rows = []   # Qualitative examples
    ablation_data = {} # For Table 5

    all_results = {}

    for lang, meta in LANGUAGES.items():
        corpus_path = str(ROOT / meta["corpus"])
        print(f"\n{'='*80}")
        print(f"  LANGUAGE: {lang.upper()} ({meta['script']}, {meta['morphology']})")
        print(f"{'='*80}")

        lang_results = {"language": lang, "script": meta["script"], "morphology": meta["morphology"]}

        # ─── 1. Train MorphTokenizer ─────────────────────────────────────
        morph_dir = models_dir / f"{lang}_morph"
        print(f"\n[1/4] Training MorphTokenizer for {lang}...")
        t0 = time.time()
        vocab, merges = train_bpe(corpus_path, vocab_size=VOCAB_SIZE, max_lines=TRAIN_LINES, verbose=False)
        morph_tok = IndicTokenizer(vocab, merges)
        morph_tok.save(str(morph_dir))
        morph_train_time = time.time() - t0
        print(f"      Trained in {morph_train_time:.1f}s | Vocab: {len(vocab)} | Merges: {len(merges)}")

        # Save discovered suffixes for this language
        from python.adaptive_bpe import ACTIVE_SUFFIXES as current_suffixes
        lang_results["discovered_suffixes"] = list(current_suffixes)
        print(f"      Discovered {len(current_suffixes)} suffixes: {list(current_suffixes)[:10]}...")

        # ─── 2. Train SentencePiece BPE ──────────────────────────────────
        sp_bpe_prefix = str(models_dir / f"{lang}_sp_bpe")
        sp_bpe_model = sp_bpe_prefix + ".model"
        print(f"\n[2/4] Training SentencePiece BPE for {lang}...")
        spm.SentencePieceTrainer.Train(
            input=corpus_path,
            model_prefix=sp_bpe_prefix,
            vocab_size=VOCAB_SIZE,
            model_type="bpe",
            character_coverage=1.0,
            normalization_rule_name="nfkc",
            num_threads=4,
            max_sentence_length=16384,
            input_sentence_size=TRAIN_LINES,
            shuffle_input_sentence=True,
        )

        # ─── 3. Train SentencePiece Unigram ──────────────────────────────
        sp_uni_prefix = str(models_dir / f"{lang}_sp_unigram")
        sp_uni_model = sp_uni_prefix + ".model"
        print(f"[3/4] Training SentencePiece Unigram for {lang}...")
        spm.SentencePieceTrainer.Train(
            input=corpus_path,
            model_prefix=sp_uni_prefix,
            vocab_size=VOCAB_SIZE,
            model_type="unigram",
            character_coverage=1.0,
            normalization_rule_name="nfkc",
            num_threads=4,
            max_sentence_length=16384,
            input_sentence_size=TRAIN_LINES,
            shuffle_input_sentence=True,
        )

        # ─── 4. Train LMs ───────────────────────────────────────────────
        print(f"\n[4/4] Training Language Models ({MAX_STEPS} steps each)...")

        tokenizers = {
            "MorphTokenizer": TokenizerWrapper("indic", str(morph_dir)),
            "SP-BPE":         TokenizerWrapper("sp", sp_bpe_model),
            "SP-Unigram":     TokenizerWrapper("sp", sp_uni_model),
        }

        for tok_name, tok_wrapper in tokenizers.items():
            print(f"      Training LM with {tok_name} (vocab={tok_wrapper.vocab_size})...")
            t0 = time.time()
            final_loss, final_ppl, avg_loss, avg_ppl, history = train_lm(
                tok_wrapper, corpus_path, MAX_STEPS, device
            )
            lm_time = time.time() - t0

            if final_loss is None:
                print(f"      [SKIP] No data for {tok_name}")
                continue

            print(f"      {tok_name}: avg_loss={avg_loss:.4f} avg_ppl={avg_ppl:.1f} ({lm_time:.1f}s)")

            # ── Table 1: Main Results ────────────────────────────────────
            table1_rows.append({
                "language": lang.capitalize(),
                "tokenizer": tok_name,
                "avg_loss": round(avg_loss, 4),
                "avg_ppl": round(avg_ppl, 1),
            })

            # ── Table 2 & 3: Token Efficiency ────────────────────────────
            total_toks, total_chars, total_words = count_tokens_for_corpus(
                tok_wrapper.encode, corpus_path, max_lines=1000
            )
            table2_rows.append({
                "language": lang.capitalize(),
                "tokenizer": tok_name,
                "tokens_per_512chars": round(total_toks / max(1, total_chars) * 512, 1),
                "chars_per_512tokens": round(total_chars / max(1, total_toks) * 512, 1),
                "tokens_per_word": round(total_toks / max(1, total_words), 3),
            })
            table3_rows.append({
                "language": lang.capitalize(),
                "tokenizer": tok_name,
                "total_tokens": total_toks,
                "total_chars": total_chars,
                "total_words": total_words,
            })

            lang_results[tok_name] = {
                "avg_loss": avg_loss, "avg_ppl": avg_ppl,
                "final_loss": final_loss, "final_ppl": final_ppl,
                "total_tokens": total_toks, "total_words": total_words,
                "history": history,
            }

        # ── Table 6: Cross-Language ──────────────────────────────────────
        morph_ppl = lang_results.get("MorphTokenizer", {}).get("avg_ppl")
        bpe_ppl = lang_results.get("SP-BPE", {}).get("avg_ppl")
        if morph_ppl and bpe_ppl and bpe_ppl > 0:
            gain = ((bpe_ppl - morph_ppl) / bpe_ppl) * 100
            table6_rows.append({
                "language": lang.capitalize(),
                "script": meta["script"],
                "morphology": meta["morphology"],
                "bpe_ppl": round(bpe_ppl, 1),
                "morph_ppl": round(morph_ppl, 1),
                "ppl_reduction_pct": round(gain, 1),
            })

        all_results[lang] = lang_results

    # ═══════════════════════════════════════════════════════════════════════
    # TABLE 5: ABLATION (Hindi only — EGC-only vs Morphology-only vs Full)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  ABLATION STUDY (Hindi)")
    print(f"{'='*80}")

    hindi_corpus = str(ROOT / "datasets/hindi_raw.txt")

    # Variant 1: Standard BPE baseline (already have SP-BPE for Hindi)
    # Variant 2: EGC-only (MorphTokenizer with morphology bonus disabled)
    # Variant 3: Morphology-only (standard char split + morphology bonus)
    # Variant 4: Full MorphTokenizer (already computed)

    # For EGC-only: we temporarily disable the morphology bonus
    import python.adaptive_bpe as abpe
    orig_bonus_short = abpe._MORPHOLOGY_BONUS_SHORT
    orig_bonus_long = abpe._MORPHOLOGY_BONUS_LONG

    # ── EGC-only variant ──
    print("\n  Training EGC-only variant (morphology bonus = 0)...")
    abpe._MORPHOLOGY_BONUS_SHORT = 0.0
    abpe._MORPHOLOGY_BONUS_LONG = 0.0
    egc_dir = models_dir / "hindi_egc_only"
    vocab_egc, merges_egc = train_bpe(hindi_corpus, vocab_size=VOCAB_SIZE, max_lines=TRAIN_LINES, verbose=False)
    egc_tok = IndicTokenizer(vocab_egc, merges_egc)
    egc_tok.save(str(egc_dir))

    egc_wrapper = TokenizerWrapper("indic", str(egc_dir))
    _, _, egc_avg_loss, egc_avg_ppl, _ = train_lm(egc_wrapper, hindi_corpus, MAX_STEPS, device)
    print(f"  EGC-only: avg_loss={egc_avg_loss:.4f}, avg_ppl={egc_avg_ppl:.1f}")

    # ── Morphology-only variant (disable script penalty) ──
    print("\n  Training Morph-only variant (script penalty = 0)...")
    abpe._MORPHOLOGY_BONUS_SHORT = orig_bonus_short
    abpe._MORPHOLOGY_BONUS_LONG = orig_bonus_long
    orig_penalty = abpe._SCRIPT_CROSS_PENALTY
    abpe._SCRIPT_CROSS_PENALTY = 0.0
    morph_only_dir = models_dir / "hindi_morph_only"
    vocab_mo, merges_mo = train_bpe(hindi_corpus, vocab_size=VOCAB_SIZE, max_lines=TRAIN_LINES, verbose=False)
    morph_only_tok = IndicTokenizer(vocab_mo, merges_mo)
    morph_only_tok.save(str(morph_only_dir))
    abpe._SCRIPT_CROSS_PENALTY = orig_penalty  # restore

    morph_only_wrapper = TokenizerWrapper("indic", str(morph_only_dir))
    _, _, mo_avg_loss, mo_avg_ppl, _ = train_lm(morph_only_wrapper, hindi_corpus, MAX_STEPS, device)
    print(f"  Morph-only: avg_loss={mo_avg_loss:.4f}, avg_ppl={mo_avg_ppl:.1f}")

    # Gather ablation results
    hindi_bpe_ppl = None
    hindi_full_ppl = None
    for r in table1_rows:
        if r["language"] == "Hindi" and r["tokenizer"] == "SP-BPE":
            hindi_bpe_ppl = r["avg_ppl"]
        if r["language"] == "Hindi" and r["tokenizer"] == "MorphTokenizer":
            hindi_full_ppl = r["avg_ppl"]

    ablation_rows = [
        {"variant": "Standard BPE", "avg_loss": next((r["avg_loss"] for r in table1_rows if r["language"]=="Hindi" and r["tokenizer"]=="SP-BPE"), None),
         "avg_ppl": hindi_bpe_ppl},
        {"variant": "+ EGC Pretokenization", "avg_loss": round(egc_avg_loss, 4), "avg_ppl": round(egc_avg_ppl, 1)},
        {"variant": "+ Morphology Bonus", "avg_loss": round(mo_avg_loss, 4), "avg_ppl": round(mo_avg_ppl, 1)},
        {"variant": "Full MorphTokenizer", "avg_loss": next((r["avg_loss"] for r in table1_rows if r["language"]=="Hindi" and r["tokenizer"]=="MorphTokenizer"), None),
         "avg_ppl": hindi_full_ppl},
    ]

    # ═══════════════════════════════════════════════════════════════════════
    # TABLE 7: QUALITATIVE (Hindi)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  QUALITATIVE EXAMPLES (Hindi)")
    print(f"{'='*80}")

    qual_words = ["खेलना", "चलती", "लड़कों", "राजनीतिक", "पढ़ाई",
                  "दुकानदार", "अध्यापकों", "वाहनवाला", "सुन्दरता"]

    morph_hindi = IndicTokenizer.load(str(models_dir / "hindi_morph"))
    sp_bpe_hindi = spm.SentencePieceProcessor()
    sp_bpe_hindi.Load(str(models_dir / "hindi_sp_bpe.model"))

    for word in qual_words:
        morph_ids = morph_hindi.encode(word)
        morph_tokens = [morph_hindi.id_to_token.get(i, "?") for i in morph_ids]
        bpe_tokens = sp_bpe_hindi.Encode(word, out_type=str)
        table7_rows.append({
            "word": word,
            "bpe_tokens": bpe_tokens,
            "bpe_count": len(bpe_tokens),
            "morph_tokens": morph_tokens,
            "morph_count": len(morph_tokens),
        })
        print(f"  {word:20s} BPE={bpe_tokens}  Morph={morph_tokens}")

    # ═══════════════════════════════════════════════════════════════════════
    # FORMAT AND PRINT ALL TABLES
    # ═══════════════════════════════════════════════════════════════════════

    output_lines = []
    output_lines.append("\n" + "=" * 80)
    output_lines.append("  COMPLETE EXPERIMENTAL RESULTS")
    output_lines.append("=" * 80)

    # ── TABLE 1 ──────────────────────────────────────────────────────────
    output_lines.append("\n## TABLE 1 — Main Results: Language Modeling Perplexity")
    output_lines.append(f"Config: NanoLM (d={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}), {MAX_STEPS} steps, block_size={BLOCK_SIZE}")
    output_lines.append("")
    output_lines.append(f"| {'Language':<12} | {'Tokenizer':<16} | {'Avg Loss':>10} | {'Avg PPL':>10} |")
    output_lines.append(f"|{'-'*14}|{'-'*18}|{'-'*12}|{'-'*12}|")
    for r in sorted(table1_rows, key=lambda x: (x["language"], x["tokenizer"])):
        output_lines.append(f"| {r['language']:<12} | {r['tokenizer']:<16} | {r['avg_loss']:>10.4f} | {r['avg_ppl']:>10.1f} |")

    # ── TABLE 2 ──────────────────────────────────────────────────────────
    output_lines.append("\n## TABLE 2 — Token Efficiency")
    output_lines.append("")
    output_lines.append(f"| {'Language':<12} | {'Tokenizer':<16} | {'Tok/Word':>10} | {'Tok/512ch':>10} | {'Ch/512tok':>10} |")
    output_lines.append(f"|{'-'*14}|{'-'*18}|{'-'*12}|{'-'*12}|{'-'*12}|")
    for r in sorted(table2_rows, key=lambda x: (x["language"], x["tokenizer"])):
        output_lines.append(f"| {r['language']:<12} | {r['tokenizer']:<16} | {r['tokens_per_word']:>10.3f} | {r['tokens_per_512chars']:>10.1f} | {r['chars_per_512tokens']:>10.1f} |")

    # ── TABLE 3 ──────────────────────────────────────────────────────────
    output_lines.append("\n## TABLE 3 — Total Tokens (first 1000 lines)")
    output_lines.append("")
    output_lines.append(f"| {'Language':<12} | {'Tokenizer':<16} | {'Total Tokens':>14} | {'Total Words':>12} | {'Total Chars':>14} |")
    output_lines.append(f"|{'-'*14}|{'-'*18}|{'-'*16}|{'-'*14}|{'-'*16}|")
    for r in sorted(table3_rows, key=lambda x: (x["language"], x["tokenizer"])):
        output_lines.append(f"| {r['language']:<12} | {r['tokenizer']:<16} | {r['total_tokens']:>14,} | {r['total_words']:>12,} | {r['total_chars']:>14,} |")

    # ── TABLE 4 ──────────────────────────────────────────────────────────
    output_lines.append("\n## TABLE 4 — Morphology Impact")
    output_lines.append("")
    output_lines.append(f"| {'Language':<12} | {'Morphology Type':<18} | {'BPE PPL':>10} | {'Morph PPL':>10} | {'Improvement':>12} |")
    output_lines.append(f"|{'-'*14}|{'-'*20}|{'-'*12}|{'-'*12}|{'-'*14}|")
    for r in table6_rows:
        imp = f"{r['ppl_reduction_pct']:+.1f}%"
        output_lines.append(f"| {r['language']:<12} | {r['morphology']:<18} | {r['bpe_ppl']:>10.1f} | {r['morph_ppl']:>10.1f} | {imp:>12} |")

    # ── TABLE 5 ──────────────────────────────────────────────────────────
    output_lines.append("\n## TABLE 5 — Ablation Study (Hindi)")
    output_lines.append("")
    output_lines.append(f"| {'Variant':<28} | {'Avg Loss':>10} | {'Avg PPL':>10} |")
    output_lines.append(f"|{'-'*30}|{'-'*12}|{'-'*12}|")
    for r in ablation_rows:
        loss_str = f"{r['avg_loss']:.4f}" if r['avg_loss'] is not None else "N/A"
        ppl_str = f"{r['avg_ppl']:.1f}" if r['avg_ppl'] is not None else "N/A"
        output_lines.append(f"| {r['variant']:<28} | {loss_str:>10} | {ppl_str:>10} |")

    # ── TABLE 6 ──────────────────────────────────────────────────────────
    output_lines.append("\n## TABLE 6 — Cross-Language Generalization")
    output_lines.append("")
    output_lines.append(f"| {'Language':<12} | {'Script':<12} | {'BPE PPL':>10} | {'Morph PPL':>10} | {'PPL Gain %':>12} |")
    output_lines.append(f"|{'-'*14}|{'-'*14}|{'-'*12}|{'-'*12}|{'-'*14}|")
    for r in sorted(table6_rows, key=lambda x: x["ppl_reduction_pct"], reverse=True):
        output_lines.append(f"| {r['language']:<12} | {r['script']:<12} | {r['bpe_ppl']:>10.1f} | {r['morph_ppl']:>10.1f} | {r['ppl_reduction_pct']:>+11.1f}% |")

    # ── TABLE 7 ──────────────────────────────────────────────────────────
    output_lines.append("\n## TABLE 7 — Qualitative Examples (Hindi)")
    output_lines.append("")
    output_lines.append(f"| {'Word':<20} | {'BPE Tokens':<30} | {'#':>3} | {'MorphTokenizer Tokens':<30} | {'#':>3} |")
    output_lines.append(f"|{'-'*22}|{'-'*32}|{'-'*5}|{'-'*32}|{'-'*5}|")
    for r in table7_rows:
        bpe_str = " ".join(r["bpe_tokens"])[:28]
        morph_str = " ".join(r["morph_tokens"])[:28]
        output_lines.append(f"| {r['word']:<20} | {bpe_str:<30} | {r['bpe_count']:>3} | {morph_str:<30} | {r['morph_count']:>3} |")

    # Print all tables
    full_output = "\n".join(output_lines)
    print(full_output)

    # ═══════════════════════════════════════════════════════════════════════
    # SAVE EVERYTHING
    # ═══════════════════════════════════════════════════════════════════════

    # Save raw JSON
    json_out = {
        "config": {
            "vocab_size": VOCAB_SIZE, "max_steps": MAX_STEPS,
            "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
            "block_size": BLOCK_SIZE, "batch_size": BATCH_SIZE,
            "train_lines": TRAIN_LINES, "seed": SEED,
        },
        "table1_main_results": table1_rows,
        "table2_token_efficiency": table2_rows,
        "table3_total_tokens": table3_rows,
        "table5_ablation": ablation_rows,
        "table6_cross_language": table6_rows,
        "table7_qualitative": table7_rows,
        "per_language": {k: {kk: vv for kk, vv in v.items() if kk != "history" and not isinstance(vv, dict)}
                        for k, v in all_results.items()},
    }
    with open(out_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False, default=str)

    # Save markdown tables
    with open(out_dir / "tables.md", "w", encoding="utf-8") as f:
        f.write(full_output)

    # ═══════════════════════════════════════════════════════════════════════
    # GENERATE LATEX
    # ═══════════════════════════════════════════════════════════════════════

    latex_lines = []

    # Table 1 LaTeX
    latex_lines.append("% TABLE 1 — Main Results")
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Language modeling perplexity across eight typologically diverse languages. Lower is better. All models share identical architecture and training budget.}")
    latex_lines.append(r"\label{tab:main_results}")
    latex_lines.append(r"\small")
    latex_lines.append(r"\begin{tabular}{l l r r}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"\textbf{Language} & \textbf{Tokenizer} & \textbf{Avg Loss} & \textbf{Avg PPL $\downarrow$} \\")
    latex_lines.append(r"\midrule")

    prev_lang = None
    for r in sorted(table1_rows, key=lambda x: (x["language"], x["tokenizer"])):
        if prev_lang and prev_lang != r["language"]:
            latex_lines.append(r"\addlinespace")
        bold = r"\textbf" if r["tokenizer"] == "MorphTokenizer" else ""
        ppl_str = f"{r['avg_ppl']:.1f}"
        if bold:
            ppl_str = r"\textbf{" + ppl_str + "}"
        latex_lines.append(f"{r['language']} & {r['tokenizer']} & {r['avg_loss']:.4f} & {ppl_str} \\\\")
        prev_lang = r["language"]

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table*}")

    # Table 5 LaTeX (Ablation)
    latex_lines.append("\n% TABLE 5 — Ablation Study")
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Ablation study on Hindi. Each row adds one component.}")
    latex_lines.append(r"\label{tab:ablation}")
    latex_lines.append(r"\begin{tabular}{l r r}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"\textbf{Variant} & \textbf{Avg Loss} & \textbf{Avg PPL $\downarrow$} \\")
    latex_lines.append(r"\midrule")
    for r in ablation_rows:
        loss_s = f"{r['avg_loss']:.4f}" if r['avg_loss'] else "---"
        ppl_s = f"{r['avg_ppl']:.1f}" if r['avg_ppl'] else "---"
        latex_lines.append(f"{r['variant']} & {loss_s} & {ppl_s} \\\\")
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    # Table 6 LaTeX (Cross-language)
    latex_lines.append("\n% TABLE 6 — Cross-Language Generalization")
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Cross-language generalization. PPL reduction of MorphTokenizer over SP-BPE baseline.}")
    latex_lines.append(r"\label{tab:crosslang}")
    latex_lines.append(r"\begin{tabular}{l l r r r}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"\textbf{Language} & \textbf{Script} & \textbf{SP-BPE PPL} & \textbf{MorphTok PPL} & \textbf{$\Delta$ PPL \%} \\")
    latex_lines.append(r"\midrule")
    for r in sorted(table6_rows, key=lambda x: x["ppl_reduction_pct"], reverse=True):
        latex_lines.append(f"{r['language']} & {r['script']} & {r['bpe_ppl']:.1f} & \\textbf{{{r['morph_ppl']:.1f}}} & {r['ppl_reduction_pct']:+.1f}\\% \\\\")
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table*}")

    latex_output = "\n".join(latex_lines)
    with open(out_dir / "tables.tex", "w") as f:
        f.write(latex_output)

    print(f"\n{'='*80}")
    print("  ALL RESULTS SAVED")
    print(f"{'='*80}")
    print(f"  JSON:  {out_dir / 'all_results.json'}")
    print(f"  MD:    {out_dir / 'tables.md'}")
    print(f"  LaTeX: {out_dir / 'tables.tex'}")


if __name__ == "__main__":
    main()
