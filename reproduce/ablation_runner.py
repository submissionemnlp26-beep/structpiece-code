"""
Ablation Study Runner
======================
Trains IndicTokenizer variants with components selectively disabled
and measures the impact on TPW, BPT, and suffix isolation.

Variants:
    1. Full model (morphology bonus + script penalty + script-aware pretokenizer)
    2. No morphology bonus
    3. No script penalty
    4. Pure BPE (neither bonus nor penalty)

Usage:
    uv run python experiments/ablation_runner.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

from rich.console import Console
from rich.table import Table as RichTable
from tqdm import tqdm

# ── make python/ importable ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from pretokenizer import pretokenize, normalize
from adaptive_bpe import (
    HINDI_SUFFIXES,
    IndicTokenizer,
    _build_word_freqs,
    _count_pairs,
    _merge_pair,
    morphology_bonus as _orig_morph_bonus,
    script_penalty as _orig_script_penalty,
)

CORPUS = ROOT / "datasets" / "hindi_raw.txt"
EXPERIMENTS_DIR = ROOT / "experiments"
ABLATION_DIR = EXPERIMENTS_DIR / "ablation_models"

VOCAB_SIZE = 4000          # fixed for ablation
TRAIN_MAX_LINES = 10000
EVAL_LINES = 5000

console = Console()


# ── helpers ──────────────────────────────────────────────────────────────────

def load_lines(path: Path, n: int) -> list[str]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
                if len(lines) >= n:
                    break
    return lines


WordRepr = Tuple[str, ...]
WordFreqs = Dict[WordRepr, int]


def train_ablation(
    corpus_path: str,
    vocab_size: int,
    use_morph_bonus: bool,
    use_script_penalty: bool,
    max_lines: int | None = None,
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """Modified BPE training with toggleable components."""
    word_freqs = _build_word_freqs(corpus_path, max_lines=max_lines)

    base_vocab: set[str] = set()
    for word in word_freqs:
        for ch in word:
            base_vocab.add(ch)

    vocab_list = sorted(base_vocab)
    vocab: Dict[str, int] = {"<unk>": 0, "<pad>": 1, " ": 2}
    for i, tok in enumerate(vocab_list, start=3):
        vocab[tok] = i

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []

    merges: List[Tuple[str, str]] = []
    for step in tqdm(range(num_merges), desc="BPE merges", leave=False):
        pair_counts = _count_pairs(word_freqs)
        if not pair_counts:
            break

        best_pair = None
        best_score = float("-inf")

        for pair, freq in pair_counts.items():
            score = float(freq)
            merged = pair[0] + pair[1]
            if use_morph_bonus:
                score += _orig_morph_bonus(merged)
            if use_script_penalty:
                score += _orig_script_penalty(pair[0], pair[1])
            if score > best_score:
                best_score = score
                best_pair = pair

        if best_pair is None:
            break

        merges.append(best_pair)
        merged_token = best_pair[0] + best_pair[1]
        vocab[merged_token] = len(vocab)

        new_wf: WordFreqs = {}
        for word, freq in word_freqs.items():
            new_word = _merge_pair(word, best_pair)
            new_wf[new_word] = new_wf.get(new_word, 0) + freq
        word_freqs = new_wf

    return vocab, merges


def evaluate(tok: IndicTokenizer, lines: list[str]) -> dict:
    total_tokens = 0
    total_words = 0
    total_bytes = 0
    for line in lines:
        words = line.split()
        if not words:
            continue
        ids = tok.encode(line)
        total_tokens += len(ids)
        total_words += len(words)
        total_bytes += len(line.encode("utf-8"))

    tpw = total_tokens / total_words if total_words else 0
    bpt = total_bytes / total_tokens if total_tokens else 0
    return {
        "tokens_per_word": round(tpw, 3),
        "bytes_per_token": round(bpt, 3),
    }


def suffix_isolation_pct(vocab: Dict[str, int]) -> float:
    """% of HINDI_SUFFIXES that appear as standalone vocab items."""
    found = sum(1 for s in HINDI_SUFFIXES if s in vocab)
    return round(found / len(HINDI_SUFFIXES) * 100, 1)


# ── main ─────────────────────────────────────────────────────────────────────

VARIANTS = [
    ("Full model",              True,  True),
    ("No morphology bonus",     False, True),
    ("No script penalty",       True,  False),
    ("Pure BPE (neither)",      False, False),
]


def main():
    console.rule("[bold cyan]Ablation Study")
    os.makedirs(ABLATION_DIR, exist_ok=True)
    eval_lines = load_lines(CORPUS, EVAL_LINES)
    console.print(f"Eval lines: {len(eval_lines)}, vocab target: {VOCAB_SIZE}")

    all_results = []
    full_tpw = None

    for label, morph, penalty in VARIANTS:
        console.print(f"\n[yellow]Training: {label}[/yellow]")
        tag = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_dir = ABLATION_DIR / tag

        vocab, merges = train_ablation(
            corpus_path=str(CORPUS),
            vocab_size=VOCAB_SIZE,
            use_morph_bonus=morph,
            use_script_penalty=penalty,
            max_lines=TRAIN_MAX_LINES,
        )

        tok = IndicTokenizer(vocab, merges)
        tok.save(str(model_dir))

        metrics = evaluate(tok, eval_lines)
        suf_pct = suffix_isolation_pct(vocab)

        if full_tpw is None:
            full_tpw = metrics["tokens_per_word"]

        delta_tpw = metrics["tokens_per_word"] - full_tpw

        result = {
            "variant": label,
            "morph_bonus": morph,
            "script_penalty": penalty,
            "vocab_size": VOCAB_SIZE,
            **metrics,
            "suffix_isolation_pct": suf_pct,
            "delta_tpw": round(delta_tpw, 3),
        }
        all_results.append(result)
        console.print(
            f"  [green]{label}: TPW={metrics['tokens_per_word']:.3f}, "
            f"BPT={metrics['bytes_per_token']:.3f}, "
            f"Suffix={suf_pct}%[/green]"
        )

    # ── save JSON ────────────────────────────────────────────────────────
    out_path = EXPERIMENTS_DIR / "results_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    console.print(f"\n[bold green]Results saved → {out_path}[/bold green]")

    # ── rich table ───────────────────────────────────────────────────────
    tbl = RichTable(title=f"Ablation Study (vocab={VOCAB_SIZE})")
    tbl.add_column("Variant", style="cyan")
    tbl.add_column("TPW ↓", justify="right")
    tbl.add_column("BPT ↑", justify="right")
    tbl.add_column("Suffix %", justify="right")
    tbl.add_column("ΔTPW", justify="right")
    for r in all_results:
        tbl.add_row(
            r["variant"],
            f"{r['tokens_per_word']:.3f}",
            f"{r['bytes_per_token']:.3f}",
            f"{r['suffix_isolation_pct']}",
            f"{r['delta_tpw']:+.3f}",
        )
    console.print(tbl)

    # ── LaTeX ────────────────────────────────────────────────────────────
    latex = generate_latex(all_results)
    latex_path = EXPERIMENTS_DIR / "latex_ablation.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    console.print(f"[bold green]LaTeX table → {latex_path}[/bold green]")
    console.print(latex)


def generate_latex(results: list[dict]) -> str:
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study isolating the effect of each Adaptive BPE component (vocab = 4,000).}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Configuration} & \textbf{Tokens/Word $\downarrow$} & \textbf{Bytes/Token $\uparrow$} "
        r"& \textbf{Suffix Isolation (\%)} & \textbf{$\Delta$ TPW} \\"
    )
    lines.append(r"\midrule")

    for r in results:
        name = r["variant"]
        bold = r["variant"] == "Full model"
        tpw_cell = f"\\textbf{{{r['tokens_per_word']:.3f}}}" if bold else f"{r['tokens_per_word']:.3f}"
        bpt_cell = f"\\textbf{{{r['bytes_per_token']:.3f}}}" if bold else f"{r['bytes_per_token']:.3f}"
        suf_cell = f"\\textbf{{{r['suffix_isolation_pct']}}}" if bold else f"{r['suffix_isolation_pct']}"
        delta_cell = "---" if bold else f"{r['delta_tpw']:+.3f}"
        lines.append(f"{name} & {tpw_cell} & {bpt_cell} & {suf_cell} & {delta_cell} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
