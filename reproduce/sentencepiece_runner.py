"""
SentencePiece Baseline Experiment
==================================
Trains SentencePiece BPE and Unigram models on the same Hindi corpus,
evaluates TPW & BPT, and saves results + LaTeX tables.

Usage:
    uv run python experiments/sentencepiece_runner.py
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import sentencepiece as spm
from rich.console import Console
from rich.table import Table as RichTable

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "datasets" / "hindi_raw.txt"
EXPERIMENTS_DIR = ROOT / "experiments"
SP_MODEL_DIR = EXPERIMENTS_DIR / "sp_models"

VOCAB_SIZES = [1000, 2000, 4000, 8000]
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


def eval_sp_model(sp_model_path: str, lines: list[str]) -> dict:
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)

    total_tokens = 0
    total_words = 0
    total_bytes = 0

    t0 = time.time()
    for line in lines:
        words = line.split()
        if not words:
            continue
        pieces = sp.Encode(line, out_type=int)
        total_tokens += len(pieces)
        total_words += len(words)
        total_bytes += len(line.encode("utf-8"))
    elapsed = time.time() - t0

    tpw = total_tokens / total_words if total_words else 0
    bpt = total_bytes / total_tokens if total_tokens else 0
    speed = total_tokens / elapsed if elapsed else 0

    return {
        "tokens_per_word": round(tpw, 3),
        "bytes_per_token": round(bpt, 3),
        "speed_tok_per_sec": round(speed, 2),
        "total_tokens": total_tokens,
        "total_words": total_words,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]SentencePiece Baseline Experiment")

    os.makedirs(SP_MODEL_DIR, exist_ok=True)
    eval_lines = load_lines(CORPUS, EVAL_LINES)
    console.print(f"Loaded {len(eval_lines)} eval lines from {CORPUS}")

    all_results = []

    for model_type in ["bpe", "unigram"]:
        for vs in VOCAB_SIZES:
            tag = f"sp_{model_type}_vs{vs}"
            prefix = str(SP_MODEL_DIR / tag)

            console.print(f"\n[yellow]Training {model_type.upper()} vocab={vs}...[/yellow]")

            spm.SentencePieceTrainer.Train(
                input=str(CORPUS),
                model_prefix=prefix,
                vocab_size=vs,
                model_type=model_type,
                character_coverage=1.0,
                normalization_rule_name="nfkc",
                num_threads=4,
                max_sentence_length=8192,
                shuffle_input_sentence=True,
                input_sentence_size=10000,
            )

            model_path = prefix + ".model"
            metrics = eval_sp_model(model_path, eval_lines)

            display_name = f"SP-{model_type.upper()}"
            result = {
                "tokenizer": display_name,
                "model_type": model_type,
                "vocab_size": vs,
                **metrics,
            }
            all_results.append(result)
            console.print(
                f"  [green]{display_name} vs={vs}: "
                f"TPW={metrics['tokens_per_word']:.3f}, "
                f"BPT={metrics['bytes_per_token']:.3f}[/green]"
            )

    # ── save JSON ────────────────────────────────────────────────────────
    out_path = EXPERIMENTS_DIR / "results_sentencepiece.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    console.print(f"\n[bold green]Results saved → {out_path}[/bold green]")

    # ── print table ──────────────────────────────────────────────────────
    tbl = RichTable(title="SentencePiece Baseline Results")
    tbl.add_column("Tokenizer", style="cyan")
    tbl.add_column("Vocab", justify="right")
    tbl.add_column("TPW ↓", justify="right")
    tbl.add_column("BPT ↑", justify="right")
    for r in all_results:
        tbl.add_row(r["tokenizer"], str(r["vocab_size"]),
                     f"{r['tokens_per_word']:.3f}", f"{r['bytes_per_token']:.3f}")
    console.print(tbl)

    # ── generate LaTeX ───────────────────────────────────────────────────
    latex = generate_latex(all_results)
    latex_path = EXPERIMENTS_DIR / "latex_sentencepiece.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    console.print(f"[bold green]LaTeX table → {latex_path}[/bold green]")
    console.print(latex)


def generate_latex(results: list[dict]) -> str:
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{SentencePiece baselines (BPE and Unigram) on the Hindi evaluation corpus.}")
    lines.append(r"\label{tab:sentencepiece}")
    lines.append(r"\begin{tabular}{llcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Tokenizer} & \textbf{Vocab Size} & \textbf{Tokens/Word $\downarrow$} & \textbf{Bytes/Token $\uparrow$} \\")
    lines.append(r"\midrule")

    for r in results:
        name = r["tokenizer"].replace("-", r"\text{-}")
        lines.append(
            f"{name} & {r['vocab_size']:,} & {r['tokens_per_word']:.3f} & {r['bytes_per_token']:.3f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
