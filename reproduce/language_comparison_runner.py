"""
English vs Hindi Tokenization Comparison
==========================================
Downloads a small English sample (from WikiText-2 raw) and compares
TPW using tiktoken (cl100k_base) on English vs Hindi.
This quantifies the tokenization "fairness gap".

Usage:
    uv run python experiments/language_comparison_runner.py
"""

import json
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table as RichTable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

CORPUS_HI = ROOT / "datasets" / "hindi_raw.txt"
EXPERIMENTS_DIR = ROOT / "experiments"
EVAL_LINES = 5000

console = Console()

# ── try imports ──────────────────────────────────────────────────────────────
try:
    import tiktoken
except ImportError:
    console.print("[red]tiktoken not installed – aborting.[/red]")
    sys.exit(1)

from adaptive_bpe import IndicTokenizer


# ── helpers ──────────────────────────────────────────────────────────────────

def load_lines(path: Path, n: int) -> list[str]:
    lines: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
                if len(lines) >= n:
                    break
    return lines


def download_english_sample(n: int = 5000) -> list[str]:
    """Return n non-empty lines of English text (WikiText-2-raw)."""
    cache_path = EXPERIMENTS_DIR / "english_sample.txt"
    if cache_path.exists():
        return load_lines(cache_path, n)

    console.print("[yellow]Downloading WikiText-2-raw-v1 via HuggingFace datasets...[/yellow]")
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    lines = []
    for row in ds:
        text = row["text"].strip()
        if len(text) >= 15:
            lines.append(text)
            if len(lines) >= n:
                break

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    console.print(f"  Cached {len(lines)} English lines → {cache_path}")
    return lines


def eval_tokenizer(name: str, encode_fn, lines: list[str]) -> dict:
    total_tok = 0
    total_words = 0
    total_bytes = 0
    t0 = time.time()
    for line in lines:
        words = line.split()
        if not words:
            continue
        ids = encode_fn(line)
        total_tok += len(ids)
        total_words += len(words)
        total_bytes += len(line.encode("utf-8"))
    elapsed = time.time() - t0

    tpw = total_tok / total_words if total_words else 0
    bpt = total_bytes / total_tok if total_tok else 0
    return {
        "tokenizer": name,
        "tokens_per_word": round(tpw, 3),
        "bytes_per_token": round(bpt, 3),
        "total_tokens": total_tok,
        "total_words": total_words,
        "speed_tok_per_sec": round(total_tok / elapsed, 2) if elapsed else 0,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]English vs Hindi Tokenization Comparison")

    hi_lines = load_lines(CORPUS_HI, EVAL_LINES)
    en_lines = download_english_sample(EVAL_LINES)
    console.print(f"Hindi lines: {len(hi_lines)}, English lines: {len(en_lines)}")

    enc = tiktoken.get_encoding("cl100k_base")

    results = []

    # tiktoken on English
    r = eval_tokenizer("tiktoken (English)", enc.encode, en_lines)
    r["language"] = "English"
    results.append(r)
    console.print(f"  [green]tiktoken English: TPW={r['tokens_per_word']:.3f}[/green]")

    # tiktoken on Hindi
    r = eval_tokenizer("tiktoken (Hindi)", enc.encode, hi_lines)
    r["language"] = "Hindi"
    results.append(r)
    console.print(f"  [green]tiktoken Hindi: TPW={r['tokens_per_word']:.3f}[/green]")

    # IndicTokenizer on Hindi
    best_model = ROOT / "models" / "vs_8000"
    if (best_model / "indic_tokenizer_vocab.json").exists():
        indic_tok = IndicTokenizer.load(str(best_model))
        r = eval_tokenizer("IndicTokenizer (Hindi)", indic_tok.encode, hi_lines)
        r["language"] = "Hindi"
        results.append(r)
        console.print(f"  [green]IndicTokenizer Hindi: TPW={r['tokens_per_word']:.3f}[/green]")
    else:
        console.print("[yellow]IndicTokenizer model not found – skipping.[/yellow]")

    # Compute fairness gap
    en_tpw = results[0]["tokens_per_word"]
    hi_tpw = results[1]["tokens_per_word"]
    gap = hi_tpw / en_tpw
    console.print(f"\n[bold magenta]Tokenization fairness gap: Hindi needs {gap:.2f}× more tokens than English with tiktoken[/bold magenta]")

    # ── save ─────────────────────────────────────────────────────────────
    out_path = EXPERIMENTS_DIR / "results_language_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "fairness_gap_ratio": round(gap, 3)}, f, ensure_ascii=False, indent=2)
    console.print(f"[bold green]Results saved → {out_path}[/bold green]")

    # ── rich table ───────────────────────────────────────────────────────
    tbl = RichTable(title="English vs Hindi Tokenization")
    tbl.add_column("Tokenizer", style="cyan")
    tbl.add_column("Language")
    tbl.add_column("TPW ↓", justify="right")
    tbl.add_column("BPT ↑", justify="right")
    for r in results:
        tbl.add_row(r["tokenizer"], r["language"],
                     f"{r['tokens_per_word']:.3f}", f"{r['bytes_per_token']:.3f}")
    console.print(tbl)

    # ── LaTeX ────────────────────────────────────────────────────────────
    latex = generate_latex(results, gap)
    latex_path = EXPERIMENTS_DIR / "latex_language_comparison.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    console.print(f"[bold green]LaTeX table → {latex_path}[/bold green]")
    console.print(latex)


def generate_latex(results: list[dict], gap: float) -> str:
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cross-language tokenization efficiency comparison. "
                 f"Hindi requires {gap:.1f}$\\times$ more tokens than English with the same tiktoken tokenizer.}}"[:-1])
    lines.append(r"\label{tab:lang_compare}")
    lines.append(r"\begin{tabular}{llcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Tokenizer} & \textbf{Language} & \textbf{Tokens/Word $\downarrow$} & \textbf{Bytes/Token $\uparrow$} \\")
    lines.append(r"\midrule")
    for r in results:
        name = r["tokenizer"]
        lines.append(f"{name} & {r['language']} & {r['tokens_per_word']:.3f} & {r['bytes_per_token']:.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
