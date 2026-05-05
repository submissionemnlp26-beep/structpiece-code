"""
Comprehensive Experiment Runner for IndicTokenizer
====================================================

Runs all experiments for the research paper:
1. Train tokenizers at multiple vocab sizes
2. Evaluate each against baselines (tiktoken, etc.)
3. Measure C++ vs Python inference speed
4. Analyze morphological coverage
5. Save all results to experiments/results.json
"""

import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from rich.console import Console
from rich.table import Table

# Add python/ to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from pretokenizer import pretokenize, grapheme_clusters, normalize
from adaptive_bpe import train_bpe, IndicTokenizer, HINDI_SUFFIXES

try:
    import tiktoken
except ImportError:
    tiktoken = None

console = Console()

# ─── Configuration ───────────────────────────────────────────────────────────

CORPUS_PATH = "datasets/hindi_raw.txt"
EXPERIMENTS_DIR = "experiments"
MODELS_BASE_DIR = "models"
CPP_TOKENIZER_BIN = "cpp/build/tokenizer_test"

VOCAB_SIZES = [1000, 2000, 4000, 8000]
TRAIN_MAX_LINES = 10000        # lines used for training (each experiment)
EVAL_LINES = 5000              # lines used for evaluation
SPEED_BENCH_LINES = 1000       # lines for speed benchmarking
SPEED_BENCH_REPEATS = 3        # repeat speed benchmarks for stability

# Curated test sentences for qualitative analysis
QUALITATIVE_SENTENCES = [
    "भारत एक विविधताओं से भरा देश है।",
    "मशीन लर्निंग ने प्राकृतिक भाषा प्रसंस्करण में क्रांति ला दी है।",
    "राजनीतिक उत्तराधिकारी की घोषणा ने पार्टी में हलचल मचा दी।",
    "किसानों की समस्याओं का समाधान सरकार की प्राथमिकता होनी चाहिए।",
    "विज्ञान और प्रौद्योगिकी के क्षेत्र में भारत ने उल्लेखनीय प्रगति की है।",
    "नमस्कार, आप कैसे हैं?",
    "123 रुपये का सामान ₹100 में मिल गया।",
    "हिन्दी (Hindi) विश्व की चौथी सबसे अधिक बोली जाने वाली भाषा है।",
]


# ─── Helper Functions ────────────────────────────────────────────────────────

def load_corpus_lines(path: str, n: int) -> List[str]:
    """Load first n non-empty lines from corpus."""
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
                if len(lines) >= n:
                    break
    return lines


def count_words(lines: List[str]) -> int:
    """Count total whitespace-separated words."""
    return sum(len(line.split()) for line in lines)


def tokenize_lines_timed(tokenize_fn, lines: List[str], repeats: int = 1) -> Tuple[int, float]:
    """Tokenize lines and return (total_tokens, avg_seconds)."""
    total_tokens = 0
    total_time = 0.0
    for _ in range(repeats):
        t0 = time.time()
        tokens_count = 0
        for line in lines:
            tokens = tokenize_fn(line)
            tokens_count += len(tokens)
        t1 = time.time()
        total_time += (t1 - t0)
        total_tokens = tokens_count  # same each repeat
    avg_time = total_time / repeats
    return total_tokens, avg_time


# ─── Experiment 1: Multi-Vocab-Size Training & Evaluation ────────────────────

def run_vocab_size_experiments(corpus_path: str, eval_lines: List[str]) -> List[Dict[str, Any]]:
    """Train tokenizers at different vocab sizes and evaluate each."""
    console.rule("[bold cyan]Experiment 1: Vocab Size Sweep")
    results = []

    for vs in VOCAB_SIZES:
        console.print(f"\n[yellow]Training vocab_size={vs}...[/yellow]")
        model_dir = os.path.join(MODELS_BASE_DIR, f"vs_{vs}")

        vocab, merges = train_bpe(
            corpus_path=corpus_path,
            vocab_size=vs,
            max_lines=TRAIN_MAX_LINES,
            verbose=True,
        )
        tok = IndicTokenizer(vocab, merges)
        tok.save(model_dir)

        # Evaluate
        total_tokens = 0
        total_words = 0
        total_bytes = 0
        t0 = time.time()
        for line in eval_lines:
            words = line.split()
            if not words:
                continue
            ids = tok.encode(line)
            total_tokens += len(ids)
            total_words += len(words)
            total_bytes += len(line.encode('utf-8'))
        duration = time.time() - t0

        tpw = total_tokens / total_words if total_words else 0
        bpt = total_bytes / total_tokens if total_tokens else 0
        speed = total_tokens / duration if duration else 0

        result = {
            "vocab_size": vs,
            "actual_vocab_size": len(vocab),
            "num_merges": len(merges),
            "tokens_per_word": round(tpw, 3),
            "bytes_per_token": round(bpt, 3),
            "speed_tok_per_sec": round(speed, 2),
            "total_tokens": total_tokens,
            "total_words": total_words,
        }
        results.append(result)
        console.print(f"  [green]vs={vs}: {tpw:.3f} tok/word, {bpt:.3f} bytes/tok[/green]")

    return results


# ─── Experiment 2: Baseline Comparison ───────────────────────────────────────

def run_baseline_comparison(eval_lines: List[str], best_model_dir: str) -> List[Dict[str, Any]]:
    """Compare IndicTokenizer against tiktoken baseline."""
    console.rule("[bold cyan]Experiment 2: Baseline Comparison")
    results = []

    # Load best IndicTokenizer
    indic_tok = IndicTokenizer.load(best_model_dir)

    def eval_tok(name, encode_fn):
        total_tokens = 0
        total_words = 0
        total_bytes = 0
        t0 = time.time()
        for line in eval_lines:
            words = line.split()
            if not words:
                continue
            ids = encode_fn(line)
            total_tokens += len(ids)
            total_words += len(words)
            total_bytes += len(line.encode('utf-8'))
        duration = time.time() - t0
        return {
            "tokenizer": name,
            "tokens_per_word": round(total_tokens / total_words, 3) if total_words else 0,
            "bytes_per_token": round(total_bytes / total_tokens, 3) if total_tokens else 0,
            "speed_tok_per_sec": round(total_tokens / duration, 2) if duration else 0,
            "total_tokens": total_tokens,
            "total_words": total_words,
        }

    # IndicTokenizer
    r = eval_tok("IndicTokenizer", indic_tok.encode)
    results.append(r)
    console.print(f"  [green]IndicTokenizer: {r['tokens_per_word']} tok/word[/green]")

    # tiktoken
    if tiktoken:
        enc = tiktoken.get_encoding("cl100k_base")
        r = eval_tok("tiktoken (cl100k_base)", enc.encode)
        results.append(r)
        console.print(f"  [green]tiktoken: {r['tokens_per_word']} tok/word[/green]")

        # o200k_base (GPT-4o)
        try:
            enc2 = tiktoken.get_encoding("o200k_base")
            r = eval_tok("tiktoken (o200k_base)", enc2.encode)
            results.append(r)
            console.print(f"  [green]tiktoken o200k: {r['tokens_per_word']} tok/word[/green]")
        except Exception:
            pass

    return results


# ─── Experiment 3: C++ vs Python Speed Benchmark ─────────────────────────────

def run_speed_benchmark(bench_lines: List[str], best_model_dir: str) -> Dict[str, Any]:
    """Benchmark Python vs C++ tokenizer inference speed."""
    console.rule("[bold cyan]Experiment 3: C++ vs Python Speed")

    results = {"python": {}, "cpp": {}}

    # Python benchmark
    indic_tok = IndicTokenizer.load(best_model_dir)
    total_tokens, avg_time = tokenize_lines_timed(
        indic_tok.encode, bench_lines, repeats=SPEED_BENCH_REPEATS
    )
    py_tps = total_tokens / avg_time if avg_time > 0 else 0
    results["python"] = {
        "total_tokens": total_tokens,
        "avg_time_sec": round(avg_time, 4),
        "tokens_per_sec": round(py_tps, 2),
        "lines": len(bench_lines),
    }
    console.print(f"  [green]Python: {py_tps:,.0f} tok/sec ({avg_time:.4f}s)[/green]")

    # C++ benchmark
    cpp_bin = os.path.abspath(CPP_TOKENIZER_BIN)
    cpp_model_dir = os.path.abspath(best_model_dir)

    if os.path.exists(cpp_bin):
        # Write bench lines to a temp file
        bench_file = "/tmp/indic_bench_input.txt"
        with open(bench_file, 'w', encoding='utf-8') as f:
            for line in bench_lines:
                f.write(line + '\n')

        # Build a small C++ benchmark wrapper
        cpp_bench_src = os.path.join(os.path.dirname(cpp_bin), "..", "speed_bench.cpp")
        cpp_bench_bin = os.path.join(os.path.dirname(cpp_bin), "speed_bench")

        # Write the benchmark source
        with open(cpp_bench_src, 'w') as f:
            f.write('''
#include "tokenizer_engine.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: speed_bench <model_dir> <input_file> [repeats]\\n";
        return 1;
    }
    std::string model_dir = argv[1];
    std::string input_file = argv[2];
    int repeats = argc > 3 ? std::atoi(argv[3]) : 3;

    indic::TokenizerEngine engine;
    if (!engine.load(model_dir)) {
        std::cerr << "Failed to load model\\n";
        return 1;
    }

    // Load lines
    std::vector<std::string> lines;
    std::ifstream ifs(input_file);
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty()) lines.push_back(line);
    }

    long total_tokens = 0;
    double total_ms = 0;

    for (int r = 0; r < repeats; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        long tokens = 0;
        for (const auto& l : lines) {
            auto ids = engine.encode(l);
            tokens += ids.size();
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        total_ms += ms;
        total_tokens = tokens;
    }

    double avg_ms = total_ms / repeats;
    double tps = (total_tokens / avg_ms) * 1000.0;

    // Output as JSON
    std::cout << "{"
              << "\\"total_tokens\\":" << total_tokens << ","
              << "\\"avg_time_ms\\":" << avg_ms << ","
              << "\\"tokens_per_sec\\":" << (long)tps << ","
              << "\\"lines\\":" << lines.size() << ","
              << "\\"repeats\\":" << repeats
              << "}\\n";
    return 0;
}
''')

        # Compile
        compile_cmd = (
            f"c++ -std=c++17 -O2 -o {cpp_bench_bin} {cpp_bench_src} "
            f"{os.path.dirname(cpp_bench_src)}/unicode_utils.cpp "
            f"{os.path.dirname(cpp_bench_src)}/tokenizer_engine.cpp "
            f"-I{os.path.dirname(cpp_bench_src)}"
        )
        ret = os.system(compile_cmd + " 2>/dev/null")

        if ret == 0 and os.path.exists(cpp_bench_bin):
            # Run benchmark
            cmd = f"{cpp_bench_bin} {cpp_model_dir} {bench_file} {SPEED_BENCH_REPEATS}"
            try:
                output = subprocess.check_output(cmd, shell=True, text=True, timeout=60)
                # Parse JSON output (skip the "Loaded vocab..." line)
                for line in output.strip().split('\n'):
                    if line.startswith('{'):
                        cpp_result = json.loads(line)
                        results["cpp"] = {
                            "total_tokens": cpp_result["total_tokens"],
                            "avg_time_sec": round(cpp_result["avg_time_ms"] / 1000, 4),
                            "tokens_per_sec": cpp_result["tokens_per_sec"],
                            "lines": cpp_result["lines"],
                        }
                        console.print(f"  [green]C++: {cpp_result['tokens_per_sec']:,} tok/sec "
                                      f"({cpp_result['avg_time_ms']:.1f}ms)[/green]")
                        break
            except Exception as e:
                console.print(f"  [red]C++ benchmark error: {e}[/red]")
        else:
            console.print(f"  [red]C++ compile failed[/red]")
    else:
        console.print(f"  [yellow]C++ binary not found at {cpp_bin}[/yellow]")

    # Compute speedup
    if results["python"].get("tokens_per_sec") and results["cpp"].get("tokens_per_sec"):
        speedup = results["cpp"]["tokens_per_sec"] / results["python"]["tokens_per_sec"]
        results["speedup"] = round(speedup, 2)
        console.print(f"  [bold magenta]C++ speedup: {speedup:.2f}x[/bold magenta]")

    return results


# ─── Experiment 4: Morphological Analysis ────────────────────────────────────

def run_morphological_analysis(eval_lines: List[str], best_model_dir: str) -> Dict[str, Any]:
    """Analyze how well the tokenizer captures Hindi morphology."""
    console.rule("[bold cyan]Experiment 4: Morphological Analysis")

    tok = IndicTokenizer.load(best_model_dir)
    vocab = tok.vocab

    # Count suffix tokens in vocabulary
    suffix_tokens = {}
    for token_str in vocab:
        for suf in HINDI_SUFFIXES:
            if token_str.endswith(suf) and len(token_str) >= len(suf):
                if suf not in suffix_tokens:
                    suffix_tokens[suf] = []
                suffix_tokens[suf].append(token_str)

    # Check which suffixes are standalone vocab items
    standalone_suffixes = [s for s in HINDI_SUFFIXES if s in vocab]

    # Token length distribution
    token_lengths = [len(t) for t in vocab if t not in ("<unk>", "<pad>", " ")]
    grapheme_lengths = []
    for t in vocab:
        if t not in ("<unk>", "<pad>", " "):
            grapheme_lengths.append(len(grapheme_clusters(t)))

    # Analyze tokenization of sample words with known morphology
    morph_test_words = {
        "खेलना": "खेल + ना (to play)",
        "चलती": "चल + ती (walks, fem.)",
        "लड़कों": "लड़क + ों (boys, oblique)",
        "सुन्दरता": "सुन्दर + ता (beauty)",
        "राजनीतिक": "राजनीति + क (political)",
        "पढ़ाई": "पढ़ + ाई (study)",
        "दुकानदार": "दुकान + दार (shopkeeper)",
        "अध्यापकों": "अध्यापक + ों (teachers)",
        "वाहनवाला": "वाहन + वाला (vehicle person)",
    }

    morph_results = {}
    for word, description in morph_test_words.items():
        ids = tok.encode(word)
        tokens_decoded = [tok.id_to_token.get(i, "?") for i in ids]
        morph_results[word] = {
            "description": description,
            "tokens": tokens_decoded,
            "num_tokens": len(ids),
        }
        console.print(f"  {word:20s} → {tokens_decoded} ({description})")

    return {
        "standalone_suffixes": standalone_suffixes,
        "suffix_coverage": len(standalone_suffixes) / len(HINDI_SUFFIXES),
        "total_suffixes_defined": len(HINDI_SUFFIXES),
        "avg_token_chars": round(np.mean(token_lengths), 2) if token_lengths else 0,
        "avg_token_graphemes": round(np.mean(grapheme_lengths), 2) if grapheme_lengths else 0,
        "max_token_chars": max(token_lengths) if token_lengths else 0,
        "morphological_words": morph_results,
        "token_length_distribution": {
            "mean": round(np.mean(token_lengths), 2),
            "median": round(float(np.median(token_lengths)), 2),
            "std": round(float(np.std(token_lengths)), 2),
        },
    }


# ─── Experiment 5: Qualitative Analysis ──────────────────────────────────────

def run_qualitative_analysis(best_model_dir: str) -> List[Dict[str, Any]]:
    """Show side-by-side tokenization of sample sentences."""
    console.rule("[bold cyan]Experiment 5: Qualitative Analysis")

    tok = IndicTokenizer.load(best_model_dir)
    results = []

    enc = tiktoken.get_encoding("cl100k_base") if tiktoken else None

    for sent in QUALITATIVE_SENTENCES:
        entry = {"sentence": sent}

        # IndicTokenizer
        ids = tok.encode(sent)
        tokens = [tok.id_to_token.get(i, "?") for i in ids]
        entry["indic_tokens"] = tokens
        entry["indic_count"] = len(ids)

        # tiktoken
        if enc:
            tik_ids = enc.encode(sent)
            tik_tokens = [enc.decode([i]) for i in tik_ids]
            entry["tiktoken_tokens"] = tik_tokens
            entry["tiktoken_count"] = len(tik_ids)

        results.append(entry)
        console.print(f"  [dim]{sent[:60]}...[/dim]")
        console.print(f"    Indic : {len(ids)} tokens")
        if enc:
            console.print(f"    tiktoken: {len(tik_ids)} tokens")

    return results


# ─── Experiment 6: Corpus Statistics ─────────────────────────────────────────

def compute_corpus_stats(corpus_path: str, max_lines: int = 50000) -> Dict[str, Any]:
    """Compute statistics about the corpus."""
    console.rule("[bold cyan]Experiment 6: Corpus Statistics")

    lines = load_corpus_lines(corpus_path, max_lines)
    total_chars = sum(len(l) for l in lines)
    total_words = count_words(lines)
    total_bytes = sum(len(l.encode('utf-8')) for l in lines)

    # Character distribution
    from pretokenizer import is_devanagari as is_deva_char
    deva_chars = sum(1 for l in lines for c in l if is_deva_char(c))

    # Word length distribution
    word_lengths = [len(w) for l in lines for w in l.split()]
    avg_word_len = np.mean(word_lengths) if word_lengths else 0

    stats = {
        "total_lines": len(lines),
        "total_characters": total_chars,
        "total_words": total_words,
        "total_bytes": total_bytes,
        "avg_line_length_chars": round(total_chars / len(lines), 1),
        "avg_word_length_chars": round(float(avg_word_len), 2),
        "devanagari_char_ratio": round(deva_chars / total_chars, 4) if total_chars else 0,
        "bytes_per_char": round(total_bytes / total_chars, 3) if total_chars else 0,
    }

    console.print(f"  Lines: {stats['total_lines']:,}")
    console.print(f"  Words: {stats['total_words']:,}")
    console.print(f"  Chars: {stats['total_characters']:,}")
    console.print(f"  Devanagari ratio: {stats['devanagari_char_ratio']:.2%}")

    return stats


# ─── Generate Charts ─────────────────────────────────────────────────────────

def generate_charts(all_results: Dict[str, Any], output_dir: str):
    """Generate publication-quality charts from experiment results."""
    console.rule("[bold cyan]Generating Charts")

    os.makedirs(output_dir, exist_ok=True)

    # Set style
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    # ── Chart 1: Tokens/Word vs Vocab Size ──────────────────────────────
    vs_results = all_results.get("vocab_size_sweep", [])
    if vs_results:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 4.5))

        sizes = [r["vocab_size"] for r in vs_results]
        tpw = [r["tokens_per_word"] for r in vs_results]
        bpt = [r["bytes_per_token"] for r in vs_results]

        color1 = '#2196F3'
        color2 = '#FF5722'

        ax1.plot(sizes, tpw, 'o-', color=color1, linewidth=2, markersize=8, label='Tokens/Word')
        ax1.set_xlabel('Vocabulary Size')
        ax1.set_ylabel('Tokens per Word', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        ax2.plot(sizes, bpt, 's--', color=color2, linewidth=2, markersize=8, label='Bytes/Token')
        ax2.set_ylabel('Bytes per Token', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        # Add tiktoken baseline
        baseline = all_results.get("baseline_comparison", [])
        for b in baseline:
            if "tiktoken" in b.get("tokenizer", "").lower() and "cl100k" in b.get("tokenizer", "").lower():
                ax1.axhline(y=b["tokens_per_word"], color=color1, linestyle=':', alpha=0.5,
                            label=f'tiktoken ({b["tokens_per_word"]})')
                break

        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=9)
        ax1.set_title('Effect of Vocabulary Size on Tokenization Efficiency')
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join(output_dir, 'vocab_size_sweep.pdf'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(output_dir, 'vocab_size_sweep.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        console.print("  [green]✓ vocab_size_sweep.pdf[/green]")

    # ── Chart 2: Baseline Comparison Bar Chart ──────────────────────────
    baseline = all_results.get("baseline_comparison", [])
    if baseline:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        names = [b["tokenizer"] for b in baseline]
        tpw = [b["tokens_per_word"] for b in baseline]
        bpt = [b["bytes_per_token"] for b in baseline]

        colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']

        # Tokens/Word
        bars1 = axes[0].barh(names, tpw, color=colors[:len(names)], edgecolor='white', height=0.5)
        axes[0].set_xlabel('Tokens per Word (↓ better)')
        axes[0].set_title('Token Efficiency')
        for bar, val in zip(bars1, tpw):
            axes[0].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                         f'{val:.2f}', va='center', fontsize=10, fontweight='bold')

        # Bytes/Token
        bars2 = axes[1].barh(names, bpt, color=colors[:len(names)], edgecolor='white', height=0.5)
        axes[1].set_xlabel('Bytes per Token (↑ better)')
        axes[1].set_title('Compression Ratio')
        for bar, val in zip(bars2, bpt):
            axes[1].text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                         f'{val:.2f}', va='center', fontsize=10, fontweight='bold')

        fig.suptitle('IndicTokenizer vs Baselines', fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'baseline_comparison.pdf'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(output_dir, 'baseline_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        console.print("  [green]✓ baseline_comparison.pdf[/green]")

    # ── Chart 3: C++ vs Python Speed ────────────────────────────────────
    speed = all_results.get("speed_benchmark", {})
    if speed.get("python") and speed.get("cpp"):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        langs = ['Python', 'C++']
        speeds = [speed["python"]["tokens_per_sec"], speed["cpp"]["tokens_per_sec"]]
        colors = ['#FF9800', '#4CAF50']

        bars = ax.bar(langs, speeds, color=colors, edgecolor='white', width=0.5)
        for bar, val in zip(bars, speeds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speeds)*0.02,
                    f'{val:,.0f}', ha='center', fontsize=11, fontweight='bold')

        speedup = speed.get("speedup", 0)
        ax.set_ylabel('Tokens per Second')
        ax.set_title(f'Inference Speed: C++ is {speedup:.1f}× faster than Python')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'speed_comparison.pdf'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(output_dir, 'speed_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        console.print("  [green]✓ speed_comparison.pdf[/green]")

    console.print("[bold green]All charts saved.[/bold green]")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold magenta]IndicTokenizer — Full Experiment Suite", style="bold magenta")
    console.print(f"Corpus: {CORPUS_PATH}")
    console.print(f"Vocab sizes to test: {VOCAB_SIZES}")
    console.print(f"Train lines: {TRAIN_MAX_LINES}, Eval lines: {EVAL_LINES}")

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    # Load evaluation data
    eval_lines = load_corpus_lines(CORPUS_PATH, EVAL_LINES)
    bench_lines = load_corpus_lines(CORPUS_PATH, SPEED_BENCH_LINES)
    console.print(f"Loaded {len(eval_lines)} eval lines, {len(bench_lines)} bench lines")

    all_results: Dict[str, Any] = {}

    # Corpus stats
    all_results["corpus_stats"] = compute_corpus_stats(CORPUS_PATH)

    # Experiment 1: Vocab size sweep
    all_results["vocab_size_sweep"] = run_vocab_size_experiments(CORPUS_PATH, eval_lines)

    # Use best (largest) vocab for remaining experiments
    best_model_dir = os.path.join(MODELS_BASE_DIR, f"vs_{VOCAB_SIZES[-1]}")

    # Experiment 2: Baseline comparison
    all_results["baseline_comparison"] = run_baseline_comparison(eval_lines, best_model_dir)

    # Experiment 3: Speed benchmark
    all_results["speed_benchmark"] = run_speed_benchmark(bench_lines, best_model_dir)

    # Experiment 4: Morphological analysis
    all_results["morphological_analysis"] = run_morphological_analysis(eval_lines, best_model_dir)

    # Experiment 5: Qualitative analysis
    all_results["qualitative_analysis"] = run_qualitative_analysis(best_model_dir)

    # Save results
    results_path = os.path.join(EXPERIMENTS_DIR, "results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    console.print(f"\n[bold green]All results saved to {results_path}[/bold green]")

    # Generate charts
    charts_dir = os.path.join(EXPERIMENTS_DIR, "figures")
    generate_charts(all_results, charts_dir)

    console.rule("[bold green]All Experiments Complete!", style="bold green")


if __name__ == "__main__":
    main()
