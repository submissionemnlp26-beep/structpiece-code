"""
Evaluation Utilities for IndicTokenizer
Compares custom IndicTokenizer against standard tokenizers like tiktoken,
SentencePiece, and HuggingFace BPE.
"""

import time
import argparse
import os
from typing import Dict, Any, List

# ── Soft dependencies for baselines ──────────────────────────────────────────

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    import sentencepiece as spm
except ImportError:
    spm = None

try:
    from transformers import PreTrainedTokenizerFast
except ImportError:
    PreTrainedTokenizerFast = None

# ── Custom IndicTokenizer ────────────────────────────────────────────────────

try:
    from adaptive_bpe import IndicTokenizer
except ImportError:
    IndicTokenizer = None


# ─── Data loading ────────────────────────────────────────────────────────────

def load_test_data(file_path: str, num_lines: int = 1000) -> List[str]:
    """Load a sample of text for evaluation."""
    lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(num_lines):
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    lines.append(line.strip())
    except FileNotFoundError:
        print(f"Test data file not found: {file_path}")
    return lines


# ─── Evaluation core ─────────────────────────────────────────────────────────

def evaluate_tokenizer(name: str, tokenize_fn, text_samples: List[str]) -> Dict[str, Any]:
    """
    Evaluates a single tokenizer on a list of string samples.
    tokenize_fn: A callable that accepts a string and returns a list of tokens/ids.
    """
    total_tokens = 0
    total_words = 0
    total_bytes = 0

    start_time = time.time()

    for text in text_samples:
        words = text.split()
        if not words:
            continue

        tokens = tokenize_fn(text)

        total_tokens += len(tokens)
        total_words += len(words)
        total_bytes += len(text.encode('utf-8'))

    duration = time.time() - start_time

    tokens_per_word = total_tokens / total_words if total_words > 0 else 0

    # Corpus compression ratio (how efficiently it compresses bytes into tokens)
    bytes_per_token = total_bytes / total_tokens if total_tokens > 0 else 0

    tokens_per_sec = total_tokens / duration if duration > 0 else 0

    # Token fertility: average number of tokens per word
    # (same as tokens_per_word but renamed for clarity)
    token_fertility = tokens_per_word

    return {
        "Tokenizer": name,
        "Tokens/Word": round(tokens_per_word, 3),
        "Bytes/Token": round(bytes_per_token, 3),
        "Token Fertility": round(token_fertility, 3),
        "Speed (tok/sec)": round(tokens_per_sec, 2)
    }


def print_results(results: List[Dict[str, Any]]):
    """Prints the evaluation results in a markdown table format."""
    headers = ["Tokenizer", "Tokens/Word", "Bytes/Token", "Token Fertility", "Speed (tok/sec)"]
    col_widths = [25, 15, 15, 16, 18]

    # Header
    header_line = "| " + " | ".join(
        f"{h:<{w}}" for h, w in zip(headers, col_widths)
    ) + " |"
    separator = "| " + " | ".join(
        "-" * w for w in col_widths
    ) + " |"

    print(f"\n{header_line}")
    print(separator)
    for r in results:
        row = "| " + " | ".join(
            f"{str(r.get(h, 'N/A')):<{w}}" for h, w in zip(headers, col_widths)
        ) + " |"
        print(row)
    print()


def print_winner_analysis(results: List[Dict[str, Any]]):
    """Print a summary showing which tokenizer wins on each metric."""
    if len(results) < 2:
        return

    print("── Winner Analysis ──")

    # Lower Tokens/Word is better
    best_tpw = min(results, key=lambda r: r["Tokens/Word"])
    print(f"  Best Tokens/Word  : {best_tpw['Tokenizer']} ({best_tpw['Tokens/Word']})")

    # Higher Bytes/Token is better (more compression)
    best_bpt = max(results, key=lambda r: r["Bytes/Token"])
    print(f"  Best Bytes/Token  : {best_bpt['Tokenizer']} ({best_bpt['Bytes/Token']})")

    # Lower Token Fertility is better
    best_fert = min(results, key=lambda r: r["Token Fertility"])
    print(f"  Best Fertility    : {best_fert['Tokenizer']} ({best_fert['Token Fertility']})")

    # Higher speed is better
    best_speed = max(results, key=lambda r: r["Speed (tok/sec)"])
    print(f"  Fastest           : {best_speed['Tokenizer']} ({best_speed['Speed (tok/sec)']} tok/s)")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate tokenizers on a sample of text.")
    parser.add_argument("--test-file", type=str, default="datasets/hindi_raw.txt",
                        help="Path to text file for testing")
    parser.add_argument("--lines", type=int, default=5000,
                        help="Number of lines to evaluate")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory containing trained IndicTokenizer model")
    args = parser.parse_args()

    text_samples = load_test_data(args.test_file, args.lines)
    if not text_samples:
        print("No test data available for evaluation. Exiting.")
        return

    print(f"Loaded {len(text_samples)} lines for evaluation.")

    results = []

    # ─── 1. Custom IndicTokenizer ────────────────────────────────────────
    indic_loaded = False
    if IndicTokenizer is not None:
        vocab_path = os.path.join(args.model_dir, "indic_tokenizer_vocab.json")
        if os.path.exists(vocab_path):
            try:
                indic_tok = IndicTokenizer.load(args.model_dir)
                res = evaluate_tokenizer("IndicTokenizer (ours)", indic_tok.encode, text_samples)
                results.append(res)
                indic_loaded = True
                print("  ✓ IndicTokenizer loaded successfully.")
            except Exception as e:
                print(f"  ✗ Error loading IndicTokenizer: {e}")
        else:
            print(f"  ⊘ IndicTokenizer model not found at {args.model_dir}/")
            print(f"    Train first: uv run python python/adaptive_bpe.py")
    else:
        print("  ⊘ adaptive_bpe module not importable (check PYTHONPATH).")

    # ─── 2. Baseline: tiktoken (GPT-4 / cl100k_base) ────────────────────
    if tiktoken:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            res = evaluate_tokenizer("tiktoken (cl100k_base)", enc.encode, text_samples)
            results.append(res)
            print("  ✓ tiktoken loaded.")
        except Exception as e:
            print(f"  ✗ Error evaluating tiktoken: {e}")
    else:
        print("  ⊘ tiktoken not installed. Skipping.")

    # ─── 3. Baseline: SentencePiece (if a model exists) ──────────────────
    sp_model_path = os.path.join(args.model_dir, "sentencepiece.model")
    if spm and os.path.exists(sp_model_path):
        try:
            sp = spm.SentencePieceProcessor()
            sp.Load(sp_model_path)
            res = evaluate_tokenizer("SentencePiece", sp.Encode, text_samples)
            results.append(res)
            print("  ✓ SentencePiece loaded.")
        except Exception as e:
            print(f"  ✗ Error evaluating SentencePiece: {e}")
    else:
        if spm:
            print(f"  ⊘ SentencePiece model not found at {sp_model_path}. Skipping.")
        else:
            print("  ⊘ sentencepiece not installed. Skipping.")

    # ─── 4. Baseline: HuggingFace BPE tokenizer ─────────────────────────
    hf_model_path = os.path.join(args.model_dir, "hf_tokenizer.json")
    if PreTrainedTokenizerFast and os.path.exists(hf_model_path):
        try:
            hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=hf_model_path)
            res = evaluate_tokenizer("HuggingFace BPE", hf_tokenizer.encode, text_samples)
            results.append(res)
            print("  ✓ HuggingFace BPE loaded.")
        except Exception as e:
            print(f"  ✗ Error evaluating HuggingFace BPE: {e}")
    else:
        if PreTrainedTokenizerFast:
            print(f"  ⊘ HF tokenizer not found at {hf_model_path}. Skipping.")
        else:
            print("  ⊘ transformers not installed. Skipping.")

    # ─── Print results ───────────────────────────────────────────────────
    if results:
        print_results(results)
        print_winner_analysis(results)

        if indic_loaded and len(results) > 1:
            indic_result = results[0]
            others = results[1:]
            best_other_tpw = min(r["Tokens/Word"] for r in others)
            improvement = ((best_other_tpw - indic_result["Tokens/Word"]) / best_other_tpw) * 100
            if improvement > 0:
                print(f"🎉 IndicTokenizer uses {improvement:.1f}% fewer tokens per word "
                      f"than the best baseline!")
            else:
                print(f"📊 IndicTokenizer uses {-improvement:.1f}% more tokens per word "
                      f"than the best baseline. Consider tuning vocab size or morphology rules.")
    else:
        print("No tokenizers were evaluated. Check your dependencies and model files.")


if __name__ == "__main__":
    main()
