"""
Context Window Efficiency
=========================
Given a fixed context window (e.g., 128 or 512 tokens), this script measures
how many words each tokenizer can fit inside on average.
A highly compressing tokenizer will pack more linguistic information into
the same fixed LM context.
"""

import sys
import json
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lm.tokenizer_adapter import TokenizerWrapper

def count_words(text):
    return len([w for w in text.strip().split() if w])

def count_characters(text):
    return len(text)

def run_context_efficiency(corpus_path: str, window_size: int = 512) -> dict:
    # Load tokenizers
    indic_tok = TokenizerWrapper("indic", str(_ROOT / "models"))

    sp_bpe_tok = TokenizerWrapper(
        "sp",
        str(_ROOT / "experiments" / "sp_models" / "sp_bpe_vs8000.model")
    )

    sp_unigram_tok = TokenizerWrapper(
        "sp",
        str(_ROOT / "experiments" / "sp_models" / "sp_unigram_vs8000.model")
    )

    tokenizers = {
        "IndicTokenizer": indic_tok,
        "SentencePiece-BPE": sp_bpe_tok,
        "SentencePiece-Unigram": sp_unigram_tok,
}

    # Read corpus once into a list of lines
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    text = "".join(lines)

    results = {}
    print(f"Evaluating Tokenizers for Context Efficiency (Window Size: {window_size} tokens)")
    print("-" * 60)

    for name, tok in tokenizers.items():
        tokens = tok.encode(text)
        print(f"{name}: total tokens = {len(tokens)}")

        # Chop into contiguous non-overlapping windows of `window_size`
        words_per_window = []
        chars_per_window = []
        for i in range(0, len(tokens) - window_size + 1, window_size):
            window = tokens[i : i+window_size]
            decoded = tok.decode(window).strip()
            if not decoded:
                continue

            words = count_words(decoded)
            chars = count_characters(decoded.replace(" ", ""))
            words_per_window.append(words)
            chars_per_window.append(chars)

        if not words_per_window:
            avg_words = 0.0
            avg_chars = 0.0
        else:
            avg_words = sum(words_per_window) / len(words_per_window)
            avg_chars = sum(chars_per_window) / len(chars_per_window)

        print(f"{name}: processed {len(words_per_window)} windows")

        if "Indic" in name:
            key = "indic"
        elif "BPE" in name:
            key = "sp_bpe"
        else:
            key = "sp_unigram"
        results[key] = {
            "words_per_window": round(avg_words, 2),
            "chars_per_window": round(avg_chars, 2)
        }

        print(f"[{name}] {window_size} tokens -> {avg_words:.2f} words, {avg_chars:.2f} chars")

    return results

def main() -> None:
    corpus = str(_ROOT / "datasets" / "hindi_raw.txt")
    results = run_context_efficiency(corpus, window_size=512)

    final_out = _ROOT / "experiments" / "results_context.json"
    with open(final_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved context efficiency results to: {final_out}")

if __name__ == "__main__":
    main()
