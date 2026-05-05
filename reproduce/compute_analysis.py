"""
Compute Analysis for Tokenizers
===============================
In a Transformer, the cost of self-attention grows quadratically with
sequence length `n`. Specifically, for one layer, the attention FLOPs 
are roughly proportional to `n^2 * d_model`.

This script compares the theoretical compute cost to process the exact
same text (1000 words) using IndicTokenizer vs. SentencePiece.
"""

import sys
import json
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lm.tokenizer_adapter import TokenizerWrapper

def run_compute_analysis(corpus_path: str, d_model: int = 256) -> list[dict]:
    # Load tokenizers
    indic_tok = TokenizerWrapper("indic", str(_ROOT / "models"))
    sp_tok = TokenizerWrapper("sp", str(_ROOT / "experiments" / "sp_models" / "sp_bpe_vs8000.model"))
    
    tokenizers = {
        "IndicTokenizer": indic_tok,
        "SentencePiece": sp_tok
    }
    
    # We want to process exactly 1000 words, for example
    with open(corpus_path, "r", encoding="utf-8") as f:
        # read enough text to get 1000 words
        text = ""
        word_count = 0
        for line in f:
            words = line.split()
            if word_count + len(words) > 1000:
                needed = 1000 - word_count
                text += " ".join(words[:needed])
                break
            else:
                text += line
                word_count += len(words)
                
    results = []
    print(f"Theoretical Compute Analysis (d_model={d_model}, text=1000 words)")
    print("-" * 65)
    
    for name, tok in tokenizers.items():
        tokens = tok.encode(text)
        n = len(tokens)
        
        # Self-attention compute approx: 
        #   Q @ K.T -> n * n * d_model
        #   A @ V   -> n * n * d_model
        # Total per layer ~ 2 * n**2 * d_model operations
        flops_approx = 2 * (n ** 2) * d_model
        
        results.append({
            "tokenizer": name,
            "text_words": 1000,
            "sequence_length_n": n,
            "flops_approx": flops_approx
        })
        
        flops_mil = flops_approx / 1_000_000
        print(f"[{name}] Tokens: {n} | FLOPs: ~{flops_mil:.2f} M")
        
    return results

def main() -> None:
    corpus = str(_ROOT / "datasets" / "hindi_raw.txt")
    results = run_compute_analysis(corpus)
    
    final_out = _ROOT / "experiments" / "results_compute.json"
    with open(final_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved compute efficiency comparison to: {final_out}")

if __name__ == "__main__":
    main()
