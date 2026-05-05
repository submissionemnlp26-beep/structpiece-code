"""
Attention Entropy Analysis
==========================
Computes the Shannon entropy of attention weights for models
trained with different tokenizers. A model with more morphologically
meaningful tokens might exhibit different attention sparsity 
(lower entropy = sharper attention).
"""

import sys
import json
import torch
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from lm.tokenizer_adapter import TokenizerWrapper
from lm.model import NanoLM

def calc_attention_entropy(att_weights: torch.Tensor) -> float:
    """
    Computes Shannon entropy over the last dimension (T) of the attention map.
    att_weights is shape (B, n_heads, T, T).
    """
    # Exclude padded/masked probabilities which are 0 -> log(0)
    # Add a tiny epsilon to avoid -inf in log
    eps = 1e-9
    att_weights = att_weights + eps
    
    entropy_per_token = -torch.sum(att_weights * torch.log2(att_weights), dim=-1)
    return float(entropy_per_token.mean().item())

def run_attention_analysis(corpus_path: str) -> list[dict]:
    # Load tokenizers
    indic_tok = TokenizerWrapper("indic", str(_ROOT / "models"))
    sp_tok = TokenizerWrapper("sp", str(_ROOT / "experiments" / "sp_models" / "sp_bpe_vs8000.model"))
    
    tokenizers = {
        "IndicTokenizer": indic_tok,
        "SentencePiece": sp_tok
    }
    
    # Initialize un-trained models just to measure forward-pass prior / initialization entropy
    # (Or you could load trained checkpoints, but for a standalone quick analysis 
    #  without checkpoints, we use randomly initialized NanoLMs)
    print("Computing attention entropy (random init, purely tokenizer structure)...")
    print("-" * 65)
    
    # Sample text
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = " ".join(f.read().split()[:200]) # 200 words
        
    results = []
    
    for name, tok in tokenizers.items():
        # Encode test sequence
        tokens = tok.encode(text)
        
        # limit to block_size 128
        context = min(len(tokens), 128)
        x = torch.tensor(tokens[:context]).unsqueeze(0) # (1, T)
        
        model = NanoLM(
            vocab_size=tok.vocab_size, 
            block_size=128,
            d_model=256,
            n_heads=4,
            n_layers=2
        )
        model.eval()
        
        with torch.no_grad():
            _, _, all_att_weights = model(x)
            
        layer_entropies = []
        for i, att in enumerate(all_att_weights):
            ent = calc_attention_entropy(att)
            layer_entropies.append(ent)
            
        avg_ent = sum(layer_entropies) / len(layer_entropies)
        
        results.append({
            "tokenizer": name,
            "layer_entropies": [round(e, 3) for e in layer_entropies],
            "avg_entropy": round(avg_ent, 3)
        })
        print(f"[{name}] Avg Entropy: {avg_ent:.3f} bits")
        
    return results

def main() -> None:
    corpus = str(_ROOT / "datasets" / "hindi_raw.txt")
    results = run_attention_analysis(corpus)
    
    final_out = _ROOT / "experiments" / "results_attention.json"
    with open(final_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved attention analysis results to: {final_out}")

if __name__ == "__main__":
    main()
