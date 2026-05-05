#!/usr/bin/env python3
"""
Vocabulary Surgery Initialization for LLaMA-3.2-1B
==================================================
Implements Phase 2: Architectural Surgery via Geometric Projection.
Projects the 128k base LLaMA embeddings into a 32k StructPiece vocabulary
using mean-pooled initialization (Equation 6 and 7 in the paper).

Hardware requirement: ~80GB VRAM to load LLaMA-3.2-1B and perform the
surgery if running in full precision, or ~40GB if using BF16/DeepSpeed.
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from adaptive_bpe import IndicTokenizer

def get_greedy_decomposition(target_str: str, base_vocab: dict) -> list[str]:
    """Greedy left-to-right decomposition using the base vocabulary."""
    # Simplified greedy matching logic
    pieces = []
    i = 0
    n = len(target_str)
    while i < n:
        best_match = None
        best_len = 0
        # Search for longest matching prefix in base vocab
        for j in range(i + 1, n + 1):
            sub = target_str[i:j]
            if sub in base_vocab and (j - i) > best_len:
                best_match = sub
                best_len = j - i
        
        if best_match:
            pieces.append(best_match)
            i += best_len
        else:
            # Fallback for unmatched characters
            pieces.append(target_str[i])
            i += 1
    return pieces

def main():
    parser = argparse.ArgumentParser(description="LLaMA Vocabulary Surgery")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--structpiece-vocab", type=str, required=True, help="Path to StructPiece models dir")
    parser.add_argument("--output", type=str, required=True, help="Output directory for surgical checkpoint")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}...")
    # This will require significant RAM/VRAM
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    print(f"Loading StructPiece target vocabulary from {args.structpiece_vocab}...")
    structpiece = IndicTokenizer.load(args.structpiece_vocab)
    
    base_vocab = base_tokenizer.get_vocab()
    target_vocab = structpiece.vocab
    
    base_embeddings = model.model.embed_tokens.weight.data
    base_lm_head = model.lm_head.weight.data
    
    embed_dim = base_embeddings.shape[1]
    target_vocab_size = len(target_vocab)
    
    print(f"Base vocab size:   {len(base_vocab)}")
    print(f"Target vocab size: {target_vocab_size}")
    print(f"Embedding dim:     {embed_dim}")
    print("Performing Mean-Pooled Surgery (Equation 6 & 7)...")
    
    new_embeddings = torch.zeros((target_vocab_size, embed_dim), dtype=torch.bfloat16)
    new_lm_head = torch.zeros((target_vocab_size, embed_dim), dtype=torch.bfloat16)
    
    unmatched_count = 0
    
    # Create reverse lookup for base vocab to get IDs
    base_str_to_id = {k: v for k, v in base_vocab.items()}
    
    for token_str, target_id in tqdm(target_vocab.items()):
        # Try direct match first
        if token_str in base_str_to_id:
            base_id = base_str_to_id[token_str]
            new_embeddings[target_id] = base_embeddings[base_id]
            new_lm_head[target_id] = base_lm_head[base_id]
        else:
            # Equation 6: Mean-pool over constituent base tokens
            pieces = get_greedy_decomposition(token_str, base_str_to_id)
            piece_ids = [base_str_to_id[p] for p in pieces if p in base_str_to_id]
            
            if piece_ids:
                mean_emb = base_embeddings[piece_ids].mean(dim=0)
                mean_head = base_lm_head[piece_ids].mean(dim=0)
                new_embeddings[target_id] = mean_emb
                new_lm_head[target_id] = mean_head
            else:
                # Equation 7: Xavier Initialization fallback (Z term)
                unmatched_count += 1
                nn.init.xavier_uniform_(new_embeddings[target_id:target_id+1])
                nn.init.xavier_uniform_(new_lm_head[target_id:target_id+1])
    
    print(f"Surgery complete.")
    print(f"Mean-pooled matched tokens: {target_vocab_size - unmatched_count}")
    print(f"Xavier-initialized fallback tokens: {unmatched_count} ({unmatched_count/target_vocab_size*100:.1f}%)")
    
    # Inject new embeddings back into model
    model.model.embed_tokens.weight.data = new_embeddings
    model.lm_head.weight.data = new_lm_head
    model.config.vocab_size = target_vocab_size
    
    print(f"Saving structural checkpoint to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    print("Done! Ready for continual pre-training manifold alignment.")

if __name__ == "__main__":
    main()
