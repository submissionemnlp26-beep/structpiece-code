#!/usr/bin/env python3
"""
Vocabulary Surgery Initialization for LLaMA-3.2-1B
==================================================
Implements Phase 2: Architectural Surgery via Geometric Projection.
Projects the 128k base LLaMA embeddings into a 32k StructPiece vocabulary
using mean-pooled initialization (Equation 6 and 7 in the paper).

Hardware requirement: ~80GB VRAM to load LLaMA-3.2-1B and perform the
surgery if running in full precision, or ~40GB if using BF16/DeepSpeed.

Usage:
    PYTHONPATH=. python reproduce/scripts/surgery_init.py \
        --base-model meta-llama/Llama-3.2-1B \
        --structpiece-vocab reproduce/models/multilingual_structpiece \
        --output reproduce/results/surgery_init/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Resolve project root and add python/ to path for adaptive_bpe import
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from adaptive_bpe import IndicTokenizer

def build_prefix_trie(vocab: dict) -> dict:
    """Build a prefix trie from vocabulary for O(max_len) lookup per position."""
    trie = {}
    for token in vocab:
        node = trie
        for ch in token:
            node = node.setdefault(ch, {})
        node["_end"] = True
    return trie


def get_greedy_decomposition(target_str: str, base_vocab: dict,
                              trie: dict = None) -> list[str]:
    """
    Greedy left-to-right decomposition using the base vocabulary.

    Uses a prefix trie for O(n * max_token_len) complexity instead of
    O(n^2 * |V|). Falls back to single-character pieces for unmatched spans.
    """
    if trie is None:
        trie = build_prefix_trie(base_vocab)

    pieces = []
    i = 0
    n = len(target_str)
    while i < n:
        node = trie
        best_end = -1
        j = i
        while j < n and target_str[j] in node:
            node = node[target_str[j]]
            j += 1
            if "_end" in node:
                best_end = j

        if best_end > i:
            pieces.append(target_str[i:best_end])
            i = best_end
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
    
    # Build prefix trie once for O(n * max_token_len) decomposition
    vocab_trie = build_prefix_trie(base_str_to_id)
    
    for token_str, target_id in tqdm(target_vocab.items()):
        # Try direct match first
        if token_str in base_str_to_id:
            base_id = base_str_to_id[token_str]
            new_embeddings[target_id] = base_embeddings[base_id]
            new_lm_head[target_id] = base_lm_head[base_id]
        else:
            # Equation 6: Mean-pool over constituent base tokens
            pieces = get_greedy_decomposition(token_str, base_str_to_id, trie=vocab_trie)
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
