"""
Minimal Language Model Training Loop
====================================
Trains the NanoLM on a TextDataset. Saves step-wise loss to JSON.
No checkpointing, no validation loops. Keeps things as simple as possible.

Usage
-----
    python -m lm.train \
        --tokenizer indic \
        --model-path models \
        --corpus datasets/hindi_raw.txt \
        --output experiments/lm_results_indic.json
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lm.tokenizer_adapter import TokenizerWrapper
from lm.dataset import TextDataset
from lm.model import NanoLM


def train(
    tokenizer_type: str,
    model_path: str,
    corpus_path: str,
    output_path: str,
    block_size: int = 128,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    epochs: int = 1,
    max_steps: int = -1,
) -> None:
    torch.manual_seed(42)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Setup tokenizer & dataset
    tok = TokenizerWrapper(tokenizer_type, model_path)
    print(f"Loaded tokenizer: {repr(tok)}")

    ds = TextDataset(corpus_path, tok, block_size=block_size)
    print(f"Loaded dataset: {len(ds)} samples")
    pin_memory = False if device == "mps" else (device != "cpu")
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory = pin_memory
    )

    # 2. Setup model
    model = NanoLM(
        vocab_size=tok.vocab_size,
        block_size=block_size,
        d_model=256,
        n_heads=4,
        n_layers=2,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # 3. Training loop
    model.train()
    history = []

    global_step = 0
    t0 = time.time()

    for epoch in range(epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}")
        model.train()
        for x, y in pbar:
            if max_steps > 0 and global_step >= max_steps:
                break

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            _, loss, _ = model(x, targets=y)
            loss.backward()

            # gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            loss_val = loss.item()
            ppl = math.exp(loss_val) if loss_val < 20 else float("inf")
            history.append({
                "step": global_step,
                "loss": loss_val,
                "perplexity": ppl,
                "epoch": epoch
            })

            pbar.set_postfix({"loss": f"{loss_val:.4f}","ppl": f"{ppl:.2f}"})
            global_step += 1
        epoch_losses = [h["loss"] for h in history if h["epoch"] == epoch]
        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        if max_steps > 0 and global_step >= max_steps:
             break

    t1 = time.time()
    print(f"Training finished in {t1 - t0:.2f}s")

    model.eval()
    final_loss = history[-1]["loss"] if history else None
    final_ppl = history[-1]["perplexity"] if history else None

    # 4. Save results
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "tokenizer": tokenizer_type,
            "vocab_size": tok.vocab_size,
            "time_seconds": t1 - t0,
            "final_loss": final_loss,
            "final_perplexity": final_ppl,
            "history": history
        }, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True, choices=["indic", "sp"])
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--corpus", type=str, default="datasets/hindi_raw.txt")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    train(
        tokenizer_type=args.tokenizer,
        model_path=args.model_path,
        corpus_path=args.corpus,
        output_path=args.output,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )
