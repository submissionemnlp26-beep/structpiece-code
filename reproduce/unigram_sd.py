import os, sys, time, math, torch, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from lm.tokenizer_adapter import TokenizerWrapper
from experiments.run_core_reeval import train_lm, compute_bpc, compute_token_stats

LANGS = ["arabic", "english", "estonian", "finnish", "georgian", "hindi", "hungarian", "tamil", "turkish"]

def get_corpus(lang):
    c = ROOT / f"datasets/{lang}_clean.txt"
    if c.exists(): return str(c)
    r = ROOT / f"datasets/{lang}_raw.txt"
    if r.exists(): return str(r)
    if lang == "english":
        e = ROOT / "experiments/english_sample.txt"
        if e.exists(): return str(e)
    return None

import time

print("Computing Unigram multi-seed standard deviations...")
print(f"{'Language':<12} {'Avg Loss':<15} {'Token PPL':<15} {'BPC'}")
print("-" * 60)

for lang in LANGS:
    corpus = get_corpus(lang)
    if not corpus: 
        print(f"Skipping {lang}, corpus not found.")
        continue
    
    mp1 = ROOT / f"experiments/fresh_results/models/{lang}_sp_unigram.model"
    mp2 = ROOT / f"experiments/modern_baselines/models/{lang}_sp_unigram.model"
    
    if mp1.exists():
        model_path = mp1
    elif mp2.exists():
        model_path = mp2
    else:
         print(f"Skipping {lang}, no SP-Unigram model found.")
         continue
         
    tok_wrapper = TokenizerWrapper("sp", str(model_path))
    st = compute_token_stats(tok_wrapper.encode, corpus, max_lines=1000)
    tok_per_char = st["tok_per_char"]
    
    losses, ppls, bpcs = [], [], []
    for s in [42, 123, 777]:
        torch.manual_seed(s)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
        np.random.seed(s)
        
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        res = train_lm(tok_wrapper, corpus, max_steps=500, device=device)
        l = res["avg_loss"]
        p = res["avg_ppl"]
        b = compute_bpc(l, tok_per_char)
        
        losses.append(l)
        ppls.append(p)
        bpcs.append(b)
        
    l_mean, l_sd = np.mean(losses), np.std(losses)
    p_mean, p_sd = np.mean(ppls), np.std(ppls)
    b_mean, b_sd = np.mean(bpcs), np.std(bpcs)
    
    print(f"{lang.capitalize():<12} {l_mean:.4f} \u00b1 {l_sd:.2f}    {p_mean:.1f} \u00b1 {p_sd:.1f}    {b_mean:.3f} \u00b1 {b_sd:.3f}")
