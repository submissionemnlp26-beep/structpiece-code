import os, sys, random
from pathlib import Path
import torch
import torch.nn as nn
from datasets import load_dataset
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from adaptive_bpe import IndicTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"

with open(ROOT / "experiments" / "english_sample.txt", "r", encoding="utf-8") as f:
    texts = [t.strip() for t in f if len(t.strip()) > 30]
train_texts = texts[:2000]
val_texts = texts[2000:2500]

train_file = str(ROOT / "en_train.txt")
with open(train_file, "w", encoding="utf-8") as f:
    f.write("\n".join(train_texts))

print("Training SP-BPE...")
spm.SentencePieceTrainer.Train(
    input=train_file,
    model_prefix=str(ROOT / "en_sp"),
    vocab_size=4000,
    model_type="bpe",
    character_coverage=0.9995
)
sp_tok = spm.SentencePieceProcessor()
sp_tok.Load(str(ROOT / "en_sp.model"))

print("Training MorphTokenizer...")
from python.adaptive_bpe import train_bpe
vocab, merges = train_bpe(train_file, vocab_size=4000, max_lines=2000, verbose=False)
morph_tok = IndicTokenizer(vocab, merges)

class NanoLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, max_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(embed_dim, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, vocab_size)
    def forward(self, x):
        _, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        emb = self.embed(x) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        return self.fc(self.transformer(emb, mask, is_causal=True))

def create_batches(tok, corpus, block_size=128, batch_size=32, is_morph=False):
    all_ids = []
    for txt in corpus:
        if is_morph:
            all_ids.extend(tok.encode(txt))
        else:
            all_ids.extend(tok.Encode(txt, out_type=int))
    batches = []
    for i in range(0, len(all_ids) - block_size, block_size):
        batches.append(all_ids[i:i+block_size+1])
    # truncate to multiple of batch_size
    batches = batches[:(len(batches)//batch_size)*batch_size]
    final = []
    for i in range(0, len(batches), batch_size):
        b = torch.tensor(batches[i:i+batch_size], dtype=torch.long)
        final.append((b[:, :-1], b[:, 1:]))
    return final

print("Creating batches...")
m_train_dl = create_batches(morph_tok, train_texts, is_morph=True)
m_val_dl = create_batches(morph_tok, val_texts, is_morph=True)
s_train_dl = create_batches(sp_tok, train_texts, is_morph=False)
s_val_dl = create_batches(sp_tok, val_texts, is_morph=False)

def train_lm(train_dl, val_dl, vocab_size, name):
    print(f"Training {name} LM...")
    torch.manual_seed(42)
    model = NanoLM(vocab_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    # 500 batches
    batches_done = 0
    while batches_done < 500 and train_dl:
        for x, y in train_dl:
            if batches_done >= 500: break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            opt.step()
            batches_done += 1
            
    model.eval()
    total_loss, steps = 0, 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += loss_fn(logits.view(-1, vocab_size), y.view(-1)).item()
            steps += 1
            
    avg_loss = total_loss / steps if steps > 0 else 0
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl

m_loss, m_ppl = train_lm(m_train_dl, m_val_dl, len(morph_tok.vocab), "MorphTokenizer")
s_loss, s_ppl = train_lm(s_train_dl, s_val_dl, sp_tok.GetPieceSize(), "SP-BPE")

print(f"\n✅ ENGLISH RESULTS")
print(f"MorphTokenizer -> Loss: {m_loss:.4f} | PPL: {m_ppl:.1f}")
print(f"SP-BPE         -> Loss: {s_loss:.4f} | PPL: {s_ppl:.1f}")

import time

print("\n--- EFFICIENCY & TOTALS (First 1000 lines) ---")
total_words = 0
total_chars = 0
morph_tokens_count = 0
sp_tokens_count = 0
for line in train_texts[:1000]:
    total_words += len(line.split())
    total_chars += len(line)
    morph_tokens_count += len(morph_tok.encode(line))
    sp_tokens_count += len(sp_tok.Encode(line, out_type=int))

print(f"Total Words: {total_words}, Total Chars: {total_chars}")
print(f"MorphTokenizer Total Tokens: {morph_tokens_count}")
print(f"SP-BPE Total Tokens: {sp_tokens_count}")
print(f"MorphTok - Tok/Word: {morph_tokens_count/total_words:.3f}, Tok/512ch: {(morph_tokens_count/total_chars)*512:.1f}, Ch/512tok: {(total_chars/morph_tokens_count)*512:.1f}")
print(f"SP-BPE   - Tok/Word: {sp_tokens_count/total_words:.3f}, Tok/512ch: {(sp_tokens_count/total_chars)*512:.1f}, Ch/512tok: {(total_chars/sp_tokens_count)*512:.1f}")

print("\n--- SPEED TEST ---")
t0 = time.time()
for line in train_texts: morph_tok.encode(line)
m_time = (time.time() - t0) * 1000
t0 = time.time()
for line in train_texts: sp_tok.Encode(line, out_type=int)
s_time = (time.time() - t0) * 1000

total_kch = sum(len(x) for x in train_texts) / 1000
print(f"MorphTokenizer ms/1kch: {m_time/total_kch:.2f}")
print(f"SP-BPE ms/1kch: {s_time/total_kch:.2f}")
print(f"Overhead: {((m_time/total_kch - s_time/total_kch)/(s_time/total_kch))*100:.1f}%")

print("\n--- QUALITATIVE ---")
test_words = ["archaeologists", "unbelievably", "internationalization", "disenfranchisement", "playing", "characterization"]
for w in test_words:
    m_t = [morph_tok.id_to_token[i] for i in morph_tok.encode(w)]
    b_t = sp_tok.Encode(w, out_type=str)
    print(f"{w:20} BPE: {b_t} ({len(b_t)}) | MORPH: {m_t} ({len(m_t)})")

import shutil
out_models_dir = ROOT / "experiments" / "fresh_results" / "models"
out_models_dir.mkdir(parents=True, exist_ok=True)
morph_tok.save(str(out_models_dir / "english_morph"))
shutil.copy(str(ROOT / "en_sp.model"), str(out_models_dir / "english_sp_bpe.model"))
print(f"Saved english models to {out_models_dir}")

