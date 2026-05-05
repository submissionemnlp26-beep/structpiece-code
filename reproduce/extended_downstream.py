"""
Extended Downstream Evaluation: POS Tagging + Text Classification
=================================================================
Evaluates MorphTokenizer vs SP-BPE on:
  1. POS Tagging (XTREME/udpos) — Hindi, Arabic, Turkish
  2. Text Classification (XNLI — 3-way NLI) — Hindi, Arabic, Turkish

Uses BiLSTM models to isolate tokenizer signal.
Outputs JSON results for Tables 12 & 13.
"""
import sys, json, time, math, random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from adaptive_bpe import IndicTokenizer
import sentencepiece as spm
from datasets import load_dataset

MODELS_DIR = ROOT / "experiments" / "fresh_results" / "models"
PAD_TAG_ID = -100

# ── Tokenizer loaders ────────────────────────────────────────────────────────

def load_morph(lang):
    return IndicTokenizer.load(str(MODELS_DIR / f"{lang}_morph"))

def load_sp_bpe(lang):
    sp = spm.SentencePieceProcessor()
    sp.Load(str(MODELS_DIR / f"{lang}_sp_bpe.model"))
    return sp

# ── Label alignment for token classification ─────────────────────────────────

def tokenize_align_morph(morph_tok, words, tags):
    all_ids, all_labels = [], []
    for word, tag in zip(words, tags):
        ids = morph_tok.encode(word)
        if not ids:
            continue
        all_ids.extend(ids)
        all_labels.append(tag)
        all_labels.extend([PAD_TAG_ID] * (len(ids) - 1))
    return all_ids, all_labels

def tokenize_align_sp(sp_tok, words, tags):
    all_ids, all_labels = [], []
    for word, tag in zip(words, tags):
        ids = sp_tok.Encode(word, out_type=int)
        if not ids:
            continue
        all_ids.extend(ids)
        all_labels.append(tag)
        all_labels.extend([PAD_TAG_ID] * (len(ids) - 1))
    return all_ids, all_labels

# ── Sentence tokenization for classification ─────────────────────────────────

def tokenize_sentence_morph(morph_tok, text, max_len=256):
    ids = morph_tok.encode(text)
    return ids[:max_len]

def tokenize_sentence_sp(sp_tok, text, max_len=256):
    ids = sp_tok.Encode(text, out_type=int)
    return ids[:max_len]

# ── Datasets ─────────────────────────────────────────────────────────────────

class SeqLabelDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        ids, labels = self.examples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

class ClassifDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        ids, label = self.examples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_seq(batch):
    ids_list, labels_list = zip(*batch)
    return (pad_sequence(ids_list, batch_first=True, padding_value=0),
            pad_sequence(labels_list, batch_first=True, padding_value=PAD_TAG_ID))

def collate_clf(batch):
    ids_list, labels = zip(*batch)
    return pad_sequence(ids_list, batch_first=True, padding_value=0), torch.stack(labels)

# ── Models ───────────────────────────────────────────────────────────────────

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, num_tags, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
    def forward(self, x):
        emb = self.dropout(self.embed(x))
        return self.fc(self.lstm(emb)[0])

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    def forward(self, x):
        out, _ = self.lstm(self.embed(x))
        # Mean pool over sequence
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.fc(pooled)

# ── Training routines ────────────────────────────────────────────────────────

def train_eval_tagger(train_data, val_data, vocab_size, num_tags, epochs=10, lr=1e-3, bs=16, seeds=3):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_dl = DataLoader(SeqLabelDataset(train_data), batch_size=bs, shuffle=True, collate_fn=collate_seq)
    val_dl = DataLoader(SeqLabelDataset(val_data), batch_size=bs, shuffle=False, collate_fn=collate_seq)
    
    seed_accs = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        model = BiLSTMTagger(vocab_size, num_tags).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TAG_ID)
        
        for epoch in range(epochs):
            model.train()
            for ids, labels in train_dl:
                ids, labels = ids.to(device), labels.to(device)
                opt.zero_grad()
                loss = loss_fn(model(ids).view(-1, num_tags), labels.view(-1))
                loss.backward()
                opt.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for ids, labels in val_dl:
                ids, labels = ids.to(device), labels.to(device)
                preds = model(ids).argmax(dim=-1)
                mask = labels != PAD_TAG_ID
                correct += (preds[mask] == labels[mask]).sum().item()
                total += mask.sum().item()
        
        acc = correct / total if total > 0 else 0.0
        seed_accs.append(acc)
    return sum(seed_accs) / len(seed_accs)

def train_eval_classifier(train_data, val_data, vocab_size, num_classes, epochs=10, lr=1e-3, bs=16, seeds=3):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_dl = DataLoader(ClassifDataset(train_data), batch_size=bs, shuffle=True, collate_fn=collate_clf)
    val_dl = DataLoader(ClassifDataset(val_data), batch_size=bs, shuffle=False, collate_fn=collate_clf)
    
    seed_accs = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        model = BiLSTMClassifier(vocab_size, num_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            for ids, labels in train_dl:
                ids, labels = ids.to(device), labels.to(device)
                opt.zero_grad()
                loss = loss_fn(model(ids), labels)
                loss.backward()
                opt.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for ids, labels in val_dl:
                ids, labels = ids.to(device), labels.to(device)
                preds = model(ids).argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total if total > 0 else 0.0
        seed_accs.append(acc)
    return sum(seed_accs) / len(seed_accs)

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    results = {"pos": [], "classif": []}
    
    configs = [("english", "English")]
    
    # ═══════════════════════════════════════════════════════════════════════
    # PART 1: POS TAGGING (XTREME/udpos)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  PART 1: POS TAGGING (XTREME/udpos)")
    print("="*70)
    
    for lang_key, hf_name in configs:
        print(f"\n--- {lang_key.upper()} ---")
        try:
            ds = load_dataset("xtreme", f"udpos.{hf_name}")
            train_split = ds["train"].select(range(min(2000, len(ds["train"]))))
            val_split = ds["validation"].select(range(min(500, len(ds["validation"]))))
            
            # Determine number of unique POS tags
            all_tags = set()
            for ex in train_split:
                all_tags.update(ex["pos_tags"])
            num_tags = max(all_tags) + 1
            print(f"  Train={len(train_split)}, Val={len(val_split)}, Tags={num_tags}")
            
            morph_tok = load_morph(lang_key)
            sp_tok = load_sp_bpe(lang_key)
            morph_v = len(morph_tok.vocab)
            sp_v = sp_tok.GetPieceSize()
            
            # Tokenize
            print(f"  Tokenizing...")
            m_train = [(ids, lbl) for ids, lbl in 
                       [tokenize_align_morph(morph_tok, ex["tokens"], ex["pos_tags"]) for ex in train_split]
                       if ids]
            m_val = [(ids, lbl) for ids, lbl in 
                     [tokenize_align_morph(morph_tok, ex["tokens"], ex["pos_tags"]) for ex in val_split]
                     if ids]
            s_train = [(ids, lbl) for ids, lbl in 
                       [tokenize_align_sp(sp_tok, ex["tokens"], ex["pos_tags"]) for ex in train_split]
                       if ids]
            s_val = [(ids, lbl) for ids, lbl in 
                     [tokenize_align_sp(sp_tok, ex["tokens"], ex["pos_tags"]) for ex in val_split]
                     if ids]
            
            # Train + eval
            print(f"  Training MorphTokenizer tagger...")
            t0 = time.time()
            m_acc = train_eval_tagger(m_train, m_val, morph_v, num_tags)
            print(f"    MorphTokenizer POS Acc={m_acc:.4f} ({time.time()-t0:.1f}s)")
            
            print(f"  Training SP-BPE tagger...")
            t0 = time.time()
            s_acc = train_eval_tagger(s_train, s_val, sp_v, num_tags)
            print(f"    SP-BPE POS Acc={s_acc:.4f} ({time.time()-t0:.1f}s)")
            
            results["pos"].append({
                "language": lang_key.capitalize(),
                "morph_acc": round(m_acc * 100, 1),
                "sp_acc": round(s_acc * 100, 1),
                "delta": round((m_acc - s_acc) * 100, 1)
            })
        except Exception as e:
            print(f"  SKIPPED {lang_key}: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PART 2: TEXT CLASSIFICATION (XNLI — 3-way NLI)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  PART 2: TEXT CLASSIFICATION (XNLI)")
    print("="*70)
    
    for lang_key, hf_name in configs:
        hf_code = {"english": "en"}[lang_key]
        print(f"\n--- {lang_key.upper()} ---")
        try:
            ds = load_dataset("xnli", hf_code)
            train_split = ds["train"].select(range(min(2000, len(ds["train"]))))
            val_split = ds["validation"].select(range(min(500, len(ds["validation"]))))
            num_classes = 3  # entailment, neutral, contradiction
            print(f"  Train={len(train_split)}, Val={len(val_split)}, Classes={num_classes}")
            
            morph_tok = load_morph(lang_key)
            sp_tok = load_sp_bpe(lang_key)
            morph_v = len(morph_tok.vocab)
            sp_v = sp_tok.GetPieceSize()
            
            # For NLI, concatenate premise + hypothesis
            def make_text(ex):
                return ex["premise"] + " " + ex["hypothesis"]
            
            print(f"  Tokenizing...")
            m_train = [(tokenize_sentence_morph(morph_tok, make_text(ex)), ex["label"]) for ex in train_split]
            m_val = [(tokenize_sentence_morph(morph_tok, make_text(ex)), ex["label"]) for ex in val_split]
            m_train = [(ids, lbl) for ids, lbl in m_train if ids]
            m_val = [(ids, lbl) for ids, lbl in m_val if ids]
            
            s_train = [(tokenize_sentence_sp(sp_tok, make_text(ex)), ex["label"]) for ex in train_split]
            s_val = [(tokenize_sentence_sp(sp_tok, make_text(ex)), ex["label"]) for ex in val_split]
            s_train = [(ids, lbl) for ids, lbl in s_train if ids]
            s_val = [(ids, lbl) for ids, lbl in s_val if ids]
            
            # Train + eval
            print(f"  Training MorphTokenizer classifier...")
            t0 = time.time()
            m_acc = train_eval_classifier(m_train, m_val, morph_v, num_classes)
            print(f"    MorphTokenizer XNLI Acc={m_acc:.4f} ({time.time()-t0:.1f}s)")
            
            print(f"  Training SP-BPE classifier...")
            t0 = time.time()
            s_acc = train_eval_classifier(s_train, s_val, sp_v, num_classes)
            print(f"    SP-BPE XNLI Acc={s_acc:.4f} ({time.time()-t0:.1f}s)")
            
            results["classif"].append({
                "language": lang_key.capitalize(),
                "morph_acc": round(m_acc * 100, 1),
                "sp_acc": round(s_acc * 100, 1),
                "delta": round((m_acc - s_acc) * 100, 1)
            })
        except Exception as e:
            print(f"  SKIPPED {lang_key}: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Save Results
    # ═══════════════════════════════════════════════════════════════════════
    out_path = ROOT / "experiments" / "fresh_results" / "english_extended_downstream_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("  TABLE 12 — POS TAGGING")
    print(f"{'='*70}")
    for r in results["pos"]:
        print(f"| {r['language']:<10} | MorphTokenizer | {r['morph_acc']:>6.1f}% |")
        print(f"| {'':<10} | SP-BPE         | {r['sp_acc']:>6.1f}% |")
    
    print(f"\n{'='*70}")
    print("  TABLE 13 — TEXT CLASSIFICATION (XNLI)")
    print(f"{'='*70}")
    for r in results["classif"]:
        print(f"| {r['language']:<10} | MorphTokenizer | {r['morph_acc']:>6.1f}% |")
        print(f"| {'':<10} | SP-BPE         | {r['sp_acc']:>6.1f}% |")
    
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
