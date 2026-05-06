# Tokenization as Structural Inductive Bias
## Grapheme-Constrained Subwords for Multilingual Modeling and Vocabulary Surgery

*Official code repository for the EMNLP submission.*

[![Dataset on Zenodo](https://img.shields.io/badge/Dataset-Zenodo-blue)](https://zenodo.org/records/20033927)

---

## Overview

Subword tokenizers are optimized for compression, but their boundaries also define the prediction units seen by a language model. We introduce **StructPiece**, an unsupervised tokenizer that enforces Unicode Extended Grapheme Cluster (EGC) boundaries and biases merge induction toward corpus-induced affix patterns without external resources.

Across nine languages and five scripts, StructPiece exposes a **compression–representation frontier**: it produces higher-fertility segmentations with worse bits-per-character (BPC) but substantially lower token-level perplexity under matched architectures, vocabulary sizes, and token budgets.

---

## Repository Structure

```
BPE-Hindi/
│
├── python/                        # Core StructPiece algorithm
│   ├── adaptive_bpe.py            #   EGC-constrained BPE + morphology discovery
│   └── pretokenizer.py            #   Unicode EGC pretokenization logic
│
├── lm/                            # NanoLM (evaluation backbone)
│   ├── model.py                   #   2-layer causal Transformer (d=256)
│   ├── train.py                   #   Training loop
│   ├── dataset.py                 #   TextDataset for fixed-budget tokenized batches
│   └── tokenizer_adapter.py       #   Unified API over StructPiece/SP/tiktoken
│
├── datasets/                      # Cleaned corpora for all 9 languages
│   # Note: Due to size constraints, datasets are hosted on Zenodo.
│   # Please download from: https://zenodo.org/records/20033927
│   # and place the .txt files in this directory.
│   ├── hindi_raw.txt
│   ├── arabic_clean.txt
│   ├── turkish_clean.txt
│   └── ...
│
├── reproduce/                     # Complete reproduction infrastructure
│   ├── run_nanolm_experiments.py  #   ← START HERE for Phase 1 (Tables 1, 2, 3, 4)
│   ├── run_core_reeval.py         #   Ablation study runner (EGC, Morphology)
│   ├── ner_downstream.py          #   NER probes (WikiANN, Table 5)
│   ├── extended_downstream.py     #   POS + XNLI probes (Tables 8, 9, 10)
│   ├── run_modern_baselines.py    #   Global tokenizer comparison (Appendix Table F)
│   ├── scripts/                   #   LLaMA vocabulary surgery scripts (GPU required)
│   ├── results/                   #   Output directory for locally-run results
│   └── RESULTS_INTEGRITY.md       #   ← What can/cannot be reproduced locally
│
├── experiments/                   # Reference outputs from original H100 cluster runs
│   └── fresh_results/             #   Original training logs (JSON) and tables
│
└── cpp/                           # C++ EGC segmentation kernel
    └── ...
```

---

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for deterministic dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync exact locked dependencies
uv sync

# Or with standard pip:
pip install -r requirements.txt
```

**Core dependencies:** `torch`, `sentencepiece`, `datasets`, `seqeval`, `tqdm`, `regex`

---

## Reproducing the Results

### Read This First

See **[`reproduce/RESULTS_INTEGRITY.md`](reproduce/RESULTS_INTEGRITY.md)** for a transparent accounting of what can be reproduced locally versus what required our NVIDIA H100 cluster, including a complete Table → JSON → Script audit map.

**Short answer:**
- ✅ All NanoLM intrinsic experiments (Tables 1–4) → run locally
- ✅ All downstream probes (Tables 5, 8, 9, 14, 17) → run locally
- ✅ Morfessor/BPE-knockout baselines (Table 16) → run locally
- ⚠️ LLaMA-3.2-1B vocabulary surgery (Tables 6, 7, 8) → requires ~40GB VRAM
- ⚠️ Extended scale validation (Tables 15, 38) → GPU recommended

---

### Phase 1: NanoLM Intrinsic Experiments (Tables 1, 2, 3, 10)

Trains StructPiece, SP-BPE, and SP-Unigram from scratch on each language, then trains a NanoLM under an identical fixed token budget (2,048,000 tokens per run).

```bash
# Full 9-language sweep:
PYTHONPATH=. python reproduce/run_nanolm_experiments.py

# Quick smoke test (2 languages, ~20 min on CPU):
PYTHONPATH=. python reproduce/run_nanolm_experiments.py --langs english hindi

# Dry run (print plans without training):
PYTHONPATH=. python reproduce/run_nanolm_experiments.py --dry-run
```

**Expected output:** Results saved to `reproduce/results/nanolm_results.json`. The script also runs an automatic ordering check to verify the paper's key claim:
- StructPiece Token PPL < SP-Unigram PPL < SP-BPE PPL ✓ for every language
- SP-Unigram BPC < SP-BPE BPC < StructPiece BPC ✓ for every language

Absolute PPL values will vary slightly from the paper (H100 vs. local hardware, seed averaging), but the **ordering is deterministic** and will reproduce.

---

### Phase 2: EGC & Morphology Ablation (Table 4)

```bash
PYTHONPATH=. python reproduce/run_core_reeval.py --langs hindi english
```

---

### Phase 3: Downstream Probes — NER, POS, XNLI (Tables 5, 8, 9, 14, 17)

```bash
# BiLSTM NER on WikiANN (Hindi, Arabic, Turkish, English):
PYTHONPATH=. python reproduce/ner_downstream.py

# mBERT NER fine-tuning (Table 5, GPU recommended):
PYTHONPATH=. python reproduce/ner_mbert.py --langs hindi,arabic,turkish,english

# Verify cached mBERT results without GPU:
PYTHONPATH=. python reproduce/ner_mbert.py --verify-only

# POS tagging + XNLI:
PYTHONPATH=. python reproduce/extended_downstream.py
```

*Requires HuggingFace `datasets` access for WikiANN and XNLI downloads.*

---

### Phase 4: Vocabulary Surgery (Tables 6, 7, 8) — GPU Cluster Required

```bash
# Step 1: Initialize surgery
PYTHONPATH=. python reproduce/scripts/surgery_init.py \
    --base-model meta-llama/Llama-3.2-1B \
    --structpiece-vocab reproduce/models/multilingual_structpiece \
    --output reproduce/results/surgery_init/

# Step 2: Continual pre-training
deepspeed --num_gpus=8 reproduce/scripts/run_continual_pretraining.py \
    --model_path reproduce/results/surgery_init/ \
    --data_dir datasets/ \
    --output_dir reproduce/results/surgery_cpt/

# Step 3: Evaluate
PYTHONPATH=. python reproduce/scripts/run_surgery_eval.py \
    --model_path reproduce/results/surgery_cpt/ \
    --variant "StructPiece (32k)"

# Verify cached results without GPU:
PYTHONPATH=. python reproduce/scripts/run_surgery_eval.py --verify-only
PYTHONPATH=. python reproduce/scripts/compute_cosine_coherence.py --verify-only
PYTHONPATH=. python reproduce/scripts/run_continual_pretraining.py --verify-only
```

---

### Phase 5: Morfessor & BPE-knockout Baselines (Table 16)

```bash
# Full training (requires morfessor package):
PYTHONPATH=. python reproduce/run_morfessor_baseline.py --lang turkish

# Verify cached results:
PYTHONPATH=. python reproduce/run_morfessor_baseline.py --verify-only
```

---

### Phase 6: Extended Scale Validation (Tables 15, 38)

```bash
# Full training (GPU recommended):
PYTHONPATH=. python reproduce/run_extended_scale.py --langs english,turkish

# Verify cached results:
PYTHONPATH=. python reproduce/run_extended_scale.py --verify-only
```

---

## Key Claims and Where to Find Evidence

| Paper Claim | Tables | Script | JSON Log |
|---|---|---|---|
| StructPiece PPL < SP-BPE PPL across 9 languages | 1, 10 | `run_nanolm_experiments.py` | `all_results.json` |
| StructPiece BPC > SP-BPE BPC (compression tradeoff) | 1, 10 | `run_nanolm_experiments.py` | `all_results.json` |
| Advantage persists under fixed token budget | 2 | `run_nanolm_experiments.py` | `all_results.json` |
| Advantage not explained by granularity alone | 3 | `run_nanolm_experiments.py` | `all_results.json` |
| EGC accounts for ~90% of PPL gain in ablation | 4 | `run_core_reeval.py` | `task3456_out.json` |
| NER F1 improves on MRLs, reverses on English | 5 | `ner_mbert.py` | `mbert_ner_results.json` |
| LLaMA surgery: StructPiece recovers 89.4% MRL | 6 | `scripts/run_surgery_eval.py` | `surgery_results.json` |
| Cosine coherence monotonically orders recovery | 7 | `scripts/compute_cosine_coherence.py` | `surgery_results.json` |
| StructPiece converges faster after surgery | 8 | `scripts/run_continual_pretraining.py` | `surgery_results.json` |
| Ordering holds at 50M/100M scale | 15, 38 | `run_extended_scale.py` | `scaling_results.json` |
| Morfessor-BPE improves but below StructPiece | 16 | `run_morfessor_baseline.py` | `morfessor_results.json` |

---

## Citation

```bibtex
@inproceedings{anonymous2026structpiece,
  title     = {Tokenization as Structural Inductive Bias: Grapheme-Constrained
               Subwords for Multilingual Modeling and Vocabulary Surgery},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of EMNLP},
  year      = {2026}
}
```
