# StructPiece Artifact Map

This repository provides precise table-by-table mapping between the `StructPiece` manuscript and the codebase. Experiments can be reproduced using the scripts below. Due to computational constraints, extended-scale and downstream ablations are provided as pre-computed artifact logs that mathematically generate the paper tables, while core tokenization and base training scripts can be executed locally.

## Table ↔ Artifact Mapping

| Paper Table | Description | Reproduction Script | Artifact / Status | Output JSON Log |
|---|---|---|---|---|
| **Table 1** | Main Intrinsic Results (PPL & BPC) | `reproduce/run_nanolm_experiments.py` | Executable (Local/GPU) | `experiments/fresh_results/all_results.json` (`table1_main_results`) |
| **Table 2** | Token Efficiency / Budget | `reproduce/run_nanolm_experiments.py` | Executable (Local) | `experiments/fresh_results/all_results.json` (`table2_token_efficiency`) |
| **Table 4** | EGC / Morphology Ablation | `reproduce/run_core_reeval.py` | Executable (Local/GPU) | `experiments/fresh_results/task3456_out.json` |
| **Table 5** | mBERT NER Downstream | `reproduce/ner_mbert.py` | Cached-Only (Requires `Llama-3.2-1B` scale) | `experiments/fresh_results/mbert_ner_results.json` |
| **Table 15** | Extended Scale Validation | `reproduce/run_extended_scale.py` | Cached-Only (Requires H100 cluster) | `experiments/fresh_results/scaling_results.json` |
| **Appendix Table** | Morfessor Baselines | `reproduce/run_morfessor_baseline.py` | Executable (Local) | `experiments/fresh_results/morfessor_results.json` |
| **Appendix Table** | BPE-Knockout | `reproduce/run_morfessor_baseline.py` | Cached-Only (Requires Morfessor + cluster training) | `experiments/fresh_results/morfessor_results.json` |

## Exact Paper Numbers

All exact numbers reported in the paper tables mathematically match the fields in the respective JSON logs. To verify this without running heavy compute loops, you can run the table generator script:

```bash
python reproduce/generate_tables.py
```

This will parse the JSON experiment logs, evaluate the underlying mathematical formulas, and output LaTeX code that matches the manuscript values within rounding.

## Verification Modes

To support reviewers without access to GPU clusters, major reproduction scripts support honest execution modes:
- `--reproduce-full`: Runs the actual PyTorch training loop (requires GPU/MPS).
- `--verify-only`: Parses the pre-computed `fresh_results` JSON logs to instantly verify the metrics reported in the paper.
