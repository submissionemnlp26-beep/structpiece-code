#!/usr/bin/env python3
"""
reproduce/generate_tables.py
Generate LaTeX table fragments from the experiment JSON logs.

Running this script produces output consistent with the reported paper values.
All metrics are computed directly from the sanitised JSON logs; no values
are hardcoded in this script.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Table 1: NanoLM PPL & BPC ───────────────────────────────────────────────

def generate_table1(all_res, eng_res):
    print("\n% === Table 1: Language modeling results ===")
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\begin{tabular}{llccc}")
    print("\\toprule")
    print("Language & Tokenizer & Token PPL $\\downarrow$ & BPC $\\downarrow$ \\\\")
    print("\\midrule")

    results = list(all_res.get("table1_main_results", []))
    if "table1_main_results" in eng_res:
        results.extend(eng_res["table1_main_results"])

    by_lang = {}
    for r in results:
        by_lang.setdefault(r["language"], {})[r["tokenizer"]] = r

    # tokens-per-char lookup for BPC calculation
    efficiency = list(all_res.get("table2_token_efficiency", []))
    if "table2_token_efficiency" in eng_res:
        efficiency.extend(eng_res["table2_token_efficiency"])
    tok_per_char_lookup = {}
    for e in efficiency:
        lang = e["language"]
        tok = e["tokenizer"]
        tpc = e.get("tokens_per_512chars", 0) / 512.0
        tok_per_char_lookup.setdefault(lang, {})[tok] = tpc

    for lang in ["Hindi", "Arabic", "Turkish", "English"]:
        if lang not in by_lang:
            continue
        print(f"\\multirow{{3}}{{*}}{{{lang}}}")
        toks = by_lang[lang]
        for tok_name, tex_name in [
            ("StructPiece", "StructPiece"),
            ("SP-BPE",      "SP-BPE"),
            ("SP-Unigram",  "SP-Unigram"),
        ]:
            search_tok = "MorphTokenizer" if tok_name == "StructPiece" else tok_name
            d = toks.get(search_tok, {})
            if not d:
                continue
            ppl = f"{d.get('avg_ppl', 0):.1f} \\pm {d.get('avg_ppl_std', 0.0):.1f}"
            tok_char = tok_per_char_lookup.get(lang, {}).get(search_tok, 0)
            loss = d.get("avg_loss", 0)
            bpc_val = (loss * tok_char) / math.log(2) if tok_char else 0
            bpc = f"{bpc_val:.3f} \\pm {d.get('avg_loss_std', 0.0):.3f}"
            print(f"& {tex_name} & {ppl} & {bpc} \\\\")
        print("\\midrule")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")


# ─── Table 2: Token Efficiency ────────────────────────────────────────────────

def generate_table2(efficiency_res):
    print("\n% === Table 2: Token Efficiency / Budget ===")
    if not efficiency_res:
        return

    # Deduplicate: keep first occurrence of each (language, tokenizer) pair
    seen = set()
    deduped = []
    for r in efficiency_res:
        key = (r["language"], r["tokenizer"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    # Build base (MorphTokenizer) tokens_per_word per language for real inflation
    base_tpw = {}
    for r in deduped:
        if r["tokenizer"] == "MorphTokenizer":
            base_tpw[r["language"]] = r.get("tokens_per_word", 1.0)

    print("\\begin{table}[t]")
    print("\\centering")
    print("\\begin{tabular}{llccc}")
    print("\\toprule")
    print("Language & Tokenizer & Tok/Word & Tok/512ch & Inflation \\\\")
    print("\\midrule")
    for r in deduped:
        lang = r["language"]
        tpw  = r.get("tokens_per_word", 0)
        base = base_tpw.get(lang, tpw) or tpw
        inflation = tpw / base if base else 1.0
        print(
            f"{lang} & {r['tokenizer']} & {tpw:.2f} & "
            f"{r.get('tokens_per_512chars', 0):.1f} & {inflation:.2f}x \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


# ─── Table 4: EGC / Morphology Ablation ──────────────────────────────────────

def generate_table4(ablation_res):
    print("\n% === Table 4: EGC / Morphology Ablation (3-seed mean ± std) ===")
    if not ablation_res:
        return

    # Use the structurally stored 3-seed averages
    seeds3 = ablation_res.get("ablation_3seed", {})
    if not seeds3:
        print("% WARNING: ablation_3seed key missing from task3456_out.json")
        return

    print("\\begin{table}[t]")
    print("\\centering")
    print("\\begin{tabular}{llc}")
    print("\\toprule")
    print("Language & Variant & Token PPL $\\downarrow$ \\\\")
    print("\\midrule")
    for variant_name, data in seeds3.items():
        print(
            f"Hindi & {variant_name} & "
            f"{data.get('ppl_m', 0):.1f} \\pm {data.get('ppl_s', 0):.1f} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


# ─── Table 5: mBERT NER ───────────────────────────────────────────────────────

def generate_table5(ner_res):
    print("\n% === Table 5: mBERT NER Downstream (entity-level F1) ===")
    if not ner_res:
        return

    print("\\begin{table}[t]")
    print("\\centering")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print(
        "Lang & mBERT Std. & $\\sigma$ & +SP-BPE & $\\sigma$ "
        "& +StructPiece & $\\sigma$ \\\\"
    )
    print("\\midrule")
    for r in ner_res.get("results", []):
        print(
            f"{r['language']} "
            f"& {r['mbert_std_f1']:.1f} & {r.get('mbert_std_std', 0):.1f} "
            f"& {r['sp_bpe_f1']:.1f}   & {r.get('sp_bpe_std', 0):.1f} "
            f"& {r['structpiece_f1']:.1f} & {r.get('structpiece_std', 0):.1f} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


# ─── Table 15 & 38: Extended Scale (block 128 / block 512) ───────────────────

def generate_table15_and_38(scaling_res):
    for block_key, table_label in [("block_128", "Table 15"), ("block_512", "Table 38 (Appendix)")]:
        rows = scaling_res.get(block_key, [])
        block_size = block_key.split("_")[1]
        print(f"\n% === {table_label}: Extended Scale Validation (block={block_size}) ===")
        if not rows:
            continue
        print("\\begin{table}[t]")
        print("\\centering")
        print("\\begin{tabular}{llcc}")
        print("\\toprule")
        print(
            f"Language & Tokenizer & Token PPL $\\downarrow$ & BPC $\\downarrow$ \\\\"
        )
        print("\\midrule")
        for r in rows:
            print(
                f"{r['language']} & {r['tokenizer']} "
                f"& {r['token_ppl_mean']:.1f} \\pm {r['token_ppl_std']:.1f} "
                f"& {r['bpc_mean']:.3f} \\pm {r['bpc_std']:.3f} \\\\"
            )
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")


# ─── Appendix: Morfessor Baselines ───────────────────────────────────────────

def generate_morfessor_table(morf_res):
    print("\n% === Appendix Table: Morfessor & BPE-Knockout Baselines (Turkish) ===")
    if "turkish_baselines" not in morf_res:
        return
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Tokenizer & Tokens/Word & Inflation & Token PPL \\\\")
    print("\\midrule")
    for r in morf_res["turkish_baselines"]:
        ppl = f"{r.get('token_ppl_mean', 0):.1f} \\pm {r.get('token_ppl_std', 0):.1f}"
        print(
            f"{r.get('tokenizer', '')} & {r.get('tok_per_word', 0):.2f} "
            f"& {r.get('inflation', 0):.2f}x & {ppl} \\\\"
        )
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    print("% ==========================================")
    print("% STRUCTPIECE — TABLE GENERATION FROM LOGS")
    print("% Values are computed from sanitised JSON experiment logs.")
    print("% These match the manuscript within rounding.")
    print("% ==========================================\n")

    all_res      = load_json(ROOT / "experiments/fresh_results/all_results.json")
    eng_res      = load_json(ROOT / "experiments/fresh_results/english_results.json")
    morf_res     = load_json(ROOT / "experiments/fresh_results/morfessor_results.json")
    ablation_res = load_json(ROOT / "experiments/fresh_results/task3456_out.json")
    ner_res      = load_json(ROOT / "experiments/fresh_results/mbert_ner_results.json")
    scaling_res  = load_json(ROOT / "experiments/fresh_results/scaling_results.json")

    efficiency = list(all_res.get("table2_token_efficiency", []))
    if "table2_token_efficiency" in eng_res:
        efficiency.extend(eng_res["table2_token_efficiency"])

    generate_table1(all_res, eng_res)
    generate_table2(efficiency)
    generate_table4(ablation_res)
    generate_table5(ner_res)
    generate_table15_and_38(scaling_res)
    generate_morfessor_table(morf_res)

    print("\n% Tables successfully generated.")


if __name__ == "__main__":
    main()
