"""
Multilingual Evaluation Pipeline
================================
Trains MorphTokenizer and SentencePiece baselines across 8 typologically diverse languages.
Evaluates Language Modeling Perplexity to generate table results for the paper.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import sentencepiece as spm
from rich.console import Console

# Add python/ and lm/ to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from python.adaptive_bpe import train_bpe, IndicTokenizer
from lm.train import train

console = Console()

LANGUAGES = [
    "hindi", "arabic", "tamil", "finnish", 
    "hungarian", "estonian", "georgian", "turkish"
]
VOCAB_SIZE = 8000
TRAIN_MAX_LINES = 10000

def get_corpus_path(lang: str) -> str:
    if lang == "hindi":
        return str(ROOT / "datasets" / "hindi_raw.txt")
    return str(ROOT / "datasets" / f"{lang}_clean.txt")


def generate_latex(results: list) -> str:
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Language modeling perplexity across eight typologically diverse languages.}")
    lines.append(r"\label{tab:multilingual_ppl}")
    lines.append(r"\begin{tabular}{l ccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Language} & \textbf{Standard BPE (PPL)} & \textbf{MorphTokenizer (PPL)} & \textbf{\% Reduction} \\")
    lines.append(r"\midrule")

    for r in results:
        lang = r["language"].capitalize()
        bpe_ppl = f"{r['sp_ppl']:.1f}" if r["sp_ppl"] != float('inf') else "High"
        morph_ppl = f"{r['morph_ppl']:.1f}" if r["morph_ppl"] != float('inf') else "High"
        
        red = 0.0
        if r['sp_ppl'] > 0 and r['sp_ppl'] != float('inf') and r['morph_ppl'] != float('inf'):
            red = ((r['sp_ppl'] - r['morph_ppl']) / r['sp_ppl']) * 100

        red_str = f"{-red:.1f}\\%" if red > 0 else f"+{-red:.1f}\\%"
        
        lines.append(f"{lang} & {bpe_ppl} & \\textbf{{{morph_ppl}}} & {red_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    console.rule("[bold magenta]MorphTokenizer Multilingual Evaluation Suite")
    
    models_dir = ROOT / "experiments" / "multilingual_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    final_results = []
    
    for lang in LANGUAGES:
        console.rule(f"[cyan]Processing Language: {lang.upper()}")
        corpus = get_corpus_path(lang)
        
        if not os.path.exists(corpus):
            console.print(f"[red]Missing dataset for {lang} at {corpus}[/red]")
            continue
            
        # 1. Train MorphTokenizer Context
        morph_dir = models_dir / f"{lang}_morph"
        if not (morph_dir / "indic_tokenizer_vocab.json").exists():
            console.print(f"[yellow]Training MorphTokenizer for {lang}...[/yellow]")
            vocab, merges = train_bpe(corpus, vocab_size=VOCAB_SIZE, max_lines=TRAIN_MAX_LINES, verbose=False)
            tok = IndicTokenizer(vocab, merges)
            tok.save(str(morph_dir))
            
        # 2. Train SentencePiece Context
        sp_prefix = str(models_dir / f"{lang}_sp_bpe")
        sp_model = sp_prefix + ".model"
        if not os.path.exists(sp_model):
            console.print(f"[yellow]Training SentencePiece BPE for {lang}...[/yellow]")
            spm.SentencePieceTrainer.Train(
                input=corpus,
                model_prefix=sp_prefix,
                vocab_size=VOCAB_SIZE,
                model_type="bpe",
                character_coverage=1.0,
                normalization_rule_name="nfkc",
            )
            
        # 3. Lang Model Training (SentencePiece)
        sp_out = str(models_dir / f"lm_results_{lang}_sp.json")
        console.print(f"[green]Evaluating SentencePiece LM for {lang}...[/green]")
        train(
            tokenizer_type="sp",
            model_path=sp_model,
            corpus_path=corpus,
            output_path=sp_out,
            block_size=128,
            batch_size=32,
            learning_rate=5e-4,
            epochs=args.epochs,
            max_steps=args.max_steps
        )
        
        with open(sp_out, "r") as f:
            sp_res = json.load(f)
            sp_ppl = sp_res["final_perplexity"]
            
        # 4. Lang Model Training (MorphTokenizer)
        morph_out = str(models_dir / f"lm_results_{lang}_morph.json")
        console.print(f"[green]Evaluating MorphTokenizer LM for {lang}...[/green]")
        train(
            tokenizer_type="indic",
            model_path=str(morph_dir),
            corpus_path=corpus,
            output_path=morph_out,
            block_size=128,
            batch_size=32,
            learning_rate=5e-4,
            epochs=args.epochs,
            max_steps=args.max_steps
        )
        
        with open(morph_out, "r") as f:
            morph_res = json.load(f)
            morph_ppl = morph_res["final_perplexity"]
            
        console.print(f"  [bold]Results ({lang}): SP PPL={sp_ppl:.1f} vs Morph PPL={morph_ppl:.1f}[/bold]")
        
        final_results.append({
            "language": lang,
            "sp_ppl": sp_ppl,
            "morph_ppl": morph_ppl
        })
        
    # Final aggregations
    out_json = str(ROOT / "experiments" / "multilingual_results.json")
    with open(out_json, "w") as f:
        json.dump(final_results, f, indent=2)
        
    latex = generate_latex(final_results)
    out_latex = str(ROOT / "experiments" / "multilingual_latex.tex")
    with open(out_latex, "w") as f:
        f.write(latex)
        
    console.print("\n[bold green]Evaluation Suite Complete![/bold green]")
    console.print(f"Results JSON  : {out_json}")
    console.print(f"LaTeX Table   : {out_latex}")
    print("\n" + latex)

if __name__ == "__main__":
    main()
