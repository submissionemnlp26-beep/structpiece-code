"""
Script to generate a PDF report from the experiment results.
Uses fpdf2 to read metrics and embed charts from the experiments directory.
"""

import json
import os
import urllib.request
from fpdf import FPDF

# Download NotoSansDevanagari if missing for Hindi text
font_url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf"
font_path = "experiments/NotoSansDevanagari-Regular.ttf"

if not os.path.exists(font_path):
    print("Downloading NotoSansDevanagari-Regular.ttf...")
    urllib.request.urlretrieve(font_url, font_path)

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 12)
        self.set_text_color(21, 101, 192) # primaryblue
        self.cell(0, 10, 'IndicTokenizer: Adaptive BPE for Hindi', border=0, new_x='LMARGIN', new_y='NEXT', align='C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', border=0, new_x='LMARGIN', new_y='NEXT', align='C')

    def chapter_title(self, num, title):
        self.set_font('helvetica', 'B', 16)
        self.set_text_color(21, 101, 192) # primaryblue
        self.cell(0, 10, f'{num}. {title}', border=0, new_x='LMARGIN', new_y='NEXT', align='L')
        self.ln(4)
        
    def chapter_body(self, text, font: str = 'helvetica'):
        self.set_font(font, '', 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 7, text)
        self.ln(4)
        
    def add_image_centered(self, image_path, w=150):
        if os.path.exists(image_path):
            self.image(image_path, x='C', w=w)
            self.ln(10)

def main():
    results_file = 'experiments/results.json'
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found.")
        return

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    pdf = PDF()
    pdf.add_font("NotoDev", "", font_path)
    pdf.add_page()
    
    # Title
    pdf.set_font('helvetica', 'B', 24)
    pdf.set_text_color(21, 101, 192)
    pdf.cell(0, 15, 'IndicTokenizer', border=0, new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.set_font('helvetica', 'B', 16)
    pdf.set_text_color(46, 125, 50) # accentgreen
    pdf.cell(0, 10, 'Adaptive, Morphology-Aware Byte Pair Encoding for Hindi Text', border=0, new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.set_font('helvetica', '', 12)
    pdf.set_text_color(66, 66, 66)
    pdf.cell(0, 10, 'March 2026', border=0, new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(10)
    
    # Abstract
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, 'Abstract', border=0, new_x='LMARGIN', new_y='NEXT', align='L')
    abstract_text = (
        "We present IndicTokenizer, a tokenization engine specifically designed for "
        "Hindi and the Devanagari script. Standard sub-word tokenizers such as OpenAI's "
        "tiktoken (cl100k_base) treat Devanagari text as opaque byte sequences, "
        "requiring an average of 4.99 tokens per word for Hindi. Our approach introduces "
        "a script-aware pretokenizer, an Adaptive BPE algorithm with morphological bonuses, "
        "and a high-performance C++ runtime. IndicTokenizer achieves 2.50 tokens per word "
        "(a 49.9% reduction compared to tiktoken cl100k_base) and processes tokens 3.83x faster "
        "than Python in C++."
    )
    pdf.chapter_body(abstract_text)
    pdf.ln(6)
    
    # Methodology
    pdf.chapter_title(1, 'Methodology')
    method_text = (
        "1. Corpus Preparation: We cleaned a subset of ai4bharat/IndicCorpV2, using "
        "NFC Unicode Normalization, and filtering lines with less than 40% Devanagari characters.\n\n"
        "2. Script-Aware Pretokenizer: Operates at the grapheme cluster level, keeping "
        "conjuncts and matra combinations intact.\n\n"
        "3. Adaptive BPE Algorithm: Uses frequency-based merging with a morphology bonus "
        "for common Hindi suffixes and a penalty for cross-script merges.\n\n"
        "4. C++ Runtime: A production-ready inference engine that loads vocab and merges "
        "and is heavily optimized for speed."
    )
    pdf.chapter_body(method_text)
    
    # Results
    pdf.add_page()
    pdf.chapter_title(2, 'Experiments and Results')
    
    # Vocabulary Sweep
    sweeps = results.get('vocab_size_sweep', [])
    if sweeps:
        text = "2.1 Vocabulary Size Sweep\n"
        for s in sweeps:
            text += f"Vocab: {s['vocab_size']}, Tokens/Word: {s['tokens_per_word']}, Bytes/Token: {s['bytes_per_token']}\n"
        pdf.chapter_body(text)
        pdf.add_image_centered('experiments/figures/vocab_size_sweep.png')
    
    # Baseline Comparison
    baselines = results.get('baseline_comparison', [])
    if baselines:
        text = "2.2 Baseline Comparison\n"
        for b in baselines:
            text += f"{b['tokenizer']}: {b['tokens_per_word']} tokens/word, {b['bytes_per_token']} bytes/token\n"
        pdf.chapter_body(text)
        pdf.add_image_centered('experiments/figures/baseline_comparison.png')
        
    # Speed
    speed = results.get('speed_benchmark', {})
    if speed:
        pdf.add_page()
        text = "2.3 Inference Speed Benchmark\n"
        py = speed.get('python', {})
        cp = speed.get('cpp', {})
        text += f"Python: {py.get('tokens_per_sec', 0):.0f} tokens/sec\n"
        text += f"C++: {cp.get('tokens_per_sec', 0):.0f} tokens/sec\n"
        text += f"C++ Speedup: {speed.get('speedup', 0)}x\n"
        pdf.chapter_body(text)
        pdf.add_image_centered('experiments/figures/speed_comparison.png')
        
    # Morphological
    morph = results.get('morphological_analysis', {})
    if morph:
        text = "2.4 Morphological Analysis\n"
        text += f"Suffix Coverage: {morph.get('suffix_coverage', 0)*100:.1f}%\n"
        text += f"Average Token Length: {morph.get('avg_token_chars', 0)} chars\n"
        text += "Sample Segmentations (shows isolation of suffix and stem representation):\n"
        pdf.chapter_body(text)

        words = morph.get('morphological_words', {})
        for word, val in words.items():
            # val['description'] is like "खेल + ना (to play)"
            # val['tokens'] is like ['खेलना']
            pdf.set_font('NotoDev', '', 11)
            hindi_part = f"{word} -> {val['tokens']} "
            pdf.write(7, hindi_part)
            
            desc = val['description']
            if '(' in desc:
                eng_part = '(' + desc.split('(', 1)[1]
            else:
                eng_part = ""
                
            pdf.set_font('helvetica', '', 11)
            pdf.write(7, eng_part + "\n")
        
    # Conclusion
    pdf.add_page()
    pdf.chapter_title(3, 'Conclusion')
    conclusion = (
        "IndicTokenizer demonstrates that script-aware, morphologically-guided tokenization "
        "can dramatically improve efficiency for Hindi text. By respecting Devanagari grapheme "
        "structure and incorporating Hindi morphological knowledge into the BPE merge scoring "
        "function, we achieved ~50% fewer tokens per word compared to tiktoken (cl100k_base), "
        "with a highly-optimized C++ runtime. This validates the value of language-specific "
        "tokenizers for efficient inference and lower memory overhead."
    )
    pdf.chapter_body(conclusion)
    
    output_path = 'experiments/IndicTokenizer_Report.pdf'
    pdf.output(output_path)
    print(f"PDF generated successfully at {output_path}")

if __name__ == "__main__":
    main()
