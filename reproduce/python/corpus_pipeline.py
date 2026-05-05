"""
Corpus Pipeline for IndicTokenizer
Builds a clean Hindi text corpus from ai4bharat/IndicCorpV2.
"""

import os
import argparse
import unicodedata
from datasets import load_dataset
from tqdm import tqdm


def is_devanagari(char: str) -> bool:
    """Check if a character falls in the Devanagari Unicode block."""
    # Devanagari block: u0900 - u097F
    return '\u0900' <= char <= '\u097F'


def clean_text(text: str) -> str | None:
    """
    Applies cleaning rules to a single text line.
    Returns the cleaned line or None if it should be discarded.
    """
    if not text:
        return None

    text = text.strip()

    # 1. Remove empty strings & discard very short lines (<15 chars)
    if len(text) < 15:
        return None

    # 2. Unicode normalization (NFC)
    text = unicodedata.normalize('NFC', text)

    # 3. Discard lines where <40% of characters are Devanagari
    # Calculate non-whitespace length to be more accurate, or just total characters
    # We'll use total characters as a simple heuristic
    devanagari_count = sum(1 for char in text if is_devanagari(char))

    if devanagari_count / len(text) < 0.4:
        return None

    return text


def build_corpus(
    dataset_name: str,
    config_name: str,
    output_file: str,
    target_lines: int
):
    """Downloads, cleans, and saves the text corpus."""
    print(f"Loading dataset: {dataset_name} (config: {config_name})")

    try:
        # We use streaming = True because IndicCorpV2 is humongous
        # The configuration name for IndicCorpV2 is 'indiccorp_v2' and the split is 'hin_Deva' etc.
        # It turns out 'hin_Deva' is the split name.
        if dataset_name == "ai4bharat/IndicCorpV2":
            # For IndicCorpV2, the config is 'indiccorp_v2' and the split is the language.
            dataset = load_dataset(dataset_name, "indiccorp_v2", split=config_name, streaming=True)
        else:
            dataset = load_dataset(dataset_name, name=config_name, split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Note: Make sure the config name is correct (e.g., 'hi' or 'hin_Deva').")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Writing clean corpus to {output_file} (Target: {target_lines} lines)")

    written_lines = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        # Using a tqdm progress bar up to target_lines
        with tqdm(total=target_lines, desc="Processing lines") as pbar:
            for item in dataset:
                raw_text = item.get("text", "")

                # Split by newline in case text contains multiple sentences
                lines = raw_text.split('\n')

                for line in lines:
                    cleaned_line = clean_text(line)

                    if cleaned_line:
                        f.write(cleaned_line + '\n')
                        written_lines += 1
                        pbar.update(1)

                        if written_lines >= target_lines:
                            break

                if written_lines >= target_lines:
                    break

    print(f"Done! Successfully wrote {written_lines} lines to {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and build the Hindi text corpus.")
    parser.add_argument("--dataset", type=str, default="ai4bharat/IndicCorpV2", help="HuggingFace dataset name")
    parser.add_argument("--config", type=str, default="hi", help="Dataset config/subset (e.g., 'hi' or 'hin_Deva')")
    parser.add_argument("--output", type=str, default="datasets/hindi_raw.txt", help="Output text file path")
    parser.add_argument("--lines", type=int, default=1_000_000, help="Target number of cleaned lines to write")

    args = parser.parse_args()

    build_corpus(
        dataset_name=args.dataset,
        config_name=args.config,
        output_file=args.output,
        target_lines=args.lines
    )
