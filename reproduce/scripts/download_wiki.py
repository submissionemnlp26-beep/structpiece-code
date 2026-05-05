from datasets import load_dataset
import unicodedata
import re
from tqdm import tqdm


TARGET_KB = 450


def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 20 or len(text) > 5000:
        return None

    return text


def build_dataset(lang_code, output_file):
    print(f"\nDownloading {lang_code}...")

    ds = load_dataset(
        "wikimedia/wikipedia",
        f"20231101.{lang_code}",
        split="train",
        streaming=True   # 🔥 important fix
    )

    target_bytes = TARGET_KB * 1024
    total_bytes = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc=f"{lang_code}"):

            text = clean_text(row["text"])
            if not text:
                continue

            line = text + "\n"
            encoded = line.encode("utf-8")

            if total_bytes + len(encoded) > target_bytes:
                break

            f.write(line)
            total_bytes += len(encoded)

    print(f"Saved {output_file} → {total_bytes/1024:.2f} KB")


if __name__ == "__main__":
    build_dataset("fi", "datasets/finnish_clean.txt")
    build_dataset("hu", "datasets/hungarian_clean.txt")
    build_dataset("ta", "datasets/tamil_clean.txt")
