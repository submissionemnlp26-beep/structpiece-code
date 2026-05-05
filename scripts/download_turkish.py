from datasets import load_dataset

out_path = "datasets/turkish_raw.txt"

max_lines = 750000
count = 0

# STREAMING = avoids HF script issues
ds = load_dataset("mc4", "tr", split="train", streaming=True)

with open(out_path, "w", encoding="utf-8") as f:
    for row in ds:
        if count >= max_lines:
            break

        text = row["text"].strip()

        if len(text) < 20:
            continue
        if len(text.split()) < 3:
            continue

        f.write(text + "\n")
        count += 1

print(f"Saved {count} lines to {out_path}")
