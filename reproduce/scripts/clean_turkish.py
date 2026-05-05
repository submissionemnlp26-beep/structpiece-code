import re

in_path = "datasets/turkish_raw.txt"
out_path = "datasets/turkish_clean.txt"

max_lines = 750000  # match Hindi size
count = 0

def is_valid_line(text):
    # Basic length checks
    if len(text) < 20:
        return False
    if len(text.split()) < 3:
        return False

    # Remove lines with too many numbers
    if sum(c.isdigit() for c in text) > len(text) * 0.3:
        return False

    # Remove lines with weird symbols
    if re.search(r"[^\w\s\.,!?;:()'\"-]", text):
        return False

    return True


with open(in_path, "r", encoding="utf-8") as f_in, \
     open(out_path, "w", encoding="utf-8") as f_out:

    for line in f_in:
        if count >= max_lines:
            break

        text = line.strip()

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        if not is_valid_line(text):
            continue

        f_out.write(text + "\n")
        count += 1

print(f"Saved {count} clean lines to {out_path}")
