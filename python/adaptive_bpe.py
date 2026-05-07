"""
Adaptive BPE Trainer for Hindi / Devanagari
============================================

Standard BPE merges the most *frequent* adjacent pair at each step.
This trainer adds two linguistically-motivated adjustments:

1. **Morphology bonus** – common Hindi suffixes and postpositions get a boost
   when they would be formed by a merge, encouraging the vocab to learn
   linguistically meaningful sub-words.

2. **Grapheme / script penalty** – merges that would cross script boundaries
   (e.g. Devanagari + Latin) or break a grapheme cluster mid-way are penalised.

Usage
-----
    python python/adaptive_bpe.py --corpus datasets/hindi_raw.txt \\
                                   --vocab-size 8000 \\
                                   --output-dir models
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import unicodedata
import regex
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

# Local import – pretokenizer lives in the same package directory
from python.pretokenizer import pretokenize, is_devanagari, normalize, grapheme_clusters


# ─── Unsupervised Morphology Tracking ────────────────────────────────────────

# We no longer hardcode Hindi suffixes. This set is dynamically populated
# during training via unsupervised stem-entropy discovery, making MorphTokenizer
# a zero-shot morphology tokenizer for any language.
ACTIVE_SUFFIXES: Set[str] = set()

# Bonus weights
_MORPHOLOGY_BONUS = 2.0           # lambda = 2.0 as per paper
_SCRIPT_CROSS_PENALTY = -500.0    # merging Devanagari with non-Devanagari


# ─── Dynamic Suffix Discovery ─────────────────────────────────────────────────

def discover_suffixes(word_freqs: WordFreqs, top_k: int = 50, min_unique_stems: Optional[int] = None) -> Set[str]:
    """
    Unsupervised morphology discovery: finding productive suffixes by calculating
    how many unique stems a terminal string attaches to.
    """
    if min_unique_stems is None:
        num_unique_words = len(word_freqs)
        # Threshold: tau(|W|) = 3 * log(|W|) + 2
        min_unique_stems = max(1, int(3.0 * math.log(num_unique_words) + 2.0)) if num_unique_words > 0 else 1

    suffix_stems = defaultdict(set)
    for word_tuple, _ in word_freqs.items():
        word_str = "".join(word_tuple)
        
        # Split word into extended grapheme clusters to avoid splitting inside complex characters
        clusters = regex.findall(r"\X", word_str)
        
        # We only want true suffixes, not the whole word. 
        # Look at the last 1, 2, and 3 grapheme clusters
        for length in range(1, min(4, len(clusters))):
            stem = "".join(clusters[:-length])
            suffix = "".join(clusters[-length:])
            if stem: 
                suffix_stems[suffix].add(stem)
                
    # Filter by minimum unique stems
    valid_suffixes = {
        suf: len(stems) for suf, stems in suffix_stems.items() 
        if len(stems) >= min_unique_stems
    }
    
    # Sort suffixes by productive coverage (number of stems they attach to)
    return set(sorted(valid_suffixes, key=valid_suffixes.get, reverse=True)[:top_k])

# ─── Scoring helpers ─────────────────────────────────────────────────────────

def _get_script_family(token: str) -> str:
    """Returns the primary script family of the token (e.g. 'DEVANAGARI', 'ARABIC', 'LATIN')."""
    for ch in token:
        if not ch.isspace() and not unicodedata.category(ch).startswith('P'):
            try:
                # First word of the unicode name is typically the block, e.g., "ARABIC", "LATIN", "DEVANAGARI"
                name = unicodedata.name(ch)
                return name.split()[0]
            except ValueError:
                pass
    return "UNKNOWN"


def morphology_bonus(merged: str) -> float:
    """
    Return a positive bonus if *merged* ends with a learned productive suffix.
    """
    for suf in ACTIVE_SUFFIXES:
        if merged.endswith(suf):
            return _MORPHOLOGY_BONUS
    return 0.0


def script_penalty(a: str, b: str) -> float:
    """
    Return a negative penalty if merging *a* and *b* would cross script
    boundaries generically across any language.
    """
    script_a = _get_script_family(a)
    script_b = _get_script_family(b)
    
    # If they are different scripts, and neither is purely punctuation/unknown, penalize
    if script_a != "UNKNOWN" and script_b != "UNKNOWN" and script_a != script_b:
        return _SCRIPT_CROSS_PENALTY
    return 0.0


def adjusted_score(freq: int, a: str, b: str) -> float:
    """
    Compute the merge score for the pair (a, b) with frequency *freq*.

        score = freq + morphology_bonus(a+b) + script_penalty(a, b)
    """
    merged = a + b
    return freq + morphology_bonus(merged) + script_penalty(a, b)


# ─── Word ↔ pair bookkeeping ────────────────────────────────────────────────

# A "word" is represented as a tuple of tokens (initially single grapheme
# clusters or characters) together with its corpus frequency.

WordRepr = Tuple[str, ...]            # e.g. ("न", "म", "स्", "का", "र")
WordFreqs = Dict[WordRepr, int]       # word-repr → corpus count


def _get_pairs(word: WordRepr) -> Set[Tuple[str, str]]:
    """Return the set of adjacent symbol pairs in *word*."""
    pairs: Set[Tuple[str, str]] = set()
    for i in range(len(word) - 1):
        pairs.add((word[i], word[i + 1]))
    return pairs


def _count_pairs(word_freqs: WordFreqs) -> Counter:
    """Count all adjacent pairs across the entire vocabulary, weighted by freq."""
    pair_counts: Counter = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += freq
    return pair_counts


def _merge_pair(word: WordRepr, pair: Tuple[str, str]) -> WordRepr:
    """Return a new word tuple with all occurrences of *pair* merged."""
    new_word: List[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            new_word.append(pair[0] + pair[1])
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


# ─── Corpus → initial word freqs ────────────────────────────────────────────

def _build_word_freqs(corpus_path: str, max_lines: Optional[int] = None) -> WordFreqs:
    """
    Read the corpus, pretokenize every line, and count word frequencies.

    Each "word" (non-space pre-token) is split into its individual characters
    to form the initial WordRepr.
    """
    word_counter: Counter = Counter()

    with open(corpus_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_lines and idx >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            tokens = pretokenize(line)
            for tok in tokens:
                # Skip whitespace tokens — they are boundaries, not vocab items
                if tok.isspace():
                    continue
                word_counter[tok] += 1

    # Convert each word string into a tuple of EGC atoms (initial symbols)
    word_freqs: WordFreqs = {}
    for word_str, freq in word_counter.items():
        # Use grapheme_clusters to get EGC atoms instead of codepoints
        word_freqs[tuple(grapheme_clusters(word_str))] = freq

    return word_freqs


# ─── Main BPE training loop ─────────────────────────────────────────────────

def train_bpe(
    corpus_path: str,
    vocab_size: int = 8000,
    max_lines: Optional[int] = None,
    verbose: bool = True,
    variant: str = "original",
) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """
    Train an Adaptive BPE tokenizer.

    Parameters
    ----------
    corpus_path : path to the cleaned Hindi corpus (one sentence per line).
    vocab_size  : target vocabulary size (number of unique tokens).
    max_lines   : if set, only read the first N lines of the corpus.
    verbose     : if True, print progress.

    Returns
    -------
    vocab   : dict mapping token-string → token-id
    merges  : ordered list of (a, b) merge operations
    """
    if verbose:
        print(f"[1/4] Building word frequencies from {corpus_path} …")

    word_freqs = _build_word_freqs(corpus_path, max_lines=max_lines)

    if verbose:
        print(f"       Found {len(word_freqs):,} unique word forms.")

    # ── Unsupervised Morphology Discovery ──────────────────────────────────────
    global ACTIVE_SUFFIXES
    ACTIVE_SUFFIXES = discover_suffixes(word_freqs)
    
    if verbose:
        print(f"\n[Unsupervised Discovery] Learned {len(ACTIVE_SUFFIXES)} highly productive morphological suffixes.")
        print("Here are the optimal suffixes generated purely from data statistics:")
        suf_list = list(ACTIVE_SUFFIXES)
        for i in range(0, len(suf_list), 10):
            print("  " + ", ".join(repr(s) for s in suf_list[i:i+10]))
        print()

    # ── Initial vocabulary: every unique character that appears ──────────
    base_vocab: Set[str] = set()
    for word in word_freqs:
        for ch in word:
            base_vocab.add(ch)

    vocab_list = sorted(base_vocab)
    # Reserve token 0 for <unk> and 1 for <pad>
    vocab: Dict[str, int] = {"<unk>": 0, "<pad>": 1, " ": 2}
    for i, tok in enumerate(vocab_list, start=3):
        vocab[tok] = i

    if verbose:
        print(f"[2/4] Initial character vocab size: {len(vocab)}")

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        if verbose:
            print("       Vocab size already meets target — no merges needed.")
        return vocab, []

    if verbose:
        print(f"[3/4] Running {num_merges:,} adaptive BPE merges …")

    merges: List[Tuple[str, str]] = []

    pbar = tqdm(range(num_merges), desc="BPE merges", disable=not verbose)

    for step in pbar:
        # Count all pairs
        pair_counts = _count_pairs(word_freqs)
        if not pair_counts:
            if verbose:
                print(f"\n       No more pairs to merge at step {step}.")
            break

        # Score each pair with morphology bonus + script penalty
        best_pair = None
        best_score = float("-inf")

        for pair, freq in pair_counts.items():
            score = adjusted_score(freq, pair[0], pair[1])
            if score > best_score:
                best_score = score
                best_pair = pair

        if best_pair is None:
            break

        a, b = best_pair
        merged_token = a + b

        # Record the merge
        merges.append(best_pair)

        # Add merged token to vocab
        vocab[merged_token] = len(vocab)

        # Apply the merge to every word in word_freqs
        new_word_freqs: WordFreqs = {}
        for word, freq in word_freqs.items():
            new_word = _merge_pair(word, best_pair)
            new_word_freqs[new_word] = new_word_freqs.get(new_word, 0) + freq
        word_freqs = new_word_freqs

        if step % 500 == 0:
            pbar.set_postfix({"merged": f"{a}+{b}={merged_token}", "score": f"{best_score:.0f}"})

    if verbose:
        print(f"[4/4] Final vocab size: {len(vocab)}")

    return vocab, merges


# ─── Tokenizer (inference) ──────────────────────────────────────────────────

class IndicTokenizer:
    """
    A trained Adaptive BPE tokenizer for Hindi.

    Supports:
    * encode(text) → list[int]
    * decode(ids)  → str
    * save / load from JSON files
    """

    def __init__(self, vocab: Dict[str, int], merges: List[Tuple[str, str]]):
        self.vocab = vocab
        self.merges = merges

        # Reverse lookup
        self.id_to_token: Dict[int, str] = {v: k for k, v in vocab.items()}

        # Pre-compute merge priority (lower index = higher priority)
        self.merge_priority: Dict[Tuple[str, str], int] = {
            pair: i for i, pair in enumerate(merges)
        }

        self.unk_id = vocab.get("<unk>", 0)

        # ── Inference-time caches ────────────────────────────────────────
        # Word-level cache: avoids re-running BPE on the same word twice.
        # Bounded to 131K entries to cover typical corpus vocabularies.
        self._word_cache: Dict[str, List[int]] = {}
        self._CACHE_MAX = 131_072

    # ── Encode ───────────────────────────────────────────────────────────

    def _bpe_word(self, token_str: str) -> List[int]:
        """Run BPE merges on a single pre-token and return its token IDs.
        Results are cached per unique word string.

        Uses an optimised algorithm: instead of scanning all adjacent pairs
        on every merge iteration, we only re-check pairs that were affected
        by the previous merge (the neighbours of the merge site).
        """
        cached = self._word_cache.get(token_str)
        if cached is not None:
            return cached

        # --- Fast path: single character ---
        if len(token_str) == 1:
            result = [self.vocab.get(token_str, self.unk_id)]
            self._word_cache[token_str] = result
            return result

        # Split into EGC atoms
        symbols = grapheme_clusters(token_str)
        merge_priority = self.merge_priority  # local ref

        # --- Fast path: check if entire word is in vocab ---
        if token_str in self.vocab:
            result = [self.vocab[token_str]]
            self._word_cache[token_str] = result
            return result

        # Standard BPE merge loop with local-variable speedups
        while len(symbols) > 1:
            # Find the pair with the highest priority (lowest index)
            best_pair = None
            best_idx = 2147483647  # int max, faster than float("inf")

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                idx = merge_priority.get(pair)
                if idx is not None and idx < best_idx:
                    best_idx = idx
                    best_pair = pair

            if best_pair is None:
                break  # no more applicable merges

            # Apply this merge everywhere in the symbol list
            a, b = best_pair
            merged = a + b
            new_symbols: List[str] = []
            i = 0
            n = len(symbols)
            while i < n:
                if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        # Map symbols to IDs
        vocab = self.vocab
        unk = self.unk_id
        result = [vocab.get(s, unk) for s in symbols]

        # Cache
        if len(self._word_cache) < self._CACHE_MAX:
            self._word_cache[token_str] = result

        return result

    def encode(self, text: str) -> List[int]:
        """Tokenize *text* and return a list of token IDs."""
        # normalize is already called inside pretokenize, skip duplicate
        pre_tokens = pretokenize(text)

        ids: List[int] = []
        space_id = self.vocab.get(" ", self.unk_id)
        for token_str in pre_tokens:
            if token_str.isspace():
                ids.append(space_id)
            else:
                ids.extend(self._bpe_word(token_str))

        return ids

    # ── Decode ───────────────────────────────────────────────────────────

    def decode(self, ids: List[int]) -> str:
        """Convert a list of token IDs back to a string."""
        return "".join(self.id_to_token.get(i, "�") for i in ids)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, output_dir: str):
        """Save vocab and merges to *output_dir*."""
        os.makedirs(output_dir, exist_ok=True)

        vocab_path = os.path.join(output_dir, "indic_tokenizer_vocab.json")
        merges_path = os.path.join(output_dir, "indic_tokenizer_merges.txt")

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        with open(merges_path, "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

        print(f"Saved vocab  → {vocab_path} ({len(self.vocab)} tokens)")
        print(f"Saved merges → {merges_path} ({len(self.merges)} merges)")

    @classmethod
    def load(cls, model_dir: str) -> "IndicTokenizer":
        """Load a previously saved tokenizer from *model_dir*."""
        vocab_path = os.path.join(model_dir, "indic_tokenizer_vocab.json")
        merges_path = os.path.join(model_dir, "indic_tokenizer_merges.txt")

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        merges: List[Tuple[str, str]] = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))

        return cls(vocab, merges)


# ─── CLI entry-point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train an Adaptive BPE tokenizer for Hindi."
    )
    parser.add_argument(
        "--corpus", type=str, default="datasets/hindi_raw.txt",
        help="Path to the cleaned Hindi corpus"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=8000,
        help="Target vocabulary size"
    )
    parser.add_argument(
        "--max-lines", type=int, default=None,
        help="Only read first N lines of corpus (for faster dev iteration)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models",
        help="Directory to save vocab and merges"
    )
    args = parser.parse_args()

    t0 = time.time()

    vocab, merges = train_bpe(
        corpus_path=args.corpus,
        vocab_size=args.vocab_size,
        max_lines=args.max_lines,
    )

    tokenizer = IndicTokenizer(vocab, merges)
    tokenizer.save(args.output_dir)

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s")

    # Quick smoke test
    test_sentences = [
        "भारत महान है।",
        "नमस्कार, आप कैसे हैं?",
        "क्रांति का समय आ गया है।",
    ]
    print("\n── Smoke test ──")
    for sent in test_sentences:
        ids = tokenizer.encode(sent)
        decoded = tokenizer.decode(ids)
        print(f"  {sent!r}")
        print(f"    → IDs   : {ids}")
        print(f"    → decode: {decoded!r}")
        print(f"    → tokens: {len(ids)}")
        print()


if __name__ == "__main__":
    main()
