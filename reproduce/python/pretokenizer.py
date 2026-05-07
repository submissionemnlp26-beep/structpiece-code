"""
Script-Aware Pretokenizer for Hindi / Devanagari
Splits text into atomic grapheme-cluster-based tokens that are safe for BPE merging.

Key design decisions
--------------------
* Uses the `regex` library's \\X pattern to correctly identify Unicode Extended
  Grapheme Clusters (EGC).  This keeps conjunct consonants like क्र and
  consonant+matra combinations like कि as *single* units.
* Splits aggressively on whitespace, punctuation, and digit/script boundaries
  so the BPE trainer never accidentally merges across word or type boundaries.
* NFC-normalizes everything first so that identical-looking text always has
  the same byte representation.
"""

import os
import sys
import ctypes
import unicodedata
from functools import lru_cache
from typing import List

import regex as re


# ─── Character classifiers ──────────────────────────────────────────────────

# Standard punctuation set for Hindi text (includes Devanagari danda/double-danda)
PUNCT = set(".,!?;:।॥()[]{}\"'–—-/\\@#$%^&*+=<>~`|""''…")

# Devanagari Unicode range
_DEVA_START = 0x0900
_DEVA_END = 0x097F

# Devanagari Extended
_DEVA_EXT_START = 0xA8E0
_DEVA_EXT_END = 0xA8FF

# Vedic Extensions (sometimes appear in Hindi corpora)
_VEDIC_START = 0x1CD0
_VEDIC_END = 0x1CFF


def is_devanagari(char: str) -> bool:
    """Return True if *char* is in the Devanagari or Devanagari Extended block."""
    cp = ord(char)
    return (_DEVA_START <= cp <= _DEVA_END or
            _DEVA_EXT_START <= cp <= _DEVA_EXT_END or
            _VEDIC_START <= cp <= _VEDIC_END)


def is_punctuation(char: str) -> bool:
    """Return True if *char* is a known punctuation character."""
    if char in PUNCT:
        return True
    # Fall back to Unicode General Category
    cat = unicodedata.category(char)
    return cat.startswith("P") or cat.startswith("S")


def is_digit(char: str) -> bool:
    """Return True for ASCII and Devanagari digits."""
    return char.isdigit()  # handles both ASCII 0-9 and ०-९


# ─── Core helpers ────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """NFC-normalize *text*."""
    return unicodedata.normalize("NFC", text)


# Pre-compiled EGC regex (avoids re.compile on every call)
_EGC_PATTERN = re.compile(r"\X")

# Load C++ EGC kernel
_LIB_DIR = os.path.join(os.path.dirname(__file__), "..", "cpp", "build")
_ext = ".dylib" if sys.platform == "darwin" else ".so"
_LIB_PATH = os.path.join(_LIB_DIR, f"libindic_tokenizer{_ext}")

try:
    _lib = ctypes.CDLL(_LIB_PATH)
    _lib.c_split_grapheme_clusters.argtypes = [ctypes.c_char_p]
    _lib.c_split_grapheme_clusters.restype = ctypes.c_void_p
    _lib.c_free_string.argtypes = [ctypes.c_void_p]
    _lib.c_free_string.restype = None
    _C_API_AVAILABLE = True  # use C++ kernel for Devanagari/Arabic/Latin
except OSError:
    _C_API_AVAILABLE = False


def grapheme_clusters(text: str) -> List[str]:
    """Split *text* into Unicode Extended Grapheme Clusters.
    
    Uses the compiled C++ kernel for Devanagari/Arabic/Latin.
    Automatically falls back to regex \\X for Abugida scripts (e.g. Tamil)
    where the C++ kernel's simplified combining rules are insufficient.
    """
    # C++ kernel is optimized for Devanagari/Arabic/Latin;
    # fallback to regex for Tamil (0x0B80-0x0BFF) and Georgian (0x10A0-0x10FF)
    if _C_API_AVAILABLE and not any(
        0x0B80 <= ord(c) <= 0x0BFF or 0x10A0 <= ord(c) <= 0x10FF for c in text
    ):
        if not text:
            return []
        encoded = text.encode("utf-8")
        ptr = _lib.c_split_grapheme_clusters(encoded)
        if not ptr:
            return []
        c_str = ctypes.cast(ptr, ctypes.c_char_p).value
        decoded = c_str.decode("utf-8")
        _lib.c_free_string(ptr)
        return decoded.split('\x1F')

    return _EGC_PATTERN.findall(text)


# ─── Type tagging (for boundary detection) ───────────────────────────────────

# Use integer type tags instead of string comparisons for speed
_TYPE_DEVA  = 0
_TYPE_DIGIT = 1
_TYPE_PUNCT = 2
_TYPE_SPACE = 3
_TYPE_OTHER = 4

# Set of break-types that always become their own token
_BREAK_TYPES = {_TYPE_SPACE, _TYPE_PUNCT}


@lru_cache(maxsize=4096)
def _cluster_type(cluster: str) -> int:
    """Determine the script/type category of a grapheme cluster.
    Cached for repeated lookups on the same cluster string."""
    first = cluster[0]
    cp = ord(first)

    # Fast-path: ASCII space/newline/tab
    if cp <= 0x20 or cp == 0x09 or cp == 0x0A or cp == 0x0D:
        if first.isspace():
            return _TYPE_SPACE

    # Fast-path: ASCII digits 0-9
    if 0x30 <= cp <= 0x39:
        return _TYPE_DIGIT

    # Fast-path: common ASCII punctuation
    if first in PUNCT:
        return _TYPE_PUNCT

    # Devanagari range check (much faster than unicodedata)
    if (_DEVA_START <= cp <= _DEVA_END or
        _DEVA_EXT_START <= cp <= _DEVA_EXT_END or
        _VEDIC_START <= cp <= _VEDIC_END):
        return _TYPE_DEVA

    # Devanagari digits
    if 0x0966 <= cp <= 0x096F:
        return _TYPE_DIGIT

    # General space check
    if first.isspace():
        return _TYPE_SPACE

    # General digit check
    if first.isdigit():
        return _TYPE_DIGIT

    # Unicode category fallback for punctuation
    cat = unicodedata.category(first)
    if cat[0] == 'P' or cat[0] == 'S':
        return _TYPE_PUNCT

    return _TYPE_OTHER


# ─── Main pretokenize function ───────────────────────────────────────────────

def pretokenize(text: str) -> List[str]:
    """
    Split *text* into pre-tokens suitable for BPE training.

    Returns a list of strings where each element is one of:
    * a Devanagari word (sequence of Devanagari grapheme clusters)
    * a run of digits
    * a single punctuation character
    * a single whitespace character
    * a run of "other" (Latin, etc.) characters

    Boundaries are inserted whenever the *type* of the current cluster differs
    from the previous one, or when we hit punctuation / whitespace (which always
    become their own tokens).
    """
    text = normalize(text)
    clusters = grapheme_clusters(text)

    if not clusters:
        return []

    tokens: List[str] = []
    current_parts: List[str] = []  # accumulate parts, join once
    current_type = -1  # sentinel

    for c in clusters:
        ctype = _cluster_type(c)

        # Whitespace and punctuation always break and become their own token
        if ctype in _BREAK_TYPES:
            if current_parts:
                tokens.append("".join(current_parts))
                current_parts = []
                current_type = -1
            tokens.append(c)
            continue

        # Type change → flush
        if ctype != current_type:
            if current_parts:
                tokens.append("".join(current_parts))
            current_parts = [c]
            current_type = ctype
        else:
            current_parts.append(c)

    if current_parts:
        tokens.append("".join(current_parts))

    return tokens


def pretokenize_corpus(lines: List[str]) -> List[List[str]]:
    """Pretokenize an entire corpus (list of lines) into lists of pre-tokens."""
    return [pretokenize(line) for line in lines]


# ─── CLI entry-point ─────────────────────────────────────────────────────────

def main():
    """Quick sanity-check when run directly."""
    tests = [
        "कि",
        "क्रांति",
        "ज्ञान",
        "नमस्कार",
        "भारत महान है।",
        "123 रुपये cost ₹123",
        "हिन्दी (Hindi) एक भाषा है।",
    ]
    for t in tests:
        result = pretokenize(t)
        print(f"{t!r:40s} → {result}")


if __name__ == "__main__":
    main()
