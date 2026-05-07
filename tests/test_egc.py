import pytest
from python.pretokenizer import grapheme_clusters

def test_hindi_egc():
    assert grapheme_clusters("कक्षा") == ["क", "क्षा"]
    assert grapheme_clusters("क्ष") == ["क्ष"]
    assert grapheme_clusters("क्षा") == ["क्षा"]
    assert grapheme_clusters("कि") == ["कि"]

def test_arabic_egc():
    assert grapheme_clusters("قُ") == ["قُ"]
    assert grapheme_clusters("بِ") == ["بِ"]
    assert grapheme_clusters("مّ") == ["مّ"]

def test_tamil_egc():
    assert grapheme_clusters("ம்") == ["ம்"]
    assert grapheme_clusters("மா") == ["மா"]
    assert grapheme_clusters("க") == ["க"]

def test_georgian_latin_egc():
    assert grapheme_clusters("aé") == ["a", "é"] # Latin extended
    assert grapheme_clusters("á") == ["á"]
    assert grapheme_clusters("ა") == ["ა"] # Georgian letter

def test_merge_file_invariant():
    """Verify that no indic_tokenizer_merges.txt contains intra-EGC merges."""
    import os
    def count_egc(s):
        return len(grapheme_clusters(s))
        
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file == 'indic_tokenizer_merges.txt':
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        line_s = line.strip()
                        if not line_s:
                            continue
                        parts = line_s.split()
                        if len(parts) == 2:
                            p1_clean = parts[0].replace('▁', '')
                            p2_clean = parts[1].replace('▁', '')
                            
                            p1_count = count_egc(p1_clean)
                            p2_count = count_egc(p2_clean)
                            comb_count = count_egc(p1_clean + p2_clean)
                            
                            assert comb_count == p1_count + p2_count, f"Intra-EGC merge found in {filepath} line {i}: '{parts[0]}' + '{parts[1]}'"
