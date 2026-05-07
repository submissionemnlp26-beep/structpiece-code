/**
 * unicode_utils.cpp — Fast UTF-8 ↔ Unicode Code-Point Utilities
 * ==============================================================
 *
 * Provides low-level helpers for:
 *   - Decoding a UTF-8 byte stream into a vector of code points.
 *   - Encoding a vector of code points back into a UTF-8 string.
 *   - Counting the number of code points in a UTF-8 string without
 *     allocating a vector.
 *   - Checking whether a code point belongs to the Devanagari block.
 *   - Splitting a UTF-8 string into Extended Grapheme Clusters (simplified
 *     approach using Devanagari combining rules).
 *
 * Build (standalone test):
 *   c++ -std=c++17 -O2 -DUNICODE_UTILS_MAIN -o unicode_test unicode_utils.cpp
 */

#include "unicode_utils.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace indic {

// ─── UTF-8 Decoding ─────────────────────────────────────────────────────────

std::vector<uint32_t> utf8_to_codepoints(const std::string& utf8) {
    std::vector<uint32_t> cps;
    cps.reserve(utf8.size());  // over-estimate is fine

    size_t i = 0;
    const size_t len = utf8.size();
    const auto* data = reinterpret_cast<const uint8_t*>(utf8.data());

    while (i < len) {
        uint32_t cp = 0;
        uint8_t leading = data[i];

        if (leading < 0x80) {
            // 1-byte (ASCII)
            cp = leading;
            i += 1;
        } else if ((leading & 0xE0) == 0xC0) {
            // 2-byte
            if (i + 1 >= len) throw std::runtime_error("Truncated UTF-8 (2-byte)");
            cp = (uint32_t(leading & 0x1F) << 6) |
                 (uint32_t(data[i + 1] & 0x3F));
            i += 2;
        } else if ((leading & 0xF0) == 0xE0) {
            // 3-byte
            if (i + 2 >= len) throw std::runtime_error("Truncated UTF-8 (3-byte)");
            cp = (uint32_t(leading & 0x0F) << 12) |
                 (uint32_t(data[i + 1] & 0x3F) << 6) |
                 (uint32_t(data[i + 2] & 0x3F));
            i += 3;
        } else if ((leading & 0xF8) == 0xF0) {
            // 4-byte
            if (i + 3 >= len) throw std::runtime_error("Truncated UTF-8 (4-byte)");
            cp = (uint32_t(leading & 0x07) << 18) |
                 (uint32_t(data[i + 1] & 0x3F) << 12) |
                 (uint32_t(data[i + 2] & 0x3F) << 6) |
                 (uint32_t(data[i + 3] & 0x3F));
            i += 4;
        } else {
            // Invalid leading byte — skip it
            i += 1;
            continue;
        }
        cps.push_back(cp);
    }
    return cps;
}

// ─── UTF-8 Encoding ─────────────────────────────────────────────────────────

std::string codepoints_to_utf8(const std::vector<uint32_t>& cps) {
    std::string out;
    out.reserve(cps.size() * 3);  // rough estimate for Devanagari

    for (uint32_t cp : cps) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x110000) {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        // else: skip invalid code points silently
    }
    return out;
}

// ─── Code-point count ───────────────────────────────────────────────────────

size_t utf8_codepoint_count(const std::string& utf8) {
    size_t count = 0;
    for (size_t i = 0; i < utf8.size(); ) {
        uint8_t c = static_cast<uint8_t>(utf8[i]);
        if      (c < 0x80)           i += 1;
        else if ((c & 0xE0) == 0xC0) i += 2;
        else if ((c & 0xF0) == 0xE0) i += 3;
        else if ((c & 0xF8) == 0xF0) i += 4;
        else                         i += 1;  // skip invalid
        ++count;
    }
    return count;
}

// ─── Devanagari checks ──────────────────────────────────────────────────────

bool is_devanagari(uint32_t cp) {
    return (cp >= 0x0900 && cp <= 0x097F);
}

bool is_devanagari_vowel_sign(uint32_t cp) {
    // Devanagari dependent vowel signs (matras): 093A-094F
    // Also Anusvara 0902, Visarga 0903, Chandrabindu 0901
    return (cp >= 0x093A && cp <= 0x094F) ||
           cp == 0x0901 || cp == 0x0902 || cp == 0x0903;
}

bool is_devanagari_halant(uint32_t cp) {
    return cp == 0x094D;
}

bool is_devanagari_consonant(uint32_t cp) {
    return (cp >= 0x0915 && cp <= 0x0939) || cp == 0x0958 ||
           (cp >= 0x0959 && cp <= 0x095F);
}

bool is_devanagari_nukta(uint32_t cp) {
    return cp == 0x093C;
}

// ─── Arabic checks ──────────────────────────────────────────────────────────

bool is_arabic(uint32_t cp) {
    return (cp >= 0x0600 && cp <= 0x06FF);
}

bool is_arabic_diacritic(uint32_t cp) {
    return (cp >= 0x0610 && cp <= 0x061A) ||
           (cp >= 0x064B && cp <= 0x065F) ||
           cp == 0x0670;
}

// ─── Simplified Grapheme Cluster Splitting ──────────────────────────────────

std::vector<std::string> split_grapheme_clusters(const std::string& utf8) {
    auto cps = utf8_to_codepoints(utf8);
    if (cps.empty()) return {};

    std::vector<std::string> clusters;
    std::vector<uint32_t> current_cluster;
    current_cluster.push_back(cps[0]);

    for (size_t i = 1; i < cps.size(); ++i) {
        uint32_t cp = cps[i];
        uint32_t prev = cps[i - 1];

        // A code point extends the current cluster if:
        //   - It is a Halant (virama) — the next consonant will also join
        //   - It is a vowel sign / matra
        //   - It is an Anusvara, Visarga, or Chandrabindu
        //   - It is a Nukta
        //   - The previous character was a Halant (consonant joining)
        bool extends = false;

        if (is_devanagari(cp)) {
            if (is_devanagari_halant(cp) ||
                is_devanagari_vowel_sign(cp) ||
                is_devanagari_nukta(cp)) {
                extends = true;
            }
            // A consonant after Halant joins (conjunct)
            if (is_devanagari_halant(prev) && is_devanagari_consonant(cp)) {
                extends = true;
            }
        } else if (is_arabic(cp)) {
            if (is_arabic_diacritic(cp)) {
                extends = true;
            }
        }

        if (extends) {
            current_cluster.push_back(cp);
        } else {
            clusters.push_back(codepoints_to_utf8(current_cluster));
            current_cluster.clear();
            current_cluster.push_back(cp);
        }
    }

    if (!current_cluster.empty()) {
        clusters.push_back(codepoints_to_utf8(current_cluster));
    }

    return clusters;
}

}  // namespace indic

// ─── Standalone test ────────────────────────────────────────────────────────

#ifdef UNICODE_UTILS_MAIN
#include <iostream>

int main() {
    using namespace indic;

    // Test 1: UTF-8 round-trip
    {
        std::string hindi = "नमस्कार दुनिया";
        auto cps = utf8_to_codepoints(hindi);
        std::string rt = codepoints_to_utf8(cps);
        std::cout << "Round-trip test: "
                  << (rt == hindi ? "PASS" : "FAIL") << "\n";
        std::cout << "  Code points: " << cps.size() << "\n";
        std::cout << "  utf8_codepoint_count: "
                  << utf8_codepoint_count(hindi) << "\n";
    }

    // Test 2: Grapheme cluster splitting
    {
        struct TestCase { std::string input; size_t expected_clusters; };
        TestCase tests[] = {
            {"कि",        1},  // consonant + matra → one cluster
            {"क्रांति",    2},  // conjunct + anusvara + ti
            {"नमस्कार",   3},  // न + म + स्कार
            {"भारत",      3},  // भा + र + त
        };

        std::cout << "\nGrapheme cluster tests:\n";
        for (auto& tc : tests) {
            auto clusters = split_grapheme_clusters(tc.input);
            bool ok = (clusters.size() == tc.expected_clusters);
            std::cout << "  \"" << tc.input << "\" → "
                      << clusters.size() << " clusters";
            if (!ok) {
                std::cout << " (expected " << tc.expected_clusters << ")";
            }
            std::cout << " [";
            for (size_t i = 0; i < clusters.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << "\"" << clusters[i] << "\"";
            }
            std::cout << "]\n";
        }
    }

    // Test 3: Devanagari detection
    {
        auto cps = utf8_to_codepoints("क A");
        std::cout << "\nDevanagari checks:\n";
        std::cout << "  'क' (0x" << std::hex << cps[0] << std::dec << "): "
                  << (is_devanagari(cps[0]) ? "yes" : "no") << "\n";
        std::cout << "  'A' (0x" << std::hex << cps[2] << std::dec << "): "
                  << (is_devanagari(cps[2]) ? "yes" : "no") << "\n";
    }

    std::cout << "\nAll tests completed.\n";
    return 0;
}
#endif
