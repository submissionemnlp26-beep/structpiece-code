/**
 * unicode_utils.h — Public API for UTF-8 / Devanagari utilities
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace indic {

// ─── UTF-8 ↔ Code-Point conversion ─────────────────────────────────────────

/// Decode a UTF-8 string into Unicode code points.
std::vector<uint32_t> utf8_to_codepoints(const std::string& utf8);

/// Encode Unicode code points into a UTF-8 string.
std::string codepoints_to_utf8(const std::vector<uint32_t>& cps);

/// Count the number of code points in a UTF-8 string (without allocation).
size_t utf8_codepoint_count(const std::string& utf8);

// ─── Devanagari character classification ────────────────────────────────────

bool is_devanagari(uint32_t cp);
bool is_devanagari_vowel_sign(uint32_t cp);
bool is_devanagari_halant(uint32_t cp);
bool is_devanagari_consonant(uint32_t cp);
bool is_devanagari_nukta(uint32_t cp);

// ─── Grapheme cluster splitting ─────────────────────────────────────────────

/// Split a UTF-8 string into Extended Grapheme Clusters
/// (simplified Devanagari-aware approach).
std::vector<std::string> split_grapheme_clusters(const std::string& utf8);

}  // namespace indic
