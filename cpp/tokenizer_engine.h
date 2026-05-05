/**
 * tokenizer_engine.h — Public API for the BPE tokenizer runtime
 */
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace indic {

// A single merge rule: merge pair.first + pair.second
using MergePair = std::pair<std::string, std::string>;

/**
 * TokenizerEngine — High-performance BPE tokenizer for Hindi text.
 *
 * Loads vocabulary + merge rules produced by the Python trainer
 * (adaptive_bpe.py) and performs fast encoding / decoding.
 */
class TokenizerEngine {
public:
    TokenizerEngine() = default;

    /// Load vocabulary and merges from the specified directory.
    /// Expects files: indic_tokenizer_vocab.json, indic_tokenizer_merges.txt
    bool load(const std::string& model_dir);

    /// Encode a UTF-8 string into token IDs.
    std::vector<int> encode(const std::string& text) const;

    /// Decode token IDs back into a UTF-8 string.
    std::string decode(const std::vector<int>& ids) const;

    /// Return the vocabulary size.
    size_t vocab_size() const { return token_to_id_.size(); }

private:
    // Pre-tokenize text into word-level chunks.
    std::vector<std::string> pretokenize(const std::string& text) const;

    // Apply BPE merges to a single word (list of symbols).
    std::vector<std::string> apply_bpe(const std::string& word) const;

    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;

    // Merge rules stored in priority order (index = priority)
    std::vector<MergePair> merges_;

    // Fast lookup: concatenated pair key → merge priority
    // Key format: pair.first + "\x00" + pair.second
    std::unordered_map<std::string, int> merge_priority_;

    int unk_id_ = 0;
};

}  // namespace indic
