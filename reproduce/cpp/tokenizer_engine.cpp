/**
 * tokenizer_engine.cpp — High-Performance BPE Tokenizer Runtime
 * ==============================================================
 *
 * Loads a vocabulary + merge table produced by the Python trainer and
 * performs BPE encoding / decoding entirely in C++ for maximum speed.
 *
 * Build (standalone test):
 *   c++ -std=c++17 -O2 -DTOKENIZER_ENGINE_MAIN \
 *       -o tokenizer_test tokenizer_engine.cpp unicode_utils.cpp
 */

#include "tokenizer_engine.h"
#include "unicode_utils.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace indic {

// ─── Helpers ────────────────────────────────────────────────────────────────

static std::string make_pair_key(const std::string& a, const std::string& b) {
    // Use null byte as separator (cannot appear in valid UTF-8 text)
    return a + '\0' + b;
}

// Simple punctuation check (mirrors the Python pretokenizer)
static bool is_punct(uint32_t cp) {
    static const std::unordered_set<uint32_t> puncts = {
        '.', ',', '!', '?', ';', ':', 0x0964 /* । */, 0x0965 /* ॥ */,
        '(', ')', '[', ']', '{', '}', '"', '\'',
        '-', '/', '\\', '@', '#', '$', '%', '^', '&', '*', '+', '=',
        '<', '>', '~', '`', '|',
    };
    return puncts.count(cp) > 0;
}

// ─── JSON Parsing (minimal, for vocab file) ─────────────────────────────────

// We only need to parse a flat { "key": int, ... } JSON object.
// This avoids pulling in a full JSON library.

static bool parse_vocab_json(const std::string& path,
                             std::unordered_map<std::string, int>& vocab) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "Cannot open vocab file: " << path << "\n";
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());

    // State machine: look for "key": value patterns
    size_t i = 0;
    const size_t len = content.size();

    auto skip_ws = [&]() {
        while (i < len && (content[i] == ' ' || content[i] == '\n' ||
                           content[i] == '\r' || content[i] == '\t'))
            ++i;
    };

    skip_ws();
    if (i >= len || content[i] != '{') return false;
    ++i;

    while (i < len) {
        skip_ws();
        if (content[i] == '}') break;
        if (content[i] == ',') { ++i; continue; }

        // Expect a string key
        if (content[i] != '"') return false;
        ++i;

        std::string key;
        while (i < len && content[i] != '"') {
            if (content[i] == '\\' && i + 1 < len) {
                ++i;
                switch (content[i]) {
                    case '"':  key += '"';  break;
                    case '\\': key += '\\'; break;
                    case 'n':  key += '\n'; break;
                    case 't':  key += '\t'; break;
                    case 'u': {
                        // Parse \\uXXXX
                        if (i + 4 < len) {
                            std::string hex_str = content.substr(i + 1, 4);
                            uint32_t cp = static_cast<uint32_t>(
                                std::stoul(hex_str, nullptr, 16));
                            // Encode as UTF-8
                            std::vector<uint32_t> tmp = {cp};
                            key += codepoints_to_utf8(tmp);
                            i += 4;
                        }
                        break;
                    }
                    default: key += content[i]; break;
                }
            } else {
                key += content[i];
            }
            ++i;
        }
        if (i < len) ++i;  // skip closing quote

        skip_ws();
        if (i >= len || content[i] != ':') return false;
        ++i;
        skip_ws();

        // Parse integer value
        bool neg = false;
        if (i < len && content[i] == '-') { neg = true; ++i; }
        int val = 0;
        while (i < len && content[i] >= '0' && content[i] <= '9') {
            val = val * 10 + (content[i] - '0');
            ++i;
        }
        if (neg) val = -val;

        vocab[key] = val;
    }

    return true;
}

// ─── TokenizerEngine implementation ─────────────────────────────────────────

bool TokenizerEngine::load(const std::string& model_dir) {
    std::string vocab_path = model_dir + "/indic_tokenizer_vocab.json";
    std::string merges_path = model_dir + "/indic_tokenizer_merges.txt";

    // 1. Load vocabulary
    if (!parse_vocab_json(vocab_path, token_to_id_)) {
        return false;
    }

    // Build reverse map
    for (auto& [tok, id] : token_to_id_) {
        id_to_token_[id] = tok;
    }

    auto it = token_to_id_.find("<unk>");
    unk_id_ = (it != token_to_id_.end()) ? it->second : 0;

    // 2. Load merge rules
    std::ifstream mfs(merges_path);
    if (!mfs.is_open()) {
        std::cerr << "Cannot open merges file: " << merges_path << "\n";
        return false;
    }

    std::string line;
    int priority = 0;
    while (std::getline(mfs, line)) {
        if (line.empty()) continue;
        // Each line is: tokenA tokenB  (separated by first space)
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;

        std::string a = line.substr(0, sp);
        std::string b = line.substr(sp + 1);

        merges_.push_back({a, b});
        merge_priority_[make_pair_key(a, b)] = priority++;
    }

    std::cout << "Loaded vocab: " << token_to_id_.size()
              << " tokens, " << merges_.size() << " merges\n";
    return true;
}

// ─── Pre-tokenization ───────────────────────────────────────────────────────

std::vector<std::string> TokenizerEngine::pretokenize(const std::string& text) const {
    // Split on whitespace, punctuation, and script boundaries
    auto cps = utf8_to_codepoints(text);
    if (cps.empty()) return {};

    enum CType { CT_DEVA, CT_DIGIT, CT_PUNCT, CT_SPACE, CT_OTHER };

    auto classify = [](uint32_t cp) -> CType {
        if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r')
            return CT_SPACE;
        if (is_punct(cp))
            return CT_PUNCT;
        if ((cp >= '0' && cp <= '9') || (cp >= 0x0966 && cp <= 0x096F))
            return CT_DIGIT;
        if (is_devanagari(cp))
            return CT_DEVA;
        return CT_OTHER;
    };

    std::vector<std::string> tokens;
    std::vector<uint32_t> current;
    CType current_type = classify(cps[0]);

    for (size_t i = 0; i < cps.size(); ++i) {
        uint32_t cp = cps[i];
        CType ct = classify(cp);

        if (ct == CT_SPACE || ct == CT_PUNCT) {
            if (!current.empty()) {
                tokens.push_back(codepoints_to_utf8(current));
                current.clear();
            }
            tokens.push_back(codepoints_to_utf8({cp}));
            current_type = ct;
            continue;
        }

        if (ct != current_type && !current.empty()) {
            tokens.push_back(codepoints_to_utf8(current));
            current.clear();
        }
        current.push_back(cp);
        current_type = ct;
    }

    if (!current.empty()) {
        tokens.push_back(codepoints_to_utf8(current));
    }

    return tokens;
}

// ─── BPE application ────────────────────────────────────────────────────────

std::vector<std::string> TokenizerEngine::apply_bpe(const std::string& word) const {
    // Start with Extended Grapheme Clusters (EGCs)
    std::vector<std::string> symbols = split_grapheme_clusters(word);

    while (symbols.size() > 1) {
        // Find the pair with the lowest merge priority (highest rank)
        int best_priority = std::numeric_limits<int>::max();
        int best_pos = -1;

        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            std::string key = make_pair_key(symbols[i], symbols[i + 1]);
            auto it = merge_priority_.find(key);
            if (it != merge_priority_.end() && it->second < best_priority) {
                best_priority = it->second;
                best_pos = static_cast<int>(i);
            }
        }

        if (best_pos < 0) break;  // no applicable merge

        // Apply the merge at every occurrence of this pair
        std::string merged = symbols[best_pos] + symbols[best_pos + 1];
        std::vector<std::string> new_symbols;
        new_symbols.reserve(symbols.size());

        size_t i = 0;
        while (i < symbols.size()) {
            if (i + 1 < symbols.size() &&
                symbols[i] == merges_[best_priority].first &&
                symbols[i + 1] == merges_[best_priority].second) {
                new_symbols.push_back(merged);
                i += 2;
            } else {
                new_symbols.push_back(symbols[i]);
                i += 1;
            }
        }
        symbols = std::move(new_symbols);
    }

    return symbols;
}

// ─── Encoding ───────────────────────────────────────────────────────────────

std::vector<int> TokenizerEngine::encode(const std::string& text) const {
    auto pre_tokens = pretokenize(text);
    std::vector<int> ids;
    ids.reserve(pre_tokens.size());

    for (const auto& tok : pre_tokens) {
        // Whitespace → space token
        if (tok.size() == 1 && (tok[0] == ' ' || tok[0] == '\t' ||
                                tok[0] == '\n' || tok[0] == '\r')) {
            auto it = token_to_id_.find(" ");
            ids.push_back(it != token_to_id_.end() ? it->second : unk_id_);
            continue;
        }

        // Apply BPE
        auto symbols = apply_bpe(tok);
        for (const auto& sym : symbols) {
            auto it = token_to_id_.find(sym);
            ids.push_back(it != token_to_id_.end() ? it->second : unk_id_);
        }
    }

    return ids;
}

// ─── Decoding ───────────────────────────────────────────────────────────────

std::string TokenizerEngine::decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) {
            result += it->second;
        } else {
            result += "\xEF\xBF\xBD";  // U+FFFD replacement character
        }
    }
    return result;
}

}  // namespace indic

// ─── Standalone test ────────────────────────────────────────────────────────

#ifdef TOKENIZER_ENGINE_MAIN
int main(int argc, char* argv[]) {
    using namespace indic;

    std::string model_dir = "models";
    if (argc > 1) {
        model_dir = argv[1];
    }

    TokenizerEngine engine;
    if (!engine.load(model_dir)) {
        std::cerr << "Failed to load model from " << model_dir << "\n";
        std::cerr << "Usage: " << argv[0] << " [model_dir]\n";
        std::cerr << "  Train the model first:  python python/adaptive_bpe.py\n";
        return 1;
    }

    std::cout << "Vocab size: " << engine.vocab_size() << "\n\n";

    // Test sentences
    std::vector<std::string> tests = {
        "भारत महान है।",
        "नमस्कार, आप कैसे हैं?",
        "क्रांति का समय आ गया है।",
    };

    for (const auto& sent : tests) {
        auto ids = engine.encode(sent);
        std::string decoded = engine.decode(ids);

        std::cout << "Input  : " << sent << "\n";
        std::cout << "IDs    :";
        for (int id : ids) std::cout << " " << id;
        std::cout << "\n";
        std::cout << "Decoded: " << decoded << "\n";
        std::cout << "Tokens : " << ids.size() << "\n\n";
    }

    return 0;
}
#endif
