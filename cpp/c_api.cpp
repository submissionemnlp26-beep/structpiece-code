#include "unicode_utils.h"
#include <cstring>
#include <vector>
#include <string>

extern "C" {

// Splits the input UTF-8 text into grapheme clusters.
// Returns a single dynamically allocated C-string where clusters are 
// separated by the Unit Separator character (0x1F).
// The caller is responsible for freeing the returned string using c_free_string.
char* c_split_grapheme_clusters(const char* text) {
    if (!text) return nullptr;
    std::string input(text);
    std::vector<std::string> clusters = indic::split_grapheme_clusters(input);
    
    std::string result;
    for (size_t i = 0; i < clusters.size(); ++i) {
        result += clusters[i];
        if (i + 1 < clusters.size()) {
            result += '\x1F';
        }
    }
    
    char* out = new char[result.size() + 1];
    std::strcpy(out, result.c_str());
    return out;
}

// Frees the memory allocated by c_split_grapheme_clusters.
void c_free_string(char* str) {
    delete[] str;
}

}
