
#include "tokenizer_engine.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: speed_bench <model_dir> <input_file> [repeats]\n";
        return 1;
    }
    std::string model_dir = argv[1];
    std::string input_file = argv[2];
    int repeats = argc > 3 ? std::atoi(argv[3]) : 3;

    indic::TokenizerEngine engine;
    if (!engine.load(model_dir)) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    // Load lines
    std::vector<std::string> lines;
    std::ifstream ifs(input_file);
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty()) lines.push_back(line);
    }

    long total_tokens = 0;
    double total_ms = 0;

    for (int r = 0; r < repeats; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        long tokens = 0;
        for (const auto& l : lines) {
            auto ids = engine.encode(l);
            tokens += ids.size();
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        total_ms += ms;
        total_tokens = tokens;
    }

    double avg_ms = total_ms / repeats;
    double tps = (total_tokens / avg_ms) * 1000.0;

    // Output as JSON
    std::cout << "{"
              << "\"total_tokens\":" << total_tokens << ","
              << "\"avg_time_ms\":" << avg_ms << ","
              << "\"tokens_per_sec\":" << (long)tps << ","
              << "\"lines\":" << lines.size() << ","
              << "\"repeats\":" << repeats
              << "}\n";
    return 0;
}
