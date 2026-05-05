cmake_minimum_required(VERSION 3.16)
project(IndicTokenizerCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── Unicode utilities test ──────────────────────────────────────────────────

add_executable(unicode_test unicode_utils.cpp)
target_compile_definitions(unicode_test PRIVATE UNICODE_UTILS_MAIN)

# ─── Tokenizer engine test ──────────────────────────────────────────────────

add_executable(tokenizer_test
    tokenizer_engine.cpp
    unicode_utils.cpp
)
target_compile_definitions(tokenizer_test PRIVATE TOKENIZER_ENGINE_MAIN)

# ─── Shared library (for FFI / Python bindings if needed) ────────────────────

add_library(indic_tokenizer SHARED
    unicode_utils.cpp
    tokenizer_engine.cpp
)
target_compile_definitions(indic_tokenizer PRIVATE INDIC_TOKENIZER_SHARED)
