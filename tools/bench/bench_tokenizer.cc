#include "infeng/tokenizer/tokenizer.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
std::vector<std::string> sample_prompts() {
  return {
      "Explain the benefits of CPU-first inference engines in two sentences.",
      "List five use cases for LoRA adapters in enterprise AI.",
      "Summarize the latest benchmark results for TinyLlama models.",
      "Describe a debugging workflow for fused GEMM kernels."};
}
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <tokenizer.json> [threads]" << std::endl;
    return 1;
  }
  const std::string model_path = argv[1];
  const std::size_t threads = (argc > 2) ? static_cast<std::size_t>(std::stoul(argv[2])) : 4;

  infeng::tokenizer::Tokenizer tokenizer(model_path);
  tokenizer.enable_prefix_cache(true);

  const auto prompts = sample_prompts();

  std::size_t total_tokens = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (const auto& prompt : prompts) {
    tokenizer.encode_stream_begin(prompt, threads);
    infeng::tokenizer::Batch batch{};
    while (tokenizer.encode_stream_next(&batch)) {
      total_tokens += batch.len;
    }
    tokenizer.encode_stream_end();
  }
  auto end = std::chrono::high_resolution_clock::now();
  const double seconds = std::chrono::duration<double>(end - start).count();
  const double tokens_per_second = total_tokens / seconds;

  std::cout << "bench,threads,tokens_per_s\n";
  std::cout << "tokenizer," << threads << "," << tokens_per_second << "\n";
  return 0;
}
