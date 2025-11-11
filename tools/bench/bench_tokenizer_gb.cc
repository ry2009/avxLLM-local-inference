#include "infeng/tokenizer/tokenizer.h"

#include <benchmark/benchmark.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

namespace {

std::vector<std::string> read_prompts(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open prompts file: " + path);
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  if (lines.empty()) {
    throw std::runtime_error("Prompts file empty: " + path);
  }
  return lines;
}

std::string g_model_path;
std::vector<std::string> g_prompts;
std::size_t g_threads = 4;
std::size_t g_chunk_len = 4096;
std::size_t g_prefix_k = 128;
std::size_t g_prefix_capacity = 2048;
std::filesystem::path g_reports_dir{"reports"};

void WriteCsv(double tokens_per_sec) {
  std::filesystem::create_directories(g_reports_dir);
  const auto csv_path = g_reports_dir / "tokenizer.csv";
  const bool existed = std::filesystem::exists(csv_path);
  std::ofstream out(csv_path, existed ? std::ios::app : std::ios::trunc);
  if (!existed) {
    out << "bench,threads,tokens_per_s\n";
  }
  out << "tokenizer," << g_threads << "," << tokens_per_sec << "\n";
}

static void BM_Tokenizer(benchmark::State& state) {
  infeng::tokenizer::Tokenizer tokenizer(g_model_path);
  tokenizer.enable_prefix_cache(true);
  tokenizer.set_prefix_params(g_prefix_k, g_prefix_capacity);
  tokenizer.set_thread_override(g_threads);

  std::size_t iter_tokens = 0;
  auto wall_start = std::chrono::high_resolution_clock::now();
  for (auto _ : state) {
    for (const auto& prompt : g_prompts) {
      tokenizer.encode_stream_begin(prompt, g_threads);
      infeng::tokenizer::Batch batch{};
      while (tokenizer.encode_stream_next(&batch)) {
        iter_tokens += batch.len;
      }
      tokenizer.encode_stream_end();
    }
  }
  const auto wall_end = std::chrono::high_resolution_clock::now();
  const double seconds = std::chrono::duration<double>(wall_end - wall_start).count();
  const double tokens_per_sec = seconds > 0.0 ? static_cast<double>(iter_tokens) / seconds : 0.0;
  state.counters["tokens_per_s"] = benchmark::Counter(static_cast<double>(iter_tokens), benchmark::Counter::kIsRate);
  state.SetItemsProcessed(static_cast<long long>(iter_tokens));
  WriteCsv(tokens_per_sec);
}

}  // namespace

int main(int argc, char** argv) {
  std::string prompts_path = "tools/bench/prompts.txt";
  g_model_path = "tokenizer.json";

  std::vector<char*> bm_argv;
  bm_argv.reserve(argc);
  for (int i = 0; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.starts_with("--model=")) {
      g_model_path = arg.substr(8);
      continue;
    }
    if (arg.starts_with("--input_file=")) {
      prompts_path = arg.substr(13);
      continue;
    }
    if (arg.starts_with("--threads=")) {
      g_threads = static_cast<std::size_t>(std::stoul(arg.substr(10)));
      continue;
    }
    if (arg.starts_with("--chunk_len=")) {
      g_chunk_len = static_cast<std::size_t>(std::stoul(arg.substr(12)));
      continue;
    }
    if (arg.starts_with("--prefix_k=")) {
      g_prefix_k = static_cast<std::size_t>(std::stoul(arg.substr(11)));
      continue;
    }
    if (arg.starts_with("--prefix_cache_entries=")) {
      g_prefix_capacity = static_cast<std::size_t>(std::stoul(arg.substr(24)));
      continue;
    }
    bm_argv.push_back(argv[i]);
  }
  bm_argv.push_back(nullptr);

  try {
    g_prompts = read_prompts(prompts_path);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;
    return 1;
  }

  int bm_argc = static_cast<int>(bm_argv.size() - 1);
  benchmark::Initialize(&bm_argc, bm_argv.data());
  benchmark::RegisterBenchmark("Tokenizer", BM_Tokenizer)->Iterations(1);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
