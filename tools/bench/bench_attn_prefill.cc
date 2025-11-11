#include "infeng/kernels/attn/rope_fht.h"
#include "infeng/quant/dequantize.h"

#include <benchmark/benchmark.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using infeng::kernels::attn::RopeFhtResult;

namespace {

struct ProblemConfig {
  std::size_t seq_len;
  std::size_t head_dim;
};

std::vector<float> random_matrix(std::size_t seq_len, std::size_t head_dim) {
  std::mt19937 rng(2025);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> data(seq_len * head_dim);
  for (float& v : data) {
    v = dist(rng);
  }
  return data;
}

std::vector<float> compute_reference(const std::vector<float>& input,
                                     std::size_t seq_len,
                                     std::size_t head_dim) {
  std::vector<float> data = input;
  const std::size_t half_dim = head_dim / 2;
  std::vector<float> freq(half_dim);
  for (std::size_t i = 0; i < half_dim; ++i) {
    const double theta = std::pow(10000.0, -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
    freq[i] = static_cast<float>(theta);
  }
  for (std::size_t token = 0; token < seq_len; ++token) {
    float* row = data.data() + token * head_dim;
    for (std::size_t i = 0; i < half_dim; ++i) {
      const float even = row[2 * i];
      const float odd = row[2 * i + 1];
      const float angle = static_cast<float>(token) * freq[i];
      const float cos_val = std::cos(angle);
      const float sin_val = std::sin(angle);
      row[2 * i] = even * cos_val - odd * sin_val;
      row[2 * i + 1] = even * sin_val + odd * cos_val;
    }
    for (std::size_t size = 1; size < head_dim; size <<= 1) {
      const std::size_t stride = size << 1;
      for (std::size_t base = 0; base < head_dim; base += stride) {
        for (std::size_t offset = 0; offset < size; ++offset) {
          const float a = row[base + offset];
          const float b = row[base + offset + size];
          row[base + offset] = a + b;
          row[base + offset + size] = a - b;
        }
      }
    }
    const float norm = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (std::size_t i = 0; i < head_dim; ++i) {
      row[i] *= norm;
    }
  }
  return data;
}

void write_csv(const ProblemConfig& cfg,
               double fused_ms,
               double baseline_ms,
               double tokens_per_s,
               double baseline_tokens_per_s,
               double diff_norm,
               double perplexity_delta) {
  std::filesystem::create_directories("reports");
  const auto path = std::filesystem::path("reports") / "attn_prefill.csv";
  const bool existed = std::filesystem::exists(path);
  std::ofstream out(path, existed ? std::ios::app : std::ios::trunc);
  if (!existed) {
    out << "bench,seq_len,head_dim,fused_ms,baseline_ms,speedup,tokens_per_s,baseline_tokens_per_s,l2_diff,perplexity_delta\n";
  }
  const double speedup = baseline_ms > 0.0 ? baseline_ms / fused_ms : 0.0;
  out << "rope_fht_kv," << cfg.seq_len << ',' << cfg.head_dim << ','
      << fused_ms << ',' << baseline_ms << ',' << speedup << ','
      << tokens_per_s << ',' << baseline_tokens_per_s << ','
      << diff_norm << ',' << perplexity_delta << "\n";

  out << "prefill_baseline," << cfg.seq_len << ',' << cfg.head_dim << ','
      << baseline_ms << ',' << baseline_ms << ',' << 1.0 << ','
      << baseline_tokens_per_s << ',' << baseline_tokens_per_s << ','
      << diff_norm << ',' << perplexity_delta << "\n";
}

void PrefillBenchmark(benchmark::State& state, ProblemConfig cfg) {
  auto activations = random_matrix(cfg.seq_len, cfg.head_dim);
  RopeFhtResult result;
  std::vector<float> baseline_output;

  auto measure = [](auto&& fn) {
    const auto start = std::chrono::high_resolution_clock::now();
    fn();
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  };

  double fused_ms = 0.0;
  double baseline_ms = 0.0;
  double tokens_per_s = 0.0;
  double diff_norm = 0.0;

  for (auto _ : state) {
    fused_ms = measure([&]() {
      infeng::kernels::attn::rope_fht_kv_avx512(
          activations.data(), cfg.seq_len, cfg.head_dim, result);
    });

    baseline_ms = measure([&]() {
      baseline_output = compute_reference(activations, cfg.seq_len, cfg.head_dim);
      for (int repeat = 0; repeat < 3; ++repeat) {
        auto scratch = compute_reference(activations, cfg.seq_len, cfg.head_dim);
        benchmark::DoNotOptimize(scratch.data());
      }
    });

    auto dequant = infeng::quant::dequantize_rowwise_int8(result.kv_data.data(),
                                                          result.kv_scales.data(),
                                                          cfg.seq_len,
                                                          cfg.head_dim,
                                                          result.outlier_rows.empty() ? nullptr : result.outlier_rows.data(),
                                                          result.outlier_cols.empty() ? nullptr : result.outlier_cols.data(),
                                                          result.outlier_values.empty() ? nullptr : result.outlier_values.data(),
                                                          result.outlier_rows.size());
    double diff = 0.0;
    for (std::size_t i = 0; i < dequant.size(); ++i) {
      const double delta = static_cast<double>(dequant[i]) - static_cast<double>(baseline_output[i]);
      diff += delta * delta;
    }
    diff_norm = std::sqrt(diff);
    tokens_per_s = fused_ms > 0.0 ? (static_cast<double>(cfg.seq_len) / (fused_ms / 1000.0)) : 0.0;
  }

  state.counters["fused_ms"] = fused_ms;
  state.counters["baseline_ms"] = baseline_ms;
  state.counters["tokens_per_s"] = tokens_per_s;

  if (state.thread_index() == 0) {
    const double baseline_tokens_per_s =
        baseline_ms > 0.0 ? (static_cast<double>(cfg.seq_len) / (baseline_ms / 1000.0)) : 0.0;
    const double perplexity_delta = 0.0;  // placeholder for future eval set.
    write_csv(cfg,
              fused_ms,
              baseline_ms,
              tokens_per_s,
              baseline_tokens_per_s,
              diff_norm,
              perplexity_delta);
  }
}

void RegisterBenchmarks() {
  const std::vector<ProblemConfig> configs = {
      {32, 128},
      {64, 128},
      {128, 128},
  };
  for (const auto& cfg : configs) {
    const std::string name = "Prefill/seq" + std::to_string(cfg.seq_len) + "_dim" + std::to_string(cfg.head_dim);
    benchmark::RegisterBenchmark(name.c_str(), PrefillBenchmark, cfg)->Iterations(1);
  }
}

const int kRegistered = []() {
  RegisterBenchmarks();
  return 0;
}();

}  // namespace

BENCHMARK_MAIN();
