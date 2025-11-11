#include "infeng/kernels/gemm/fused_base_lora.h"
#include "infeng/kernels/gemm/pack.h"
#include "infeng/kernels/ref/gemm_ref.h"

#include <benchmark/benchmark.h>

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace infeng::kernels::gemm;

namespace {

struct ProblemConfig {
  std::size_t rows;
  std::size_t cols;
  std::size_t batch;
  std::size_t rank;
};

struct RunMetrics {
  double fused_ms{0.0};
  double baseline_ms{0.0};
  double base_ms{0.0};
  double speedup{0.0};
  double lora_tax_pct{0.0};
  float diff{0.0f};
};

std::vector<float> random_matrix(std::size_t rows, std::size_t cols, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> data(rows * cols);
  for (float& v : data) {
    v = dist(rng);
  }
  return data;
}

void write_csv(const ProblemConfig& cfg,
               double fused_ms,
               double baseline_ms,
               double base_ms,
               double speedup,
               double lora_tax_pct,
               double fused_tokens_per_s,
               double baseline_tokens_per_s,
               double l2_diff) {
  std::filesystem::create_directories("reports");
  const auto path = std::filesystem::path("reports") / "gemm_lora.csv";
  const bool existed = std::filesystem::exists(path);
  std::ofstream out(path, existed ? std::ios::app : std::ios::trunc);
  if (!existed) {
    out << "bench,rows,cols,batch,rank,fused_ms,baseline_ms,base_ms,speedup,lora_tax_pct,"
           "tokens_per_s,baseline_tokens_per_s,l2_diff\n";
  }
  out << "fused_base_lora,"
      << cfg.rows << ',' << cfg.cols << ',' << cfg.batch << ',' << cfg.rank << ','
      << fused_ms << ',' << baseline_ms << ',' << base_ms << ','
      << speedup << ',' << lora_tax_pct << ','
      << fused_tokens_per_s << ',' << baseline_tokens_per_s << ',' << l2_diff << "\n";

  out << "baseline_two_call,"
      << cfg.rows << ',' << cfg.cols << ',' << cfg.batch << ',' << cfg.rank << ','
      << baseline_ms << ',' << baseline_ms << ',' << base_ms << ','
      << 1.0 << ',' << 0.0 << ','
      << baseline_tokens_per_s << ',' << baseline_tokens_per_s << ',' << l2_diff << "\n";
}

RunMetrics run_problem(const ProblemConfig& cfg) {
  std::mt19937 rng(1234);

  auto W_data = random_matrix(cfg.rows, cfg.cols, rng);
  auto X_data = random_matrix(cfg.cols, cfg.batch, rng);
  auto A_data = random_matrix(cfg.rows, cfg.rank, rng);
  auto B_data = random_matrix(cfg.rank, cfg.cols, rng);

  PackedRowwiseInt8Matrix W_qpack = pack_matrix_rowwise_int8(W_data.data(), cfg.rows, cfg.cols, cfg.cols);
  PackedMatrix W_fpack = pack_matrix(W_data.data(), cfg.rows, cfg.cols, cfg.cols);
  PackedMatrix X_pack = pack_matrix(X_data.data(), cfg.cols, cfg.batch, cfg.batch);
  PackedMatrix A_pack = pack_matrix(A_data.data(), cfg.rows, cfg.rank, cfg.rank);
  PackedMatrix B_pack = pack_matrix(B_data.data(), cfg.rank, cfg.cols, cfg.cols);

  RowwiseInt8MatView W_q = make_view(W_qpack);
  MatView W_float = make_view(W_fpack);
  MatView X = make_view(X_pack);

  LoRAView adapter{make_view(A_pack), make_view(B_pack), cfg.rank};

  std::vector<float> y_fused(cfg.rows * cfg.batch, 0.0f);
  std::vector<float> y_baseline(cfg.rows * cfg.batch, 0.0f);
  std::vector<float> y_base(cfg.rows * cfg.batch, 0.0f);

  MutableMatView Y_fused = make_mutable_view(y_fused.data(), cfg.rows, cfg.batch, cfg.batch);
  MutableMatView Y_baseline = make_mutable_view(y_baseline.data(), cfg.rows, cfg.batch, cfg.batch);
  MutableMatView Y_base = make_mutable_view(y_base.data(), cfg.rows, cfg.batch, cfg.batch);

  FusedGemmContext fused_ctx;
  FusedGemmContext base_ctx;

  auto measure = [](auto&& fn) {
    const auto start = std::chrono::high_resolution_clock::now();
    fn();
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  };

  const double fused_ms = measure([&]() {
    std::fill(y_fused.begin(), y_fused.end(), 0.0f);
    fused_base_lora_gemm(W_q, X, &adapter, 1, Y_fused, &fused_ctx);
  });

  const double baseline_ms = measure([&]() {
    std::fill(y_baseline.begin(), y_baseline.end(), 0.0f);
    infeng::kernels::ref::fused_base_lora_ref(W_float, X, &adapter, 1, Y_baseline);
  });

  const double base_ms = measure([&]() {
    std::fill(y_base.begin(), y_base.end(), 0.0f);
    fused_base_lora_gemm(W_q, X, nullptr, 0, Y_base, &base_ctx);
  });

  float diff = 0.0f;
  for (std::size_t idx = 0; idx < y_fused.size(); ++idx) {
    const float delta = y_fused[idx] - y_baseline[idx];
    diff += delta * delta;
  }
  diff = std::sqrt(diff);

  const double speedup = baseline_ms > 0.0 ? baseline_ms / fused_ms : 0.0;
  const double lora_tax_pct = base_ms > 0.0 ? (fused_ms / base_ms - 1.0) * 100.0 : 0.0;

  return RunMetrics{
      fused_ms,
      baseline_ms,
      base_ms,
      speedup,
      lora_tax_pct,
      diff,
  };
}

void GemmBenchmark(benchmark::State& state, ProblemConfig cfg) {
  RunMetrics metrics{};
  for (auto _ : state) {
    metrics = run_problem(cfg);
  }

  const double tokens = static_cast<double>(cfg.rows) * static_cast<double>(cfg.batch);
  const double fused_tokens_per_s = metrics.fused_ms > 0.0 ? tokens / (metrics.fused_ms / 1000.0) : 0.0;
  const double baseline_tokens_per_s =
      metrics.baseline_ms > 0.0 ? tokens / (metrics.baseline_ms / 1000.0) : 0.0;

  state.counters["fused_ms"] = metrics.fused_ms;
  state.counters["baseline_ms"] = metrics.baseline_ms;
  state.counters["base_ms"] = metrics.base_ms;
  state.counters["speedup"] = metrics.speedup;
  state.counters["lora_tax_pct"] = metrics.lora_tax_pct;
  state.counters["l2_diff"] = metrics.diff;

  if (state.thread_index() == 0) {
    write_csv(cfg,
              metrics.fused_ms,
              metrics.baseline_ms,
              metrics.base_ms,
              metrics.speedup,
              metrics.lora_tax_pct,
              fused_tokens_per_s,
              baseline_tokens_per_s,
              static_cast<double>(metrics.diff));
  }
}

void RegisterBenchmarks() {
  const std::array<std::size_t, 4> ranks = {8, 16, 32, 64};
  const std::array<std::size_t, 3> batches = {1, 4, 8};
  const std::size_t rows = 4096;
  const std::size_t cols = 4096;

  for (std::size_t rank : ranks) {
    for (std::size_t batch : batches) {
      const std::string name = "FusedGemm/rows" + std::to_string(rows) + "/cols" + std::to_string(cols) +
                               "/rank" + std::to_string(rank) + "/batch" + std::to_string(batch);
      benchmark::RegisterBenchmark(name.c_str(), GemmBenchmark, ProblemConfig{rows, cols, batch, rank})
          ->Iterations(1);
    }
  }
}

const int kRegistered = []() {
  RegisterBenchmarks();
  return 0;
}();

}  // namespace

BENCHMARK_MAIN();
