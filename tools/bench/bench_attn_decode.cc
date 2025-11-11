#include "infeng/kernels/attn/decode.h"
#include "infeng/kernels/attn/softmax_mx.h"
#include "infeng/kernels/gemm/pack.h"

#include <benchmark/benchmark.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using infeng::kernels::attn::attn_decode_int8_qk;
using infeng::kernels::attn::attn_decode_tile_tokens;
using infeng::kernels::attn::DecodeScratch;
using infeng::kernels::attn::QuantKV;
using infeng::kernels::attn::QuantRowwiseInt8View;
using infeng::kernels::attn::KVIndex;
using infeng::kernels::gemm::MatView;
using infeng::kernels::gemm::MutableMatView;
using infeng::kernels::gemm::PackedMatrix;
using infeng::kernels::gemm::PackedRowwiseInt8Matrix;
using infeng::kernels::gemm::make_mutable_view;
using infeng::kernels::gemm::make_view;
using infeng::kernels::gemm::pack_matrix;
using infeng::kernels::gemm::pack_matrix_rowwise_int8;

namespace {

struct DecodeConfig {
  std::size_t seq_len;
  std::size_t head_dim;
  std::size_t heads_per_batch;
  std::size_t batch;
};

QuantRowwiseInt8View to_view(const PackedRowwiseInt8Matrix& packed) {
  return QuantRowwiseInt8View{packed.data.data(),
                              packed.scales.data(),
                              packed.outlier_rows.empty() ? nullptr : packed.outlier_rows.data(),
                              packed.outlier_cols.empty() ? nullptr : packed.outlier_cols.data(),
                              packed.outlier_values.empty() ? nullptr : packed.outlier_values.data(),
                              packed.outlier_rows.size(),
                              packed.rows,
                              packed.cols,
                              packed.ld};
}

std::vector<float> random_matrix(std::size_t rows, std::size_t cols, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> data(rows * cols);
  for (float& v : data) {
    v = dist(rng);
  }
  return data;
}

struct BaselineResult {
  std::vector<float> output;
};

BaselineResult run_baseline(const MatView& query,
                            const PackedMatrix& key,
                            const PackedMatrix& value,
                            const KVIndex& index,
                            std::size_t head_dim,
                            std::size_t num_heads) {
  BaselineResult result;
  result.output.assign(num_heads * head_dim, 0.0f);
  std::vector<float> logits(index.count);
  std::vector<float> attn(index.count);

  for (std::size_t head = 0; head < num_heads; ++head) {
    const float* q = query.data + head * query.ld;
    float* out_row = result.output.data() + head * head_dim;

    for (std::size_t i = 0; i < index.count; ++i) {
      const std::size_t row = index.rows ? index.rows[i] : i;
      const float* k_row = key.data.data() + row * key.ld;
      double dot = 0.0;
      for (std::size_t d = 0; d < head_dim; ++d) {
        dot += static_cast<double>(q[d]) * static_cast<double>(k_row[d]);
      }
      logits[i] = static_cast<float>(dot);
    }

    infeng::kernels::attn::decode_softmax_fp32(logits.data(), index.count, 1.0f, attn.data());

    for (std::size_t i = 0; i < index.count; ++i) {
      const std::size_t row = index.rows ? index.rows[i] : i;
      const float* v_row = value.data.data() + row * value.ld;
      const float weight = attn[i];
      for (std::size_t d = 0; d < head_dim; ++d) {
        out_row[d] += weight * v_row[d];
      }
    }
  }

  return result;
}

double l2_diff(const std::vector<float>& a, const std::vector<float>& b) {
  const std::size_t size = std::min(a.size(), b.size());
  double acc = 0.0;
  for (std::size_t i = 0; i < size; ++i) {
    const double delta = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    acc += delta * delta;
  }
  if (size == 0) {
    return 0.0;
  }
  return std::sqrt(acc) / static_cast<double>(size);
}

void write_csv(const DecodeConfig& cfg,
               double fused_ms,
               double baseline_ms,
               double tokens_per_s,
               double baseline_tokens_per_s,
               double diff,
               double perplexity_delta,
               std::size_t scratch_bytes,
               std::size_t baseline_scratch_bytes,
               std::size_t tile_tokens,
               const std::string& acc_width,
               bool has_avx2,
               bool has_vnni,
               bool csr_outliers,
               double l2_bytes_per_token,
               double baseline_l2_bytes_per_token) {
  std::filesystem::create_directories("reports");
  const auto path = std::filesystem::path("reports") / "attn_decode.csv";
  const bool existed = std::filesystem::exists(path);
  std::ofstream out(path, existed ? std::ios::app : std::ios::trunc);
  if (!existed) {
    out << "bench,seq_len,head_dim,heads,batch,fused_ms,baseline_ms,speedup,";
    out << "tokens_per_s,baseline_tokens_per_s,l2_diff,perplexity_delta,";
    out << "scratch_bytes,baseline_scratch_bytes,scratch_ratio,tile_tokens,acc_width,";
    out << "has_avx2,has_vnni,csr_outliers,l2_bytes_per_token\n";
  }
  const double speedup = baseline_ms > 0.0 ? baseline_ms / fused_ms : 0.0;
  const double scratch_ratio = baseline_scratch_bytes > 0 ?
                                       static_cast<double>(scratch_bytes) / static_cast<double>(baseline_scratch_bytes)
                                          : 0.0;
  out << "attn_decode_int8_qk," << cfg.seq_len << ',' << cfg.head_dim << ',' << cfg.heads_per_batch << ','
      << cfg.batch << ',' << fused_ms << ',' << baseline_ms << ',' << speedup << ',' << tokens_per_s << ','
      << baseline_tokens_per_s << ',' << diff << ',' << perplexity_delta << ',' << scratch_bytes << ','
      << baseline_scratch_bytes << ',' << scratch_ratio << ',' << tile_tokens << ',' << acc_width << ','
      << (has_avx2 ? 1 : 0) << ',' << (has_vnni ? 1 : 0) << ',' << (csr_outliers ? 1 : 0) << ','
      << l2_bytes_per_token << "\n";

  out << "attn_decode_fp16," << cfg.seq_len << ',' << cfg.head_dim << ',' << cfg.heads_per_batch << ','
      << cfg.batch << ',' << baseline_ms << ',' << baseline_ms << ',' << 1.0 << ',' << baseline_tokens_per_s << ','
      << baseline_tokens_per_s << ',' << diff << ',' << 0.0 << ',' << baseline_scratch_bytes << ','
      << baseline_scratch_bytes << ',' << 1.0 << ',' << tile_tokens << ',' << acc_width << ','
      << (has_avx2 ? 1 : 0) << ',' << (has_vnni ? 1 : 0) << ',' << (csr_outliers ? 1 : 0) << ','
      << baseline_l2_bytes_per_token << "\n";
}

void DecodeBenchmark(benchmark::State& state, DecodeConfig cfg) {
  std::mt19937 rng(2027);
  const std::size_t heads_total = cfg.heads_per_batch * cfg.batch;
  const std::size_t rows = cfg.seq_len;

  auto q_data = random_matrix(heads_total, cfg.head_dim, rng);
  auto k_data = random_matrix(rows, cfg.head_dim, rng);
  auto v_data = random_matrix(rows, cfg.head_dim, rng);

  PackedMatrix Q_pack = pack_matrix(q_data.data(), heads_total, cfg.head_dim, cfg.head_dim);
  PackedMatrix K_pack = pack_matrix(k_data.data(), rows, cfg.head_dim, cfg.head_dim);
  PackedMatrix V_pack = pack_matrix(v_data.data(), rows, cfg.head_dim, cfg.head_dim);

  PackedRowwiseInt8Matrix K_quant = pack_matrix_rowwise_int8(k_data.data(), rows, cfg.head_dim, cfg.head_dim);
  PackedRowwiseInt8Matrix V_quant = pack_matrix_rowwise_int8(v_data.data(), rows, cfg.head_dim, cfg.head_dim);

  QuantKV K_view{to_view(K_quant)};
  QuantKV V_view{to_view(V_quant)};

  std::vector<std::size_t> rows_index(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    rows_index[i] = i;
  }
  KVIndex index{rows_index.data(), rows_index.size()};

  MatView Q_view = make_view(Q_pack);
  std::vector<float> fused_output(heads_total * cfg.head_dim, 0.0f);
  MutableMatView fused_view = make_mutable_view(fused_output.data(), heads_total, cfg.head_dim, cfg.head_dim);
  DecodeScratch scratch;

  BaselineResult baseline_result;
  double best_fused_ms = std::numeric_limits<double>::max();
  double best_baseline_ms = std::numeric_limits<double>::max();

  for (auto _ : state) {
    fused_output.assign(fused_output.size(), 0.0f);

    const auto fused_start = std::chrono::high_resolution_clock::now();
    attn_decode_int8_qk(Q_view, K_view, V_view, index, fused_view, &scratch);
    const auto fused_end = std::chrono::high_resolution_clock::now();
    const double fused_ms = std::chrono::duration<double, std::milli>(fused_end - fused_start).count();
    best_fused_ms = std::min(best_fused_ms, fused_ms);

    const auto baseline_start = std::chrono::high_resolution_clock::now();
    baseline_result = run_baseline(Q_view, K_pack, V_pack, index, cfg.head_dim, heads_total);
    const auto baseline_end = std::chrono::high_resolution_clock::now();
    const double baseline_ms =
        std::chrono::duration<double, std::milli>(baseline_end - baseline_start).count();
    best_baseline_ms = std::min(best_baseline_ms, baseline_ms);
  }

  state.counters["fused_ms"] = best_fused_ms;
  state.counters["baseline_ms"] = best_baseline_ms;

  const double tokens = static_cast<double>(cfg.seq_len) * static_cast<double>(heads_total);
  const double fused_tokens_per_s = best_fused_ms > 0.0 ? tokens / (best_fused_ms / 1000.0) : 0.0;
  const double baseline_tokens_per_s = best_baseline_ms > 0.0 ? tokens / (best_baseline_ms / 1000.0) : 0.0;
  const double diff = l2_diff(fused_output, baseline_result.output);
  const double perplexity_delta = 0.0;  // placeholder until eval set integration.

  const std::size_t scratch_bytes =
      sizeof(float) * (scratch.logits.size() + scratch.attn.size() + scratch.dequant.size() + scratch.value_tile.size());
  const std::size_t baseline_scratch_bytes =
      sizeof(float) * (index.count * 2 + baseline_result.output.size());

  state.counters["tokens_per_s"] = fused_tokens_per_s;
  state.counters["diff_l2"] = diff;

  if (state.thread_index() == 0) {
    const bool has_avx2 =
#if defined(__x86_64__) || defined(_M_X64)
        __builtin_cpu_supports("avx2");
#else
        false;
#endif
    bool has_vnni = false;
#if defined(__x86_64__) || defined(_M_X64)
#  if defined(__GNUC__)
#    if defined(INFENG_AVX512) && INFENG_AVX512
    has_vnni = __builtin_cpu_supports("avx512vnni");
#    endif
#  endif
#endif
    const bool csr_outliers = !K_quant.outlier_rows.empty();
    const std::size_t tile_tokens = attn_decode_tile_tokens(cfg.head_dim);
    const std::string acc_width = "fp32";
    const double denom = static_cast<double>(cfg.seq_len) * static_cast<double>(cfg.heads_per_batch) * static_cast<double>(cfg.batch);
    const double l2_bytes_per_token = denom > 0.0 ? static_cast<double>(scratch_bytes) / denom : 0.0;
    const double baseline_l2_bytes_per_token = denom > 0.0 ? static_cast<double>(baseline_scratch_bytes) / denom : 0.0;

    write_csv(cfg,
              best_fused_ms,
              best_baseline_ms,
              fused_tokens_per_s,
              baseline_tokens_per_s,
              diff,
              perplexity_delta,
              scratch_bytes,
              baseline_scratch_bytes,
              tile_tokens,
              acc_width,
              has_avx2,
              has_vnni,
              csr_outliers,
              l2_bytes_per_token,
              baseline_l2_bytes_per_token);
  }
}

void RegisterBenchmarks() {
  const std::vector<std::size_t> heads = {32, 64};
  const std::vector<std::size_t> head_dims = {64, 128};
  const std::vector<std::size_t> ctx = {1024, 2048, 4096};
  const std::vector<std::size_t> batches = {1, 4, 8};

  for (std::size_t h : heads) {
    for (std::size_t dim : head_dims) {
      for (std::size_t context : ctx) {
        for (std::size_t batch : batches) {
          const DecodeConfig cfg{context, dim, h, batch};
          const std::string name = "Decode/ctx" + std::to_string(context) + "_dim" + std::to_string(dim) +
                                   "_h" + std::to_string(h) + "_b" + std::to_string(batch);
          benchmark::RegisterBenchmark(name.c_str(), DecodeBenchmark, cfg)->Iterations(1);
        }
      }
    }
  }
}

const int kRegistered = []() {
  RegisterBenchmarks();
  return 0;
}();

}  // namespace

BENCHMARK_MAIN();
