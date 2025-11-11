#include "infeng/kernels/attn/decode.h"
#include "infeng/kernels/attn/softmax_mx.h"
#include "infeng/kernels/gemm/pack.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using infeng::kernels::attn::DecodeKernelPath;
using infeng::kernels::attn::attn_decode_get_kernel_path;
using infeng::kernels::attn::attn_decode_int8_qk;
using infeng::kernels::attn::attn_decode_tile_tokens;
using infeng::kernels::attn::DecodeScratch;
using infeng::kernels::attn::KVIndex;
using infeng::kernels::attn::QuantKV;
using infeng::kernels::attn::QuantRowwiseInt8View;
using infeng::kernels::gemm::MatView;
using infeng::kernels::gemm::MutableMatView;
using infeng::kernels::gemm::PackedMatrix;
using infeng::kernels::gemm::PackedRowwiseInt8Matrix;
using infeng::kernels::gemm::make_mutable_view;
using infeng::kernels::gemm::make_view;
using infeng::kernels::gemm::pack_matrix;
using infeng::kernels::gemm::pack_matrix_rowwise_int8;

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

std::vector<float> seeded_random(std::size_t rows, std::size_t cols, std::uint32_t seed) {
  std::mt19937 rng(seed);
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

}  // namespace

int main() {
  const std::size_t seq_len = 256;
  const std::size_t head_dim = 64;
  const std::size_t heads = 8;

auto q_data = seeded_random(heads, head_dim, 123u);
auto k_data = seeded_random(seq_len, head_dim, 456u);
auto v_data = seeded_random(seq_len, head_dim, 789u);

  PackedMatrix Q_pack = pack_matrix(q_data.data(), heads, head_dim, head_dim);
  PackedMatrix K_pack = pack_matrix(k_data.data(), seq_len, head_dim, head_dim);
  PackedMatrix V_pack = pack_matrix(v_data.data(), seq_len, head_dim, head_dim);

  PackedRowwiseInt8Matrix K_quant = pack_matrix_rowwise_int8(k_data.data(), seq_len, head_dim, head_dim);
  PackedRowwiseInt8Matrix V_quant = pack_matrix_rowwise_int8(v_data.data(), seq_len, head_dim, head_dim);

  QuantKV K_view{to_view(K_quant)};
  QuantKV V_view{to_view(V_quant)};

  std::vector<std::size_t> rows(seq_len);
  for (std::size_t i = 0; i < seq_len; ++i) {
    rows[i] = i;
  }

  KVIndex index{rows.data(), rows.size()};

  MatView Q_view = make_view(Q_pack);
  std::vector<float> fused_output(heads * head_dim, 0.0f);
  MutableMatView fused_view = make_mutable_view(fused_output.data(), heads, head_dim, head_dim);

  DecodeScratch scratch;
  attn_decode_int8_qk(Q_view, K_view, V_view, index, fused_view, &scratch);

  BaselineResult baseline = run_baseline(Q_view, K_pack, V_pack, index, head_dim, heads);

  const double diff = l2_diff(fused_output, baseline.output);
  assert(diff <= 1e-3);

  const DecodeKernelPath path = attn_decode_get_kernel_path();
#if defined(__x86_64__) || defined(_M_X64)
  if (__builtin_cpu_supports("avx2")) {
    assert(path == DecodeKernelPath::kAvx2 || path == DecodeKernelPath::kAvx512Vnni);
  } else {
    assert(path == DecodeKernelPath::kScalar);
  }
#else
  assert(path == DecodeKernelPath::kScalar);
#endif

  const std::size_t expected_tile = (head_dim <= 64) ? 256 : 128;
  assert(attn_decode_tile_tokens(head_dim) == expected_tile);

  const std::size_t fused_scratch =
      scratch.logits.size() + scratch.attn.size() + scratch.dequant.size() + scratch.value_tile.size();
  const std::size_t baseline_scratch = index.count * 2 + baseline.output.size();
  assert(static_cast<double>(fused_scratch) <= 1.25 * static_cast<double>(baseline_scratch));

  return 0;
}
