#include "infeng/kernels/attn/decode.h"

#include "infeng/kernels/attn/kv_dequant.h"
#include "infeng/kernels/attn/softmax_mx.h"
#include "infeng/quant/dequantize.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

#include <immintrin.h>
#include <xmmintrin.h>

namespace infeng::kernels::attn {

namespace {

std::atomic<DecodeKernelPath> g_forced_path{DecodeKernelPath::kAuto};
std::atomic<DecodeKernelPath> g_last_path{DecodeKernelPath::kAuto};
std::atomic<std::size_t> g_tile_override{0};
std::once_flag g_detect_once;
DecodeKernelPath g_detected_path = DecodeKernelPath::kScalar;
bool g_has_avx2 = false;
bool g_has_vnni = false;

struct OutlierSpan {
  std::size_t start{0};
  std::uint16_t count{0};
};

constexpr std::size_t kTileTokens64 = 256;
constexpr std::size_t kTileTokens128 = 128;

inline void set_flush_zero_modes() {
#if defined(__x86_64__) || defined(_M_X64)
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif
}

DecodeKernelPath detect_kernel_path() {
  std::call_once(g_detect_once, [] {
#if defined(__x86_64__) || defined(_M_X64)
    g_has_avx2 = __builtin_cpu_supports("avx2");
#if INFENG_AVX512
    g_has_vnni = __builtin_cpu_supports("avx512vnni");
#else
    g_has_vnni = false;
#endif
    if (g_has_vnni) {
      g_detected_path = DecodeKernelPath::kAvx512Vnni;
    } else if (g_has_avx2) {
      g_detected_path = DecodeKernelPath::kAvx2;
    } else {
      g_detected_path = DecodeKernelPath::kScalar;
    }
#else
    g_has_avx2 = false;
    g_has_vnni = false;
    g_detected_path = DecodeKernelPath::kScalar;
#endif
  });
  return g_detected_path;
}

void ensure_capacity(DecodeScratch* scratch, std::size_t logits, std::size_t head_dim) {
  if (!scratch) {
    return;
  }
  if (scratch->logits.size() < logits) {
    scratch->logits.resize(logits);
  }
  if (scratch->attn.size() < logits) {
    scratch->attn.resize(logits);
  }
  if (scratch->dequant.size() < head_dim) {
    scratch->dequant.resize(head_dim);
  }
  if (scratch->value_tile.size() < head_dim) {
    scratch->value_tile.resize(head_dim);
  }
}

inline __m256i cvtepi8_epi16_avx2(__m256i value) {
  const __m128i lo = _mm256_castsi256_si128(value);
  const __m128i hi = _mm256_extracti128_si256(value, 1);
  const __m128i lo16 = _mm_cvtepi8_epi16(lo);
  const __m128i hi16 = _mm_cvtepi8_epi16(hi);
  __m256i result = _mm256_castsi128_si256(lo16);
  result = _mm256_inserti128_si256(result, hi16, 1);
  return result;
}

inline float horizontal_add(__m256 value) {
  __m128 low = _mm256_castps256_ps128(value);
  __m128 high = _mm256_extractf128_ps(value, 1);
  low = _mm_add_ps(low, high);
  __m128 shuf = _mm_movehdup_ps(low);
  low = _mm_add_ps(low, shuf);
  shuf = _mm_movehl_ps(shuf, low);
  low = _mm_add_ss(low, shuf);
  return _mm_cvtss_f32(low);
}

inline float dot_product_avx2(const std::int8_t* q8,
                              const std::int8_t* k8,
                              std::size_t head_dim) {
  alignas(32) int buffer[8];
  std::int64_t accum = 0;
  std::size_t d = 0;
  for (; d + 32 <= head_dim; d += 32) {
    const __m256i q_chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + d));
    const __m256i k_chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k8 + d));

    const __m256i q16 = cvtepi8_epi16_avx2(q_chunk);
    const __m256i k16 = cvtepi8_epi16_avx2(k_chunk);

    const __m256i prod = _mm256_madd_epi16(k16, q16);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(buffer), prod);
    for (int i = 0; i < 8; ++i) {
      accum += static_cast<std::int64_t>(buffer[i]);
    }
  }

  for (; d < head_dim; ++d) {
    accum += static_cast<std::int64_t>(q8[d]) * static_cast<std::int64_t>(k8[d]);
  }
  return static_cast<float>(accum);
}

void pack_queries_avx2(const infeng::kernels::gemm::MatView& query,
                       std::vector<std::int8_t>& packed,
                       std::vector<float>& scales,
                       std::vector<float>& residuals) {
  const std::size_t num_heads = query.rows;
  const std::size_t head_dim = query.cols;
  packed.resize(num_heads * head_dim);
  scales.resize(num_heads);
  residuals.resize(num_heads * head_dim);

  for (std::size_t head = 0; head < num_heads; ++head) {
    const float* q_row = query.data + head * query.ld;
    float max_abs = 0.0f;
    for (std::size_t d = 0; d < head_dim; ++d) {
      max_abs = std::max(max_abs, std::fabs(q_row[d]));
    }
    float scale = max_abs > 0.0f ? max_abs / 127.0f : 1e-8f;
    scales[head] = scale;
    const float inv_scale = 1.0f / scale;

    std::int8_t* dst = packed.data() + head * head_dim;
    float* residual_row = residuals.data() + head * head_dim;
    for (std::size_t d = 0; d < head_dim; ++d) {
      const float quant = std::round(q_row[d] * inv_scale);
      const int q_val = static_cast<int>(quant);
      const std::int8_t q_int8 = static_cast<std::int8_t>(std::clamp(q_val, -127, 127));
      dst[d] = q_int8;
      residual_row[d] = q_row[d] - static_cast<float>(q_int8) * scale;
    }
  }
}

void compute_outlier_spans(const QuantRowwiseInt8View& view,
                           const std::size_t* rows,
                           std::size_t tile_len,
                           std::vector<OutlierSpan>& spans,
                           std::size_t& cursor) {
  const std::uint32_t* out_rows = view.outlier_rows;
  const std::size_t total = view.num_outliers;
  std::size_t local_cursor = cursor;

  for (std::size_t t = 0; t < tile_len; ++t) {
    const std::size_t row = rows[t];
    if (local_cursor < total && static_cast<std::size_t>(out_rows[local_cursor]) > row) {
      local_cursor = 0;
    }
    while (local_cursor < total && static_cast<std::size_t>(out_rows[local_cursor]) < row) {
      ++local_cursor;
    }
    const std::size_t start = local_cursor;
    while (local_cursor < total && static_cast<std::size_t>(out_rows[local_cursor]) == row) {
      ++local_cursor;
    }
    spans[t] = OutlierSpan{start, static_cast<std::uint16_t>(local_cursor - start)};
  }
  cursor = local_cursor;
}

void pack_k_tile(const QuantRowwiseInt8View& view,
                 const std::size_t* rows,
                 std::size_t tile_len,
                 std::size_t head_dim,
                 std::vector<std::int8_t>& packed,
                 std::vector<float>& row_scales,
                 std::vector<OutlierSpan>& spans,
                 std::size_t& outlier_cursor) {
  packed.resize(tile_len * head_dim);
  row_scales.resize(tile_len);
  spans.resize(tile_len);

  for (std::size_t t = 0; t < tile_len; ++t) {
    const std::size_t row = rows[t];
    const std::int8_t* src = view.data + row * view.ld;
    std::memcpy(packed.data() + t * head_dim, src, head_dim);
    float scale = quant::fp16_to_float(view.scales[row]);
    if (scale == 0.0f) {
      scale = 1e-8f;
    }
    row_scales[t] = scale;
  }

  if (view.outlier_rows && view.outlier_cols && view.outlier_values && view.num_outliers > 0) {
    compute_outlier_spans(view, rows, tile_len, spans, outlier_cursor);
  } else {
    for (std::size_t t = 0; t < tile_len; ++t) {
      spans[t] = OutlierSpan{};
    }
  }
}

void softmax_tile(std::vector<float>& logits, std::size_t length, std::vector<float>& weights) {
  weights.resize(length);
  if (length == 0) {
    return;
  }

  float max_logit = -std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < length; ++i) {
    max_logit = std::max(max_logit, logits[i]);
  }

  float sum = 0.0f;
  for (std::size_t i = 0; i < length; ++i) {
    const float val = std::exp(logits[i] - max_logit);
    weights[i] = val;
    sum += val;
  }

  const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
  for (std::size_t i = 0; i < length; ++i) {
    weights[i] *= inv_sum;
  }
}

void attn_decode_reference(const infeng::kernels::gemm::MatView& query,
                           const QuantKV& key,
                           const QuantKV& value,
                           const KVIndex& index,
                           const infeng::kernels::gemm::MutableMatView& output,
                           DecodeScratch* scratch) {
  if (query.rows == 0 || index.count == 0 || query.cols == 0) {
    return;
  }

  const std::size_t num_heads = query.rows;
  const std::size_t head_dim = query.cols;
  ensure_capacity(scratch, index.count, head_dim);

  std::vector<float> local_logits;
  std::vector<float> local_attn;
  std::vector<float>* logits_ptr = scratch ? &scratch->logits : &local_logits;
  std::vector<float>* attn_ptr = scratch ? &scratch->attn : &local_attn;
  if (!scratch) {
    local_logits.resize(index.count);
    local_attn.resize(index.count);
  }

  auto& logits = *logits_ptr;
  auto& attn = *attn_ptr;

  const BlockConfig block_config{head_dim, 64};

  for (std::size_t head = 0; head < num_heads; ++head) {
    const float* q = query.data + head * query.ld;
    float* out_row = output.data + head * output.ld;
    std::fill(out_row, out_row + head_dim, 0.0f);

    for (std::size_t i = 0; i < index.count; ++i) {
      const std::size_t row = index.rows ? index.rows[i] : i;
      kv_dequant_row_fp32(key.matrix, row, scratch->dequant.data(), block_config);
      kv_apply_outliers_fp32(key.matrix, row, scratch->dequant.data(), block_config);

      double dot = 0.0;
      for (std::size_t d = 0; d < head_dim; ++d) {
        dot += static_cast<double>(q[d]) * static_cast<double>(scratch->dequant[d]);
      }
      logits[i] = static_cast<float>(dot);
    }

    decode_softmax_fp32(logits.data(), index.count, 1.0f, attn.data());

    for (std::size_t i = 0; i < index.count; ++i) {
      const std::size_t row = index.rows ? index.rows[i] : i;
      kv_dequant_row_fp32(value.matrix, row, scratch->dequant.data(), block_config);
      kv_apply_outliers_fp32(value.matrix, row, scratch->dequant.data(), block_config);
      const float weight = attn[i];
      for (std::size_t d = 0; d < head_dim; ++d) {
        out_row[d] += weight * scratch->dequant[d];
      }
    }
  }
}

void attn_decode_avx2(const infeng::kernels::gemm::MatView& query,
                      const QuantKV& key,
                      const QuantKV& value,
                      const KVIndex& index,
                      const infeng::kernels::gemm::MutableMatView& output,
                      DecodeScratch* scratch) {
  if (query.rows == 0 || index.count == 0 || query.cols == 0) {
    return;
  }

  const std::size_t num_heads = query.rows;
  const std::size_t head_dim = query.cols;
  const std::size_t total_tokens = index.count;

  ensure_capacity(scratch, total_tokens, head_dim);

  std::vector<std::int8_t> q_packed;
  std::vector<float> q_scales;
  std::vector<float> q_residuals;
  pack_queries_avx2(query, q_packed, q_scales, q_residuals);

  const std::size_t tile_len_max = attn_decode_tile_tokens(head_dim);
  std::vector<std::size_t> tile_rows(tile_len_max);
  std::vector<std::int8_t> k_tile;
  std::vector<float> k_scales;
  std::vector<OutlierSpan> k_outliers;

  const QuantRowwiseInt8View& key_view = key.matrix;
  const BlockConfig block_config{head_dim, 64};

  for (std::size_t head = 0; head < num_heads; ++head) {
    float* out_row = output.data + head * output.ld;
    std::fill(out_row, out_row + head_dim, 0.0f);

    const float* q_head = query.data + head * query.ld;
    const std::int8_t* q8 = q_packed.data() + head * head_dim;
    const float q_scale = q_scales[head];
    const float* q_residual = q_residuals.data() + head * head_dim;

    float* logits_all = scratch->logits.data();
    float* weights_all = scratch->attn.data();

    std::size_t outlier_cursor = 0;

    // Pass 1: compute logits for all tokens.
    for (std::size_t base = 0; base < total_tokens; base += tile_len_max) {
      const std::size_t tile_len = std::min(tile_len_max, total_tokens - base);
      for (std::size_t t = 0; t < tile_len; ++t) {
        tile_rows[t] = index.rows ? index.rows[base + t] : (base + t);
      }

      pack_k_tile(key_view, tile_rows.data(), tile_len, head_dim, k_tile, k_scales, k_outliers, outlier_cursor);

      for (std::size_t t = 0; t < tile_len; ++t) {
        const std::int8_t* k_row = k_tile.data() + t * head_dim;
        float dot = dot_product_avx2(q8, k_row, head_dim);
        dot *= (q_scale * k_scales[t]);

        const __m256 scale_vec = _mm256_set1_ps(k_scales[t]);
        __m256 residual_acc = _mm256_setzero_ps();
        std::size_t rd = 0;
        for (; rd + 8 <= head_dim; rd += 8) {
          const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(k_row + rd));
          const __m128i ints16 = _mm_cvtepi8_epi16(bytes);
          const __m256i ints32 = _mm256_cvtepi16_epi32(ints16);
          __m256 k_vals = _mm256_cvtepi32_ps(ints32);
          k_vals = _mm256_mul_ps(k_vals, scale_vec);
          const __m256 residual_vec = _mm256_loadu_ps(q_residual + rd);
          residual_acc = _mm256_fmadd_ps(residual_vec, k_vals, residual_acc);
        }
        float residual_correction = horizontal_add(residual_acc);
        for (; rd < head_dim; ++rd) {
          residual_correction += q_residual[rd] * static_cast<float>(k_row[rd]) * k_scales[t];
        }
        dot += residual_correction;

        const OutlierSpan span = k_outliers[t];
        if (span.count > 0) {
          for (std::size_t idx = 0; idx < span.count; ++idx) {
            const std::size_t global_idx = span.start + idx;
            const std::size_t col = static_cast<std::size_t>(key_view.outlier_cols[global_idx]);
            const float stored = quant::fp16_to_float(key_view.outlier_values[global_idx]);
            const float base_val = static_cast<float>(k_row[col]) * k_scales[t];
            dot += q_head[col] * (stored - base_val);
          }
        }
        logits_all[base + t] = dot;
      }
    }

    decode_softmax_fp32(logits_all, total_tokens, 1.0f, weights_all);

    // Pass 2: accumulate PV using softmax weights.
    for (std::size_t base = 0; base < total_tokens; base += tile_len_max) {
      const std::size_t tile_len = std::min(tile_len_max, total_tokens - base);
      for (std::size_t t = 0; t < tile_len; ++t) {
        tile_rows[t] = index.rows ? index.rows[base + t] : (base + t);
      }

      for (std::size_t t = 0; t < tile_len; ++t) {
        const float weight = weights_all[base + t];
        if (weight == 0.0f) {
          continue;
        }
        const std::size_t row = tile_rows[t];
        kv_dequant_row_fp32(value.matrix, row, scratch->value_tile.data(), block_config);
        kv_apply_outliers_fp32(value.matrix, row, scratch->value_tile.data(), block_config);

        const __m256 w_vec = _mm256_set1_ps(weight);
        std::size_t d = 0;
        for (; d + 8 <= head_dim; d += 8) {
          __m256 out_vec = _mm256_loadu_ps(out_row + d);
          __m256 val_vec = _mm256_loadu_ps(scratch->value_tile.data() + d);
          __m256 res = _mm256_fmadd_ps(val_vec, w_vec, out_vec);
          _mm256_storeu_ps(out_row + d, res);
        }
        for (; d < head_dim; ++d) {
          out_row[d] += weight * scratch->value_tile[d];
        }
      }
    }
  }
}

DecodeKernelPath run_scalar_kernel(const infeng::kernels::gemm::MatView& query,
                                   const QuantKV& key,
                                   const QuantKV& value,
                                   const KVIndex& index,
                                   const infeng::kernels::gemm::MutableMatView& output,
                                   DecodeScratch* scratch) {
  attn_decode_reference(query, key, value, index, output, scratch);
  return DecodeKernelPath::kScalar;
}

DecodeKernelPath run_avx2_kernel(const infeng::kernels::gemm::MatView& query,
                                 const QuantKV& key,
                                 const QuantKV& value,
                                 const KVIndex& index,
                                 const infeng::kernels::gemm::MutableMatView& output,
                                 DecodeScratch* scratch) {
  attn_decode_avx2(query, key, value, index, output, scratch);
  return DecodeKernelPath::kAvx2;
}

DecodeKernelPath run_vnni_kernel(const infeng::kernels::gemm::MatView& query,
                                 const QuantKV& key,
                                 const QuantKV& value,
                                 const KVIndex& index,
                                 const infeng::kernels::gemm::MutableMatView& output,
                                 DecodeScratch* scratch) {
#if INFENG_AVX512
  // TODO: implement dedicated VNNI path. For now, reuse AVX2 implementation.
  attn_decode_avx2(query, key, value, index, output, scratch);
  return g_has_avx2 ? DecodeKernelPath::kAvx2 : DecodeKernelPath::kScalar;
#else
  (void)query;
  (void)key;
  (void)value;
  (void)index;
  (void)output;
  (void)scratch;
  return DecodeKernelPath::kScalar;
#endif
}

}  // namespace

void attn_decode_int8_qk(const infeng::kernels::gemm::MatView& query,
                         const QuantKV& key,
                         const QuantKV& value,
                         const KVIndex& index,
                         const infeng::kernels::gemm::MutableMatView& output,
                         DecodeScratch* scratch) {
  set_flush_zero_modes();

  DecodeKernelPath desired = g_forced_path.load(std::memory_order_relaxed);
  if (desired == DecodeKernelPath::kAuto) {
    desired = detect_kernel_path();
  } else {
    // Ensure capabilities are populated for fallbacks.
    detect_kernel_path();
  }

  DecodeKernelPath actual = DecodeKernelPath::kScalar;

  switch (desired) {
    case DecodeKernelPath::kAvx512Vnni:
      if (g_has_vnni) {
        actual = run_vnni_kernel(query, key, value, index, output, scratch);
        break;
      }
      [[fallthrough]];
    case DecodeKernelPath::kAvx2:
      if (g_has_avx2) {
        actual = run_avx2_kernel(query, key, value, index, output, scratch);
        break;
      }
      [[fallthrough]];
    case DecodeKernelPath::kScalar:
    case DecodeKernelPath::kAuto:
    default:
      actual = run_scalar_kernel(query, key, value, index, output, scratch);
      break;
  }

  g_last_path.store(actual, std::memory_order_relaxed);
}

void attn_decode_set_kernel_path(DecodeKernelPath path) {
  g_forced_path.store(path, std::memory_order_relaxed);
}

DecodeKernelPath attn_decode_get_kernel_path() {
  return g_last_path.load(std::memory_order_relaxed);
}

const char* attn_decode_kernel_path_name(DecodeKernelPath path) {
  switch (path) {
    case DecodeKernelPath::kScalar:
      return "scalar";
    case DecodeKernelPath::kAvx2:
      return "avx2";
    case DecodeKernelPath::kAvx512Vnni:
      return "avx512_vnni";
    case DecodeKernelPath::kAuto:
    default:
      return "auto";
  }
}

void attn_decode_set_tile_tokens_override(std::size_t tokens) {
  g_tile_override.store(tokens, std::memory_order_relaxed);
}

std::size_t attn_decode_tile_tokens(std::size_t head_dim) {
  const std::size_t override = g_tile_override.load(std::memory_order_relaxed);
  if (override > 0) {
    return override;
  }
  return head_dim <= 64 ? kTileTokens64 : kTileTokens128;
}

}  // namespace infeng::kernels::attn
