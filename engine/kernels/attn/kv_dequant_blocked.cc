#include "infeng/kernels/attn/kv_dequant.h"

#include "infeng/quant/dequantize.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>

#if INFENG_ENABLE_AVX512
#include <immintrin.h>
#endif

namespace infeng::kernels::attn {

namespace {

inline float fp16_to_float(std::uint16_t value) {
  return quant::fp16_to_float(value);
}

inline void scalar_dequant(const QuantRowwiseInt8View& view,
                           std::size_t row,
                           float* dst,
                           const BlockConfig& config) {
  const std::size_t cols = config.head_dim;
  const std::int8_t* data = view.data + row * view.ld;
  const float scale = fp16_to_float(view.scales[row]);
  const float actual_scale = scale == 0.0f ? 1e-8f : scale;
  for (std::size_t c = 0; c < cols; ++c) {
    dst[c] = static_cast<float>(data[c]) * actual_scale;
  }
}

inline void apply_outliers_scalar(const QuantRowwiseInt8View& view,
                                  std::size_t row,
                                  float* dst,
                                  const BlockConfig& config) {
  (void)config;
  if (!view.outlier_rows || !view.outlier_cols || !view.outlier_values || view.num_outliers == 0) {
    return;
  }
  const std::uint32_t* rows = view.outlier_rows;
  const std::uint32_t target = static_cast<std::uint32_t>(row);
  const std::uint32_t* begin = rows;
  const std::uint32_t* end = rows + view.num_outliers;
  const std::uint32_t* lower = std::lower_bound(begin, end, target);
  if (lower == end || *lower != target) {
    return;
  }
  const std::uint32_t* upper = std::upper_bound(lower, end, target);
  const std::size_t start = static_cast<std::size_t>(lower - begin);
  const std::size_t finish = static_cast<std::size_t>(upper - begin);
  for (std::size_t idx = start; idx < finish; ++idx) {
    const std::size_t col = static_cast<std::size_t>(view.outlier_cols[idx]);
    if (col < config.head_dim) {
      dst[col] = fp16_to_float(view.outlier_values[idx]);
    }
  }
}

}  // namespace

void kv_dequant_row_fp32(const QuantRowwiseInt8View& view,
                         std::size_t row,
                         float* dst,
                         const BlockConfig& config) {
  if (!dst || row >= view.rows) {
    return;
  }

  scalar_dequant(view, row, dst, config);
}

void kv_apply_outliers_fp32(const QuantRowwiseInt8View& view,
                            std::size_t row,
                            float* dst,
                            const BlockConfig& config) {
  if (!dst || row >= view.rows) {
    return;
  }
  apply_outliers_scalar(view, row, dst, config);
}

}  // namespace infeng::kernels::attn
