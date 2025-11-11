#include "infeng/quant/packers/int4_rowwise.h"

#include <algorithm>
#include <cmath>

namespace infeng::quant::packers {

namespace {

std::uint16_t float_to_fp16(float value) {
  union {
    float f;
    std::uint32_t u;
  } in{value};

  const std::uint32_t sign = (in.u >> 16) & 0x8000u;
  std::uint32_t mantissa = in.u & 0x007FFFFFu;
  std::int32_t exponent = static_cast<std::int32_t>((in.u >> 23) & 0xFF) - 127 + 15;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    mantissa |= 0x00800000u;
    const std::uint32_t shift = static_cast<std::uint32_t>(1 - exponent);
    const std::uint32_t fraction = mantissa >> (shift + 13);
    return static_cast<std::uint16_t>(sign | fraction);
  }
  if (exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  }

  const std::uint16_t fraction = static_cast<std::uint16_t>((mantissa + 0x00001000u) >> 13);
  return static_cast<std::uint16_t>(sign | (exponent << 10) | fraction);
}

inline void store_nibble(std::vector<std::uint8_t>& buffer,
                         std::size_t byte_index,
                         bool high_half,
                         std::uint8_t value) {
  const std::uint8_t masked = static_cast<std::uint8_t>(value & 0x0Fu);
  if (high_half) {
    buffer[byte_index] = static_cast<std::uint8_t>((buffer[byte_index] & 0x0Fu) | (masked << 4));
  } else {
    buffer[byte_index] = static_cast<std::uint8_t>((buffer[byte_index] & 0xF0u) | masked);
  }
}

}  // namespace

RowwiseInt4Packed pack_rowwise_int4(const float* src,
                                    std::size_t rows,
                                    std::size_t cols,
                                    std::size_t ld,
                                    float outlier_threshold) {
  RowwiseInt4Packed packed;
  packed.rows = rows;
  packed.cols = cols;
  packed.block = 64;
  packed.outlier_threshold = outlier_threshold;

  const std::size_t bytes_per_row = (cols + 1) / 2;
  packed.data.assign(rows * bytes_per_row, 0);
  packed.scales.resize(rows);

  for (std::size_t r = 0; r < rows; ++r) {
    const float* row = src + r * ld;
    float max_abs = 0.0f;
    for (std::size_t c = 0; c < cols; ++c) {
      const float value = row[c];
      if (outlier_threshold > 0.0f && std::fabs(value) > outlier_threshold) {
        continue;
      }
      max_abs = std::max(max_abs, std::fabs(value));
    }
    if (max_abs < 1e-8f) {
      max_abs = 1e-8f;
    }
    const float inv_scale = 7.0f / max_abs;
    packed.scales[r] = float_to_fp16(max_abs / 7.0f);

    for (std::size_t c = 0; c < cols; ++c) {
      const float value = row[c];
      if (outlier_threshold > 0.0f && std::fabs(value) > outlier_threshold) {
        packed.outlier_rows.push_back(static_cast<std::uint32_t>(r));
        packed.outlier_cols.push_back(static_cast<std::uint16_t>(c));
        packed.outlier_values.push_back(float_to_fp16(value));
        continue;
      }
      const float scaled = value * inv_scale;
      const int quant = std::clamp(static_cast<int>(std::round(scaled)), -8, 7);
      const std::uint8_t stored = static_cast<std::uint8_t>(quant & 0x0F);
      const std::size_t byte_index = r * bytes_per_row + (c >> 1);
      const bool high_half = (c & 1) != 0;
      store_nibble(packed.data, byte_index, high_half, stored);
    }
  }

  return packed;
}

}  // namespace infeng::quant::packers
