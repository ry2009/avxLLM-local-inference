#include "infeng/quant/packers/int8_rowwise.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace infeng::quant::packers {

namespace {

std::uint16_t float_to_fp16(float value) {
  // IEEE 754 half conversion (round-to-nearest-even)
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
  } else if (exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  }

  const std::uint16_t fraction = static_cast<std::uint16_t>((mantissa + 0x00001000u) >> 13);
  return static_cast<std::uint16_t>(sign | (exponent << 10) | fraction);
}

}  // namespace

RowwiseInt8Packed pack_rowwise_int8(const float* src,
                                    std::size_t rows,
                                    std::size_t cols,
                                    std::size_t ld,
                                    float outlier_threshold) {
  RowwiseInt8Packed packed;
  packed.rows = rows;
  packed.cols = cols;
  packed.block = 64;
  packed.outlier_threshold = outlier_threshold;
  packed.data.resize(rows * cols);
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
    const float inv_scale = 127.0f / max_abs;
    packed.scales[r] = float_to_fp16(max_abs / 127.0f);
    for (std::size_t c = 0; c < cols; ++c) {
      const float value = row[c];
      if (outlier_threshold > 0.0f && std::fabs(value) > outlier_threshold) {
        packed.outlier_rows.push_back(static_cast<std::uint32_t>(r));
        packed.outlier_cols.push_back(static_cast<std::uint16_t>(c));
        packed.outlier_values.push_back(float_to_fp16(value));
        packed.data[r * cols + c] = 0;
        continue;
      }
      const float scaled = value * inv_scale;
      const int quant = static_cast<int>(std::round(scaled));
      packed.data[r * cols + c] = static_cast<std::int8_t>(std::clamp(quant, -127, 127));
    }
  }
  return packed;
}

}  // namespace infeng::quant::packers
