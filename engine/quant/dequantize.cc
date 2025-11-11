#include "infeng/quant/dequantize.h"

#include <cmath>
#include <limits>

namespace infeng::quant {

float fp16_to_float(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value >> 15);
  const std::uint32_t exponent = (value >> 10) & 0x1Fu;
  const std::uint32_t mantissa = value & 0x3FFu;

  if (exponent == 0) {
    if (mantissa == 0) {
      return sign ? -0.0f : 0.0f;
    }
    const float val = std::ldexp(static_cast<float>(mantissa), -24);
    return sign ? -val : val;
  }

  if (exponent == 0x1F) {
    if (mantissa == 0) {
      return sign ? -INFINITY : INFINITY;
    }
    return std::numeric_limits<float>::quiet_NaN();
  }

  const std::uint32_t adjusted_exponent = exponent - 15 + 127;
  const std::uint32_t bits = (sign << 31) | (adjusted_exponent << 23) | (mantissa << 13);

  union {
    std::uint32_t u;
    float f;
  } out{bits};
  return out.f;
}

std::vector<float> dequantize_rowwise_int8(const std::int8_t* data,
                                           const std::uint16_t* scales,
                                           std::size_t rows,
                                           std::size_t cols,
                                           const std::uint32_t* outlier_rows,
                                           const std::uint16_t* outlier_cols,
                                           const std::uint16_t* outlier_values,
                                           std::size_t num_outliers) {
  std::vector<float> output(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    const float scale = fp16_to_float(scales[r]);
    const float actual_scale = (scale == 0.0f) ? 1e-8f : scale;
    for (std::size_t c = 0; c < cols; ++c) {
      const std::int8_t q = data[r * cols + c];
      output[r * cols + c] = static_cast<float>(q) * actual_scale;
    }
  }

  if (outlier_rows && outlier_cols && outlier_values && num_outliers > 0) {
    for (std::size_t idx = 0; idx < num_outliers; ++idx) {
      const std::size_t row = static_cast<std::size_t>(outlier_rows[idx]);
      const std::size_t col = static_cast<std::size_t>(outlier_cols[idx]);
      if (row < rows && col < cols) {
        output[row * cols + col] = fp16_to_float(outlier_values[idx]);
      }
    }
  }
  return output;
}

std::vector<float> dequantize_rowwise_int4(const std::uint8_t* data,
                                           const std::uint16_t* scales,
                                           std::size_t rows,
                                           std::size_t cols,
                                           const std::uint32_t* outlier_rows,
                                           const std::uint16_t* outlier_cols,
                                           const std::uint16_t* outlier_values,
                                           std::size_t num_outliers) {
  std::vector<float> output(rows * cols);
  const std::size_t bytes_per_row = (cols + 1) / 2;
  for (std::size_t r = 0; r < rows; ++r) {
    const float scale = fp16_to_float(scales[r]);
    const float actual_scale = (scale == 0.0f) ? 1e-8f : scale;
    const std::uint8_t* row_ptr = data + r * bytes_per_row;
    for (std::size_t c = 0; c < cols; ++c) {
      const std::uint8_t byte = row_ptr[c >> 1];
      const std::uint8_t nibble = (c & 1) ? static_cast<std::uint8_t>(byte >> 4) : static_cast<std::uint8_t>(byte & 0x0F);
      const std::int8_t signed_val = static_cast<std::int8_t>(static_cast<std::int8_t>(nibble << 4)) >> 4;
      output[r * cols + c] = static_cast<float>(signed_val) * actual_scale;
    }
  }

  if (outlier_rows && outlier_cols && outlier_values && num_outliers > 0) {
    for (std::size_t idx = 0; idx < num_outliers; ++idx) {
      const std::size_t row = static_cast<std::size_t>(outlier_rows[idx]);
      const std::size_t col = static_cast<std::size_t>(outlier_cols[idx]);
      if (row < rows && col < cols) {
        output[row * cols + col] = fp16_to_float(outlier_values[idx]);
      }
    }
  }
  return output;
}

}  // namespace infeng::quant
