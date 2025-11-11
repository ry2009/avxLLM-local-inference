#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infeng::quant {

float fp16_to_float(std::uint16_t value);

std::vector<float> dequantize_rowwise_int8(const std::int8_t* data,
                                           const std::uint16_t* scales,
                                           std::size_t rows,
                                           std::size_t cols,
                                           const std::uint32_t* outlier_rows = nullptr,
                                           const std::uint16_t* outlier_cols = nullptr,
                                           const std::uint16_t* outlier_values = nullptr,
                                           std::size_t num_outliers = 0);

inline std::vector<float> dequantize_rowwise_int8(const std::vector<std::int8_t>& data,
                                                  const std::vector<std::uint16_t>& scales,
                                                  std::size_t rows,
                                                  std::size_t cols,
                                                  const std::vector<std::uint32_t>* outlier_rows = nullptr,
                                                  const std::vector<std::uint16_t>* outlier_cols = nullptr,
                                                  const std::vector<std::uint16_t>* outlier_values = nullptr) {
  if (outlier_rows && outlier_cols && outlier_values) {
    return dequantize_rowwise_int8(data.data(),
                                   scales.data(),
                                   rows,
                                   cols,
                                   outlier_rows->data(),
                                   outlier_cols->data(),
                                   outlier_values->data(),
                                   outlier_rows->size());
  }
  return dequantize_rowwise_int8(data.data(), scales.data(), rows, cols);
}

std::vector<float> dequantize_rowwise_int4(const std::uint8_t* data,
                                           const std::uint16_t* scales,
                                           std::size_t rows,
                                           std::size_t cols,
                                           const std::uint32_t* outlier_rows = nullptr,
                                           const std::uint16_t* outlier_cols = nullptr,
                                           const std::uint16_t* outlier_values = nullptr,
                                           std::size_t num_outliers = 0);

inline std::vector<float> dequantize_rowwise_int4(const std::vector<std::uint8_t>& data,
                                                  const std::vector<std::uint16_t>& scales,
                                                  std::size_t rows,
                                                  std::size_t cols,
                                                  const std::vector<std::uint32_t>* outlier_rows = nullptr,
                                                  const std::vector<std::uint16_t>* outlier_cols = nullptr,
                                                  const std::vector<std::uint16_t>* outlier_values = nullptr) {
  if (outlier_rows && outlier_cols && outlier_values) {
    return dequantize_rowwise_int4(data.data(),
                                   scales.data(),
                                   rows,
                                   cols,
                                   outlier_rows->data(),
                                   outlier_cols->data(),
                                   outlier_values->data(),
                                   outlier_rows->size());
  }
  return dequantize_rowwise_int4(data.data(), scales.data(), rows, cols);
}

}  // namespace infeng::quant
