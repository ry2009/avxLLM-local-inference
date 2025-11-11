#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infeng::quant::packers {

struct RowwiseInt4Packed {
  std::vector<std::uint8_t> data;  // Two values per byte (low nibble = even column)
  std::vector<std::uint16_t> scales;  // fp16 encoded
  std::vector<std::uint32_t> outlier_rows;
  std::vector<std::uint16_t> outlier_cols;
  std::vector<std::uint16_t> outlier_values;  // fp16 encoded
  std::size_t rows{0};
  std::size_t cols{0};
  std::size_t block{64};
  float outlier_threshold{0.0f};
};

RowwiseInt4Packed pack_rowwise_int4(const float* src,
                                    std::size_t rows,
                                    std::size_t cols,
                                    std::size_t ld,
                                    float outlier_threshold = 0.0f);

}  // namespace infeng::quant::packers
