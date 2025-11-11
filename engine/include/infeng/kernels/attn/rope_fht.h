#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infeng::kernels::attn {

struct RopeFhtResult {
  std::vector<std::int8_t> kv_data;
  std::vector<std::uint16_t> kv_scales;
  std::vector<std::uint32_t> outlier_rows;
  std::vector<std::uint16_t> outlier_cols;
  std::vector<std::uint16_t> outlier_values;
  std::size_t seq_len{0};
  std::size_t head_dim{0};
};

void rope_fht_kv_avx512(const float* input,
                        std::size_t seq_len,
                        std::size_t head_dim,
                        RopeFhtResult& result);

}  // namespace infeng::kernels::attn
#include "infeng/kernels/attn/softmax_mx.h"
