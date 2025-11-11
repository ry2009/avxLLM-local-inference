#pragma once

#include "infeng/kernels/attn/decode.h"

#include <cstddef>

namespace infeng::kernels::attn {

struct BlockConfig {
  std::size_t head_dim{0};
  std::size_t block_cols{64};
};

void kv_dequant_row_fp32(const QuantRowwiseInt8View& view,
                         std::size_t row,
                         float* dst,
                         const BlockConfig& config);

void kv_apply_outliers_fp32(const QuantRowwiseInt8View& view,
                            std::size_t row,
                            float* dst,
                            const BlockConfig& config);

}  // namespace infeng::kernels::attn

