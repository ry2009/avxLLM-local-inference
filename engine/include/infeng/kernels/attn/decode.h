#pragma once

#include "infeng/kernels/gemm/pack.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infeng::kernels::attn {

struct QuantRowwiseInt8View {
  const std::int8_t* data{nullptr};
  const std::uint16_t* scales{nullptr};
  const std::uint32_t* outlier_rows{nullptr};
  const std::uint16_t* outlier_cols{nullptr};
  const std::uint16_t* outlier_values{nullptr};
  std::size_t num_outliers{0};
  std::size_t rows{0};
  std::size_t cols{0};
  std::size_t ld{0};
};

struct QuantKV {
  QuantRowwiseInt8View matrix;
};

struct KVIndex {
  const std::size_t* rows{nullptr};
  std::size_t count{0};
};

enum class DecodeKernelPath {
  kAuto = 0,
  kScalar,
  kAvx2,
  kAvx512Vnni,
};

struct DecodeScratch {
  std::vector<float> logits;
  std::vector<float> attn;
  std::vector<float> dequant;
  std::vector<float> value_tile;
};

// Kernel dispatch controls.
void attn_decode_set_kernel_path(DecodeKernelPath path);
DecodeKernelPath attn_decode_get_kernel_path();
const char* attn_decode_kernel_path_name(DecodeKernelPath path);

void attn_decode_set_tile_tokens_override(std::size_t tokens);
std::size_t attn_decode_tile_tokens(std::size_t head_dim);

void attn_decode_int8_qk(const infeng::kernels::gemm::MatView& query,
                         const QuantKV& key,
                         const QuantKV& value,
                         const KVIndex& index,
                         const infeng::kernels::gemm::MutableMatView& output,
                         DecodeScratch* scratch);

}  // namespace infeng::kernels::attn
