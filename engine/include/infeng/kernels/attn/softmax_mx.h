#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infeng::kernels::attn {

void softmax_mx(std::vector<float>& logits);

void decode_softmax_fp32(const float* logits,
                         std::size_t length,
                         float temperature,
                         float* output);

void decode_softmax_bf16(const std::uint16_t* logits_bf16,
                         std::size_t length,
                         float temperature,
                         float* output);

}  // namespace infeng::kernels::attn

