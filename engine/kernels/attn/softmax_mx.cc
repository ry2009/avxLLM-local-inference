#include "infeng/kernels/attn/softmax_mx.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace infeng::kernels::attn {

namespace {

constexpr float kEps = 1e-6f;

inline float bf16_to_float(std::uint16_t value) {
  union {
    std::uint32_t u32;
    float f32;
  } conv{static_cast<std::uint32_t>(value) << 16};
  return conv.f32;
}

}  // namespace

void softmax_mx(std::vector<float>& logits) {
  if (logits.empty()) {
    return;
  }
  decode_softmax_fp32(logits.data(), logits.size(), 1.0f, logits.data());
}

void decode_softmax_fp32(const float* logits,
                         std::size_t length,
                         float temperature,
                         float* output) {
  if (!logits || !output || length == 0) {
    return;
  }
  const float inv_temp = temperature > kEps ? 1.0f / temperature : 1.0f;
  const float max_logit = *std::max_element(logits, logits + length);
  double sum = 0.0;
  for (std::size_t i = 0; i < length; ++i) {
    const double val = (static_cast<double>(logits[i]) - static_cast<double>(max_logit)) * inv_temp;
    const double exp_val = std::exp(val);
    output[i] = static_cast<float>(exp_val);
    sum += exp_val;
  }
  const double inv_sum = sum > kEps ? 1.0 / sum : 0.0;
  for (std::size_t i = 0; i < length; ++i) {
    output[i] = static_cast<float>(static_cast<double>(output[i]) * inv_sum);
  }
}

void decode_softmax_bf16(const std::uint16_t* logits_bf16,
                         std::size_t length,
                         float temperature,
                         float* output) {
  if (!logits_bf16 || !output || length == 0) {
    return;
  }
  std::vector<float> logits(length);
  for (std::size_t i = 0; i < length; ++i) {
    logits[i] = bf16_to_float(logits_bf16[i]);
  }
  decode_softmax_fp32(logits.data(), length, temperature, output);
}

}  // namespace infeng::kernels::attn
