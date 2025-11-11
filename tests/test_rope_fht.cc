#include "infeng/kernels/attn/rope_fht.h"
#include "infeng/quant/dequantize.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

std::vector<float> compute_reference(const std::vector<float>& input,
                                     std::size_t seq_len,
                                     std::size_t head_dim) {
  std::vector<float> data = input;
  const std::size_t half_dim = head_dim / 2;
  std::vector<float> freq(half_dim);
  for (std::size_t i = 0; i < half_dim; ++i) {
    const double theta = std::pow(10000.0, -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
    freq[i] = static_cast<float>(theta);
  }
  auto hadamard = [&](float* row) {
    for (std::size_t size = 1; size < head_dim; size <<= 1) {
      const std::size_t stride = size << 1;
      for (std::size_t base = 0; base < head_dim; base += stride) {
        for (std::size_t offset = 0; offset < size; ++offset) {
          const float a = row[base + offset];
          const float b = row[base + offset + size];
          row[base + offset] = a + b;
          row[base + offset + size] = a - b;
        }
      }
    }
    const float norm = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (std::size_t i = 0; i < head_dim; ++i) {
      row[i] *= norm;
    }
  };

  for (std::size_t token = 0; token < seq_len; ++token) {
    float* row = data.data() + token * head_dim;
    for (std::size_t i = 0; i < half_dim; ++i) {
      const float even = row[2 * i];
      const float odd = row[2 * i + 1];
      const float angle = static_cast<float>(token) * freq[i];
      const float cos_val = std::cos(angle);
      const float sin_val = std::sin(angle);
      row[2 * i] = even * cos_val - odd * sin_val;
      row[2 * i + 1] = even * sin_val + odd * cos_val;
    }
    hadamard(row);
  }
  return data;
}

std::uint16_t float_to_bf16(float value) {
  union {
    float f32;
    std::uint32_t u32;
  } conv{};
  conv.f32 = value;
  return static_cast<std::uint16_t>(conv.u32 >> 16);
}

}  // namespace

int main() {
  const std::size_t seq_len = 2;
  const std::size_t head_dim = 4;
  std::vector<float> input = {
      0.5f, -0.3f, 0.1f, 0.7f,
      -0.2f, 0.4f, 0.9f, -0.6f};

  infeng::kernels::attn::RopeFhtResult result;
  infeng::kernels::attn::rope_fht_kv_avx512(input.data(), seq_len, head_dim, result);

  assert(result.seq_len == seq_len);
  assert(result.head_dim == head_dim);
  assert(result.kv_data.size() == seq_len * head_dim);
  assert(result.kv_scales.size() == seq_len);

  auto reference = compute_reference(input, seq_len, head_dim);
  auto dequant = infeng::quant::dequantize_rowwise_int8(
      result.kv_data.data(), result.kv_scales.data(), seq_len, head_dim);

  for (std::size_t i = 0; i < dequant.size(); ++i) {
    const float diff = std::fabs(dequant[i] - reference[i]);
    assert(diff < 0.05f);
  }

  const std::vector<float> logits = {1.0f, 2.0f, 0.5f};
  std::vector<float> out_fp32(logits.size(), 0.0f);
  infeng::kernels::attn::decode_softmax_fp32(logits.data(), logits.size(), 1.0f, out_fp32.data());

  double sum = 0.0;
  for (float v : logits) {
    sum += std::exp(static_cast<double>(v) - 2.0);
  }
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const double expected = std::exp(static_cast<double>(logits[i]) - 2.0) / sum;
    assert(std::fabs(out_fp32[i] - static_cast<float>(expected)) < 1e-5f);
  }

  std::vector<std::uint16_t> logits_bf16(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    logits_bf16[i] = float_to_bf16(logits[i]);
  }
  std::vector<float> out_bf16(logits.size(), 0.0f);
  infeng::kernels::attn::decode_softmax_bf16(logits_bf16.data(), logits_bf16.size(), 1.0f, out_bf16.data());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    assert(std::fabs(out_bf16[i] - out_fp32[i]) < 1e-4f);
  }

  return 0;
}
