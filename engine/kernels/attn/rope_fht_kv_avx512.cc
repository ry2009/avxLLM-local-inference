#include "infeng/kernels/attn/rope_fht.h"

#include "infeng/quant/packers/int8_rowwise.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace infeng::kernels::attn {
namespace {

bool is_power_of_two(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

void hadamard_transform(float* data, std::size_t length) {
  if (!is_power_of_two(length) || length == 1) {
    return;
  }
  for (std::size_t size = 1; size < length; size <<= 1) {
    const std::size_t stride = size << 1;
    for (std::size_t base = 0; base < length; base += stride) {
      for (std::size_t offset = 0; offset < size; ++offset) {
        const float a = data[base + offset];
        const float b = data[base + offset + size];
        data[base + offset] = a + b;
        data[base + offset + size] = a - b;
      }
    }
  }
  const float norm = 1.0f / std::sqrt(static_cast<float>(length));
  for (std::size_t i = 0; i < length; ++i) {
    data[i] *= norm;
  }
}

float bf16_to_float(std::uint16_t value) {
  union {
    std::uint32_t u32;
    float f32;
  } conv{};
  conv.u32 = static_cast<std::uint32_t>(value) << 16;
  return conv.f32;
}

}  // namespace

void rope_fht_kv_avx512(const float* input,
                        std::size_t seq_len,
                        std::size_t head_dim,
                        RopeFhtResult& result) {
  if (!input || seq_len == 0 || head_dim == 0 || head_dim % 2 != 0) {
    result = RopeFhtResult{};
    return;
  }

  std::vector<float> transformed(seq_len * head_dim);
  std::copy(input, input + transformed.size(), transformed.begin());

  const std::size_t half_dim = head_dim / 2;
  std::vector<float> base_frequency(half_dim);
  for (std::size_t i = 0; i < half_dim; ++i) {
    const double theta = std::pow(10000.0, -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
    base_frequency[i] = static_cast<float>(theta);
  }

  for (std::size_t token = 0; token < seq_len; ++token) {
    float* row = transformed.data() + token * head_dim;
    for (std::size_t i = 0; i < half_dim; ++i) {
      const float even = row[2 * i];
      const float odd = row[2 * i + 1];
      const float angle = static_cast<float>(token) * base_frequency[i];
      const float cos_val = std::cos(angle);
      const float sin_val = std::sin(angle);
      row[2 * i] = even * cos_val - odd * sin_val;
      row[2 * i + 1] = even * sin_val + odd * cos_val;
    }
    hadamard_transform(row, head_dim);
  }

  auto packed = quant::packers::pack_rowwise_int8(transformed.data(), seq_len, head_dim, head_dim);
  result.kv_data = std::move(packed.data);
  result.kv_scales = std::move(packed.scales);
  result.outlier_rows = std::move(packed.outlier_rows);
  result.outlier_cols = std::move(packed.outlier_cols);
  result.outlier_values = std::move(packed.outlier_values);
  result.seq_len = seq_len;
  result.head_dim = head_dim;
}

}  // namespace infeng::kernels::attn
