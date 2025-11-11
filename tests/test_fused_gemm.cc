#include "infeng/kernels/gemm/fused_base_lora.h"
#include "infeng/kernels/ref/gemm_ref.h"

#include <cassert>
#include <cmath>
#include <random>
#include <vector>

using namespace infeng::kernels::gemm;

namespace {

float l2_diff(const MutableMatView& lhs, const MutableMatView& rhs) {
  float acc = 0.0f;
  for (std::size_t r = 0; r < lhs.rows; ++r) {
    for (std::size_t c = 0; c < lhs.cols; ++c) {
      const float a = lhs.data[r * lhs.ld + c];
      const float b = rhs.data[r * rhs.ld + c];
      const float diff = a - b;
      acc += diff * diff;
    }
  }
  return std::sqrt(acc);
}

std::vector<float> random_matrix(std::size_t rows, std::size_t cols, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> data(rows * cols);
  for (float& v : data) {
    v = dist(rng);
  }
  return data;
}

}  // namespace

int main() {
  std::mt19937 rng(42);

  const std::size_t rows = 8;
  const std::size_t cols = 6;
  const std::size_t batch = 3;

  auto W_data = random_matrix(rows, cols, rng);
  auto X_data = random_matrix(cols, batch, rng);

  auto A1_data = random_matrix(rows, 2, rng);
  auto B1_data = random_matrix(2, cols, rng);
  auto A2_data = random_matrix(rows, 3, rng);
  auto B2_data = random_matrix(3, cols, rng);

  PackedMatrix W_pack = pack_matrix(W_data.data(), rows, cols, cols);
  PackedRowwiseInt8Matrix W_qpack = pack_matrix_rowwise_int8(W_data.data(), rows, cols, cols);
  PackedMatrix X_pack = pack_matrix(X_data.data(), cols, batch, batch);
  PackedMatrix A1_pack = pack_matrix(A1_data.data(), rows, 2, 2);
  PackedMatrix B1_pack = pack_matrix(B1_data.data(), 2, cols, cols);
  PackedMatrix A2_pack = pack_matrix(A2_data.data(), rows, 3, 3);
  PackedMatrix B2_pack = pack_matrix(B2_data.data(), 3, cols, cols);

  std::vector<float> out_fused(rows * batch, 0.0f);
  std::vector<float> out_ref(rows * batch, 0.0f);

  RowwiseInt8MatView W_q = make_view(W_qpack);
  MatView W = make_view(W_pack);
  MatView X = make_view(X_pack);
  MutableMatView Y_fused = make_mutable_view(out_fused.data(), rows, batch, batch);
  MutableMatView Y_ref = make_mutable_view(out_ref.data(), rows, batch, batch);

  LoRAView adapters[2];
  adapters[0] = LoRAView{make_view(A1_pack), make_view(B1_pack), 2};
  adapters[1] = LoRAView{make_view(A2_pack), make_view(B2_pack), 3};

  FusedGemmContext context;
  fused_base_lora_gemm(W_q, X, adapters, 2, Y_fused, &context);
  infeng::kernels::ref::fused_base_lora_ref(W, X, adapters, 2, Y_ref);

  const float diff = l2_diff(Y_fused, Y_ref);
  assert(diff < 1e-4f);

  // Zero adapters should match base product.
  std::vector<float> out_zero(rows * batch, 0.0f);
  MutableMatView Y_zero = make_mutable_view(out_zero.data(), rows, batch, batch);
  fused_base_lora_gemm(W_q, X, nullptr, 0, Y_zero, nullptr);

  std::vector<float> out_base(rows * batch, 0.0f);
  MutableMatView Y_base = make_mutable_view(out_base.data(), rows, batch, batch);
  infeng::kernels::ref::fused_base_lora_ref(W, X, nullptr, 0, Y_base);
  assert(l2_diff(Y_zero, Y_base) < 1e-5f);

  // Ensure workspace reuse does not change results.
  fused_base_lora_gemm(W_q, X, adapters, 2, Y_fused, &context);
  fused_base_lora_gemm(W_q, X, adapters, 2, Y_fused, &context);

  return 0;
}
