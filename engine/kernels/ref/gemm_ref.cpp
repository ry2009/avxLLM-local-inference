#include "infeng/kernels/ref/gemm_ref.h"

namespace infeng::kernels::ref {

void fused_base_lora_ref(const infeng::kernels::gemm::MatView& W,
                         const infeng::kernels::gemm::MatView& X,
                         const infeng::kernels::gemm::LoRAView* loras,
                         int n_loras,
                         const infeng::kernels::gemm::MutableMatView& Y) {
  const std::size_t rows = W.rows;
  const std::size_t cols = W.cols;
  const std::size_t batch = X.cols;

  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t b = 0; b < batch; ++b) {
      float acc = 0.0f;
      for (std::size_t k = 0; k < cols; ++k) {
        acc += W.data[r * W.ld + k] * X.data[k * X.ld + b];
      }
      Y.data[r * Y.ld + b] = acc;
    }
  }

  if (!loras) {
    return;
  }

  for (int l = 0; l < n_loras; ++l) {
    const auto& adapter = loras[l];
    if (adapter.rank == 0) {
      continue;
    }
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t b = 0; b < batch; ++b) {
        float acc = 0.0f;
        for (std::size_t r = 0; r < adapter.rank; ++r) {
          float tmp = 0.0f;
          for (std::size_t k = 0; k < cols; ++k) {
            tmp += adapter.B.data[r * adapter.B.ld + k] * X.data[k * X.ld + b];
          }
          acc += adapter.A.data[row * adapter.A.ld + r] * tmp;
        }
        Y.data[row * Y.ld + b] += acc;
      }
    }
  }
}

}  // namespace infeng::kernels::ref
