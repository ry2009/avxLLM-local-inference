#pragma once

#include "infeng/kernels/gemm/pack.h"

#include <vector>

namespace infeng::kernels::gemm {

struct LoRAView {
  MatView A;
  MatView B;
  std::size_t rank{0};
};

struct FusedGemmContext {
  std::vector<float> workspace;
};

void fused_base_lora_gemm(const RowwiseInt8MatView& W,
                          const MatView& X,
                          const LoRAView* loras,
                          int n_loras,
                          const MutableMatView& Y,
                          void* ctx);

void apply_lora_update(const LoRAView& lora,
                       const MatView& X,
                       std::size_t column,
                       const MutableMatView& Y_col,
                       FusedGemmContext* ctx,
                       std::vector<float>& buffer);

}  // namespace infeng::kernels::gemm
