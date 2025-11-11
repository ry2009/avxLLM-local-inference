#pragma once

#include "infeng/kernels/gemm/fused_base_lora.h"

namespace infeng::kernels::ref {

void fused_base_lora_ref(const infeng::kernels::gemm::MatView& W,
                         const infeng::kernels::gemm::MatView& X,
                         const infeng::kernels::gemm::LoRAView* loras,
                         int n_loras,
                         const infeng::kernels::gemm::MutableMatView& Y);

}
