#pragma once

#include "infeng/kernels/gemm/fused_base_lora.h"
#include "infeng/kernels/gemm/pack.h"
#include "infeng/quant/infq_reader.h"
#include "infeng/runtime/microbatch.h"

#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace infeng::runtime {

class CpuLoraEngine {
 public:
  explicit CpuLoraEngine(std::filesystem::path infq_manifest);

  std::size_t input_dim() const { return input_dim_; }
  std::size_t output_dim() const { return output_dim_; }

  bool has_adapter(const std::string& name) const;

  void run_microbatch(Microbatch& batch) const;

 private:
  struct AdapterWeights {
    kernels::gemm::PackedRowwiseInt8Matrix A_quant;
    kernels::gemm::PackedRowwiseInt8Matrix B_quant;
    mutable kernels::gemm::PackedMatrix A_dequant;
    mutable kernels::gemm::PackedMatrix B_dequant;
    mutable std::shared_ptr<std::mutex> mutex;
    mutable bool dequantized{false};
    std::size_t rank{0};
  };

  kernels::gemm::PackedRowwiseInt8Matrix base_weight_;
  std::size_t input_dim_{0};
  std::size_t output_dim_{0};
  mutable std::unordered_map<std::string, AdapterWeights> adapters_;

  static kernels::gemm::PackedMatrix dequantize_rowwise(const kernels::gemm::PackedRowwiseInt8Matrix& src,
                                                        std::size_t rows,
                                                        std::size_t cols);
  void ensure_dequantized(AdapterWeights& weights) const;
};

}  // namespace infeng::runtime
