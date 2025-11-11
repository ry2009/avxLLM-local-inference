#include "infeng/runtime/cpu_lora_engine.h"

#include "infeng/kernels/gemm/pack.h"
#include "infeng/quant/dequantize.h"
#include "infeng/quant/infq_reader.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infeng::runtime {
namespace fs = std::filesystem;

namespace {

template <typename T>
std::vector<T> read_values(const fs::path& file, std::size_t offset, std::size_t count) {
  std::ifstream in(file, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open INFQ payload: " + file.string());
  }
  in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  if (!in.good()) {
    throw std::runtime_error("Failed to seek INFQ payload: " + file.string());
  }
  std::vector<T> data(count);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(T)));
  if (in.gcount() != static_cast<std::streamsize>(count * sizeof(T))) {
    throw std::runtime_error("Unexpected EOF while reading INFQ payload: " + file.string());
  }
  return data;
}

struct QuantizedTensor {
  std::vector<std::int8_t> data;
  std::vector<std::uint16_t> scales;
  std::vector<std::uint32_t> outlier_rows;
  std::vector<std::uint16_t> outlier_cols;
  std::vector<std::uint16_t> outlier_values;
  std::size_t rows{0};
  std::size_t cols{0};
};

std::vector<std::int8_t> decompress_int4_rows(const std::vector<std::uint8_t>& bytes,
                                             std::size_t rows,
                                             std::size_t cols) {
  std::vector<std::int8_t> out(rows * cols, 0);
  const std::size_t bytes_per_row = (cols + 1) / 2;
  for (std::size_t r = 0; r < rows; ++r) {
    const std::uint8_t* row = bytes.data() + r * bytes_per_row;
    for (std::size_t c = 0; c < cols; ++c) {
      const std::uint8_t packed = row[c >> 1];
      const std::uint8_t nibble = (c & 1) ? static_cast<std::uint8_t>(packed >> 4)
                                          : static_cast<std::uint8_t>(packed & 0x0F);
      const std::int8_t value = static_cast<std::int8_t>(static_cast<std::int8_t>(nibble << 4)) >> 4;
      out[r * cols + c] = value;
    }
  }
  return out;
}

QuantizedTensor load_tensor_quant(const fs::path& root, const quant::TensorInfo& info) {
  const fs::path file = root / info.data_file;
  QuantizedTensor tensor;
  tensor.rows = info.rows;
  tensor.cols = info.cols;
  tensor.scales = read_values<std::uint16_t>(file, info.scale_offset, info.rows);

  if (info.dtype == "int8_rowwise") {
    const std::size_t elements = info.rows * info.cols;
    tensor.data = read_values<std::int8_t>(file, info.data_offset, elements);
  } else if (info.dtype == "int4_rowwise") {
    const std::size_t bytes_per_row = (info.cols + 1) / 2;
    const std::size_t total_bytes = info.rows * bytes_per_row;
    auto compressed = read_values<std::uint8_t>(file, info.data_offset, total_bytes);
    tensor.data = decompress_int4_rows(compressed, info.rows, info.cols);
  } else {
    throw std::runtime_error("Unsupported INFQ tensor dtype: " + info.dtype);
  }

  if (info.outliers && info.outliers->count > 0) {
    const auto& meta = *info.outliers;
    const fs::path outlier_path = root / meta.data_file;
    std::ifstream out(outlier_path, std::ios::binary);
    if (!out.is_open()) {
      throw std::runtime_error("Failed to open INFQ outlier payload: " + outlier_path.string());
    }
    if (meta.align && (meta.offset % meta.align) != 0) {
      throw std::runtime_error("Outlier block alignment mismatch for file: " + meta.data_file);
    }
    const std::size_t expected_record_bytes = meta.record_bytes ? meta.record_bytes : 8;
    if (expected_record_bytes != 8) {
      throw std::runtime_error("Unsupported outlier record size in INFQ manifest");
    }
    out.seekg(static_cast<std::streamoff>(meta.offset), std::ios::beg);
    tensor.outlier_rows.resize(meta.count);
    tensor.outlier_cols.resize(meta.count);
    tensor.outlier_values.resize(meta.count);
    for (std::size_t idx = 0; idx < meta.count; ++idx) {
      std::uint32_t row = 0;
      std::uint16_t col = 0;
      std::uint16_t value = 0;
      out.read(reinterpret_cast<char*>(&row), sizeof(row));
      out.read(reinterpret_cast<char*>(&col), sizeof(col));
      out.read(reinterpret_cast<char*>(&value), sizeof(value));
      tensor.outlier_rows[idx] = row;
      tensor.outlier_cols[idx] = col;
      tensor.outlier_values[idx] = value;
    }
  }
  return tensor;
}

kernels::gemm::PackedRowwiseInt8Matrix make_packed(QuantizedTensor tensor, std::size_t rows, std::size_t cols) {
  kernels::gemm::PackedRowwiseInt8Matrix packed;
  packed.rows = rows;
  packed.cols = cols;
  packed.ld = cols;
  packed.data = std::move(tensor.data);
  packed.scales = std::move(tensor.scales);
  packed.outlier_rows = std::move(tensor.outlier_rows);
  packed.outlier_cols = std::move(tensor.outlier_cols);
  packed.outlier_values = std::move(tensor.outlier_values);
  return packed;
}

}  // namespace

kernels::gemm::PackedMatrix CpuLoraEngine::dequantize_rowwise(const kernels::gemm::PackedRowwiseInt8Matrix& src,
                                                              std::size_t rows,
                                                              std::size_t cols) {
  auto values = quant::dequantize_rowwise_int8(src.data.data(),
                                               src.scales.data(),
                                               rows,
                                               cols,
                                               src.outlier_rows.empty() ? nullptr : src.outlier_rows.data(),
                                               src.outlier_cols.empty() ? nullptr : src.outlier_cols.data(),
                                               src.outlier_values.empty() ? nullptr : src.outlier_values.data(),
                                               src.outlier_rows.size());
  return kernels::gemm::pack_matrix(values.data(), rows, cols, cols);
}

void CpuLoraEngine::ensure_dequantized(AdapterWeights& weights) const {
  if (weights.dequantized) {
    return;
  }
  if (!weights.mutex) {
    weights.mutex = std::make_shared<std::mutex>();
  }
  std::lock_guard<std::mutex> lock(*weights.mutex);
  if (weights.dequantized) {
    return;
  }
  weights.A_dequant = dequantize_rowwise(weights.A_quant, weights.A_quant.rows, weights.A_quant.cols);
  weights.B_dequant = dequantize_rowwise(weights.B_quant, weights.B_quant.rows, weights.B_quant.cols);
  weights.dequantized = true;
}

CpuLoraEngine::CpuLoraEngine(fs::path manifest_path) {
  quant::InfqModel model(manifest_path.string());
  const fs::path root = manifest_path.parent_path();

  if (model.tensors().empty()) {
    throw std::runtime_error("INFQ manifest contains no base tensors");
  }

  const auto& base = model.tensors().front();
  auto base_tensor = load_tensor_quant(root, base);
  base_weight_ = make_packed(std::move(base_tensor), base.rows, base.cols);
  output_dim_ = base.rows;
  input_dim_ = base.cols;

  for (const auto& adapter : model.adapters()) {
    AdapterWeights weights;
    weights.rank = static_cast<std::size_t>(adapter.rank);

    auto A_tensor = load_tensor_quant(root, adapter.A);
    auto B_tensor = load_tensor_quant(root, adapter.B);

    if (adapter.A.cols != weights.rank) {
      throw std::runtime_error("Adapter " + adapter.name + " has inconsistent A shape");
    }
    if (adapter.B.rows != weights.rank) {
      throw std::runtime_error("Adapter " + adapter.name + " has inconsistent B shape");
    }
    if (adapter.B.cols != static_cast<std::size_t>(input_dim_)) {
      throw std::runtime_error("Adapter " + adapter.name + " input dim mismatch");
    }
    if (adapter.A.rows != static_cast<std::size_t>(output_dim_)) {
      throw std::runtime_error("Adapter " + adapter.name + " output dim mismatch");
    }

    weights.A_quant = make_packed(std::move(A_tensor), adapter.A.rows, adapter.A.cols);
    weights.B_quant = make_packed(std::move(B_tensor), adapter.B.rows, adapter.B.cols);
    weights.mutex = std::make_shared<std::mutex>();
    adapters_.emplace(adapter.name, std::move(weights));
  }
}

bool CpuLoraEngine::has_adapter(const std::string& name) const {
  return adapters_.find(name) != adapters_.end();
}

void CpuLoraEngine::run_microbatch(Microbatch& batch) const {
  if (batch.requests.empty()) {
    return;
  }
  if (input_dim_ == 0 || output_dim_ == 0) {
    throw std::runtime_error("CpuLoraEngine not initialised with base weights");
  }

  const std::size_t batch_size = batch.requests.size();
  std::vector<float> activations(input_dim_ * batch_size);
  for (std::size_t col = 0; col < batch_size; ++col) {
    const auto& request = batch.requests[col];
    if (request.activations.size() != input_dim_) {
      throw std::runtime_error("Request activation shape mismatch");
    }
    for (std::size_t row = 0; row < input_dim_; ++row) {
      activations[row * batch_size + col] = request.activations[row];
    }
  }

  std::vector<float> outputs(output_dim_ * batch_size, 0.0f);

  auto W = kernels::gemm::make_view(base_weight_);
  auto X = kernels::gemm::make_view(activations.data(), input_dim_, batch_size, batch_size);
  auto Y = kernels::gemm::make_mutable_view(outputs.data(), output_dim_, batch_size, batch_size);

  kernels::gemm::FusedGemmContext ctx;
  kernels::gemm::fused_base_lora_gemm(W, X, nullptr, 0, Y, &ctx);

  if (!batch.adapter_signature.empty()) {
    std::unordered_map<std::string, std::size_t> adapter_index;
    adapter_index.reserve(batch.adapter_signature.size());
    std::vector<kernels::gemm::LoRAView> lora_views;
    lora_views.reserve(batch.adapter_signature.size());

    for (const auto& adapter_name : batch.adapter_signature) {
      auto it = adapters_.find(adapter_name);
      if (it == adapters_.end()) {
        throw std::runtime_error("Unknown adapter requested: " + adapter_name);
      }
      AdapterWeights& weights = it->second;
      ensure_dequantized(weights);
      kernels::gemm::LoRAView view;
      view.A = kernels::gemm::make_view(weights.A_dequant);
      view.B = kernels::gemm::make_view(weights.B_dequant);
      view.rank = weights.rank;
      adapter_index.emplace(adapter_name, lora_views.size());
      lora_views.push_back(view);
    }

    std::vector<float> scratch;
    for (std::size_t col = 0; col < batch_size; ++col) {
      const Request& request = batch.requests[col];
      const std::vector<std::string>* adapters = &request.adapters;
      std::vector<std::string> fallback;
      if (adapters->empty() && !request.adapter.empty()) {
        fallback.push_back(request.adapter);
        adapters = &fallback;
      }
      if (adapters->empty()) {
        continue;
      }

      auto Y_col = kernels::gemm::make_mutable_view(outputs.data() + col, output_dim_, 1, batch_size);
      for (const auto& adapter_name : *adapters) {
        auto idx_it = adapter_index.find(adapter_name);
        if (idx_it == adapter_index.end()) {
          throw std::runtime_error("Request adapter missing from microbatch signature: " + adapter_name);
        }
        kernels::gemm::apply_lora_update(lora_views[idx_it->second], X, col, Y_col, &ctx, scratch);
      }
    }
  }

  for (std::size_t col = 0; col < batch_size; ++col) {
    if (auto& output = batch.requests[col].output) {
      output->resize(output_dim_);
      for (std::size_t row = 0; row < output_dim_; ++row) {
        (*output)[row] = outputs[row * batch_size + col];
      }
    }
  }
}

}  // namespace infeng::runtime
