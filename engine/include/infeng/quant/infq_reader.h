#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace infeng::quant {

struct OutlierInfo {
  std::string data_file;
  std::size_t offset{0};
  std::size_t count{0};
  std::size_t record_bytes{0};
  std::size_t align{0};
  std::string layout;
};

struct TensorInfo {
  std::string name;
  std::string dtype;
  std::size_t rows{0};
  std::size_t cols{0};
  std::size_t block{0};
  std::string scale_dtype;
  std::string data_file;
  std::size_t data_offset{0};
  std::size_t scale_offset{0};
  std::optional<OutlierInfo> outliers;
};

struct AdapterInfo {
  std::string name;
  std::int32_t rank{0};
  TensorInfo A;
  TensorInfo B;
};

class InfqModel {
 public:
  explicit InfqModel(std::string manifest_path);
  ~InfqModel();

  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const std::vector<AdapterInfo>& adapters() const { return adapters_; }

 private:
  std::vector<TensorInfo> tensors_;
  std::vector<AdapterInfo> adapters_;
};

}  // namespace infeng::quant
