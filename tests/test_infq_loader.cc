#include "infeng/quant/infq_reader.h"

#include <cassert>
#include <filesystem>
#include <fstream>

int main() {
  namespace fs = std::filesystem;
  auto tmp_dir = fs::temp_directory_path() / "infq_loader_test";
  fs::create_directories(tmp_dir);
  fs::path manifest = tmp_dir / "manifest.json";
  std::ofstream out(manifest);
  out << R"({
    "version": 1,
    "endianness": "LE",
    "tensors": [
      {
        "name": "w0",
        "dtype": "int8_rowwise",
        "rows": 2,
        "cols": 2,
        "block": 64,
        "scale_dtype": "fp16",
        "layout": "rowmajor_blocked",
        "data_file": "weights.bin",
        "offset_data": 0,
        "offset_scales": 4
      }
    ],
    "adapters": [
      {
        "name": "foo",
        "rank": 1,
        "A": {
          "name": "foo.A",
          "dtype": "int8_rowwise",
          "rows": 2,
          "cols": 1,
          "block": 64,
          "scale_dtype": "fp16",
          "layout": "rowmajor_blocked",
          "data_file": "adapters.bin",
          "offset_data": 0,
          "offset_scales": 2
        },
        "B": {
          "name": "foo.B",
          "dtype": "int8_rowwise",
          "rows": 1,
          "cols": 2,
          "block": 64,
          "scale_dtype": "fp16",
          "layout": "rowmajor_blocked",
          "data_file": "adapters.bin",
          "offset_data": 4,
          "offset_scales": 6
        }
      }
    ]
  })";
  out.close();

  infeng::quant::InfqModel model(manifest.string());
  assert(model.tensors().size() == 1);
  assert(model.adapters().size() == 1);
  const auto& tensor = model.tensors()[0];
  assert(tensor.name == "w0");
  assert(tensor.rows == 2);
  assert(tensor.cols == 2);
  assert(tensor.data_file == "weights.bin");
  const auto& adapter = model.adapters()[0];
  assert(adapter.name == "foo");
  assert(adapter.rank == 1);
  assert(adapter.A.cols == 1);
  assert(adapter.B.rows == 1);

  fs::remove_all(tmp_dir);
  return 0;
}
