#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infeng::kernels::gemm {

struct MatView {
  const float* data{nullptr};
  std::size_t rows{0};
  std::size_t cols{0};
  std::size_t ld{0};
};

struct MutableMatView {
  float* data{nullptr};
  std::size_t rows{0};
  std::size_t cols{0};
  std::size_t ld{0};
};

struct PackedMatrix {
  std::vector<float> data;
  std::size_t rows{0};
  std::size_t cols{0};
  std::size_t ld{0};
};

struct RowwiseInt8MatView {
  const std::int8_t* data{nullptr};
  const std::uint16_t* scales{nullptr};
  const std::uint32_t* outlier_rows{nullptr};
  const std::uint16_t* outlier_cols{nullptr};
  const std::uint16_t* outlier_values{nullptr};
  std::size_t num_outliers{0};
  std::size_t rows{0};
  std::size_t cols{0};
  std::size_t ld{0};
};

struct PackedRowwiseInt8Matrix {
  std::vector<std::int8_t> data;
  std::vector<std::uint16_t> scales;
  std::vector<std::uint32_t> outlier_rows;
  std::vector<std::uint16_t> outlier_cols;
  std::vector<std::uint16_t> outlier_values;
  std::size_t rows{0};
  std::size_t cols{0};
  std::size_t ld{0};
};

struct RowwiseInt4MatView {
  const std::uint8_t* data{nullptr};
  const std::uint16_t* scales{nullptr};
  const std::uint32_t* outlier_rows{nullptr};
  const std::uint16_t* outlier_cols{nullptr};
  const std::uint16_t* outlier_values{nullptr};
  std::size_t num_outliers{0};
  std::size_t rows{0};
  std::size_t cols{0};
  std::size_t ld{0};
};

struct PackedRowwiseInt4Matrix {
  std::vector<std::uint8_t> data;
  std::vector<std::uint16_t> scales;
  std::vector<std::uint32_t> outlier_rows;
  std::vector<std::uint16_t> outlier_cols;
  std::vector<std::uint16_t> outlier_values;
  std::size_t rows{0};
  std::size_t cols{0};
  std::size_t ld{0};
};

PackedMatrix pack_matrix(const float* src, std::size_t rows, std::size_t cols, std::size_t ld);
PackedMatrix pack_vector(const float* src, std::size_t rows);
PackedRowwiseInt8Matrix pack_matrix_rowwise_int8(const float* src, std::size_t rows, std::size_t cols, std::size_t ld);
PackedRowwiseInt4Matrix pack_matrix_rowwise_int4(const float* src, std::size_t rows, std::size_t cols, std::size_t ld);

MatView make_view(const PackedMatrix& packed);
MatView make_view(const float* data, std::size_t rows, std::size_t cols, std::size_t ld);
MutableMatView make_mutable_view(float* data, std::size_t rows, std::size_t cols, std::size_t ld);
RowwiseInt8MatView make_view(const PackedRowwiseInt8Matrix& packed);
RowwiseInt8MatView make_view(const std::int8_t* data,
                             const std::uint16_t* scales,
                             const std::uint32_t* outlier_rows,
                             const std::uint16_t* outlier_cols,
                             const std::uint16_t* outlier_values,
                             std::size_t num_outliers,
                             std::size_t rows,
                             std::size_t cols,
                             std::size_t ld);
RowwiseInt4MatView make_view(const PackedRowwiseInt4Matrix& packed);
RowwiseInt4MatView make_view(const std::uint8_t* data,
                             const std::uint16_t* scales,
                             const std::uint32_t* outlier_rows,
                             const std::uint16_t* outlier_cols,
                             const std::uint16_t* outlier_values,
                             std::size_t num_outliers,
                             std::size_t rows,
                             std::size_t cols,
                             std::size_t ld);

}  // namespace infeng::kernels::gemm
