#include "infeng/kernels/gemm/pack.h"

#include <algorithm>
#include <stdexcept>

#include "infeng/quant/packers/int8_rowwise.h"
#include "infeng/quant/packers/int4_rowwise.h"

namespace infeng::kernels::gemm {

PackedMatrix pack_matrix(const float* src, std::size_t rows, std::size_t cols, std::size_t ld) {
  if (!src && rows && cols) {
    throw std::invalid_argument("pack_matrix: null source");
  }
  PackedMatrix packed;
  packed.rows = rows;
  packed.cols = cols;
  packed.ld = cols;
  packed.data.resize(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    const float* row_src = src + r * ld;
    float* row_dst = packed.data.data() + r * cols;
    std::copy(row_src, row_src + cols, row_dst);
  }
  return packed;
}

PackedMatrix pack_vector(const float* src, std::size_t rows) {
  return pack_matrix(src, rows, 1, 1);
}

PackedRowwiseInt8Matrix pack_matrix_rowwise_int8(const float* src,
                                                 std::size_t rows,
                                                 std::size_t cols,
                                                 std::size_t ld) {
  if (!src && rows && cols) {
    throw std::invalid_argument("pack_matrix_rowwise_int8: null source");
  }
  auto packed = quant::packers::pack_rowwise_int8(src, rows, cols, ld);
  PackedRowwiseInt8Matrix result;
  result.rows = rows;
  result.cols = cols;
  result.ld = cols;
  result.data = std::move(packed.data);
  result.scales = std::move(packed.scales);
  result.outlier_rows = std::move(packed.outlier_rows);
  result.outlier_cols = std::move(packed.outlier_cols);
  result.outlier_values = std::move(packed.outlier_values);
  return result;
}

PackedRowwiseInt4Matrix pack_matrix_rowwise_int4(const float* src,
                                                 std::size_t rows,
                                                 std::size_t cols,
                                                 std::size_t ld) {
  if (!src && rows && cols) {
    throw std::invalid_argument("pack_matrix_rowwise_int4: null source");
  }
  auto packed = quant::packers::pack_rowwise_int4(src, rows, cols, ld);
  PackedRowwiseInt4Matrix result;
  result.rows = rows;
  result.cols = cols;
  result.ld = cols;
  result.data = std::move(packed.data);
  result.scales = std::move(packed.scales);
  result.outlier_rows = std::move(packed.outlier_rows);
  result.outlier_cols = std::move(packed.outlier_cols);
  result.outlier_values = std::move(packed.outlier_values);
  return result;
}

MatView make_view(const PackedMatrix& packed) {
  return MatView{packed.data.data(), packed.rows, packed.cols, packed.cols};
}

MatView make_view(const float* data, std::size_t rows, std::size_t cols, std::size_t ld) {
  return MatView{data, rows, cols, ld};
}

MutableMatView make_mutable_view(float* data, std::size_t rows, std::size_t cols, std::size_t ld) {
  return MutableMatView{data, rows, cols, ld};
}

RowwiseInt8MatView make_view(const PackedRowwiseInt8Matrix& packed) {
  return RowwiseInt8MatView{packed.data.data(),
                            packed.scales.data(),
                            packed.outlier_rows.empty() ? nullptr : packed.outlier_rows.data(),
                            packed.outlier_cols.empty() ? nullptr : packed.outlier_cols.data(),
                            packed.outlier_values.empty() ? nullptr : packed.outlier_values.data(),
                            packed.outlier_rows.size(),
                            packed.rows,
                            packed.cols,
                            packed.ld};
}

RowwiseInt8MatView make_view(const std::int8_t* data,
                             const std::uint16_t* scales,
                             const std::uint32_t* outlier_rows,
                             const std::uint16_t* outlier_cols,
                             const std::uint16_t* outlier_values,
                             std::size_t num_outliers,
                             std::size_t rows,
                             std::size_t cols,
                             std::size_t ld) {
  return RowwiseInt8MatView{data, scales, outlier_rows, outlier_cols, outlier_values, num_outliers, rows, cols, ld};
}

RowwiseInt4MatView make_view(const PackedRowwiseInt4Matrix& packed) {
  return RowwiseInt4MatView{packed.data.data(),
                            packed.scales.data(),
                            packed.outlier_rows.empty() ? nullptr : packed.outlier_rows.data(),
                            packed.outlier_cols.empty() ? nullptr : packed.outlier_cols.data(),
                            packed.outlier_values.empty() ? nullptr : packed.outlier_values.data(),
                            packed.outlier_rows.size(),
                            packed.rows,
                            packed.cols,
                            packed.ld};
}

RowwiseInt4MatView make_view(const std::uint8_t* data,
                             const std::uint16_t* scales,
                             const std::uint32_t* outlier_rows,
                             const std::uint16_t* outlier_cols,
                             const std::uint16_t* outlier_values,
                             std::size_t num_outliers,
                             std::size_t rows,
                             std::size_t cols,
                             std::size_t ld) {
  return RowwiseInt4MatView{data, scales, outlier_rows, outlier_cols, outlier_values, num_outliers, rows, cols, ld};
}

}  // namespace infeng::kernels::gemm
