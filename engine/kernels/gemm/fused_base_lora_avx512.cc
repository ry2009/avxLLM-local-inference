#include "infeng/kernels/gemm/fused_base_lora.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <immintrin.h>

#include "infeng/quant/dequantize.h"

namespace infeng::kernels::gemm {

namespace {

std::vector<float>* acquire_workspace(void* ctx, std::size_t desired) {
  if (!ctx) {
    return nullptr;
  }
  auto* context = static_cast<FusedGemmContext*>(ctx);
  if (context->workspace.size() < desired) {
    context->workspace.resize(desired);
  }
  return &context->workspace;
}

void dequantize_row(const std::int8_t* src,
                    float scale,
                    std::size_t elems,
                    float* dst) {
#if defined(__AVX512F__)
  const __m512 scale_vec = _mm512_set1_ps(scale);
  std::size_t k = 0;
  for (; k + 64 <= elems; k += 64) {
    for (std::size_t lane = 0; lane < 64; lane += 16) {
      const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + k + lane));
      const __m512i expanded = _mm512_cvtepi8_epi32(bytes);
      __m512 floats = _mm512_cvtepi32_ps(expanded);
      floats = _mm512_mul_ps(floats, scale_vec);
      _mm512_storeu_ps(dst + k + lane, floats);
    }
  }
  for (; k < elems; ++k) {
    dst[k] = static_cast<float>(src[k]) * scale;
  }
#else
  for (std::size_t k = 0; k < elems; ++k) {
    dst[k] = static_cast<float>(src[k]) * scale;
  }
#endif
}

inline void fmadd_scaled_row(float* dst, const float* src, float scale, std::size_t len) {
  if (scale == 0.0f) {
    return;
  }
#if INFENG_ENABLE_AVX512
  std::size_t idx = 0;
  const __m512 scale_vec = _mm512_set1_ps(scale);
  for (; idx + 16 <= len; idx += 16) {
    const __m512 src_vec = _mm512_loadu_ps(src + idx);
    __m512 dst_vec = _mm512_loadu_ps(dst + idx);
    dst_vec = _mm512_fmadd_ps(scale_vec, src_vec, dst_vec);
    _mm512_storeu_ps(dst + idx, dst_vec);
  }
#elif defined(__AVX2__)
  std::size_t idx = 0;
  const __m256 scale_vec = _mm256_set1_ps(scale);
  for (; idx + 8 <= len; idx += 8) {
    const __m256 src_vec = _mm256_loadu_ps(src + idx);
    __m256 dst_vec = _mm256_loadu_ps(dst + idx);
    dst_vec = _mm256_fmadd_ps(scale_vec, src_vec, dst_vec);
    _mm256_storeu_ps(dst + idx, dst_vec);
  }
#else
  std::size_t idx = 0;
#endif
  for (; idx < len; ++idx) {
    dst[idx] += scale * src[idx];
  }
}

}  // namespace

void fused_base_lora_gemm(const RowwiseInt8MatView& W,
                          const MatView& X,
                          const LoRAView* loras,
                          int n_loras,
                          const MutableMatView& Y,
                          void* ctx) {
  assert(W.cols == X.rows);
  assert(Y.rows == W.rows);
  assert(Y.cols == X.cols);

  const std::size_t rows = W.rows;
  const std::size_t cols = W.cols;
  const std::size_t batch = X.cols;
  const std::size_t max_rank = [&]() {
    std::size_t value = 0;
    for (int idx = 0; idx < n_loras; ++idx) {
      value = std::max<std::size_t>(value, loras[idx].rank);
    }
    return value;
  }();

  const std::size_t workspace_floats = cols + max_rank * batch;
  std::vector<float> stack_workspace;
  std::vector<float>* workspace = acquire_workspace(ctx, workspace_floats ? workspace_floats : cols);
  if (!workspace) {
    stack_workspace.resize(workspace_floats ? workspace_floats : cols);
    workspace = &stack_workspace;
  }
  float* dequant_row = workspace->data();
  float* adapter_tmp = max_rank ? (workspace->data() + cols) : nullptr;
  const bool has_outliers = W.outlier_rows && W.outlier_cols && W.outlier_values && W.num_outliers > 0;
  std::size_t outlier_index = 0;

  // Phase 1: base product Y = W * X
  for (std::size_t r = 0; r < rows; ++r) {
    const float scale = quant::fp16_to_float(W.scales[r]);
    const std::int8_t* w_row = W.data + r * W.ld;
    dequantize_row(w_row, scale, cols, dequant_row);
    if (has_outliers) {
      while (outlier_index < W.num_outliers && W.outlier_rows[outlier_index] < r) {
        ++outlier_index;
      }
      while (outlier_index < W.num_outliers && W.outlier_rows[outlier_index] == r) {
        const std::size_t column = static_cast<std::size_t>(W.outlier_cols[outlier_index]);
        if (column < cols) {
          dequant_row[column] = quant::fp16_to_float(W.outlier_values[outlier_index]);
        }
        ++outlier_index;
      }
    }

    float* y_row = Y.data + r * Y.ld;
    std::fill(y_row, y_row + batch, 0.0f);
    for (std::size_t k = 0; k < cols; ++k) {
      const float coeff = dequant_row[k];
      if (coeff == 0.0f) {
        continue;
      }
      const float* x_row = X.data + k * X.ld;
      fmadd_scaled_row(y_row, x_row, coeff, batch);
    }
  }

  if (!loras || n_loras == 0) {
    return;
  }

  for (int l = 0; l < n_loras; ++l) {
    const auto& adapter = loras[l];
    if (adapter.rank == 0) {
      continue;
    }

    float* tmp = adapter_tmp;
    std::fill(tmp, tmp + adapter.rank * batch, 0.0f);
    for (std::size_t k = 0; k < cols; ++k) {
      const float* x_row = X.data + k * X.ld;
      for (std::size_t rnk = 0; rnk < adapter.rank; ++rnk) {
        const float coeff = adapter.B.data[rnk * adapter.B.ld + k];
        if (coeff == 0.0f) {
          continue;
        }
        fmadd_scaled_row(tmp + rnk * batch, x_row, coeff, batch);
      }
    }

    for (std::size_t row = 0; row < rows; ++row) {
      float* y_row = Y.data + row * Y.ld;
      for (std::size_t rnk = 0; rnk < adapter.rank; ++rnk) {
        const float coeff = adapter.A.data[row * adapter.A.ld + rnk];
        if (coeff == 0.0f) {
          continue;
        }
        const float* tmp_row = tmp + rnk * batch;
        fmadd_scaled_row(y_row, tmp_row, coeff, batch);
      }
    }
  }
}

void apply_lora_update(const LoRAView& lora,
                       const MatView& X,
                       std::size_t column,
                       const MutableMatView& Y_col,
                       FusedGemmContext* ctx,
                       std::vector<float>& buffer) {
  if (lora.rank == 0) {
    return;
  }

  (void)ctx;
  if (buffer.size() < lora.rank) {
    buffer.resize(lora.rank);
  }
  float* tmp = buffer.data();

  for (std::size_t r = 0; r < lora.rank; ++r) {
    float acc = 0.0f;
    for (std::size_t k = 0; k < X.rows; ++k) {
      const float lhs = lora.B.data[r * lora.B.ld + k];
      const float rhs = X.data[k * X.ld + column];
      acc += lhs * rhs;
    }
    tmp[r] = acc;
  }

  for (std::size_t row = 0; row < Y_col.rows; ++row) {
    float acc = 0.0f;
    for (std::size_t r = 0; r < lora.rank; ++r) {
      const float lhs = lora.A.data[row * lora.A.ld + r];
      acc += lhs * tmp[r];
    }
    Y_col.data[row * Y_col.ld] += acc;
  }
}

}  // namespace infeng::kernels::gemm
