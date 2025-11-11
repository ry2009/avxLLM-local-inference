#include "infeng/runtime/cpu_lora_engine.h"
#include "infeng/runtime/scheduler.h"
#include "infeng/quant/infq_reader.h"

#include <cmath>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;

struct TensorOffsets {
  std::size_t data{0};
  std::size_t scales{0};
};

struct QuantizedMatrix {
  std::vector<std::int8_t> data;
  std::vector<std::uint16_t> scales;
};

std::uint16_t float_to_fp16(float value) {
  union {
    float f;
    std::uint32_t u;
  } in{value};

  const std::uint32_t sign = (in.u >> 16) & 0x8000u;
  std::uint32_t mantissa = in.u & 0x007FFFFFu;
  std::int32_t exponent = static_cast<std::int32_t>((in.u >> 23) & 0xFF) - 127 + 15;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    mantissa |= 0x00800000u;
    const std::uint32_t shift = static_cast<std::uint32_t>(1 - exponent);
    const std::uint32_t fraction = mantissa >> (shift + 13);
    return static_cast<std::uint16_t>(sign | fraction);
  } else if (exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  }

  const std::uint16_t fraction = static_cast<std::uint16_t>((mantissa + 0x00001000u) >> 13);
  return static_cast<std::uint16_t>(sign | (exponent << 10) | fraction);
}

QuantizedMatrix quantize_rowwise(const std::vector<float>& src, std::size_t rows, std::size_t cols) {
  QuantizedMatrix packed;
  packed.data.resize(rows * cols);
  packed.scales.resize(rows);
  for (std::size_t r = 0; r < rows; ++r) {
    float max_abs = 0.0f;
    for (std::size_t c = 0; c < cols; ++c) {
      max_abs = std::max(max_abs, std::fabs(src[r * cols + c]));
    }
    if (max_abs < 1e-8f) {
      max_abs = 1e-8f;
    }
    const float inv_scale = 127.0f / max_abs;
    packed.scales[r] = float_to_fp16(max_abs / 127.0f);
    for (std::size_t c = 0; c < cols; ++c) {
      const float scaled = src[r * cols + c] * inv_scale;
      const int quant = static_cast<int>(std::round(scaled));
      packed.data[r * cols + c] = static_cast<std::int8_t>(std::clamp(quant, -127, 127));
    }
  }
  return packed;
}

void reset_file(const fs::path& path) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
}

TensorOffsets append_tensor(const fs::path& file, const QuantizedMatrix& packed) {
  std::ofstream out(file, std::ios::binary | std::ios::app);
  const std::size_t offset_data = static_cast<std::size_t>(out.tellp());
  out.write(reinterpret_cast<const char*>(packed.data.data()), static_cast<std::streamsize>(packed.data.size()));
  const std::size_t offset_scales = static_cast<std::size_t>(out.tellp());
  out.write(reinterpret_cast<const char*>(packed.scales.data()),
            static_cast<std::streamsize>(packed.scales.size() * sizeof(std::uint16_t)));
  return TensorOffsets{offset_data, offset_scales};
}

std::vector<float> apply_base(const std::vector<float>& weights, std::size_t rows, std::size_t cols,
                              const std::vector<float>& x) {
  std::vector<float> y(rows, 0.0f);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < cols; ++c) {
      y[r] += weights[r * cols + c] * x[c];
    }
  }
  return y;
}

std::vector<float> apply_lora(const std::vector<float>& weights,
                              const std::vector<float>& A,
                              const std::vector<float>& B,
                              std::size_t rows,
                              std::size_t cols,
                              std::size_t rank,
                              const std::vector<float>& x) {
  std::vector<float> y = apply_base(weights, rows, cols, x);
  std::vector<float> tmp(rank, 0.0f);
  for (std::size_t r = 0; r < rank; ++r) {
    for (std::size_t c = 0; c < cols; ++c) {
      tmp[r] += B[r * cols + c] * x[c];
    }
  }
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t r = 0; r < rank; ++r) {
      y[row] += A[row * rank + r] * tmp[r];
    }
  }
  return y;
}

bool close_enough(const std::vector<float>& lhs, const std::vector<float>& rhs, float tol = 1e-2f) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (std::fabs(lhs[i] - rhs[i]) > tol) {
      return false;
    }
  }
  return true;
}

bool expect_close(const char* label, const std::vector<float>& actual, const std::vector<float>& expected, float tol = 5e-2f) {
  if (close_enough(actual, expected, tol)) {
    return true;
  }
  std::cerr << label << " mismatch\nactual:";
  for (float v : actual) {
    std::cerr << ' ' << v;
  }
  std::cerr << "\nexpected:";
  for (float v : expected) {
    std::cerr << ' ' << v;
  }
  std::cerr << "\n";
  return false;
}

bool wait_for_output(const std::shared_ptr<std::vector<float>>& out, std::size_t expected_size) {
  for (int i = 0; i < 1000; ++i) {
    if (out->size() == expected_size) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return out->size() == expected_size;
}

int run_cpu_lora_test() {
  const std::size_t rows = 2;
  const std::size_t cols = 3;
  const std::size_t rank = 1;

  std::vector<float> base_weights = {1.0f, -0.5f, 0.25f,
                                     -0.75f, 0.5f, 0.1f};
  const std::vector<float> lora_A = {0.6f,
                                     -0.4f};
  const std::vector<float> lora_B = {0.2f, -0.1f, 0.3f};

  const std::vector<float> x_base = {0.1f, -0.2f, 0.05f};
  const std::vector<float> x_adapter0 = {0.5f, -0.1f, 0.3f};
  const std::vector<float> x_adapter1 = {-0.2f, 0.4f, 0.1f};

  const auto expected_base = apply_base(base_weights, rows, cols, x_base);
  const auto expected_adapter0 = apply_lora(base_weights, lora_A, lora_B, rows, cols, rank, x_adapter0);
  const auto expected_adapter1 = apply_lora(base_weights, lora_A, lora_B, rows, cols, rank, x_adapter1);

  fs::path tmp = fs::temp_directory_path() / "cpu_lora_engine_test";
  fs::create_directories(tmp);
  const fs::path weights_file = tmp / "weights.bin";
  const fs::path adapters_file = tmp / "adapters.bin";
  reset_file(weights_file);
  reset_file(adapters_file);

  const QuantizedMatrix base_quant = quantize_rowwise(base_weights, rows, cols);
  const TensorOffsets base_offsets = append_tensor(weights_file, base_quant);

  const QuantizedMatrix A_quant = quantize_rowwise(lora_A, rows, rank);
  const TensorOffsets A_offsets = append_tensor(adapters_file, A_quant);
  const QuantizedMatrix B_quant = quantize_rowwise(lora_B, rank, cols);
  const TensorOffsets B_offsets = append_tensor(adapters_file, B_quant);

  const fs::path manifest_path = tmp / "manifest.json";
  {
    std::ofstream manifest(manifest_path, std::ios::binary | std::ios::trunc);
    manifest << "{\n"
             << "  \"version\": 1,\n"
             << "  \"endianness\": \"LE\",\n"
             << "  \"tensors\": [\n"
             << "    {\n"
             << "      \"name\": \"linear.weight\",\n"
             << "      \"dtype\": \"int8_rowwise\",\n"
             << "      \"rows\": " << rows << ",\n"
             << "      \"cols\": " << cols << ",\n"
             << "      \"block\": 64,\n"
             << "      \"scale_dtype\": \"fp16\",\n"
             << "      \"layout\": \"rowmajor_blocked\",\n"
             << "      \"data_file\": \"weights.bin\",\n"
             << "      \"offset_data\": " << base_offsets.data << ",\n"
             << "      \"offset_scales\": " << base_offsets.scales << "\n"
             << "    }\n"
             << "  ],\n"
             << "  \"adapters\": [\n"
             << "    {\n"
             << "      \"name\": \"adapter.foo\",\n"
             << "      \"rank\": " << rank << ",\n"
             << "      \"A\": {\n"
             << "        \"name\": \"adapter.foo.A\",\n"
             << "        \"dtype\": \"int8_rowwise\",\n"
             << "        \"rows\": " << rows << ",\n"
             << "        \"cols\": " << rank << ",\n"
             << "        \"block\": 64,\n"
             << "        \"scale_dtype\": \"fp16\",\n"
             << "        \"layout\": \"rowmajor_blocked\",\n"
             << "        \"data_file\": \"adapters.bin\",\n"
             << "        \"offset_data\": " << A_offsets.data << ",\n"
             << "        \"offset_scales\": " << A_offsets.scales << "\n"
             << "      },\n"
             << "      \"B\": {\n"
             << "        \"name\": \"adapter.foo.B\",\n"
             << "        \"dtype\": \"int8_rowwise\",\n"
             << "        \"rows\": " << rank << ",\n"
             << "        \"cols\": " << cols << ",\n"
             << "        \"block\": 64,\n"
             << "        \"scale_dtype\": \"fp16\",\n"
             << "        \"layout\": \"rowmajor_blocked\",\n"
             << "        \"data_file\": \"adapters.bin\",\n"
             << "        \"offset_data\": " << B_offsets.data << ",\n"
             << "        \"offset_scales\": " << B_offsets.scales << "\n"
             << "      }\n"
             << "    }\n"
             << "  ]\n"
             << "}\n";
  }

  try {
    infeng::quant::InfqModel manifest_probe(manifest_path.string());
    (void)manifest_probe;
  } catch (const std::exception& ex) {
    std::cerr << "manifest exception: " << ex.what() << std::endl;
    fs::remove_all(tmp);
    return 1;
  }

  infeng::runtime::CpuLoraEngine engine(manifest_path);

  infeng::runtime::Microbatch base_batch;
  infeng::runtime::Request base_request;
  base_request.activations = x_base;
  base_request.output = std::make_shared<std::vector<float>>();
  base_batch.requests.push_back(base_request);
  engine.run_microbatch(base_batch);
  if (!expect_close("direct_base", *base_batch.requests[0].output, expected_base, 5e-2f)) {
    fs::remove_all(tmp);
    return 1;
  }

  infeng::runtime::Microbatch adapter_batch;
  adapter_batch.adapter_signature = {"adapter.foo"};
  infeng::runtime::Request adapter_request;
  adapter_request.adapter = "adapter.foo";
  adapter_request.adapters = {"adapter.foo"};
  adapter_request.activations = x_adapter0;
  adapter_request.output = std::make_shared<std::vector<float>>();
  adapter_batch.requests.push_back(adapter_request);
  engine.run_microbatch(adapter_batch);
  if (!expect_close("direct_adapter0", *adapter_batch.requests[0].output, expected_adapter0, 5e-2f)) {
    fs::remove_all(tmp);
    return 1;
  }

  infeng::runtime::Microbatch multi_batch;
  multi_batch.adapter_signature = {"adapter.foo"};
  infeng::runtime::Request multi_req0;
  multi_req0.adapter = "adapter.foo";
  multi_req0.adapters = {"adapter.foo"};
  multi_req0.activations = x_adapter0;
  multi_req0.output = std::make_shared<std::vector<float>>();
  infeng::runtime::Request multi_req1;
  multi_req1.adapter = "adapter.foo";
  multi_req1.adapters = {"adapter.foo"};
  multi_req1.activations = x_adapter1;
  multi_req1.output = std::make_shared<std::vector<float>>();
  multi_batch.requests.push_back(multi_req0);
  multi_batch.requests.push_back(multi_req1);
  engine.run_microbatch(multi_batch);
  if (!expect_close("multi_adapter0", *multi_batch.requests[0].output, expected_adapter0, 5e-2f)) {
    fs::remove_all(tmp);
    return 1;
  }
  if (!expect_close("multi_adapter1", *multi_batch.requests[1].output, expected_adapter1, 5e-2f)) {
    fs::remove_all(tmp);
    return 1;
  }

  infeng::runtime::SchedulerConfig config;
  config.max_batch_size = 2;
  config.engine = &engine;
  infeng::runtime::Scheduler scheduler(config);
  scheduler.start();

  auto sched_adapter_out0 = std::make_shared<std::vector<float>>();
  infeng::runtime::Request sched_adapter0;
  sched_adapter0.adapter = "adapter.foo";
  sched_adapter0.adapters = {"adapter.foo"};
  sched_adapter0.activations = x_adapter0;
  sched_adapter0.output = sched_adapter_out0;
  scheduler.submit(std::move(sched_adapter0));

  auto sched_adapter_out1 = std::make_shared<std::vector<float>>();
  infeng::runtime::Request sched_adapter1;
  sched_adapter1.adapter = "adapter.foo";
  sched_adapter1.adapters = {"adapter.foo"};
  sched_adapter1.activations = x_adapter1;
  sched_adapter1.output = sched_adapter_out1;
  scheduler.submit(std::move(sched_adapter1));

  scheduler.stop();

  if (!wait_for_output(sched_adapter_out0, rows) ||
      !wait_for_output(sched_adapter_out1, rows) ||
      !expect_close("sched_adapter0", *sched_adapter_out0, expected_adapter0, 5e-2f) ||
      !expect_close("sched_adapter1", *sched_adapter_out1, expected_adapter1, 5e-2f)) {
    fs::remove_all(tmp);
    return 1;
  }

  fs::remove_all(tmp);
  return 0;
}

}  // namespace

int main() {
  try {
    return run_cpu_lora_test();
  } catch (const std::exception& ex) {
    std::cerr << "exception: " << ex.what() << "\n";
    return 1;
  }
}
