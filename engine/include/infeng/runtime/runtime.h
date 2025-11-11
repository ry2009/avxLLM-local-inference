#pragma once

#include <string>

namespace infeng::runtime {

struct EngineConfig {
  int32_t num_decode_threads{0};
  int32_t num_adapter_threads{0};
  int32_t num_tokenizer_threads{0};
  int32_t max_pending_requests{0};
  bool enable_metrics{true};
  bool drain_on_shutdown{true};
};

void initialize(const EngineConfig& config);
void shutdown();

const EngineConfig& config();

std::string version();

}  // namespace infeng::runtime
