#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infeng::perf {

struct SchedZipfConfig {
  double alpha{1.1};
  std::uint32_t adapters{512};
  double mean_adapters_per_request{1.5};
  std::size_t lora_rank{32};
  double load_rps{50.0};
  double duration_s{30.0};
  double interactive_ratio{0.7};
  double prompt_tokens_mean{128.0};
  std::size_t max_batch_size{1};
  std::size_t decode_threads{4};
  std::size_t adapter_threads{4};
  bool pin_cores{false};
  double base_decode_ms{3.0};
  double adapter_decode_ms{0.2};
  double hot_reload_rate{2.0};
  double hot_reload_latency_ms{40.0};
  std::size_t request_queue_capacity{0};
  std::size_t microbatch_queue_capacity{0};
  unsigned int seed{2026};
};

struct SchedZipfResults {
  double p50_ms{0.0};
  double p95_ms{0.0};
  double base_p95_ms{0.0};
  double lora_tax_pct{0.0};
  double hot_reload_ms_p95{0.0};
  double tokens_per_s{0.0};
  double queue_ms_p95{0.0};
  double adapter_queue_ms_p95{0.0};
  double decode_util{0.0};
  double adapter_util{0.0};
  double overlap_ratio{0.0};
  std::size_t decode_threads{0};
  std::size_t adapter_threads{0};
  bool pin_cores{false};
  double effective_duration_s{0.0};
  std::size_t total_requests{0};
};

SchedZipfResults run_sched_zipf(const SchedZipfConfig& cfg);

}  // namespace infeng::perf
