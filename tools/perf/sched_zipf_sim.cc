#include "tools/perf/sched_zipf_sim.h"

#include "infeng/runtime/microbatch.h"
#include "infeng/runtime/runtime.h"
#include "infeng/runtime/scheduler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace infeng::perf {

namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

bool debug_logging() {
  static const bool enabled = std::getenv("INFENG_SCHED_ZIPF_DEBUG") != nullptr;
  return enabled;
}

void debug_log(const std::string& message) {
  if (debug_logging()) {
    std::cerr << "[sched_zipf_sim] " << message << std::endl;
  }
}

struct RequestSample {
  TimePoint submit{};
  TimePoint finish{};
  bool completed{false};
  double queue_wait_ms{0.0};
};

struct SimulationState {
  std::vector<RequestSample> requests;
  std::vector<double> request_queue_waits;
  std::vector<double> microbatch_queue_waits;
  std::vector<double> hot_reload_latencies;
  std::vector<infeng::runtime::CoreOverlapSnapshot> core_overlap;
  std::mutex mutex;
  std::condition_variable cv;
  std::atomic<std::size_t> completed{0};
  std::atomic<std::uint64_t> total_tokens{0};
  TimePoint start{};
  TimePoint end{};
};

class ZipfSampler {
 public:
  ZipfSampler(std::uint32_t n, double alpha) : cdf_(n) {
    long double norm = 0.0L;
    for (std::uint32_t i = 0; i < n; ++i) {
      norm += 1.0L / std::pow(static_cast<long double>(i + 1), alpha);
      cdf_[i] = static_cast<long double>(norm);
    }
    for (auto& value : cdf_) {
      value /= norm;
    }
  }

  std::uint32_t sample(std::mt19937_64& rng) const {
    const std::uint64_t r = rng();
    const long double denom = static_cast<long double>(std::numeric_limits<std::uint64_t>::max()) + 2.0L;
    const long double u = (static_cast<long double>(r) + 1.0L) / denom;
    auto it = std::lower_bound(cdf_.begin(), cdf_.end(), u);
    if (it == cdf_.end()) {
      return static_cast<std::uint32_t>(cdf_.size() - 1);
    }
    return static_cast<std::uint32_t>(std::distance(cdf_.begin(), it));
  }

  std::size_t size() const { return cdf_.size(); }

 private:
  std::vector<long double> cdf_;
};

std::vector<std::string> sample_adapters(std::size_t count,
                                         ZipfSampler& sampler,
                                         std::mt19937_64& rng) {
  std::vector<std::string> adapters;
  adapters.reserve(count);
  std::unordered_set<std::uint32_t> used;
  const std::size_t vocabulary = sampler.size();
  const std::size_t target = std::min<std::size_t>(count, vocabulary);
  std::size_t attempts = 0;
  while (adapters.size() < target && attempts < target * 4) {
    std::uint32_t id = sampler.sample(rng);
    ++attempts;
    if (used.insert(id).second) {
      adapters.emplace_back("adapter_" + std::to_string(id));
    }
  }
  return adapters;
}

double percentile(std::vector<double> values, double pct) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double rank = pct * static_cast<double>(values.size() - 1);
  const std::size_t idx = static_cast<std::size_t>(rank);
  const double frac = rank - static_cast<double>(idx);
  if (idx + 1 < values.size()) {
    return values[idx] * (1.0 - frac) + values[idx + 1] * frac;
  }
  return values.back();
}

struct RunResult {
  std::vector<double> latencies_ms;
  std::vector<double> queue_wait_ms;
  std::vector<double> micro_wait_ms;
  std::vector<double> hot_reload_ms;
  double tokens_per_s{0.0};
  double effective_duration_s{0.0};
  double decode_util{0.0};
  double adapter_util{0.0};
  double overlap_ratio{0.0};
};

RunResult simulate_once(const SchedZipfConfig& cfg,
                        double mean_adapters,
                        double hot_reload_rate) {
  const std::size_t total_requests = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(cfg.load_rps * cfg.duration_s)));
  SimulationState state;
  state.requests.resize(total_requests);
  debug_log("simulate_once: total_requests=" + std::to_string(total_requests));

  ZipfSampler sampler(cfg.adapters, cfg.alpha);
  std::mt19937_64 rng(cfg.seed);
  std::poisson_distribution<int> adapter_dist(mean_adapters);
  std::poisson_distribution<int> token_dist(std::max(1.0, cfg.prompt_tokens_mean));

  infeng::runtime::SchedulerConfig sched_cfg;
  sched_cfg.max_batch_size = cfg.max_batch_size;
  sched_cfg.decode_threads = cfg.decode_threads;
  sched_cfg.adapter_threads = cfg.adapter_threads;
  sched_cfg.request_queue_capacity = cfg.request_queue_capacity;
  sched_cfg.microbatch_queue_capacity = cfg.microbatch_queue_capacity;
  sched_cfg.request_queue_wait_observer = [&state](double wait_ms) {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.request_queue_waits.push_back(wait_ms);
  };
  sched_cfg.microbatch_queue_wait_observer = [&state](double wait_ms) {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.microbatch_queue_waits.push_back(wait_ms);
  };
  sched_cfg.core_overlap_observer = [&state](const infeng::runtime::CoreOverlapSnapshot& snapshot) {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.core_overlap.push_back(snapshot);
  };
  sched_cfg.core_overlap_emit_interval = std::chrono::milliseconds(200);

  if (cfg.pin_cores) {
    const auto hw = std::thread::hardware_concurrency();
    std::vector<int> decode_core_ids;
    std::vector<int> adapter_core_ids;
    for (unsigned i = 0; i < hw; ++i) {
      if (i % 2 == 0) {
        decode_core_ids.push_back(static_cast<int>(i));
      } else {
        adapter_core_ids.push_back(static_cast<int>(i));
      }
    }
    if (!decode_core_ids.empty()) {
      sched_cfg.decode_cores = decode_core_ids;
    }
    if (!adapter_core_ids.empty()) {
      sched_cfg.adapter_cores = adapter_core_ids;
    }
  }

  sched_cfg.run_microbatch_callback = [&state, &cfg](infeng::runtime::Microbatch& batch) {
    std::size_t adapter_count = 0;
    for (const auto& req : batch.requests) {
      adapter_count += req.adapters.size();
    }
    const double service_ms = cfg.base_decode_ms * static_cast<double>(batch.requests.size()) +
                              cfg.adapter_decode_ms * static_cast<double>(adapter_count);
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(service_ms));
    const auto end = Clock::now();

    {
      std::lock_guard<std::mutex> lock(state.mutex);
      for (const auto& req : batch.requests) {
        auto& sample = state.requests[req.request_id];
        sample.finish = end;
        sample.completed = true;
        sample.queue_wait_ms = req.queue_wait_ms;
      }
    }
    state.total_tokens.fetch_add(batch.total_tokens, std::memory_order_relaxed);
    state.completed.fetch_add(batch.requests.size(), std::memory_order_relaxed);
    state.cv.notify_all();
  };

  infeng::runtime::Scheduler scheduler(std::move(sched_cfg));

  std::atomic<bool> stop_hot_reload{false};
  std::thread hot_thread;
  if (hot_reload_rate > 0.0) {
    hot_thread = std::thread([&]() {
      const auto interval = std::chrono::duration<double>(1.0 / hot_reload_rate);
      while (!stop_hot_reload.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(interval);
        if (stop_hot_reload.load(std::memory_order_relaxed)) {
          break;
        }
        const auto start = Clock::now();
        std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(cfg.hot_reload_latency_ms));
        const auto end = Clock::now();
        const double lat_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.hot_reload_latencies.push_back(lat_ms);
      }
    });
  }

  state.start = Clock::now();
  std::thread producer([&]() {
    const auto start = Clock::now();
    for (std::size_t id = 0; id < total_requests; ++id) {
      auto& sample = state.requests[id];
      const auto now = Clock::now();
      sample.submit = now;
      sample.completed = false;
      sample.queue_wait_ms = 0.0;

      infeng::runtime::Request request;
      request.prompt = "p";
      request.request_id = id;
      const int tokens = std::max(1, token_dist(rng));
      request.token_count = static_cast<std::size_t>(tokens);

      const int adapters_needed = std::max(0, adapter_dist(rng));
      if (adapters_needed > 0) {
        auto adapters = sample_adapters(static_cast<std::size_t>(adapters_needed), sampler, rng);
        if (!adapters.empty()) {
          request.adapter = adapters.front();
          request.adapters = std::move(adapters);
        }
      }

      scheduler.submit(std::move(request));

      if (id + 1 < total_requests) {
        const double next_offset = static_cast<double>(id + 1) / cfg.load_rps;
        std::this_thread::sleep_until(start + std::chrono::duration<double>(next_offset));
      }
    }
  });

  producer.join();
  debug_log("producer joined");
  scheduler.stop();
  debug_log("scheduler stopped");
  {
    std::unique_lock<std::mutex> lock(state.mutex);
    const bool completed = state.cv.wait_for(
        lock,
        std::chrono::seconds(30),
        [&]() { return state.completed.load(std::memory_order_relaxed) >= total_requests; });
    if (!completed) {
      throw std::runtime_error("scheduler simulation timed out waiting for completion");
    }
    debug_log("all requests completed");
  }
  state.end = Clock::now();
  debug_log("state.end recorded");

  stop_hot_reload.store(true, std::memory_order_relaxed);
  if (hot_thread.joinable()) {
    hot_thread.join();
  }
  debug_log("hot reload thread joined");

  RunResult result;
  result.latencies_ms.reserve(total_requests);
  for (const auto& sample : state.requests) {
    if (!sample.completed) {
      continue;
    }
    const double latency_ms = std::chrono::duration<double, std::milli>(sample.finish - sample.submit).count();
    result.latencies_ms.push_back(latency_ms);
  }
  result.queue_wait_ms = state.request_queue_waits;
  result.micro_wait_ms = state.microbatch_queue_waits;
  result.hot_reload_ms = state.hot_reload_latencies;
  double total_window_ms = 0.0;
  double total_decode_ms = 0.0;
  double total_adapter_ms = 0.0;
  double total_overlap_ms = 0.0;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    for (const auto& snapshot : state.core_overlap) {
      total_window_ms += snapshot.window_ms;
      total_decode_ms += snapshot.decode_active_ms;
      total_adapter_ms += snapshot.adapter_active_ms;
      total_overlap_ms += snapshot.overlap_ms;
    }
  }
  if (total_window_ms > 0.0) {
    result.decode_util = total_decode_ms / total_window_ms;
    result.adapter_util = total_adapter_ms / total_window_ms;
    result.overlap_ratio = total_overlap_ms / total_window_ms;
  }
  const double elapsed_s = std::max(1e-6, std::chrono::duration<double>(state.end - state.start).count());
  result.tokens_per_s = static_cast<double>(state.total_tokens.load(std::memory_order_relaxed)) / elapsed_s;
  result.effective_duration_s = elapsed_s;
  return result;
}

SchedZipfResults summarise(const RunResult& baseline,
                           const RunResult& lora,
                           const SchedZipfConfig& cfg) {
  SchedZipfResults summary;
  summary.base_p95_ms = percentile(std::vector<double>(baseline.latencies_ms.begin(), baseline.latencies_ms.end()), 0.95);
  summary.p50_ms = percentile(std::vector<double>(lora.latencies_ms.begin(), lora.latencies_ms.end()), 0.50);
  summary.p95_ms = percentile(std::vector<double>(lora.latencies_ms.begin(), lora.latencies_ms.end()), 0.95);
  if (summary.base_p95_ms > 0.0) {
    summary.lora_tax_pct = (summary.p95_ms - summary.base_p95_ms) / summary.base_p95_ms * 100.0;
  }
  summary.hot_reload_ms_p95 = percentile(std::vector<double>(lora.hot_reload_ms.begin(), lora.hot_reload_ms.end()), 0.95);
  summary.queue_ms_p95 = percentile(std::vector<double>(lora.queue_wait_ms.begin(), lora.queue_wait_ms.end()), 0.95);
  summary.adapter_queue_ms_p95 = percentile(std::vector<double>(lora.micro_wait_ms.begin(), lora.micro_wait_ms.end()), 0.95);
  summary.tokens_per_s = lora.tokens_per_s;
  summary.decode_util = lora.decode_util;
  summary.adapter_util = lora.adapter_util;
  summary.overlap_ratio = lora.overlap_ratio;
  summary.decode_threads = cfg.decode_threads;
  summary.adapter_threads = cfg.adapter_threads;
  summary.pin_cores = cfg.pin_cores;
  summary.effective_duration_s = lora.effective_duration_s;
  summary.total_requests = lora.latencies_ms.size();
  return summary;
}

}  // namespace

SchedZipfResults run_sched_zipf(const SchedZipfConfig& cfg) {
  auto baseline_cfg = cfg;
  baseline_cfg.mean_adapters_per_request = 0.0;
  baseline_cfg.hot_reload_rate = 0.0;

  RunResult baseline = simulate_once(baseline_cfg, 0.0, 0.0);
  RunResult lora = simulate_once(cfg, cfg.mean_adapters_per_request, cfg.hot_reload_rate);
  return summarise(baseline, lora, cfg);
}

}  // namespace infeng::perf
