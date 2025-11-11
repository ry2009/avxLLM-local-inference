#pragma once

#include "infeng/runtime/microbatch.h"
#include "infeng/runtime/queues.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace infeng::runtime {

class CpuLoraEngine;

struct CoreOverlapSnapshot {
  double decode_active_ms{0.0};
  double adapter_active_ms{0.0};
  double overlap_ms{0.0};
  double window_ms{0.0};
  int active_decode{0};
  int active_adapter{0};
};

struct SchedulerConfig {
  std::size_t max_batch_size{4};
  std::vector<int> decode_cores;
  std::vector<int> adapter_cores;
  std::size_t decode_threads{1};
  std::size_t adapter_threads{1};
  CpuLoraEngine* engine{nullptr};
  std::size_t request_queue_capacity{0};
  std::size_t microbatch_queue_capacity{0};
  bool drain_on_stop{true};
  std::function<void(double)> request_queue_wait_observer;
  std::function<void(double)> microbatch_queue_wait_observer;
  std::function<void(Microbatch&)> run_microbatch_callback;
  std::function<void(const CoreOverlapSnapshot&)> core_overlap_observer;
  std::chrono::milliseconds core_overlap_emit_interval{std::chrono::milliseconds(500)};
};

class Scheduler {
 public:
  explicit Scheduler(SchedulerConfig config);
  ~Scheduler();

  void start();
  void stop();

  void submit(Request request);

  std::size_t pending_requests() const;

 private:
  void decode_worker();
  void adapter_worker();
  enum class ActivityType { Decode, Adapter };
  void record_activity(ActivityType type, int delta);
  void emit_core_overlap(bool force);
  void update_core_overlap_locked(std::chrono::steady_clock::time_point now);

  SchedulerConfig config_;
  MicrobatchBuilder builder_;
  std::mutex builder_mutex_;
  std::mutex core_overlap_mutex_;
  struct CoreOverlapState {
    std::chrono::steady_clock::time_point last_update{};
    std::chrono::steady_clock::time_point last_emit{};
    int active_decode{0};
    int active_adapter{0};
    double decode_ms{0.0};
    double adapter_ms{0.0};
    double overlap_ms{0.0};
  } core_overlap_state_;
  MpmcQueue<Request> request_queue_;
  MpmcQueue<Microbatch> adapter_queue_;
  std::vector<std::thread> decode_threads_;
  std::vector<std::thread> adapter_threads_;
  std::atomic<bool> running_{false};
};

}  // namespace infeng::runtime
