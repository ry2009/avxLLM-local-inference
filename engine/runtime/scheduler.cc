#include "infeng/runtime/scheduler.h"

#include "infeng/runtime/affinity.h"
#include "infeng/runtime/cpu_lora_engine.h"
#include "infeng/runtime/runtime.h"
#include "infeng/telemetry/counters.h"

#include <chrono>
#include <optional>

namespace infeng::runtime {

namespace {

bool metrics_enabled() {
  return config().enable_metrics;
}

class OptionalCounter {
 public:
  explicit OptionalCounter(const char* name) {
    if (metrics_enabled()) {
      counter_.emplace(name);
    }
  }

 private:
  std::optional<telemetry::ScopedCounter> counter_;
};

}  // namespace

Scheduler::Scheduler(SchedulerConfig config)
    : config_(std::move(config)), builder_(config_.max_batch_size) {}

Scheduler::~Scheduler() {
  stop();
}

void Scheduler::start() {
  if (running_.exchange(true)) {
    return;
  }
  if (config_.core_overlap_observer) {
    std::lock_guard<std::mutex> lock(core_overlap_mutex_);
    const auto now = std::chrono::steady_clock::now();
    core_overlap_state_ = {};
    core_overlap_state_.last_update = now;
    core_overlap_state_.last_emit = now;
  }
  if (config_.request_queue_capacity) {
    request_queue_.set_capacity(config_.request_queue_capacity);
  }
  if (config_.microbatch_queue_capacity) {
    adapter_queue_.set_capacity(config_.microbatch_queue_capacity);
  }
  auto spawn = [&](std::vector<std::thread>& threads,
                   const std::vector<int>& cores,
                   std::size_t count,
                   auto fn) {
    if (!cores.empty()) {
      for (int core : cores) {
        threads.emplace_back([=]() mutable {
          set_thread_affinity({core});
          fn();
        });
      }
    } else {
      for (std::size_t idx = 0; idx < std::max<std::size_t>(1, count); ++idx) {
        threads.emplace_back(fn);
      }
    }
  };
  spawn(decode_threads_, config_.decode_cores, config_.decode_threads, [this]() { decode_worker(); });
  spawn(adapter_threads_, config_.adapter_cores, config_.adapter_threads, [this]() { adapter_worker(); });
}

void Scheduler::stop() {
  if (!running_.exchange(false)) {
    return;
  }
  request_queue_.shutdown();
  for (auto& t : decode_threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
  decode_threads_.clear();
  adapter_queue_.shutdown();
  for (auto& t : adapter_threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
  adapter_threads_.clear();
  emit_core_overlap(true);
  if (config().enable_metrics) {
    telemetry::flush_csv("reports/scheduler_metrics.csv");
  }
}

void Scheduler::submit(Request request) {
  if (!running_) {
    start();
  }
  request.submit_time = std::chrono::steady_clock::now();
  request_queue_.push(std::move(request));
}

std::size_t Scheduler::pending_requests() const {
  return request_queue_.size();
}

void Scheduler::decode_worker() {
  while (true) {
    OptionalCounter pop_counter("scheduler.decode_pop");
    auto request = request_queue_.pop();
    if (!request.has_value()) {
      break;
    }
    record_activity(ActivityType::Decode, +1);
    if (request->submit_time.time_since_epoch().count() != 0) {
      const auto now = std::chrono::steady_clock::now();
      const double wait_ms = std::chrono::duration<double, std::milli>(now - request->submit_time).count();
      request->queue_wait_ms = wait_ms;
      if (config_.request_queue_wait_observer) {
        config_.request_queue_wait_observer(wait_ms);
      }
    }
    OptionalCounter build_counter("scheduler.build_microbatch");
    std::vector<Microbatch> ready;
    {
      std::lock_guard<std::mutex> lock(builder_mutex_);
      ready = builder_.add_request(std::move(*request));
    }
    for (auto& batch : ready) {
      OptionalCounter push_counter("scheduler.enqueue_microbatch");
      batch.enqueue_time = std::chrono::steady_clock::now();
      adapter_queue_.push(std::move(batch));
    }
    record_activity(ActivityType::Decode, -1);
  }
  std::vector<Microbatch> leftover;
  {
    std::lock_guard<std::mutex> lock(builder_mutex_);
    leftover = builder_.flush();
  }
  for (auto& batch : leftover) {
    OptionalCounter push_counter("scheduler.enqueue_flush");
    batch.enqueue_time = std::chrono::steady_clock::now();
    adapter_queue_.push(std::move(batch));
  }
}

void Scheduler::adapter_worker() {
  while (true) {
    OptionalCounter pop_counter("scheduler.adapter_pop");
    auto batch = adapter_queue_.pop();
    if (!batch.has_value()) {
      break;
    }
    record_activity(ActivityType::Adapter, +1);
    if (batch->enqueue_time.time_since_epoch().count() != 0) {
      const auto now = std::chrono::steady_clock::now();
      batch->queue_wait_ms = std::chrono::duration<double, std::milli>(now - batch->enqueue_time).count();
      if (config_.microbatch_queue_wait_observer) {
        config_.microbatch_queue_wait_observer(batch->queue_wait_ms);
      }
    }
    if (config_.engine) {
      OptionalCounter run_counter("scheduler.run_microbatch");
      config_.engine->run_microbatch(*batch);
    } else if (config_.run_microbatch_callback) {
      config_.run_microbatch_callback(*batch);
    }
    record_activity(ActivityType::Adapter, -1);
  }
}

void Scheduler::record_activity(ActivityType type, int delta) {
  if (!config_.core_overlap_observer || delta == 0) {
    return;
  }
  CoreOverlapSnapshot snapshot;
  bool should_emit = false;
  const auto now = std::chrono::steady_clock::now();
  {
    std::lock_guard<std::mutex> lock(core_overlap_mutex_);
    update_core_overlap_locked(now);
    auto& state = core_overlap_state_;
    if (type == ActivityType::Decode) {
      state.active_decode = std::max(0, state.active_decode + delta);
    } else {
      state.active_adapter = std::max(0, state.active_adapter + delta);
    }
    const auto since_emit = now - state.last_emit;
    if (since_emit >= config_.core_overlap_emit_interval) {
      const double window_ms = std::chrono::duration<double, std::milli>(since_emit).count();
      snapshot.decode_active_ms = state.decode_ms;
      snapshot.adapter_active_ms = state.adapter_ms;
      snapshot.overlap_ms = state.overlap_ms;
      snapshot.window_ms = window_ms;
      snapshot.active_decode = state.active_decode;
      snapshot.active_adapter = state.active_adapter;
      state.decode_ms = 0.0;
      state.adapter_ms = 0.0;
      state.overlap_ms = 0.0;
      state.last_emit = now;
      should_emit = true;
    }
  }
  if (should_emit) {
    config_.core_overlap_observer(snapshot);
  }
}

void Scheduler::emit_core_overlap(bool force) {
  if (!config_.core_overlap_observer) {
    return;
  }
  CoreOverlapSnapshot snapshot;
  bool should_emit = false;
  const auto now = std::chrono::steady_clock::now();
  {
    std::lock_guard<std::mutex> lock(core_overlap_mutex_);
    update_core_overlap_locked(now);
    auto& state = core_overlap_state_;
    const auto since_emit = now - state.last_emit;
    if (force || since_emit >= config_.core_overlap_emit_interval) {
      const double window_ms = std::chrono::duration<double, std::milli>(since_emit).count();
      if (force || window_ms > 0.0) {
        snapshot.decode_active_ms = state.decode_ms;
        snapshot.adapter_active_ms = state.adapter_ms;
        snapshot.overlap_ms = state.overlap_ms;
        snapshot.window_ms = window_ms;
        snapshot.active_decode = state.active_decode;
        snapshot.active_adapter = state.active_adapter;
        state.decode_ms = 0.0;
        state.adapter_ms = 0.0;
        state.overlap_ms = 0.0;
        state.last_emit = now;
        should_emit = true;
      }
    }
  }
  if (should_emit) {
    config_.core_overlap_observer(snapshot);
  }
}

void Scheduler::update_core_overlap_locked(std::chrono::steady_clock::time_point now) {
  auto& state = core_overlap_state_;
  const double delta_ms = std::chrono::duration<double, std::milli>(now - state.last_update).count();
  if (delta_ms <= 0.0) {
    state.last_update = now;
    return;
  }
  if (state.active_decode > 0) {
    state.decode_ms += delta_ms;
  }
  if (state.active_adapter > 0) {
    state.adapter_ms += delta_ms;
  }
  if (state.active_decode > 0 && state.active_adapter > 0) {
    state.overlap_ms += delta_ms;
  }
  state.last_update = now;
}

}  // namespace infeng::runtime
