#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

namespace infeng::runtime {

template <typename T>
class MpmcQueue {
 public:
  void push(T value) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_not_full_.wait(lock, [&] { return !capacity_ || queue_.size() < capacity_ || shutdown_; });
    if (shutdown_) {
      return;
    }
    queue_.push(std::move(value));
    lock.unlock();
    cv_.notify_one();
  }

  std::optional<T> pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return !queue_.empty() || shutdown_; });
    if (queue_.empty()) {
      return std::nullopt;
    }
    T value = std::move(queue_.front());
    queue_.pop();
    cv_not_full_.notify_one();
    return value;
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutdown_ = true;
    }
    cv_.notify_all();
    cv_not_full_.notify_all();
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  void set_capacity(std::size_t capacity) {
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_ = capacity;
    cv_not_full_.notify_all();
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable cv_not_full_;
  std::queue<T> queue_;
  bool shutdown_{false};
  std::size_t capacity_{0};
};

}  // namespace infeng::runtime
