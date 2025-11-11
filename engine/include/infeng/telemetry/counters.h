#pragma once

#include <chrono>
#include <string>

namespace infeng::telemetry {

class ScopedCounter {
 public:
  ScopedCounter(const std::string& name);
  ~ScopedCounter();

  double elapsed_microseconds() const;

 private:
  std::string name_;
  std::chrono::steady_clock::time_point start_;
};

void flush_csv(const std::string& path);

}  // namespace infeng::telemetry
