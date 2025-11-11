#include "infeng/telemetry/counters.h"

#include <filesystem>
#include <fstream>
#include <system_error>
#include <mutex>
#include <vector>

namespace infeng::telemetry {

namespace {
struct Sample {
  std::string name;
  double micros;
};

std::mutex g_mutex;
std::vector<Sample> g_samples;
}  // namespace

ScopedCounter::ScopedCounter(const std::string& name)
    : name_(name), start_(std::chrono::steady_clock::now()) {}

ScopedCounter::~ScopedCounter() {
  const auto end = std::chrono::steady_clock::now();
  const auto micros =
      std::chrono::duration<double, std::micro>(end - start_).count();
  std::lock_guard lock(g_mutex);
  g_samples.push_back(Sample{name_, micros});
}

double ScopedCounter::elapsed_microseconds() const {
  const auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(now - start_).count();
}

void flush_csv(const std::string& path) {
  std::lock_guard lock(g_mutex);
  const std::filesystem::path target(path);
  if (const auto parent = target.parent_path(); !parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
  }
  std::ofstream out(target);
  out << "stage,elapsed_us\n";
  for (const auto& sample : g_samples) {
    out << sample.name << "," << sample.micros << "\n";
  }
  g_samples.clear();
}

}  // namespace infeng::telemetry
