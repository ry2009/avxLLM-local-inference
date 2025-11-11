#include "infeng/runtime/runtime.h"

#include <mutex>

namespace infeng::runtime {

namespace {
std::once_flag g_init_flag;
bool g_initialized = false;
EngineConfig g_config{};
}  // namespace

void initialize(const EngineConfig& config) {
  std::call_once(g_init_flag, [&]() {
    g_config = config;
    g_initialized = true;
  });
}

const EngineConfig& config() {
  return g_config;
}

void shutdown() {
  if (!g_initialized) {
    return;
  }
  g_initialized = false;
}

std::string version() {
  return "0.1.0";
}

}  // namespace infeng::runtime
