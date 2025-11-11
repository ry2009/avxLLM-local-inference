#include "infeng/runtime/scheduler.h"

#include <cassert>
#include <chrono>
#include <thread>

using namespace infeng::runtime;

int main() {
  SchedulerConfig config;
  config.max_batch_size = 2;
  Scheduler sched(config);
  sched.start();
  for (int i = 0; i < 5; ++i) {
    Request req;
    req.prompt = "prompt" + std::to_string(i);
    req.adapter = (i % 2 == 0) ? "adapter" : "";
    if (!req.adapter.empty()) {
      req.adapters = {req.adapter};
    }
    sched.submit(req);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  sched.stop();
  return 0;
}
