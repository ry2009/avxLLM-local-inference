#include "infeng/runtime/scheduler.h"
#include "infeng/runtime/runtime.h"

#include <filesystem>
#include <fstream>

using namespace infeng::runtime;

int main() {
  EngineConfig cfg{};
  cfg.enable_metrics = true;
  initialize(cfg);

  SchedulerConfig sched_cfg;
  sched_cfg.max_batch_size = 1;
  Scheduler scheduler(sched_cfg);
  scheduler.start();

  Request req;
  req.prompt = "test";
  scheduler.submit(std::move(req));
  scheduler.stop();

  const std::filesystem::path csv("reports/scheduler_metrics.csv");
  if (!std::filesystem::exists(csv)) {
    return 1;
  }
  std::ifstream in(csv);
  if (!in.good()) {
    return 1;
  }
  shutdown();
  return 0;
}

