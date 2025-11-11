#include "tools/perf/sched_zipf_sim.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

int main() {
  infeng::perf::SchedZipfConfig cfg;
  std::vector<double> workloads = {10.0, 20.0, 30.0};
  fs::create_directories("reports");
  std::ofstream out("reports/sched_zipf_bench.csv", std::ios::trunc);
  out << "load_rps,p50_ms,p95_ms,lora_tax_pct,tokens_per_s\n";

  for (double load : workloads) {
    cfg.load_rps = load;
    auto results = infeng::perf::run_sched_zipf(cfg);
    out << load << ',' << results.p50_ms << ',' << results.p95_ms << ',' << results.lora_tax_pct << ','
        << results.tokens_per_s << "\n";
    std::cout << "load=" << load << " rps => p95_ms=" << results.p95_ms << " tokens/s="
              << results.tokens_per_s << std::endl;
  }
  return 0;
}
