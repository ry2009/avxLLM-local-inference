#include "tools/perf/sched_zipf_sim.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

namespace fs = std::filesystem;

namespace {

double parse_double(const std::unordered_map<std::string, std::string>& args,
                    const std::string& key,
                    double fallback) {
  auto it = args.find(key);
  if (it == args.end()) {
    return fallback;
  }
  return std::stod(it->second);
}

std::size_t parse_size(const std::unordered_map<std::string, std::string>& args,
                       const std::string& key,
                       std::size_t fallback) {
  auto it = args.find(key);
  if (it == args.end()) {
    return fallback;
  }
  return static_cast<std::size_t>(std::stoull(it->second));
}

bool parse_flag(const std::unordered_map<std::string, std::string>& args,
                const std::string& key,
                bool fallback) {
  auto it = args.find(key);
  if (it == args.end()) {
    return fallback;
  }
  const std::string value = it->second;
  return value == "1" || value == "true" || value == "on";
}

std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
  std::unordered_map<std::string, std::string> result;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--", 0) != 0) {
      continue;
    }
    arg = arg.substr(2);
    auto eq_pos = arg.find('=');
    if (eq_pos != std::string::npos) {
      result[arg.substr(0, eq_pos)] = arg.substr(eq_pos + 1);
    } else if (i + 1 < argc && argv[i + 1][0] != '-') {
      result[arg] = argv[++i];
    } else {
      result[arg] = "true";
    }
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  auto args = parse_args(argc, argv);
  infeng::perf::SchedZipfConfig cfg;
  cfg.alpha = parse_double(args, "alpha", cfg.alpha);
  cfg.adapters = static_cast<std::uint32_t>(parse_size(args, "adapters", cfg.adapters));
  cfg.mean_adapters_per_request = parse_double(args, "mean_adapters", cfg.mean_adapters_per_request);
  cfg.lora_rank = parse_size(args, "rank", cfg.lora_rank);
  cfg.load_rps = parse_double(args, "load_rps", cfg.load_rps);
  cfg.duration_s = parse_double(args, "duration_s", cfg.duration_s);
  cfg.interactive_ratio = parse_double(args, "interactive_ratio", cfg.interactive_ratio);
  cfg.decode_threads = parse_size(args, "decode_threads", cfg.decode_threads);
  cfg.adapter_threads = parse_size(args, "adapter_threads", cfg.adapter_threads);
  cfg.pin_cores = parse_flag(args, "pin_cores", cfg.pin_cores);
  cfg.hot_reload_rate = parse_double(args, "hot_reload_rate", cfg.hot_reload_rate);
  cfg.hot_reload_latency_ms = parse_double(args, "hot_reload_latency_ms", cfg.hot_reload_latency_ms);
  cfg.base_decode_ms = parse_double(args, "base_decode_ms", cfg.base_decode_ms);
  cfg.adapter_decode_ms = parse_double(args, "adapter_decode_ms", cfg.adapter_decode_ms);
  cfg.prompt_tokens_mean = parse_double(args, "prompt_tokens_mean", cfg.prompt_tokens_mean);
  cfg.max_batch_size = parse_size(args, "max_batch_size", cfg.max_batch_size);
  cfg.seed = static_cast<unsigned int>(parse_size(args, "seed", cfg.seed));

  const std::string out_path = args.count("out") ? args["out"] : "reports/sched_zipf.csv";

  const auto results = infeng::perf::run_sched_zipf(cfg);
  const double observed_rps = results.effective_duration_s > 0.0
                                  ? static_cast<double>(results.total_requests) / results.effective_duration_s
                                  : 0.0;

  fs::create_directories(fs::path(out_path).parent_path());
  const bool existed = fs::exists(out_path);
  std::ofstream out(out_path, std::ios::app);
  if (!existed) {
    out << "sched,alpha,adapters_N,mean_adapters_per_req,rank,load_rps,duration_s,";
    out << "decode_threads,adapter_threads,pin_cores,p50_ms,p95_ms,base_p95_ms,lora_tax_pct,hot_reload_ms_p95,";
    out << "tokens_per_s,queue_ms_p95,decode_util,adapter_util,overlap_ratio\n";
  }
  out << "scheduler_zipf," << cfg.alpha << ',' << cfg.adapters << ','
      << cfg.mean_adapters_per_request << ',' << cfg.lora_rank << ',' << observed_rps << ','
      << results.effective_duration_s << ',' << results.decode_threads << ','
      << results.adapter_threads << ',' << (results.pin_cores ? 1 : 0) << ',' << results.p50_ms << ','
      << results.p95_ms << ',' << results.base_p95_ms << ',' << results.lora_tax_pct << ','
      << results.hot_reload_ms_p95 << ',' << results.tokens_per_s << ',' << results.queue_ms_p95 << ','
      << results.decode_util << ',' << results.adapter_util << ',' << results.overlap_ratio << "\n";

  std::cout << "scheduler_zipf p95_ms=" << results.p95_ms << " tokens_per_s=" << results.tokens_per_s
            << " lora_tax_pct=" << results.lora_tax_pct << " overlap_ratio=" << results.overlap_ratio << std::endl;
  return 0;
}
