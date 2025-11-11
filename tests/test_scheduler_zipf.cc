#include "infeng/runtime/microbatch.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <string>
#include <vector>

using infeng::runtime::Microbatch;
using infeng::runtime::MicrobatchBuilder;
using infeng::runtime::Request;

namespace {

std::vector<std::string> generate_zipf_mix(std::size_t adapters,
                                           std::size_t requests,
                                           double alpha) {
  std::vector<double> weights(adapters);
  for (std::size_t i = 0; i < adapters; ++i) {
    weights[i] = 1.0 / std::pow(static_cast<double>(i + 1), alpha);
  }
  std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
  std::mt19937 rng(1234);
  std::vector<std::string> result;
  result.reserve(requests);
  for (std::size_t i = 0; i < requests; ++i) {
    const std::size_t idx = dist(rng);
    result.push_back("adapter_" + std::to_string(idx));
  }
  return result;
}

void assert_microbatch_consistency(const std::vector<Microbatch>& batches) {
  for (const auto& batch : batches) {
    if (batch.requests.empty()) {
      continue;
    }
    const std::string& adapter = batch.requests.front().adapter;
    for (const auto& req : batch.requests) {
      assert(req.adapter == adapter);
    }
  }
}

}  // namespace

int main() {
  const auto mix = generate_zipf_mix(/*adapters=*/6, /*requests=*/200, /*alpha=*/1.1);
  MicrobatchBuilder builder(/*max_batch_size=*/4);
  std::size_t dispatched_requests = 0;

  for (const auto& adapter : mix) {
    Request req;
    req.prompt = "p";
    req.adapter = adapter;
    req.adapters = {adapter};
    auto ready = builder.add_request(std::move(req));
    assert_microbatch_consistency(ready);
    for (const auto& batch : ready) {
      dispatched_requests += batch.requests.size();
    }
  }

  auto remaining = builder.flush();
  assert_microbatch_consistency(remaining);

  for (const auto& batch : remaining) {
    dispatched_requests += batch.requests.size();
  }
  assert(dispatched_requests == mix.size());

  return 0;
}
