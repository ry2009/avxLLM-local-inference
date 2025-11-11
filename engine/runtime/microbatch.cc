#include "infeng/runtime/microbatch.h"

#include <algorithm>
#include <chrono>
#include <utility>

namespace infeng::runtime {

namespace {

std::vector<std::string> deduplicate_preserve_order(const std::vector<std::string>& adapters) {
  std::vector<std::string> result;
  result.reserve(adapters.size());
  for (const auto& adapter : adapters) {
    if (std::find(result.begin(), result.end(), adapter) == result.end()) {
      result.push_back(adapter);
    }
  }
  return result;
}

std::vector<std::string> deduplicate_sorted(std::vector<std::string> values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

}  // namespace

MicrobatchBuilder::MicrobatchBuilder(std::size_t max_batch_size)
    : max_batch_size_(max_batch_size) {}

std::vector<std::string> MicrobatchBuilder::normalise_signature(const std::vector<std::string>& adapters) {
  if (adapters.empty()) {
    return {};
  }
  return deduplicate_sorted(adapters);
}

std::vector<Microbatch> MicrobatchBuilder::add_request(Request request) {
  if (!request.adapter.empty() && request.adapters.empty()) {
    request.adapters.push_back(request.adapter);
  }
  request.adapters = deduplicate_preserve_order(request.adapters);
  auto signature = normalise_signature(request.adapters);
  auto& bucket = buckets_[signature];
  bucket.push_back(std::move(request));
  std::vector<Microbatch> ready;
  while (bucket.size() >= max_batch_size_) {
    Microbatch batch;
    batch.adapter_signature = signature;
    batch.requests.insert(batch.requests.end(), bucket.begin(), bucket.begin() + max_batch_size_);
    bucket.erase(bucket.begin(), bucket.begin() + max_batch_size_);
    batch.total_tokens = 0;
    for (const auto& req : batch.requests) {
      batch.total_tokens += req.token_count;
    }
    batch.enqueue_time = std::chrono::steady_clock::now();
    ready.push_back(std::move(batch));
  }
  return ready;
}

std::vector<Microbatch> MicrobatchBuilder::flush() {
  std::vector<Microbatch> ready;
  for (auto& [signature, requests] : buckets_) {
    if (requests.empty()) {
      continue;
    }
    Microbatch batch;
    batch.adapter_signature = signature;
    batch.requests.swap(requests);
    batch.total_tokens = 0;
    for (const auto& req : batch.requests) {
      batch.total_tokens += req.token_count;
    }
    batch.enqueue_time = std::chrono::steady_clock::now();
    ready.push_back(std::move(batch));
  }
  buckets_.clear();
  return ready;
}

}  // namespace infeng::runtime
