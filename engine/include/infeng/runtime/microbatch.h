#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace infeng::runtime {

struct Request {
  std::string prompt;
  std::string adapter;
  std::vector<std::string> adapters;
  std::vector<float> activations;
  std::shared_ptr<std::vector<float>> output;
  std::size_t request_id{0};
  std::size_t token_count{0};
  std::chrono::steady_clock::time_point submit_time{};
  double queue_wait_ms{0.0};
};

struct Microbatch {
  std::vector<Request> requests;
  std::vector<std::string> adapter_signature;
  std::size_t total_tokens{0};
  std::chrono::steady_clock::time_point enqueue_time{};
  double queue_wait_ms{0.0};
};

class MicrobatchBuilder {
 public:
  explicit MicrobatchBuilder(std::size_t max_batch_size);

  std::vector<Microbatch> add_request(Request request);
  std::vector<Microbatch> flush();

 private:
  std::size_t max_batch_size_;
  std::map<std::vector<std::string>, std::vector<Request>> buckets_;

  static std::vector<std::string> normalise_signature(const std::vector<std::string>& adapters);
};

}  // namespace infeng::runtime
