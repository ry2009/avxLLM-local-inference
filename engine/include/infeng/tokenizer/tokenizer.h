#pragma once

#include <memory>
#include <string>

namespace infeng::tokenizer {

#include <cstdint>

struct Batch {
  const std::int32_t* ids{nullptr};
  std::size_t len{0};
};

class Tokenizer {
 public:
  explicit Tokenizer(const std::string& model_path);
  ~Tokenizer();

  void enable_prefix_cache(bool enable);
  void set_prefix_params(std::size_t prefix_k, std::size_t capacity);
  void set_thread_override(std::size_t threads);

  void encode_stream_begin(const std::string& text, std::size_t n_threads);
  bool encode_stream_next(Batch* out_batch);
  void encode_stream_end();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace infeng::tokenizer
