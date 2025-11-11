#include "infeng/tokenizer/tokenizer.h"

#include "c_api.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace infeng::tokenizer {

namespace {
std::size_t read_env_threads() {
  const char* value = std::getenv("INFENG_TOK_THREADS");
  if (!value || std::strlen(value) == 0) {
    return 0;
  }
  try {
    std::size_t threads = static_cast<std::size_t>(std::stoul(value));
    return threads;
  } catch (...) {
    return 0;
  }
}
}

struct Tokenizer::Impl {
  explicit Impl(const std::string& model_path)
      : env_threads(read_env_threads()) {
    handle = inf_tok_new(INF_TOK_ABI_VERSION, model_path.c_str());
    if (!handle) {
      throw std::runtime_error("Failed to initialize tokenizer");
    }
  }

  ~Impl() {
    if (handle) {
      inf_tok_free(handle);
      handle = nullptr;
    }
  }
  void check(int code, const char* context) {
    if (code >= 0) {
      return;
    }
    const char* err = inf_tok_last_error(handle);
    if (err && std::strlen(err) != 0) {
      throw std::runtime_error(std::string(context) + ": " + err);
    }
    throw std::runtime_error(std::string(context) + " failed (" + std::to_string(code) + ")");
  }

  std::size_t resolve_threads(std::size_t requested) const {
    if (requested > 0) {
      return requested;
    }
    if (env_threads > 0) {
      return env_threads;
    }
    return 0;
  }

  void enable_cache(bool enable) {
    inf_tok_set_prefix_cache(handle, enable ? 1 : 0);
  }

  void set_prefix_params(std::size_t prefix_k, std::size_t capacity) {
    inf_tok_set_prefix_params(handle, prefix_k, capacity);
  }

  void set_thread_override(std::size_t threads) {
    inf_tok_set_max_threads(handle, threads);
    env_threads = threads;
  }
  void begin(const std::string& text, std::size_t threads) {
    const std::size_t resolved = resolve_threads(threads);
    check(inf_tok_encode_stream_begin(handle, text.c_str(), resolved), "encode_stream_begin");
  }

  bool next(Batch* batch) {
    inf_tok_batch_t c_batch{};
    const int rc = inf_tok_encode_stream_next(handle, &c_batch);
    if (rc < 0) {
      check(rc, "encode_stream_next");
    }
    if (rc == 0 || c_batch.len == 0) {
      return false;
    }
    batch->ids = c_batch.ids;
    batch->len = c_batch.len;
    return true;
  }

  void end() {
    inf_tok_encode_stream_end(handle);
  }

  inf_tok_t* handle{nullptr};
  std::size_t env_threads{0};
};

Tokenizer::Tokenizer(const std::string& model_path)
    : impl_(std::make_unique<Impl>(model_path)) {}

Tokenizer::~Tokenizer() = default;

void Tokenizer::enable_prefix_cache(bool enable) {
  impl_->enable_cache(enable);
}

void Tokenizer::set_prefix_params(std::size_t prefix_k, std::size_t capacity) {
  impl_->set_prefix_params(prefix_k, capacity);
}

void Tokenizer::set_thread_override(std::size_t threads) {
  impl_->set_thread_override(threads);
}

void Tokenizer::encode_stream_begin(const std::string& text, std::size_t n_threads) {
  impl_->begin(text, n_threads);
}

bool Tokenizer::encode_stream_next(Batch* out_batch) {
  return impl_->next(out_batch);
}

void Tokenizer::encode_stream_end() {
  impl_->end();
}

}  // namespace infeng::tokenizer
