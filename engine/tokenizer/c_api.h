#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define INF_TOK_ABI_VERSION 1

typedef struct inf_tok inf_tok_t;

typedef struct inf_tok_batch {
  const std::int32_t* ids;
  std::size_t len;
} inf_tok_batch_t;

inf_tok_t* inf_tok_new(int abi_version, const char* model_path);
void inf_tok_free(inf_tok_t* tokenizer);
void inf_tok_set_prefix_cache(inf_tok_t* tokenizer, int enable);
void inf_tok_set_prefix_params(inf_tok_t* tokenizer, std::size_t prefix_k, std::size_t capacity);
void inf_tok_set_max_threads(inf_tok_t* tokenizer, std::size_t threads);
int inf_tok_encode_stream_begin(inf_tok_t* tokenizer, const char* text, std::size_t n_threads);
int inf_tok_encode_stream_next(inf_tok_t* tokenizer, inf_tok_batch_t* out_batch);
void inf_tok_encode_stream_end(inf_tok_t* tokenizer);
const char* inf_tok_last_error(const inf_tok_t* tokenizer);

#ifdef __cplusplus
}
#endif
