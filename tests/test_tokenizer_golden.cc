#include "infeng/tokenizer/tokenizer.h"
#include "tests/tokenizer_golden_data.h"

#include <cassert>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> read_prompts(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open prompts file: " + path);
  }
  std::vector<std::string> prompts;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      prompts.push_back(line);
    }
  }
  return prompts;
}

}  // namespace

int main() {
  const std::string model_path = "tests/data/test_tokenizer.json";
  const std::string prompts_path = "tests/data/tokenizer_prompts.txt";
  auto prompts = read_prompts(prompts_path);
  assert(prompts.size() == infeng::tests::kTokenizerGolden.size());

  infeng::tokenizer::Tokenizer tokenizer(model_path);
  tokenizer.enable_prefix_cache(true);
  tokenizer.set_prefix_params(64, 8);
  tokenizer.set_thread_override(0);

  for (std::size_t i = 0; i < prompts.size(); ++i) {
    const auto& prompt = prompts[i];
    tokenizer.encode_stream_begin(prompt, 0);
    std::vector<int32_t> tokens;
    infeng::tokenizer::Batch batch{};
    while (tokenizer.encode_stream_next(&batch)) {
      tokens.insert(tokens.end(), batch.ids, batch.ids + batch.len);
    }
    tokenizer.encode_stream_end();
    const auto& golden = infeng::tests::kTokenizerGolden[i];
    assert(tokens == golden);
  }

  bool threw = false;
  try {
    std::string invalid("\xff\xfe", 2);
    tokenizer.encode_stream_begin(invalid, 0);
  } catch (const std::exception&) {
    threw = true;
  }
  assert(threw);
  tokenizer.encode_stream_end();

  tokenizer.encode_stream_begin("hello cpu world", 0);
  infeng::tokenizer::Batch batch{};
  bool has = tokenizer.encode_stream_next(&batch);
  assert(has);
  auto first_ptr = batch.ids;
  auto first_len = batch.len;
  bool done = tokenizer.encode_stream_next(&batch);
  assert(!done);
  assert(first_ptr != nullptr);
  assert(first_len == 3);
  tokenizer.encode_stream_end();

  std::string large(1024 * 1024, 'a');
  tokenizer.encode_stream_begin(large, 0);
  tokenizer.encode_stream_end();

  tokenizer.encode_stream_begin(prompts[0], 0);
  tokenizer.encode_stream_end();
  tokenizer.encode_stream_begin(prompts[1], 0);
  tokenizer.encode_stream_end();

  return 0;
}
