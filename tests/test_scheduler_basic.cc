#include "infeng/runtime/microbatch.h"
#include "infeng/runtime/queues.h"

#include <cassert>
#include <thread>

using namespace infeng::runtime;

int main() {
  MpmcQueue<int> queue;
  std::thread producer([&]() {
    for (int i = 0; i < 4; ++i) {
      queue.push(i);
    }
    queue.shutdown();
  });
  int sum = 0;
  while (true) {
    auto value = queue.pop();
    if (!value.has_value()) {
      break;
    }
    sum += *value;
  }
  producer.join();
  assert(sum == 0 + 1 + 2 + 3);

  MicrobatchBuilder builder(/*max_batch_size=*/2);
  builder.add_request(Request{"prompt0", "adapterA", {"adapterA"}});
  builder.add_request(Request{"prompt1", "adapterA", {"adapterA"}});
  builder.add_request(Request{"prompt2", "adapterB", {"adapterB"}});
  builder.add_request(Request{"prompt3", "", {}});
  builder.add_request(Request{"prompt4", "adapterA", {"adapterA"}});

  auto ready = builder.flush();
  assert(ready.size() == 3);
  bool saw_base = false;
  for (const auto& batch : ready) {
    if (batch.adapter_signature.empty()) {
      saw_base = true;
    }
  }
  assert(saw_base);
  return 0;
}
