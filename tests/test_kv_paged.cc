#include "infeng/mem/kv_paged.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace std::chrono;

int main() {
  infeng::mem::KVPageSpec spec{4096, 4096, 0};
  infeng::mem::KvPagedAllocator alloc(spec, 256);

  const std::uint32_t iterations = 10000;
  std::vector<infeng::mem::KVHandle> handles;
  handles.reserve(iterations);

  auto t_start = high_resolution_clock::now();
  for (std::uint32_t i = 0; i < iterations; ++i) {
    auto handle = alloc.alloc(spec.page_size);
    assert(handle.ptr != nullptr);
    handles.push_back(handle);
  }
  auto t_mid = high_resolution_clock::now();

  for (auto& handle : handles) {
    auto ptr = alloc.map(handle, 0);
    assert(ptr != nullptr);
  }

  for (auto& handle : handles) {
    alloc.free(handle);
  }
  auto t_end = high_resolution_clock::now();

  const double alloc_time = duration<double>(t_mid - t_start).count();
  const double total_time = duration<double>(t_end - t_start).count();
  const double bytes_moved = static_cast<double>(iterations) * spec.page_size;
  const double throughput = bytes_moved / total_time / 1e9;  // GB/s

  std::cout << "alloc_time_s=" << alloc_time << " total_time_s=" << total_time
            << " throughput_gbps=" << throughput << "\n";

  assert(alloc.free_bytes() >= static_cast<std::size_t>(spec.page_size) * iterations);

  // Random fuzz with 1-4 pages per allocation
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(1, 4);
  handles.clear();
  for (int i = 0; i < 2000; ++i) {
    int pages = dist(rng);
    auto handle = alloc.alloc(static_cast<std::uint32_t>(pages * spec.page_size));
    handles.push_back(handle);
  }
  for (auto& handle : handles) {
    alloc.free(handle);
  }

  std::cout << "free_bytes=" << alloc.free_bytes() << "\n";
  return 0;
}
