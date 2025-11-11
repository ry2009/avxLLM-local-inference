#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace infeng::mem {

class Arena;

struct KVHandle {
  void* ptr{nullptr};
  std::uint32_t page_id{0};
  std::uint32_t len{0};
};

struct KVPageSpec {
  std::uint32_t page_size{4096};
  std::uint32_t align{4096};
  std::uint32_t numa_node{0};
};

class KvPagedAllocator {
 public:
  KvPagedAllocator(KVPageSpec spec, std::size_t pool_pages);
  ~KvPagedAllocator();

  KVHandle alloc(std::uint32_t len);
  void free(KVHandle handle);
  void* map(const KVHandle& handle, std::uint32_t offset);
  std::size_t free_bytes() const;

 private:
  struct Bucket;
  struct PageInfo;

  KVPageSpec spec_;
  std::size_t pool_pages_;

  std::vector<Bucket> buckets_;
  std::unique_ptr<Arena> arena_;

  mutable std::mutex mutex_;
  std::unordered_map<std::uint32_t, PageInfo> page_table_;
  std::atomic<std::uint32_t> next_page_id_{1};
};

}  // namespace infeng::mem
