#include "infeng/mem/kv_paged.h"

#include "infeng/mem/arena.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace infeng::mem {

namespace {
#if defined(__linux__) && __has_include(<numa.h>)
#define INFENG_HAS_LIBNUMA_LOCAL 1
#include <numa.h>
#else
#define INFENG_HAS_LIBNUMA_LOCAL 0
#endif

inline std::size_t detect_numa_nodes() {
#if INFENG_HAS_LIBNUMA_LOCAL
  if (numa_available() == -1) {
    return 1;
  }
  return static_cast<std::size_t>(numa_max_node() + 1);
#else
  return 1;
#endif
}

inline std::size_t clamp_node(std::size_t requested, std::size_t total) {
  if (requested >= total) {
    return total - 1;
  }
  return requested;
}

}  // namespace

struct KvPagedAllocator::Bucket {
  std::unordered_map<std::uint32_t, std::vector<void*>> bins;
};

struct KvPagedAllocator::PageInfo {
  void* ptr{nullptr};
  std::uint32_t len{0};
  std::size_t node{0};
  std::uint32_t pages{0};
};

KvPagedAllocator::KvPagedAllocator(KVPageSpec spec, std::size_t pool_pages)
    : spec_(spec), pool_pages_(pool_pages), buckets_(detect_numa_nodes()),
      arena_(std::make_unique<Arena>()) {
  if (spec_.page_size == 0 || (spec_.page_size % spec_.align) != 0) {
    throw std::invalid_argument("Invalid page spec");
  }
  const std::size_t node = clamp_node(spec_.numa_node, buckets_.size());
  for (std::size_t i = 0; i < pool_pages_; ++i) {
    void* ptr = arena_->allocate(spec_.page_size, spec_.align, static_cast<int>(node));
    buckets_[node].bins[1].push_back(ptr);
  }
}

KvPagedAllocator::~KvPagedAllocator() = default;

KVHandle KvPagedAllocator::alloc(std::uint32_t len) {
  if (len == 0) {
    throw std::invalid_argument("len must be > 0");
  }
  if (len % spec_.page_size != 0) {
    throw std::invalid_argument("len must be multiple of page_size");
  }
  const std::uint32_t pages = len / spec_.page_size;
  const std::size_t node = clamp_node(spec_.numa_node, buckets_.size());

  void* ptr = nullptr;
  {
    std::lock_guard lock(mutex_);
    auto& bin = buckets_[node].bins[pages];
    if (!bin.empty()) {
      ptr = bin.back();
      bin.pop_back();
    }
  }

  if (!ptr) {
    ptr = arena_->allocate(len, spec_.align, static_cast<int>(node));
  }

  const std::uint32_t id = next_page_id_.fetch_add(1, std::memory_order_relaxed);
  {
    std::lock_guard lock(mutex_);
    page_table_.emplace(id, PageInfo{ptr, len, node, pages});
  }
  return KVHandle{ptr, id, len};
}

void KvPagedAllocator::free(KVHandle handle) {
  if (handle.ptr == nullptr || handle.page_id == 0) {
    return;
  }
  PageInfo info;
  {
    std::lock_guard lock(mutex_);
    auto it = page_table_.find(handle.page_id);
    if (it == page_table_.end()) {
      throw std::runtime_error("invalid KVHandle");
    }
    info = it->second;
    page_table_.erase(it);
    buckets_[info.node].bins[info.pages].push_back(info.ptr);
  }
}

void* KvPagedAllocator::map(const KVHandle& handle, std::uint32_t offset) {
  if (offset >= handle.len) {
    throw std::out_of_range("offset out of range");
  }
  return static_cast<void*>(static_cast<std::uint8_t*>(handle.ptr) + offset);
}

std::size_t KvPagedAllocator::free_bytes() const {
  std::lock_guard lock(mutex_);
  std::size_t bytes = 0;
  for (const auto& bucket : buckets_) {
    for (const auto& entry : bucket.bins) {
      bytes += static_cast<std::size_t>(entry.first) * spec_.page_size * entry.second.size();
    }
  }
  return bytes;
}

}  // namespace infeng::mem
