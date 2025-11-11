#include "infeng/mem/arena.h"

#include <cstdlib>
#include <new>
#include <stdexcept>
#include <utility>

#if defined(__linux__) && __has_include(<numa.h>)
#define INFENG_HAS_LIBNUMA 1
#include <numa.h>
#else
#define INFENG_HAS_LIBNUMA 0
#endif

#if defined(__unix__)
#include <sys/mman.h>
#endif

namespace infeng::mem {

Arena::Arena() = default;

Arena::~Arena() {
  release_all();
}

void* Arena::allocate(std::size_t bytes, std::size_t alignment, int numa_node) {
  if (bytes == 0) {
    throw std::invalid_argument("Arena::allocate bytes == 0");
  }
  void* ptr = nullptr;
#if INFENG_HAS_LIBNUMA
  if (numa_node >= 0 && numa_available() == 0) {
    ptr = numa_alloc_onnode(bytes, numa_node);
  }
#endif
  if (!ptr) {
    if (posix_memalign(&ptr, alignment, bytes) != 0) {
      throw std::bad_alloc();
    }
  }
#ifdef INFENG_MLOCK
  mlock(ptr, bytes);
#endif
  blocks_.push_back(Block{ptr, bytes, numa_node});
  return ptr;
}

void Arena::release_all() {
  for (auto& block : blocks_) {
#if INFENG_HAS_LIBNUMA
    if (block.numa_node >= 0 && numa_available() == 0) {
      numa_free(block.ptr, block.bytes);
      continue;
    }
#endif
    std::free(block.ptr);
  }
  blocks_.clear();
}

}  // namespace infeng::mem
