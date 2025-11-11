#pragma once

#include <cstddef>
#include <vector>

namespace infeng::mem {

class Arena {
 public:
  Arena();
  ~Arena();

  void* allocate(std::size_t bytes, std::size_t alignment, int numa_node);

 private:
  struct Block {
    void* ptr;
    std::size_t bytes;
    int numa_node;
  };

  void release_all();

  std::vector<Block> blocks_;
};

}  // namespace infeng::mem
