#pragma once

#include <vector>

namespace infeng::runtime {

void set_thread_affinity(const std::vector<int>& cores);

}  // namespace infeng::runtime
