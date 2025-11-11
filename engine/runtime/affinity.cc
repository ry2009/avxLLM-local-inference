#include "infeng/runtime/affinity.h"

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace infeng::runtime {

void set_thread_affinity(const std::vector<int>& cores) {
#if defined(__linux__)
  if (cores.empty()) {
    return;
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int core : cores) {
    if (core >= 0) {
      CPU_SET(core, &set);
    }
  }
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
  (void)cores;
#endif
}

}  // namespace infeng::runtime
