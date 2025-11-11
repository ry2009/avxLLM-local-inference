#pragma once

#include <string>

namespace infeng::adapters {

struct AdapterDescriptor {
  std::string name;
  std::string path;
  int rank{0};
};

bool load_adapter(const AdapterDescriptor& descriptor);
void unload_adapter(const std::string& name);

}  // namespace infeng::adapters
