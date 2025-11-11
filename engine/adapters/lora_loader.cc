#include "infeng/adapters/lora_loader.h"

#include <unordered_set>

namespace infeng::adapters {

namespace {
std::unordered_set<std::string> g_loaded;
}  // namespace

bool load_adapter(const AdapterDescriptor& descriptor) {
  return g_loaded.insert(descriptor.name).second;
}

void unload_adapter(const std::string& name) {
  g_loaded.erase(name);
}

}  // namespace infeng::adapters
