#include <vector>

namespace infeng::kernels::ref {

std::vector<float> attention_ref(const std::vector<float>& q,
                                 const std::vector<float>& k,
                                 const std::vector<float>& v,
                                 int head_dim) {
  // Placeholder identity attention.
  return v;
}

}  // namespace infeng::kernels::ref
