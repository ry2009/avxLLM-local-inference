#pragma once

#include <cstdint>
#include <vector>

namespace infeng::tests {
inline const std::vector<std::vector<int32_t>> kTokenizerGolden = {
    {1, 2, 3},
    {4, 5},
    {6, 7},
    {8, 9},
    {10, 0, 0, 0, 0},
    {17},
};
}  // namespace infeng::tests
