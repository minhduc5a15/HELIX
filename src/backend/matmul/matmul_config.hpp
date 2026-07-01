#pragma once
#include <cstddef>
namespace helix {
    struct MatMulConfig {
        static constexpr size_t block_size = 256;
    };
}  // namespace helix
