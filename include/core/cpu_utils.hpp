#pragma once

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace helix {

    /**
     * @brief Cross-platform check for AVX2 and FMA support.
     * @return true if the CPU supports both AVX2 and FMA, false otherwise.
     */
    inline bool cpu_supports_avx2_fma() {
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#elif defined(_MSC_VER)
        int cpuInfo[4];
        // Check AVX2 (Leaf 7, Subleaf 0, EBX bit 5)
        __cpuidex(cpuInfo, 7, 0);
        bool avx2 = (cpuInfo[1] & (1 << 5)) != 0;
        
        // Check FMA (Leaf 1, ECX bit 12)
        __cpuid(cpuInfo, 1);
        bool fma = (cpuInfo[2] & (1 << 12)) != 0;
        
        return avx2 && fma;
#else
        return false;
#endif
    }

}  // namespace helix
