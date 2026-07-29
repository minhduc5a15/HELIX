#pragma once
#include <cstddef>
#include <string>

namespace helix {

    class AutoTuner {
    public:
        static AutoTuner& get_instance();

        // Get the FLOPs threshold (M * N * K) where OpenMP becomes faster than AVX2.
        // If not calibrated, it will perform lazy evaluation.
        size_t get_omp_threshold();

        // Explicitly perform hardware profiling.
        void calibrate();

        // Reset state (For Unit Testing only)
        void reset_for_testing();

    private:
        AutoTuner();  // Singleton

        bool is_calibrated_;
        size_t omp_threshold_;  // Unit: FLOPs (total compute volume M*N*K)

        // Cache file path.
        std::string get_cache_filepath() const;

        // Load and save from cache
        bool load_from_cache();
        void save_to_cache();
    };

    // Public API for explicit initialization (useful in Production)
    void init_autotuner();

}  // namespace helix
