#include "core/autotuner.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

using namespace helix;

class AutoTunerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Delete the cache file if it exists before each test to simulate a cold start
        if (std::filesystem::exists(".helix_autotune")) {
            std::filesystem::remove(".helix_autotune");
        }
        AutoTuner::get_instance().reset_for_testing();
    }

    void TearDown() override {
        // Cleanup after test
        if (std::filesystem::exists(".helix_autotune")) {
            std::filesystem::remove(".helix_autotune");
        }
        AutoTuner::get_instance().reset_for_testing();
    }
};

TEST_F(AutoTunerTest, LazyEvaluationCreatesCache) {
    // Initially there is no cache file
    EXPECT_FALSE(std::filesystem::exists(".helix_autotune"));

    // Call for the first time, the system will self-calibrate (Lazy Eval)
    size_t threshold = AutoTuner::get_instance().get_omp_threshold();

    // Threshold must be a large positive number
    EXPECT_GT(threshold, 0ULL);

    // Cache file must be created
    EXPECT_TRUE(std::filesystem::exists(".helix_autotune"));
}

TEST_F(AutoTunerTest, ExplicitInitCreatesCache) {
    EXPECT_FALSE(std::filesystem::exists(".helix_autotune"));

    // Explicit Init
    init_autotuner();

    EXPECT_TRUE(std::filesystem::exists(".helix_autotune"));

    size_t threshold = AutoTuner::get_instance().get_omp_threshold();
    EXPECT_GT(threshold, 0ULL);
}

TEST_F(AutoTunerTest, LoadFromCache) {
    // Create a fake cache file with a specific threshold value
    std::ofstream file(".helix_autotune");
    size_t fake_threshold = 999888777ULL;
    file << fake_threshold;
    file.close();

    // Reset just to be sure
    AutoTuner::get_instance().reset_for_testing();

    // Should load from cache instead of profiling
    size_t threshold = AutoTuner::get_instance().get_omp_threshold();
    EXPECT_EQ(threshold, fake_threshold);
}
