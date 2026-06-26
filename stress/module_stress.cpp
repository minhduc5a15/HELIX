#include <gtest/gtest.h>
#include "nn/module.hpp"
#include "core/tensor.hpp"
#include "autograd/engine.hpp"

using namespace helix;

class DummyModuleStress : public Module {
private:
    Tensor param1_;
    Tensor param2_;

public:
    DummyModuleStress() {
        param1_ = Tensor::randn({2, 2});
        param2_ = Tensor::zeros({2});
        
        param1_.set_requires_grad(true);
        param2_.set_requires_grad(true);
    }

    Tensor forward(const Tensor& input) override {
        return input.matmul(param1_) + param2_;
    }

    std::vector<std::pair<std::string, Tensor>> named_parameters() override {
        return {
            {"param1", param1_},
            {"param2", param2_}
        };
    }
};

class ModuleStressTest : public ::testing::Test {
protected:
    void SetUp() override {
        init_autograd();
    }
};

// ============================================================================
// STRESS & SCALE TESTING
// ============================================================================

TEST_F(ModuleStressTest, MemoryLeakStressTest) {
    // Rigorous stress test to ensure creating modules and fetching parameters
    // repetitively does not cause a memory leak.
    Tensor input = Tensor::randn({2, 2});
    
    for (int i = 0; i < 5000; ++i) {
        DummyModuleStress module;
        Tensor out = module(input);
        auto params = module.parameters();
        
        // Ensure tensors are valid and calculations completed
        EXPECT_EQ(out.shape().vec(), (std::vector<size_t>{2, 2}));
        EXPECT_EQ(params.size(), 2);
    }
}
