#include <gtest/gtest.h>
#include "nn/module.hpp"
#include "core/tensor.hpp"

using namespace helix;

class DummyModule : public Module {
private:
    Tensor param1_;
    Tensor param2_;

public:
    DummyModule() {
        param1_ = Tensor::randn({2, 2});
        param2_ = Tensor::zeros({2});
        
        param1_.set_requires_grad(true);
        param2_.set_requires_grad(true);
    }

    Tensor forward(const Tensor& input) override {
        // Simple dummy forward: input * param1 + param2
        return input.matmul(param1_) + param2_;
    }

    std::vector<std::pair<std::string, Tensor>> named_parameters() override {
        return {
            {"param1", param1_},
            {"param2", param2_}
        };
    }
};

#include "autograd/engine.hpp"

class ModuleTest : public ::testing::Test {
protected:
    void SetUp() override {
        init_autograd();
    }
};

TEST_F(ModuleTest, OperatorCallInvokesForward) {
    DummyModule module;
    Tensor input = Tensor::randn({2, 2});
    
    // Test operator()
    Tensor out_operator = module(input);
    
    // Test forward()
    Tensor out_forward = module.forward(input);
    
    // The outputs should match perfectly
    for (size_t i = 0; i < out_operator.shape().numel(); ++i) {
        EXPECT_FLOAT_EQ(out_operator.data_ptr()[i], out_forward.data_ptr()[i]);
    }
}

TEST_F(ModuleTest, NamedParametersCollection) {
    DummyModule module;
    
    auto named_params = module.named_parameters();
    EXPECT_EQ(named_params.size(), 2);
    
    EXPECT_EQ(named_params[0].first, "param1");
    EXPECT_EQ(named_params[0].second.shape().vec(), (std::vector<size_t>{2, 2}));
    EXPECT_TRUE(named_params[0].second.requires_grad());
    
    EXPECT_EQ(named_params[1].first, "param2");
    EXPECT_EQ(named_params[1].second.shape().vec(), (std::vector<size_t>{2}));
    EXPECT_TRUE(named_params[1].second.requires_grad());
}

TEST_F(ModuleTest, ParametersVectorCollection) {
    DummyModule module;
    
    auto params = module.parameters();
    EXPECT_EQ(params.size(), 2);
    
    EXPECT_EQ(params[0].shape().vec(), (std::vector<size_t>{2, 2}));
    EXPECT_TRUE(params[0].requires_grad());
    
    EXPECT_EQ(params[1].shape().vec(), (std::vector<size_t>{2}));
    EXPECT_TRUE(params[1].requires_grad());
}

// ============================================================================
// EDGE CASES & ERROR HANDLING
// ============================================================================

class EmptyModule : public Module {
public:
    Tensor forward(const Tensor& input) override {
        return input;
    }
    // Does not override named_parameters(), defaults to returning {}
};

TEST_F(ModuleTest, EmptyModuleParameters) {
    EmptyModule empty_mod;
    EXPECT_EQ(empty_mod.named_parameters().size(), 0);
    EXPECT_EQ(empty_mod.parameters().size(), 0);
}

// ============================================================================
// RESOURCE ALLOCATION & MEMORY SAFETY
// ============================================================================

TEST_F(ModuleTest, ShallowCopyParameters) {
    // According to Testing Principles: "Modifying an element inside a view or a 
    // shallow-copied tensor must reflect instantly inside the original tensor"
    DummyModule module;
    
    auto params = module.parameters();
    ASSERT_EQ(params.size(), 2);
    
    // params[1] is param2_ which was initialized to zeros({2})
    EXPECT_FLOAT_EQ(params[1].data_ptr()[0], 0.0f);
    
    // Mutate the returned parameter tensor in-place
    Tensor new_data = Tensor::ones({2});
    params[1].copy_(new_data);
    
    // Verify the change is reflected in the module's parameter
    auto updated_named_params = module.named_parameters();
    EXPECT_FLOAT_EQ(updated_named_params[1].second.data_ptr()[0], 1.0f);
    EXPECT_FLOAT_EQ(updated_named_params[1].second.data_ptr()[1], 1.0f);
}

TEST_F(ModuleTest, RequiresGradMutation) {
    // Verify that modifying the requires_grad state of a fetched parameter
    // modifies the actual module parameter's state.
    DummyModule module;
    
    auto params = module.parameters();
    ASSERT_TRUE(params[0].requires_grad());
    
    // Disable requires_grad on the fetched tensor
    params[0].set_requires_grad(false);
    
    // Fetch again and verify
    auto updated_params = module.parameters();
    EXPECT_FALSE(updated_params[0].requires_grad());
}

TEST_F(ModuleTest, ModuleParameters) {
    // Explicitly verify Module::parameters() behaves correctly
    DummyModule module;
    auto params = module.parameters();
    EXPECT_EQ(params.size(), 2);
    EXPECT_EQ(params[0].shape().vec(), (std::vector<size_t>{2, 2}));
    EXPECT_EQ(params[1].shape().vec(), (std::vector<size_t>{2}));
}

TEST_F(ModuleTest, RequiresGrad) {
    // Explicitly verify requires_grad setup and query
    DummyModule module;
    auto params = module.parameters();
    for (const auto& param : params) {
        EXPECT_TRUE(param.requires_grad());
    }
}



