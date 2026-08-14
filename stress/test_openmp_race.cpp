#include <memory>
#include <optional>
#include <vector>
#include <thread>
#include <gtest/gtest.h>

#define private public
#include "core/tensor.hpp"
#undef private

using namespace helix;

TEST(OpenMPRaceTest, OverlappingStridesAddInplace) {
    // Create a 1D tensor of size 5
    Tensor a = Tensor::zeros(Shape({5}));
    
    // Create a view with shape (2, 3) and strides (2, 1)
    // Memory mapping:
    // (0,0) -> 0
    // (0,1) -> 1
    // (0,2) -> 2
    // (1,0) -> 2  <- Overlap at flat index 2!
    // (1,1) -> 3
    // (1,2) -> 4
    
    auto a_impl = a.impl();
    auto view_impl = std::make_shared<TensorImpl>(
        a_impl->storage(),
        0,
        Shape({2, 3}),
        Stride({2, 1}),
        a.dtype(),
        a.device()
    );
    Tensor view(view_impl);
    
    // Check that has_internal_overlap() misses this!
    EXPECT_FALSE(view.has_internal_overlap()) << "BUG: has_internal_overlap() failed to detect strided overlap!";
    
    // Create another tensor to add
    Tensor b = Tensor::ones(Shape({2, 3}));
    
    // Perform in-place addition
    // The elements at index 2 will be added twice!
    // But since it's OpenMP, it's a data race if done concurrently.
    // At minimum, it will sum to 2.
    view.add_(b);
    
    // Expect the value at index 2 to be 2.0
    EXPECT_EQ(a.data_ptr()[2], 2.0f) << "Value should be 2.0 due to overlap, or might be random due to data race!";
}
