#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "core/tensor.hpp"

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
    auto view_impl =
        std::make_shared<TensorImpl>(a_impl->storage(), 0, Shape({2, 3}), Stride({2, 1}), a.dtype(), a.device());
    Tensor view(view_impl);

    // Check that has_internal_overlap() correctly catches this!
    EXPECT_TRUE(view.has_internal_overlap()) << "has_internal_overlap() failed to detect strided overlap!";

    // Create another tensor to add
    Tensor b = Tensor::ones(Shape({2, 3}));

    // Perform in-place addition, expect throw because of overlap
    EXPECT_THROW({ view.add_(b); }, std::runtime_error);
}
