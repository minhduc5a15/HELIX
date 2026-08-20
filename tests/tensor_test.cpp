#include "core/tensor.hpp"

#include <gtest/gtest.h>

using namespace helix;

TEST(TensorTest, DefaultConstructor) {
    Tensor t;
    EXPECT_EQ(t.rank(), 0);
    EXPECT_EQ(t.numel(), 1);
    EXPECT_EQ(t.shape().empty(), true);
}

TEST(TensorTest, ConstructorWithShape) {
    Shape s({2, 3});
    Tensor t(s);
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
}

TEST(TensorTest, ConstructorWithData) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Shape s({2, 3});
    Tensor t(data, s);

    EXPECT_EQ(t.numel(), 6);
    EXPECT_FLOAT_EQ(t.item({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.item({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(t.item({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(t.item({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(t.item({1, 1}), 5.0f);
    EXPECT_FLOAT_EQ(t.item({1, 2}), 6.0f);
}

TEST(TensorTest, SetItem) {
    Shape s({2});
    Tensor t(s);

    t.set_item({0}, 42.0f);
    t.set_item({1}, 100.0f);

    EXPECT_FLOAT_EQ(t.item({0}), 42.0f);
    EXPECT_FLOAT_EQ(t.item({1}), 100.0f);
}

TEST(TensorTest, ShallowCopy) {
    std::vector<float> data = {1.0f, 2.0f};
    Shape s({2});
    Tensor t1(data, s);
    Tensor t2 = t1;  // Shallow copy

    EXPECT_EQ(t1.data_ptr(), t2.data_ptr());

    t2.set_item({0}, 99.0f);
    EXPECT_FLOAT_EQ(t1.item({0}), 99.0f);
}

TEST(TensorTest, ShapeIntegerOverflow) {
    // 274177 * 67280421310721 = 18446744073709551617
    // 18446744073709551617 % 2^64 = 1
    Shape s({274177, 67280421310721});

    // The framework now throws std::overflow_error when computing numel()
    EXPECT_THROW({ s.numel(); }, std::overflow_error);
}

TEST(TensorTest, DTypeAPI_ReadWrite) {
    Shape s({2, 3});
    Tensor t_int32(s, DType::Int32);

    int32_t* ptr32 = t_int32.data_ptr<int32_t>();
    for (size_t i = 0; i < t_int32.numel(); ++i) {
        ptr32[i] = static_cast<int32_t>(i + 10);
    }

    EXPECT_EQ(static_cast<int32_t>(t_int32.item({0, 0})), 10);
    EXPECT_EQ(static_cast<int32_t>(t_int32.item({1, 2})), 15);

    Tensor t_int64(s, DType::Int64);
    int64_t* ptr64 = t_int64.data_ptr<int64_t>();
    for (size_t i = 0; i < t_int64.numel(); ++i) {
        ptr64[i] = static_cast<int64_t>(i + 100);
    }

    EXPECT_EQ(static_cast<int64_t>(t_int64.item({0, 0})), 100);
    EXPECT_EQ(static_cast<int64_t>(t_int64.item({1, 2})), 105);
}

TEST(TensorTest, DTypeAPI_StorageValidation) {
    Shape s({10});
    Tensor t_int64(s, DType::Int64);

    int64_t* ptr = t_int64.data_ptr<int64_t>();
    for (int i = 0; i < 10; ++i) ptr[i] = i;

    Tensor slice = t_int64.slice(0, 2, 5);  // elements 2, 3, 4

    // Check if the sliced data_ptr accounts for storage_offset * dtype_size
    int64_t* slice_ptr = slice.data_ptr<int64_t>();
    EXPECT_EQ(slice_ptr[0], 2);
    EXPECT_EQ(slice_ptr[1], 3);
    EXPECT_EQ(slice_ptr[2], 4);

    // Explicit pointer arithmetic check
    void* raw_storage = slice.impl()->storage()->data();
    int64_t* expected_ptr =
        reinterpret_cast<int64_t*>(static_cast<char*>(raw_storage) + slice.impl()->storage_offset() * 8);
    EXPECT_EQ(slice_ptr, expected_ptr);
}

TEST(TensorTest, DTypeAPI_ExceptionOnWrongTemplate) {
    Shape s({2});
    Tensor t_int32(s, DType::Int32);
    Tensor t_float(s, DType::Float32);

    // Wrong type template should throw
    EXPECT_THROW(t_int32.data_ptr<float>(), std::invalid_argument);
    EXPECT_THROW(t_float.data_ptr<int32_t>(), std::invalid_argument);

    // But correct type should not throw
    EXPECT_NO_THROW(t_int32.data_ptr<int32_t>());
    EXPECT_NO_THROW(t_float.data_ptr<float>());
}
