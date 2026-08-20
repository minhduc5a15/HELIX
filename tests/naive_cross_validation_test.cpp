#include <gtest/gtest.h>

#include <random>

#include "core/tensor.hpp"

using namespace helix;

class NaiveCrossValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::random_device rd;
        gen = std::mt19937(rd());
    }

    std::mt19937 gen;

    Tensor create_random_tensor(const Shape& shape, DType dtype) {
        Tensor t(shape, dtype);
        if (dtype == DType::Float32) {
            std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
            float* ptr = t.data_ptr<float>();
            for (size_t i = 0; i < t.numel(); ++i) {
                ptr[i] = dist(gen);
            }
        } else if (dtype == DType::Int32) {
            std::uniform_int_distribution<int32_t> dist(-100, 100);
            int32_t* ptr = t.data_ptr<int32_t>();
            for (size_t i = 0; i < t.numel(); ++i) {
                ptr[i] = dist(gen);
            }
        }
        return t;
    }

    template <typename T>
    void compute_naive_add(const T* a, const T* b, T* c, size_t numel) {
        for (size_t i = 0; i < numel; ++i) {
            c[i] = a[i] + b[i];
        }
    }

    template <typename T>
    void compute_naive_mul(const T* a, const T* b, T* c, size_t numel) {
        for (size_t i = 0; i < numel; ++i) {
            c[i] = a[i] * b[i];
        }
    }
};

TEST_F(NaiveCrossValidationTest, Float32_Add) {
    for (int i = 0; i < 100; ++i) {
        Shape s({128, 128});
        Tensor a = create_random_tensor(s, DType::Float32);
        Tensor b = create_random_tensor(s, DType::Float32);

        // HELIX
        Tensor c = a + b;

        // Naive
        Tensor c_naive(s, DType::Float32);
        compute_naive_add(a.data_ptr<float>(), b.data_ptr<float>(), c_naive.data_ptr<float>(), s.numel());

        // Compare
        const float* c_ptr = c.data_ptr<float>();
        const float* cn_ptr = c_naive.data_ptr<float>();
        for (size_t j = 0; j < s.numel(); ++j) {
            EXPECT_FLOAT_EQ(c_ptr[j], cn_ptr[j]);
        }
    }
}

TEST_F(NaiveCrossValidationTest, Int32_Add) {
    for (int i = 0; i < 100; ++i) {
        Shape s({128, 128});
        Tensor a = create_random_tensor(s, DType::Int32);
        Tensor b = create_random_tensor(s, DType::Int32);

        // HELIX
        Tensor c = a + b;

        // Naive
        Tensor c_naive(s, DType::Int32);
        compute_naive_add(a.data_ptr<int32_t>(), b.data_ptr<int32_t>(), c_naive.data_ptr<int32_t>(), s.numel());

        // Compare
        const int32_t* c_ptr = c.data_ptr<int32_t>();
        const int32_t* cn_ptr = c_naive.data_ptr<int32_t>();
        for (size_t j = 0; j < s.numel(); ++j) {
            EXPECT_EQ(c_ptr[j], cn_ptr[j]);
        }
    }
}

TEST_F(NaiveCrossValidationTest, Int32_Mul) {
    for (int i = 0; i < 100; ++i) {
        Shape s({128, 128});
        Tensor a = create_random_tensor(s, DType::Int32);
        Tensor b = create_random_tensor(s, DType::Int32);

        // HELIX
        Tensor c = a * b;

        // Naive
        Tensor c_naive(s, DType::Int32);
        compute_naive_mul(a.data_ptr<int32_t>(), b.data_ptr<int32_t>(), c_naive.data_ptr<int32_t>(), s.numel());

        // Compare
        const int32_t* c_ptr = c.data_ptr<int32_t>();
        const int32_t* cn_ptr = c_naive.data_ptr<int32_t>();
        for (size_t j = 0; j < s.numel(); ++j) {
            EXPECT_EQ(c_ptr[j], cn_ptr[j]);
        }
    }
}
