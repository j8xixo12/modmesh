#include <modmesh/modmesh.hpp>
#include <modmesh/transform/transform.hpp>
#include <random>
#include <gtest/gtest.h>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

template <typename T>
class ParsevalTest : public ::testing::Test
{
protected:
    const size_t VN = 1024;

    modmesh::SimpleArray<modmesh::Complex<T>> signal{
        modmesh::small_vector<size_t>{VN}, modmesh::Complex<T>{0.0, 0.0}};
    modmesh::SimpleArray<modmesh::Complex<T>> out{
        modmesh::small_vector<size_t>{VN}, modmesh::Complex<T>{0.0, 0.0}};

    // Set up the test fixture: generate the signal once
    void SetUp() override
    {
        std::mt19937 rng{std::random_device{}()};
        std::uniform_real_distribution<T> val_dist{-1.0, 1.0};
        for (unsigned int i = 0; i < VN; ++i)
        {
            T val = val_dist(rng);
            signal[i] = modmesh::Complex<T>{val, 0.0};
        }
    }

    // Function to verify energy conservation (Parseval's theorem)
    void verify_parseval()
    {
        T psd_sig = 0.0, psd_out = 0.0;

        for (unsigned int i = 0; i < VN; ++i)
        {
            psd_sig += signal[i].norm();
            psd_out += out[i].norm() / VN;
        }

        // TODO: When FFT / DFT uses float, the accumulation error is around 1e-3,
        // but when using double, the accumulation error is around 1e-12.
        // We need to investigate the root cause to determine whether this is an issue
        // or if there is a way to optimize it.
        // Expect the total energy in the time and frequency domains to be equal
        EXPECT_NEAR(psd_sig, psd_out, (std::is_same<T, float>::value ? (T)1e-2 : (T)1e-10));
    }
};

template <typename T>
class DeltaFunctionTest : public ::testing::Test
{
protected:
    const size_t VN = 1024;

    std::mt19937 rng{std::random_device{}()};

    modmesh::SimpleArray<modmesh::Complex<T>> signal{
        modmesh::small_vector<size_t>{VN}, modmesh::Complex<T>{0.0, 0.0}};
    modmesh::SimpleArray<modmesh::Complex<T>> out{
        modmesh::small_vector<size_t>{VN}, modmesh::Complex<T>{0.0, 0.0}};

    void SetUp() override
    {
        signal[0] = modmesh::Complex<T>{1.0, 0.0};
    }

    void verify_delta_function()
    {
        T expected_mag = static_cast<T>(1.0);

        for (unsigned int i = 0; i < VN; ++i)
        {
            T mag = out[i].norm();
            EXPECT_NEAR(mag, expected_mag, (std::is_same<T, float>::value ? (T)1e-2 : (T)1e-10));
        }
    }
};

template <typename T>
class InverseTest : public ::testing::Test
{
protected:
    const size_t VN = 1024;

    std::mt19937 rng{std::random_device{}()};

    modmesh::SimpleArray<modmesh::Complex<T>> signal{
        modmesh::small_vector<size_t>{VN}, modmesh::Complex<T>{0.0, 0.0}};
    modmesh::SimpleArray<modmesh::Complex<T>> freq_domain{
        modmesh::small_vector<size_t>{VN}, modmesh::Complex<T>{0.0, 0.0}};
    modmesh::SimpleArray<modmesh::Complex<T>> time_domain{
        modmesh::small_vector<size_t>{VN}, modmesh::Complex<T>{0.0, 0.0}};

    void SetUp() override
    {
        std::mt19937 rng{std::random_device{}()};
        std::uniform_real_distribution<T> val_dist{-1.0, 1.0};
        for (unsigned int i = 0; i < VN; ++i)
        {
            T val = val_dist(rng);
            signal[i] = modmesh::Complex<T>{val, 0.0};
        }
    }

    void verify_inverse_fft_function()
    {
        for (unsigned int i = 0; i < VN; ++i)
        {
            EXPECT_NEAR(signal[i].real(), time_domain[i].real(), (std::is_same<T, float>::value ? (T)1e-2 : (T)1e-10));
            EXPECT_NEAR(signal[i].imag(), time_domain[i].imag(), (std::is_same<T, float>::value ? (T)1e-2 : (T)1e-10));
        }
    }
};

typedef ::testing::Types<float, double> TestTypes;
TYPED_TEST_SUITE(ParsevalTest, TestTypes);
TYPED_TEST_SUITE(DeltaFunctionTest, TestTypes);
TYPED_TEST_SUITE(InverseTest, TestTypes);

TYPED_TEST(ParsevalTest, fft)
{
    modmesh::Transform::fft<modmesh::Complex, TypeParam>(this->signal, this->out);

    this->verify_parseval();
}

TYPED_TEST(ParsevalTest, dft)
{
    modmesh::Transform::dft<modmesh::Complex, TypeParam>(this->signal, this->out);

    this->verify_parseval();
}

TYPED_TEST(DeltaFunctionTest, fft)
{
    modmesh::Transform::fft<modmesh::Complex, TypeParam>(this->signal, this->out);

    this->verify_delta_function();
}

TYPED_TEST(DeltaFunctionTest, dft)
{
    modmesh::Transform::dft<modmesh::Complex, TypeParam>(this->signal, this->out);

    this->verify_delta_function();
}

TYPED_TEST(InverseTest, fft)
{
    modmesh::Transform::fft<modmesh::Complex, TypeParam>(this->signal, this->freq_domain);
    modmesh::Transform::ifft<modmesh::Complex, TypeParam>(this->freq_domain, this->time_domain);

    this->verify_inverse_fft_function();
}

TYPED_TEST(InverseTest, dft)
{
    modmesh::Transform::dft<modmesh::Complex, TypeParam>(this->signal, this->freq_domain);
    modmesh::Transform::ifft<modmesh::Complex, TypeParam>(this->freq_domain, this->time_domain);

    this->verify_inverse_fft_function();
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
