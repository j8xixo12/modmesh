#pragma once

#include <modmesh/math/math.hpp>
#include <modmesh/buffer/buffer.hpp>

namespace modmesh
{

namespace detail
{

size_t bit_reverse(size_t n, const size_t bits);

} /* end namespace detail */

struct Transform
{
public:
    Transform() = default;
    ~Transform() = default;
    Transform(const Transform & other) = delete;
    Transform(Transform && other) = delete;
    Transform& operator=(const Transform & other) = delete;
    Transform& operator=(Transform && other) = delete;

    // TODO: The template of template is too complicate, we should find a way to make it easier.
    template <template <typename> typename T1, typename T2>
    static void dft(SimpleArray<T1<T2>> const & in, SimpleArray<T1<T2>> & out)
    {
        size_t N = in.size();
        for (size_t i = 0; i < N; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                T2 tmp = -2.0 * pi<T2> * i * j / N;
                T1<T2> twiddle_factor{std::cos(tmp), std::sin(tmp)};
                out[i] += in[j] * twiddle_factor;
            }
        }
    }

    template <template <typename> typename T1, typename T2>
    static void fft(SimpleArray<T1<T2>> const & in, SimpleArray<T1<T2>> & out)
    {
        size_t N = in.size();
        const unsigned int bits = static_cast<unsigned int>(std::log2(N));

        // bit reversed reordering
        for (size_t i = 0; i < N; ++i)
        {
            out[detail::bit_reverse(i, bits)] = in[i];
        }

        // Cooly-Tukey FFT algorithm, radix-2
        for (size_t size = 2; size <= N; size *= 2)
        {
            size_t half_size = size / 2;
            T2 angle_inc = -2.0 * pi<T2> / size;

            for (size_t i = 0; i < N; i += size)
            {
                for (size_t k = 0; k < half_size; ++k)
                {
                    // Twiddle factor = exp(-2 * pi * i * k / N)
                    T2 angle = angle_inc * k;
                    T1<T2> twiddle_factor{std::cos(angle), std::sin(angle)};

                    T1<T2> even(out[i + k]);
                    T1<T2> odd(out[i + k + half_size] * twiddle_factor);

                    out[i + k] = even + odd;
                    out[i + k + half_size] = even - odd;
                }
            }
        }
    }

    template <template <typename> typename T1, typename T2>
    static void ifft(SimpleArray<T1<T2>> const & in, SimpleArray<T1<T2>> & out)
    {
        size_t N = in.size();
        SimpleArray<T1<T2>> in_conj(N);

        for (size_t i = 0; i < N; ++i)
        {
            in_conj[i] = in[i].conj();
        }

        fft<T1, T2>(in_conj, out);

        for (size_t i = 0; i < N; ++i)
        {
            out[i] = out[i].conj() / static_cast<T2>(N);
        }
    }
};

} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
