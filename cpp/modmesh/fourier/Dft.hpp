#pragma once

#include <modmesh/onedim/core.hpp>
#include <modmesh/base.hpp>
#include <modmesh/math.hpp>
#include <modmesh/buffer/buffer.hpp>

namespace modmesh
{
namespace onedim 
{
constexpr double PI = 3.1415926;
template<typename T>
void dft(SimpleArray<T> const & in, SimpleArray<T> & out_r, SimpleArray<T> & out_i)
{
    size_t N = in.size();
    // exp(-i * 2 * PI * k * n / N) = cos(-2 * PI * k * n / N) - i * sin(-2 * PI * k * n / N)
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            double tmp = 2.0 * PI * i * j / N;
    // Real part
            out_r[i] += in[j] * cos(tmp);
    // Imaginary part
            out_i[i] -= in[j] * sin(tmp);
        }
    }
}
} /* end namespace onedim */
} /* end namespace modmesh */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
