#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Helpers shared by the native dense kernels (QR with column pivoting, SVD,
 * nonsymmetric eigensolver): argument checks, scaled 2-norm, dot product,
 * and the right-hand-side staging. None of them calls a vendor LAPACK.
 *
 * @ingroup group_numerics
 */

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <solvcon/buffer/buffer.hpp>

namespace solvcon
{

namespace detail
{

template <typename T>
void dense_check_matrix(SimpleArray<T> const & a, char const * who)
{
    if (a.ndim() != 2)
    {
        throw std::invalid_argument(std::format(
            "{}: a must be a 2D SimpleArray, but got shape {}",
            who,
            detail::format_shape(a.shape())));
    }
    for (ssize_t it = 0; it < a.shape(0); ++it)
    {
        for (ssize_t jt = 0; jt < a.shape(1); ++jt)
        {
            if (!std::isfinite(a(it, jt)))
            {
                throw std::invalid_argument(std::format(
                    "{}: a contains a non-finite entry at ({}, {})", who, it, jt));
            }
        }
    }
}

/// Default relative rank tolerance max(m, n) * eps (the NumPy convention).
template <typename T>
T dense_default_rtol(ssize_t m, ssize_t n)
{
    return static_cast<T>(std::max<ssize_t>({m, n, 1})) * std::numeric_limits<T>::epsilon();
}

template <typename T>
T dense_dot(T const * x, T const * y, ssize_t count)
{
    T sum = 0;
    for (ssize_t it = 0; it < count; ++it)
    {
        sum += x[it] * y[it];
    }
    return sum;
}

/// 2-norm scaled by the largest magnitude, so it neither overflows nor
/// underflows for representable results.
template <typename T>
T dense_norm2(T const * x, ssize_t count)
{
    T scale = 0;
    for (ssize_t it = 0; it < count; ++it)
    {
        scale = std::max(scale, std::abs(x[it]));
    }
    if (scale == T(0))
    {
        return T(0);
    }
    T sum = 0;
    for (ssize_t it = 0; it < count; ++it)
    {
        T const ratio = x[it] / scale;
        sum += ratio * ratio;
    }
    return scale * std::sqrt(sum);
}

/**
 * Copy a 1D (length rows) or 2D (rows x p) right-hand side into a new
 * contiguous rows x p array.
 */
template <typename T>
SimpleArray<T> dense_stage_rhs(SimpleArray<T> const & b, ssize_t rows, char const * who)
{
    if (b.ndim() != 1 && b.ndim() != 2)
    {
        throw std::invalid_argument(std::format("{}: b must be 1D or 2D, but got {}D", who, b.ndim()));
    }
    if (b.shape(0) != rows)
    {
        throw std::invalid_argument(std::format(
            "{}: b must have {} rows, but got shape {}", who, rows, detail::format_shape(b.shape())));
    }
    ssize_t const ncol = (b.ndim() == 1) ? 1 : b.shape(1);
    SimpleArray<T> out(small_vector<ssize_t>{rows, ncol});
    for (ssize_t it = 0; it < rows; ++it)
    {
        for (ssize_t jt = 0; jt < ncol; ++jt)
        {
            out(it, jt) = (b.ndim() == 1) ? b(it) : b(it, jt);
        }
    }
    return out;
}

/// Return a rows x p result in the shape of the right-hand side b it came
/// from (1D when b was 1D).
template <typename T>
SimpleArray<T> dense_unstage_rhs(SimpleArray<T> const & out, SimpleArray<T> const & b)
{
    if (b.ndim() == 1)
    {
        return out.reshape(small_vector<ssize_t>{out.shape(0)});
    }
    return out;
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
