#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Householder QR factorization with column pivoting (rank revealing), the
 * algorithm of LAPACK geqp3 (unblocked, as dlaqp2), written natively so that
 * it needs no vendor LAPACK.
 *
 * @ingroup group_numerics
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

#include <solvcon/buffer/buffer.hpp>
#include <solvcon/linalg/dense_util.hpp>

namespace solvcon
{

/**
 * QR factorization with column pivoting A P = Q R of a real m-by-n matrix.
 *
 * Step i moves the remaining column of largest norm to position i and
 * annihilates it below the diagonal with a Householder reflector
 * H_i = I - tau_i v_i v_i^T, so that Q = H_0 H_1 ... H_{k-1} with
 * k = min(m, n). R is k-by-n upper trapezoidal with |R(i, i)| non-increasing
 * (up to the sqrt(eps) accuracy of the downdated column norms, as in LAPACK).
 * Column j of A P is column perm()[j] of A.
 *
 * The reflectors are kept transposed (one row per column of A P), so every
 * kernel loop runs over contiguous memory in the row-major SimpleArray.
 *
 * Supported element types: float, double.
 *
 * @ingroup group_numerics
 */
template <typename T>
class QrPivotFactorization
{

    static_assert(std::is_floating_point_v<T>, "QrPivotFactorization<T> requires a real floating-point T");

public:

    using value_type = T;
    using array_type = SimpleArray<value_type>;
    using index_array_type = SimpleArray<int64_t>;

    /**
     * Factorize an m-by-n matrix.
     *
     * @throws std::invalid_argument if a is not 2D or has a non-finite entry.
     */
    explicit QrPivotFactorization(array_type const & a);

    QrPivotFactorization() = delete;
    QrPivotFactorization(QrPivotFactorization const &) = default;
    QrPivotFactorization(QrPivotFactorization &&) = default;
    QrPivotFactorization & operator=(QrPivotFactorization const &) = default;
    QrPivotFactorization & operator=(QrPivotFactorization &&) = default;
    ~QrPivotFactorization() = default;

    ssize_t m() const { return m_m; }
    ssize_t n() const { return m_n; }
    ssize_t k() const { return std::min(m_m, m_n); }

    /// Column permutation: column j of A P is column perm()[j] of A.
    index_array_type const & perm() const { return m_perm; }

    /// The Householder scalars tau_i (length k).
    array_type const & tau() const { return m_tau; }

    /// The k-by-n upper trapezoidal factor R.
    array_type r() const;

    /// The m-by-k thin factor Q with orthonormal columns.
    array_type q() const;

    /**
     * Apply the full m-by-m Q from the left: return Q b.
     *
     * @param b  1D (length m) or 2D (m-by-p) SimpleArray.
     * @return   Q b in the shape of b.
     */
    array_type apply_q(array_type const & b) const;

    /// Apply Q^T from the left: return Q^T b in the shape of b.
    array_type apply_qt(array_type const & b) const;

    /**
     * Numerical rank: the number of leading diagonal entries with
     * |R(i, i)| > rtol * |R(0, 0)|. A negative rtol selects max(m, n) * eps.
     */
    ssize_t rank(value_type rtol = value_type(-1)) const;

private:

    // Apply H_i to the rows ir..m-1 of the m-by-p contiguous block w.
    void apply_reflector(ssize_t ir, array_type & w) const;

    ssize_t m_m = 0;
    ssize_t m_n = 0;
    // n-by-m: row j holds column j of the factored A P; R(i, j) = m_qrt(j, i)
    // for i <= j, and v_i(r) = m_qrt(i, r) for r > i (v_i(i) = 1 implied).
    array_type m_qrt;
    array_type m_tau;
    index_array_type m_perm;

}; /* end class QrPivotFactorization */

template <typename T>
QrPivotFactorization<T>::QrPivotFactorization(array_type const & a)
{
    detail::dense_check_matrix(a, "QrPivotFactorization");
    m_m = a.shape(0);
    m_n = a.shape(1);
    ssize_t const m = m_m;
    ssize_t const n = m_n;
    ssize_t const kmin = std::min(m, n);

    m_qrt = array_type(small_vector<ssize_t>{n, m});
    m_tau = array_type(small_vector<ssize_t>{kmin}, T(0));
    m_perm = index_array_type(small_vector<ssize_t>{n});
    std::iota(m_perm.begin(), m_perm.end(), int64_t{0});
    for (ssize_t jt = 0; jt < n; ++jt)
    {
        for (ssize_t it = 0; it < m; ++it)
        {
            m_qrt(jt, it) = a(it, jt);
        }
    }

    // vn1 holds the (downdated) norms of the trailing parts of the columns,
    // vn2 the norms at the last exact computation (LAPACK dlaqp2).
    SimpleArray<T> vn1(small_vector<ssize_t>{n});
    SimpleArray<T> vn2(small_vector<ssize_t>{n});
    for (ssize_t jt = 0; jt < n; ++jt)
    {
        vn1[jt] = detail::dense_norm2(m_qrt.vptr(jt, 0), m);
        vn2[jt] = vn1[jt];
    }
    // LAPACK takes tol3z = sqrt(dlamch('E')), where dlamch('E') = eps / 2.
    T const tol3z = std::sqrt(std::numeric_limits<T>::epsilon() / T(2));

    for (ssize_t it = 0; it < kmin; ++it)
    {
        ssize_t pvt = it;
        for (ssize_t jt = it + 1; jt < n; ++jt)
        {
            if (vn1[jt] > vn1[pvt])
            {
                pvt = jt;
            }
        }
        if (pvt != it)
        {
            std::swap_ranges(m_qrt.vptr(pvt, 0), m_qrt.vptr(pvt, 0) + m, m_qrt.vptr(it, 0));
            std::swap(m_perm[pvt], m_perm[it]);
            vn1[pvt] = vn1[it];
            vn2[pvt] = vn2[it];
        }

        // Householder reflector (LAPACK dlarfg) for x = column it, rows it..m-1.
        T * const col = m_qrt.vptr(it, 0);
        T tau = 0;
        if (it < m - 1)
        {
            T const xnorm = detail::dense_norm2(col + it + 1, m - it - 1);
            if (xnorm != T(0))
            {
                T const alpha = col[it];
                T const beta = -std::copysign(std::hypot(alpha, xnorm), alpha);
                tau = (beta - alpha) / beta;
                T const scal = T(1) / (alpha - beta);
                for (ssize_t ir = it + 1; ir < m; ++ir)
                {
                    col[ir] *= scal;
                }
                col[it] = beta;
            }
        }
        m_tau[it] = tau;

        if (tau != T(0))
        {
            for (ssize_t jt = it + 1; jt < n; ++jt)
            {
                T * const other = m_qrt.vptr(jt, 0);
                T const proj = tau * (other[it] + detail::dense_dot(col + it + 1, other + it + 1, m - it - 1));
                other[it] -= proj;
                for (ssize_t ir = it + 1; ir < m; ++ir)
                {
                    other[ir] -= proj * col[ir];
                }
            }
        }

        for (ssize_t jt = it + 1; jt < n; ++jt)
        {
            if (vn1[jt] == T(0))
            {
                continue;
            }
            T const ratio = std::abs(m_qrt(jt, it)) / vn1[jt];
            T const temp = std::max(T(1) - ratio * ratio, T(0));
            T const growth = vn1[jt] / vn2[jt];
            if (temp * growth * growth <= tol3z)
            {
                // The downdate lost too much accuracy: recompute the norm.
                vn1[jt] = (it < m - 1) ? detail::dense_norm2(m_qrt.vptr(jt, 0) + it + 1, m - it - 1) : T(0);
                vn2[jt] = vn1[jt];
            }
            else
            {
                vn1[jt] *= std::sqrt(temp);
            }
        }
    }
}

template <typename T>
typename QrPivotFactorization<T>::array_type QrPivotFactorization<T>::r() const
{
    ssize_t const kmin = k();
    array_type out(small_vector<ssize_t>{kmin, m_n}, T(0));
    for (ssize_t it = 0; it < kmin; ++it)
    {
        for (ssize_t jt = it; jt < m_n; ++jt)
        {
            out(it, jt) = m_qrt(jt, it);
        }
    }
    return out;
}

template <typename T>
void QrPivotFactorization<T>::apply_reflector(ssize_t ir, array_type & w) const
{
    T const tau = m_tau[ir];
    if (tau == T(0))
    {
        return;
    }
    ssize_t const ncol = w.shape(1);
    T const * const vec = m_qrt.vptr(ir, 0);

    // proj = tau * v^T w over rows ir..m-1, accumulated row by row so the
    // loops stay on contiguous memory.
    SimpleArray<T> proj(small_vector<ssize_t>{ncol});
    T const * const head = w.vptr(ir, 0);
    for (ssize_t jt = 0; jt < ncol; ++jt)
    {
        proj[jt] = head[jt];
    }
    for (ssize_t it = ir + 1; it < m_m; ++it)
    {
        T const * const row = w.vptr(it, 0);
        for (ssize_t jt = 0; jt < ncol; ++jt)
        {
            proj[jt] += vec[it] * row[jt];
        }
    }
    for (ssize_t jt = 0; jt < ncol; ++jt)
    {
        proj[jt] *= tau;
    }

    T * const top = w.vptr(ir, 0);
    for (ssize_t jt = 0; jt < ncol; ++jt)
    {
        top[jt] -= proj[jt];
    }
    for (ssize_t it = ir + 1; it < m_m; ++it)
    {
        T * const row = w.vptr(it, 0);
        for (ssize_t jt = 0; jt < ncol; ++jt)
        {
            row[jt] -= vec[it] * proj[jt];
        }
    }
}

template <typename T>
typename QrPivotFactorization<T>::array_type QrPivotFactorization<T>::apply_q(array_type const & b) const
{
    array_type w = detail::dense_stage_rhs(b, m_m, "QrPivotFactorization::apply_q");
    for (ssize_t ir = k() - 1; ir >= 0; --ir)
    {
        apply_reflector(ir, w);
    }
    return detail::dense_unstage_rhs(w, b);
}

template <typename T>
typename QrPivotFactorization<T>::array_type QrPivotFactorization<T>::apply_qt(array_type const & b) const
{
    array_type w = detail::dense_stage_rhs(b, m_m, "QrPivotFactorization::apply_qt");
    for (ssize_t ir = 0; ir < k(); ++ir)
    {
        apply_reflector(ir, w);
    }
    return detail::dense_unstage_rhs(w, b);
}

template <typename T>
typename QrPivotFactorization<T>::array_type QrPivotFactorization<T>::q() const
{
    ssize_t const kmin = k();
    array_type w(small_vector<ssize_t>{m_m, kmin}, T(0));
    for (ssize_t it = 0; it < kmin; ++it)
    {
        w(it, it) = T(1);
    }
    for (ssize_t ir = kmin - 1; ir >= 0; --ir)
    {
        apply_reflector(ir, w);
    }
    return w;
}

template <typename T>
ssize_t QrPivotFactorization<T>::rank(value_type rtol) const
{
    if (rtol < T(0))
    {
        rtol = detail::dense_default_rtol<T>(m_m, m_n);
    }
    ssize_t const kmin = k();
    if (kmin == 0)
    {
        return 0;
    }
    T const threshold = rtol * std::abs(m_qrt(0, 0));
    ssize_t count = 0;
    while (count < kmin && std::abs(m_qrt(count, count)) > threshold)
    {
        ++count;
    }
    return count;
}

/**
 * Free function wrapper for the QR factorization with column pivoting.
 *
 * @return (Q, R, perm): Q is m-by-k, R is k-by-n, and A[:, perm] = Q R.
 */
template <typename T>
std::tuple<SimpleArray<T>, SimpleArray<T>, SimpleArray<int64_t>> qr_pivot(SimpleArray<T> const & a)
{
    QrPivotFactorization<T> const f(a);
    return {f.q(), f.r(), f.perm()};
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
