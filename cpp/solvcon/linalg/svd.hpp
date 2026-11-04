#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Singular value decomposition by the one-sided Jacobi (Hestenes) method,
 * with the rank, pseudo-inverse, null space, and minimum-norm least squares
 * built on it. Written natively, so it needs no vendor LAPACK.
 *
 * @ingroup group_numerics
 */

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <solvcon/buffer/buffer.hpp>
#include <solvcon/linalg/dense_util.hpp>
#include <solvcon/linalg/qr_pivot.hpp>

namespace solvcon
{

/**
 * Thin singular value decomposition A = U diag(s) V^T of a real m-by-n
 * matrix, k = min(m, n): U is m-by-k, s has k non-negative entries in
 * descending order, V is n-by-k; U and V have orthonormal columns.
 *
 * The one-sided Jacobi method rotates pairs of columns of the working matrix
 * G (G = A for m >= n, G = A^T for m < n) until every pair is orthogonal to
 * the relative tolerance sqrt(len) * eps (as LAPACK gesvj), len being the
 * column length; the column norms are then the singular values. A column
 * whose norm falls to eps |A|_F or below is at the level of the backward
 * error and is not rotated further; its singular value is that norm and its
 * left singular vector comes from an orthonormal completion. Singular values
 * below eps |A|_F therefore have absolute, not relative, accuracy (as with
 * LAPACK gesdd). The columns of G are stored as rows, so the rotations run
 * over contiguous memory.
 *
 * Supported element types: float, double.
 *
 * @ingroup group_numerics
 */
template <typename T>
class SvdFactorization
{

    static_assert(std::is_floating_point_v<T>, "SvdFactorization<T> requires a real floating-point T");

public:

    using value_type = T;
    using array_type = SimpleArray<value_type>;

    /// The limit on Jacobi sweeps; quadratic convergence needs far fewer.
    static constexpr ssize_t MAX_SWEEPS = 100;

    /**
     * Factorize an m-by-n matrix.
     *
     * @throws std::invalid_argument if a is not 2D or has a non-finite entry.
     * @throws std::runtime_error if the Jacobi sweeps do not converge.
     */
    explicit SvdFactorization(array_type const & a);

    SvdFactorization() = delete;
    SvdFactorization(SvdFactorization const &) = default;
    SvdFactorization(SvdFactorization &&) = default;
    SvdFactorization & operator=(SvdFactorization const &) = default;
    SvdFactorization & operator=(SvdFactorization &&) = default;
    ~SvdFactorization() = default;

    ssize_t m() const { return m_m; }
    ssize_t n() const { return m_n; }
    ssize_t k() const { return std::min(m_m, m_n); }

    array_type const & u() const { return m_u; }
    array_type const & s() const { return m_s; }
    array_type const & v() const { return m_v; }

    /// Number of Jacobi sweeps run, including the last one that found every
    /// column pair orthogonal.
    ssize_t sweeps() const { return m_sweeps; }

    /**
     * Numerical rank: the number of s_i > rtol * s_0. A negative rtol selects
     * max(m, n) * eps.
     */
    ssize_t rank(value_type rtol = value_type(-1)) const;

    /**
     * Moore-Penrose pseudo-inverse (n-by-m), dropping s_i <= rcond * s_0. A
     * negative rcond selects max(m, n) * eps.
     */
    array_type pinv(value_type rcond = value_type(-1)) const;

    /**
     * Minimum-norm least-squares solution x of min |A x - b|, dropping
     * s_i <= rcond * s_0 (negative rcond: max(m, n) * eps).
     *
     * @param b  1D (length m) or 2D (m-by-p) SimpleArray.
     * @return   x, 1D (length n) or 2D (n-by-p) like b.
     */
    array_type solve(array_type const & b, value_type rcond = value_type(-1)) const;

    /**
     * Orthonormal basis of the null space (n-by-(n - rank)): the right
     * singular vectors of the singular values s_i <= rcond * s_0 and, for
     * m < n, the n - m right singular vectors beyond the thin V (the
     * orthogonal complement of its columns). Negative rcond: max(m, n) * eps.
     */
    array_type null_space(value_type rcond = value_type(-1)) const;

private:

    // Rotate the rows of g (ncol rows of length len) pairwise until they are
    // orthogonal; apply the same rotations to the rows of vt. A row of norm
    // at or below negligible is not rotated.
    void run_jacobi(array_type & g, array_type & vt, value_type negligible);

    // Return columns r..c-1 of an orthogonal matrix whose first r columns
    // span the columns of basis (len-by-r with orthonormal columns), as a
    // len-by-(c - r) array.
    static array_type complete_basis(array_type const & basis, ssize_t ncol);

    ssize_t m_m = 0;
    ssize_t m_n = 0;
    ssize_t m_sweeps = 0;
    array_type m_u;
    array_type m_s;
    array_type m_v;

}; /* end class SvdFactorization */

template <typename T>
void SvdFactorization<T>::run_jacobi(array_type & g, array_type & vt, value_type negligible)
{
    ssize_t const ncol = g.shape(0);
    ssize_t const len = g.shape(1);
    T const tol = std::sqrt(static_cast<T>(std::max<ssize_t>(len, 1))) * std::numeric_limits<T>::epsilon();

    for (m_sweeps = 1; m_sweeps <= MAX_SWEEPS; ++m_sweeps)
    {
        bool rotated = false;
        for (ssize_t ip = 0; ip < ncol - 1; ++ip)
        {
            for (ssize_t iq = ip + 1; iq < ncol; ++iq)
            {
                T * const gp = g.vptr(ip, 0);
                T * const gq = g.vptr(iq, 0);
                T const alpha = detail::dense_dot(gp, gp, len);
                T const beta = detail::dense_dot(gq, gq, len);
                T const gamma = detail::dense_dot(gp, gq, len);
                T const pnorm = std::sqrt(alpha);
                T const qnorm = std::sqrt(beta);
                // A column at the rounding level of A has no resolvable
                // direction: rotating it against a large column only shrinks
                // it (by about eps per sweep) and never makes it orthogonal.
                if (pnorm <= negligible || qnorm <= negligible)
                {
                    continue;
                }
                if (gamma == T(0) || std::abs(gamma) <= tol * pnorm * qnorm)
                {
                    continue;
                }
                rotated = true;

                // The rotation that zeroes the inner product of the pair
                // (Rutishauser's formulas, smaller-angle root).
                T const zeta = (beta - alpha) / (T(2) * gamma);
                T const tn = std::copysign(T(1), zeta) / (std::abs(zeta) + std::hypot(T(1), zeta));
                T const cs = T(1) / std::sqrt(T(1) + tn * tn);
                T const sn = cs * tn;
                for (ssize_t it = 0; it < len; ++it)
                {
                    T const xp = gp[it];
                    T const xq = gq[it];
                    gp[it] = cs * xp - sn * xq;
                    gq[it] = sn * xp + cs * xq;
                }
                T * const vp = vt.vptr(ip, 0);
                T * const vq = vt.vptr(iq, 0);
                for (ssize_t it = 0; it < ncol; ++it)
                {
                    T const xp = vp[it];
                    T const xq = vq[it];
                    vp[it] = cs * xp - sn * xq;
                    vq[it] = sn * xp + cs * xq;
                }
            }
        }
        if (!rotated)
        {
            return;
        }
    }
    throw std::runtime_error(std::format(
        "SvdFactorization: one-sided Jacobi did not converge in {} sweeps", MAX_SWEEPS));
}

template <typename T>
typename SvdFactorization<T>::array_type SvdFactorization<T>::complete_basis(array_type const & basis, ssize_t ncol)
{
    ssize_t const len = basis.shape(0);
    ssize_t const nbasis = basis.shape(1);
    array_type out(small_vector<ssize_t>{len, ncol - nbasis}, T(0));
    for (ssize_t it = nbasis; it < ncol; ++it)
    {
        out(it, it - nbasis) = T(1);
    }
    // The Householder Q of the basis has the basis span in its first nbasis
    // columns; the following columns are orthonormal and orthogonal to it.
    return QrPivotFactorization<T>(basis).apply_q(out);
}

template <typename T>
SvdFactorization<T>::SvdFactorization(array_type const & a)
{
    detail::dense_check_matrix(a, "SvdFactorization");
    m_m = a.shape(0);
    m_n = a.shape(1);
    bool const transposed = m_m < m_n;
    ssize_t const ncol = std::min(m_m, m_n);
    ssize_t const len = std::max(m_m, m_n);

    // Scale by a power of 2 (exact) to bring the largest magnitude into
    // [1, 2), so the dot products of the columns neither overflow nor
    // underflow.
    T amax = 0;
    for (ssize_t it = 0; it < m_m; ++it)
    {
        for (ssize_t jt = 0; jt < m_n; ++jt)
        {
            amax = std::max(amax, std::abs(a(it, jt)));
        }
    }
    int const expo = (amax > T(0)) ? std::ilogb(amax) : 0;

    // Row it of g is column it of G (G = A, or A^T when m < n).
    array_type g(small_vector<ssize_t>{ncol, len});
    for (ssize_t it = 0; it < ncol; ++it)
    {
        for (ssize_t jt = 0; jt < len; ++jt)
        {
            g(it, jt) = std::ldexp(transposed ? a(it, jt) : a(jt, it), -expo);
        }
    }
    // A column of norm at or below eps |A|_F is at the level of the backward
    // error: it is left unrotated and gets its left singular vector from the
    // orthonormal completion below.
    T const negligible = std::numeric_limits<T>::epsilon() * detail::dense_norm2(g.data(), ncol * len);
    array_type vt(small_vector<ssize_t>{ncol, ncol}, T(0));
    for (ssize_t it = 0; it < ncol; ++it)
    {
        vt(it, it) = T(1);
    }

    run_jacobi(g, vt, negligible);

    SimpleArray<T> norms(small_vector<ssize_t>{ncol});
    for (ssize_t it = 0; it < ncol; ++it)
    {
        norms[it] = detail::dense_norm2(g.vptr(it, 0), len);
    }
    small_vector<ssize_t> order(static_cast<size_t>(ncol));
    std::iota(order.begin(), order.end(), ssize_t{0});
    std::stable_sort(order.begin(), order.end(), [&norms](ssize_t lhs, ssize_t rhs)
                     { return norms[lhs] > norms[rhs]; });

    // left: len-by-ncol left singular vectors of G; right: ncol-by-ncol.
    m_s = array_type(small_vector<ssize_t>{ncol});
    array_type left(small_vector<ssize_t>{len, ncol}, T(0));
    array_type right(small_vector<ssize_t>{ncol, ncol});
    ssize_t nonzero = 0;
    for (ssize_t jt = 0; jt < ncol; ++jt)
    {
        ssize_t const src = order[jt];
        T const sval = norms[src];
        m_s[jt] = std::ldexp(sval, expo);
        if (sval > negligible)
        {
            ++nonzero;
            for (ssize_t it = 0; it < len; ++it)
            {
                left(it, jt) = g(src, it) / sval;
            }
        }
        for (ssize_t it = 0; it < ncol; ++it)
        {
            right(it, jt) = vt(src, it);
        }
    }

    // A zero or negligible singular value leaves its left singular vector
    // undefined; take an orthonormal completion of the others.
    if (nonzero < ncol)
    {
        array_type basis(small_vector<ssize_t>{len, nonzero});
        for (ssize_t it = 0; it < len; ++it)
        {
            for (ssize_t jt = 0; jt < nonzero; ++jt)
            {
                basis(it, jt) = left(it, jt);
            }
        }
        array_type const extra = complete_basis(basis, ncol);
        for (ssize_t it = 0; it < len; ++it)
        {
            for (ssize_t jt = nonzero; jt < ncol; ++jt)
            {
                left(it, jt) = extra(it, jt - nonzero);
            }
        }
    }

    // G = left diag(s) right^T; for m < n, A = G^T = right diag(s) left^T.
    // SimpleArray copy assignment needs equal sizes, so move.
    m_u = std::move(transposed ? right : left);
    m_v = std::move(transposed ? left : right);
}

template <typename T>
ssize_t SvdFactorization<T>::rank(value_type rtol) const
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
    T const threshold = rtol * m_s[0];
    ssize_t count = 0;
    while (count < kmin && m_s[count] > threshold)
    {
        ++count;
    }
    return count;
}

template <typename T>
typename SvdFactorization<T>::array_type SvdFactorization<T>::pinv(value_type rcond) const
{
    ssize_t const nrank = rank(rcond);
    array_type out(small_vector<ssize_t>{m_n, m_m}, T(0));
    for (ssize_t it = 0; it < m_n; ++it)
    {
        for (ssize_t jt = 0; jt < m_m; ++jt)
        {
            T sum = 0;
            for (ssize_t kt = 0; kt < nrank; ++kt)
            {
                sum += m_v(it, kt) * m_u(jt, kt) / m_s[kt];
            }
            out(it, jt) = sum;
        }
    }
    return out;
}

template <typename T>
typename SvdFactorization<T>::array_type SvdFactorization<T>::solve(array_type const & b, value_type rcond) const
{
    array_type const rhs = detail::dense_stage_rhs(b, m_m, "SvdFactorization::solve");
    ssize_t const ncol = rhs.shape(1);
    ssize_t const nrank = rank(rcond);

    // coef = diag(1 / s) U^T b over the retained singular values.
    array_type coef(small_vector<ssize_t>{nrank, ncol}, T(0));
    for (ssize_t it = 0; it < m_m; ++it)
    {
        for (ssize_t kt = 0; kt < nrank; ++kt)
        {
            T const weight = m_u(it, kt);
            for (ssize_t jt = 0; jt < ncol; ++jt)
            {
                coef(kt, jt) += weight * rhs(it, jt);
            }
        }
    }
    for (ssize_t kt = 0; kt < nrank; ++kt)
    {
        for (ssize_t jt = 0; jt < ncol; ++jt)
        {
            coef(kt, jt) /= m_s[kt];
        }
    }

    array_type out(small_vector<ssize_t>{m_n, ncol}, T(0));
    for (ssize_t it = 0; it < m_n; ++it)
    {
        for (ssize_t kt = 0; kt < nrank; ++kt)
        {
            T const weight = m_v(it, kt);
            for (ssize_t jt = 0; jt < ncol; ++jt)
            {
                out(it, jt) += weight * coef(kt, jt);
            }
        }
    }
    return detail::dense_unstage_rhs(out, b);
}

template <typename T>
typename SvdFactorization<T>::array_type SvdFactorization<T>::null_space(value_type rcond) const
{
    ssize_t const kmin = k();
    ssize_t const nrank = rank(rcond);
    array_type out(small_vector<ssize_t>{m_n, m_n - nrank});
    for (ssize_t it = 0; it < m_n; ++it)
    {
        for (ssize_t jt = nrank; jt < kmin; ++jt)
        {
            out(it, jt - nrank) = m_v(it, jt);
        }
    }
    if (kmin < m_n)
    {
        // m < n: the full V adds the orthogonal complement of the thin V.
        array_type const extra = complete_basis(m_v, m_n);
        for (ssize_t it = 0; it < m_n; ++it)
        {
            for (ssize_t jt = kmin; jt < m_n; ++jt)
            {
                out(it, jt - nrank) = extra(it, jt - kmin);
            }
        }
    }
    return out;
}

/**
 * Free function wrapper for the thin SVD.
 *
 * @return (U, s, V) with A = U diag(s) V^T; note V, not V^T.
 */
template <typename T>
std::tuple<SimpleArray<T>, SimpleArray<T>, SimpleArray<T>> svd(SimpleArray<T> const & a)
{
    SvdFactorization<T> const f(a);
    return {f.u(), f.s(), f.v()};
}

/// Moore-Penrose pseudo-inverse; see SvdFactorization::pinv().
template <typename T>
SimpleArray<T> pinv(SimpleArray<T> const & a, T rcond = T(-1))
{
    return SvdFactorization<T>(a).pinv(rcond);
}

/// Null space basis; see SvdFactorization::null_space().
template <typename T>
SimpleArray<T> null_space(SimpleArray<T> const & a, T rcond = T(-1))
{
    return SvdFactorization<T>(a).null_space(rcond);
}

/// Minimum-norm least-squares solution; see SvdFactorization::solve().
template <typename T>
SimpleArray<T> lstsq(SimpleArray<T> const & a, SimpleArray<T> const & b, T rcond = T(-1))
{
    return SvdFactorization<T>(a).solve(b, rcond);
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
