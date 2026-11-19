#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Dense linear algebra of the multibody engine on row-major SimpleArray<double>.
 *
 * Vectors are 1D arrays, matrices are 2D arrays.  Every array returned here is
 * C-contiguous (row-major).  The inline kernels (products, norms) run plain
 * loops that sum the inner index in increasing order.  The decompositions are
 * defined in linear.cpp, which is the only file that knows the backend.
 *
 * @ingroup group_numerics
 */

#include <algorithm>
#include <cmath>
#include <cstddef>

#include <solvcon/buffer/buffer.hpp>

namespace solvcon
{

namespace multibody
{

/// Zero-filled vector of length n.
inline SimpleArray<double> zero_vector(ssize_t n) { return SimpleArray<double>(small_vector<ssize_t>{n}, 0.0); }

/// Zero-filled m-by-n matrix.
inline SimpleArray<double> zero_matrix(ssize_t m, ssize_t n) { return SimpleArray<double>(small_vector<ssize_t>{m, n}, 0.0); }

/// n-by-n identity matrix.
inline SimpleArray<double> identity_matrix(ssize_t n)
{
    SimpleArray<double> out = zero_matrix(n, n);
    for (ssize_t k = 0; k < n; ++k)
    {
        out(k, k) = 1.0;
    }
    return out;
}

/// Number of rows of a matrix or length of a vector.
inline ssize_t nrow(SimpleArray<double> const & a) { return a.shape(0); }

/// Number of columns of a matrix (1 for a vector).
inline ssize_t ncol(SimpleArray<double> const & a) { return (a.ndim() == 1) ? 1 : a.shape(1); }

/// a x for an m-by-n matrix a and a vector x of length n.
inline SimpleArray<double> matvec(SimpleArray<double> const & a, SimpleArray<double> const & x)
{
    ssize_t const m = a.shape(0), n = a.shape(1);
    SimpleArray<double> out(small_vector<ssize_t>{m});
    double const * ap = a.data();
    double const * xp = x.data();
    for (ssize_t i = 0; i < m; ++i)
    {
        double sum = 0.0;
        for (ssize_t p = 0; p < n; ++p)
        {
            sum += ap[i * n + p] * xp[p];
        }
        out[i] = sum;
    }
    return out;
}

/// a^T x for an m-by-n matrix a and a vector x of length m.
inline SimpleArray<double> matvec_transpose(SimpleArray<double> const & a, SimpleArray<double> const & x)
{
    ssize_t const m = a.shape(0), n = a.shape(1);
    SimpleArray<double> out = zero_vector(n);
    double const * ap = a.data();
    double const * xp = x.data();
    double * op = out.data();
    for (ssize_t p = 0; p < m; ++p)
    {
        for (ssize_t j = 0; j < n; ++j)
        {
            op[j] += ap[p * n + j] * xp[p];
        }
    }
    return out;
}

/// a b for an m-by-k matrix a and a k-by-n matrix b.
inline SimpleArray<double> matmul(SimpleArray<double> const & a, SimpleArray<double> const & b)
{
    ssize_t const m = a.shape(0), k = a.shape(1), n = b.shape(1);
    SimpleArray<double> out = zero_matrix(m, n);
    double const * ap = a.data();
    double const * bp = b.data();
    double * op = out.data();
    for (ssize_t i = 0; i < m; ++i)
    {
        for (ssize_t p = 0; p < k; ++p)
        {
            double const aip = ap[i * k + p];
            for (ssize_t j = 0; j < n; ++j)
            {
                op[i * n + j] += aip * bp[p * n + j];
            }
        }
    }
    return out;
}

/// Transposed copy of a matrix.
inline SimpleArray<double> transposed(SimpleArray<double> const & a)
{
    ssize_t const m = a.shape(0), n = a.shape(1);
    SimpleArray<double> out(small_vector<ssize_t>{n, m});
    for (ssize_t i = 0; i < m; ++i)
    {
        for (ssize_t j = 0; j < n; ++j)
        {
            out(j, i) = a(i, j);
        }
    }
    return out;
}

/// Maximum absolute value of the elements (NaN if an element is NaN, 0 for no element).
inline double norm_inf(SimpleArray<double> const & v)
{
    double out = 0.0;
    for (double x : v)
    {
        if (std::isnan(x))
        {
            return x;
        }
        out = std::max(out, std::abs(x));
    }
    return out;
}

/// Euclidean norm of the elements.
inline double norm2(SimpleArray<double> const & v)
{
    double sum = 0.0;
    for (double x : v)
    {
        sum += x * x;
    }
    return std::sqrt(sum);
}

inline bool all_finite(SimpleArray<double> const & v)
{
    for (double x : v)
    {
        if (!std::isfinite(x))
        {
            return false;
        }
    }
    return true;
}

/**
 * Solve the square system a x = b by LU with partial pivoting.
 *
 * @param[in] a  n-by-n matrix.
 * @param[in] b  Vector of length n or n-by-k matrix.
 * @param[out] x  Solution with the shape of b.
 * @return False if a is singular to working precision (reciprocal condition
 *      number below machine epsilon); x is then undefined.
 */
bool lu_solve(SimpleArray<double> const & a, SimpleArray<double> const & b, SimpleArray<double> & x);

/**
 * Least-squares (m >= n) or minimum-norm (m < n) solution of a x = b for a
 * full-rank m-by-n matrix a, by QR.
 *
 * @param[in] a  m-by-n matrix.
 * @param[in] b  Vector of length m or m-by-k matrix.
 * @param[out] x  Solution, n rows.
 * @return False if the triangular factor has a reciprocal condition number
 *      below machine epsilon (a is rank deficient); x is then undefined.
 */
bool least_squares(SimpleArray<double> const & a, SimpleArray<double> const & b, SimpleArray<double> & x);

/**
 * Minimum-norm least-squares solution of a x = b by a complete orthogonal
 * decomposition (column-pivoted QR, then a QR of the leading rows of R).
 * Columns whose R diagonal is not above rel_tol * max(m, n) * |R(0, 0)|
 * count as dependent.
 *
 * @param a  m-by-n matrix.
 * @param b  Vector of length m.
 * @param rel_tol  Relative rank threshold.
 * @return Vector of length n.
 * @throws std::runtime_error if a or b has a non-finite element (a diverged
 *      state; a zero answer would hide it).
 */
SimpleArray<double> min_norm_solve(SimpleArray<double> const & a, SimpleArray<double> const & b, double rel_tol);

/**
 * Solve a x = b for a possibly redundant or singular system.  The direct
 * solve (LU for a square a, least_squares otherwise) is accepted if it
 * succeeds and |a x - b|_inf < 1e-8 |b|_inf.  Otherwise the result is
 * min_norm_solve(a, b, 1e-12).  The LU skips the condition estimate of
 * lu_solve when the smallest pivot is at least 1e-13 of the largest.
 *
 * @return Vector of length n.
 * @throws std::runtime_error if a or b has a non-finite element.
 */
SimpleArray<double> robust_solve(SimpleArray<double> const & a, SimpleArray<double> const & b);

/**
 * Singular values in descending order.
 *
 * @throws std::runtime_error if a has a non-finite element.
 */
SimpleArray<double> singular_values(SimpleArray<double> const & a);

/**
 * Orthonormal basis of the null space of an m-by-n matrix, as the columns of
 * an n-by-(n - rank) matrix.  The rank counts the singular values above
 * max(m, n) * s_max * eps.  Elements below eps in magnitude are set to zero.
 * An empty matrix gives a 0-by-0 result.
 */
SimpleArray<double> null_space(SimpleArray<double> const & a);
/// null_space() keeping the singular values above rel_tol * s_max.
SimpleArray<double> null_space(SimpleArray<double> const & a, double rel_tol);

/// Numerical rank: the number of singular values above rel_tol * s_max.
ssize_t matrix_rank(SimpleArray<double> const & a, double rel_tol);

/// Moore-Penrose pseudo-inverse (n-by-m); singular values below max(m, n) * s_max * eps are dropped.
SimpleArray<double> pseudo_inverse(SimpleArray<double> const & a);
/// pseudo_inverse() dropping the singular values not above rel_tol * s_max.
SimpleArray<double> pseudo_inverse(SimpleArray<double> const & a, double rel_tol);

/**
 * Inverse of a square matrix.
 *
 * @param[in] a  n-by-n matrix.
 * @param[out] out  The inverse.
 * @return False if a is singular to working precision (see lu_solve).
 */
bool inverse(SimpleArray<double> const & a, SimpleArray<double> & out);

/// Eigenvalues and right eigenvectors of a real square matrix.
struct EigenDecomposition
{
    SimpleArray<double> wr; ///< Real parts of the eigenvalues.
    SimpleArray<double> wi; ///< Imaginary parts of the eigenvalues.
    /**
     * n-by-n right eigenvectors in columns.  A real eigenvalue k has the
     * eigenvector in column k.  A complex pair (k, k + 1) with wi(k) > 0 has
     * the eigenvector of eigenvalue k in columns k (real part) and k + 1
     * (imaginary part).
     */
    SimpleArray<double> vr;
}; /* end struct EigenDecomposition */

/// Eigenvalues and right eigenvectors of a real nonsymmetric square matrix.
EigenDecomposition eigen_general(SimpleArray<double> const & a);

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
