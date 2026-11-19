/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/multibody/linear.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <solvcon/linalg/linalg.hpp>

namespace solvcon
{

namespace multibody
{

namespace
{

constexpr double machine_eps = std::numeric_limits<double>::epsilon();

SimpleArray<double> zero_solution(SimpleArray<double> const & a, SimpleArray<double> const & b)
{
    return (b.ndim() == 1) ? zero_vector(ncol(a)) : zero_matrix(ncol(a), ncol(b));
}

/// Column j of the right-hand side b (a vector is one column).
double rhs(SimpleArray<double> const & b, ssize_t i, ssize_t j) { return (b.ndim() == 1) ? b[i] : b(i, j); }

void set_out(SimpleArray<double> & x, ssize_t i, ssize_t j, double v)
{
    if (x.ndim() == 1)
    {
        x[i] = v;
    }
    else
    {
        x(i, j) = v;
    }
}

/**
 * LU factorization with partial pivoting, P A = L U, of a small dense
 * matrix, with solves by A and by A^T and the 1-norm reciprocal condition
 * estimate of Hager and Higham (the dgecon estimator).
 */
class DenseLu
{

public:

    explicit DenseLu(SimpleArray<double> const & a)
        : m_n(nrow(a))
        , m_lu(static_cast<size_t>(m_n * m_n))
        , m_piv(static_cast<size_t>(m_n))
    {
        for (ssize_t i = 0; i < m_n; ++i)
        {
            for (ssize_t j = 0; j < m_n; ++j)
            {
                at(i, j) = a(i, j);
            }
        }
        for (ssize_t j = 0; j < m_n; ++j)
        {
            double col = 0.0;
            for (ssize_t i = 0; i < m_n; ++i)
            {
                col += std::fabs(a(i, j));
            }
            m_anorm = std::max(m_anorm, col);
        }
        factorize();
    }

    bool singular() const { return m_singular; }

    /// Solve A x = b in place.
    void solve(std::vector<double> & x) const
    {
        for (ssize_t k = 0; k < m_n; ++k)
        {
            std::swap(x[static_cast<size_t>(k)], x[static_cast<size_t>(m_piv[static_cast<size_t>(k)])]);
        }
        for (ssize_t i = 0; i < m_n; ++i)
        {
            double sum = x[static_cast<size_t>(i)];
            for (ssize_t j = 0; j < i; ++j)
            {
                sum -= at(i, j) * x[static_cast<size_t>(j)];
            }
            x[static_cast<size_t>(i)] = sum;
        }
        for (ssize_t i = m_n - 1; i >= 0; --i)
        {
            double sum = x[static_cast<size_t>(i)];
            for (ssize_t j = i + 1; j < m_n; ++j)
            {
                sum -= at(i, j) * x[static_cast<size_t>(j)];
            }
            x[static_cast<size_t>(i)] = sum / at(i, i);
        }
    }

    /// Solve A^T x = b in place: U^T L^T P x = b.
    void solve_transpose(std::vector<double> & x) const
    {
        for (ssize_t i = 0; i < m_n; ++i)
        {
            double sum = x[static_cast<size_t>(i)];
            for (ssize_t j = 0; j < i; ++j)
            {
                sum -= at(j, i) * x[static_cast<size_t>(j)];
            }
            x[static_cast<size_t>(i)] = sum / at(i, i);
        }
        for (ssize_t i = m_n - 1; i >= 0; --i)
        {
            double sum = x[static_cast<size_t>(i)];
            for (ssize_t j = i + 1; j < m_n; ++j)
            {
                sum -= at(j, i) * x[static_cast<size_t>(j)];
            }
            x[static_cast<size_t>(i)] = sum;
        }
        for (ssize_t k = m_n - 1; k >= 0; --k)
        {
            std::swap(x[static_cast<size_t>(k)], x[static_cast<size_t>(m_piv[static_cast<size_t>(k)])]);
        }
    }

    /// min |U(i, i)| / max |U(i, i)|, 0 for a zero pivot: a cheap sign of a
    /// singular matrix (not a bound of the condition number).
    double pivot_ratio() const
    {
        if (m_singular || m_n == 0)
        {
            return 0.0;
        }
        double lo = std::fabs(at(0, 0)), hi = lo;
        for (ssize_t i = 1; i < m_n; ++i)
        {
            lo = std::min(lo, std::fabs(at(i, i)));
            hi = std::max(hi, std::fabs(at(i, i)));
        }
        return (hi > 0.0) ? lo / hi : 0.0;
    }

    /// 1 / (|A|_1 |A^-1|_1), with |A^-1|_1 estimated by Hager-Higham.
    double rcond() const
    {
        if (m_singular || m_anorm == 0.0)
        {
            return 0.0;
        }
        auto const n = static_cast<size_t>(m_n);
        std::vector<double> x(n, 1.0 / static_cast<double>(m_n));
        double est = 0.0;
        for (int iter = 0; iter < 5; ++iter)
        {
            std::vector<double> y(x);
            solve(y);
            est = 0.0;
            for (double v : y)
            {
                est += std::fabs(v);
            }
            std::vector<double> z(n);
            for (size_t i = 0; i < n; ++i)
            {
                z[i] = (y[i] >= 0.0) ? 1.0 : -1.0;
            }
            solve_transpose(z);
            size_t jmax = 0;
            double zx = 0.0;
            for (size_t i = 0; i < n; ++i)
            {
                zx += z[i] * x[i];
                if (std::fabs(z[i]) > std::fabs(z[jmax]))
                {
                    jmax = i;
                }
            }
            if (iter > 0 && std::fabs(z[jmax]) <= zx)
            {
                break;
            }
            std::fill(x.begin(), x.end(), 0.0);
            x[jmax] = 1.0;
        }
        // Higham's alternating vector guards against a too-small estimate.
        std::vector<double> alt(n);
        for (size_t i = 0; i < n; ++i)
        {
            double const sign = (i % 2 == 0) ? 1.0 : -1.0;
            alt[i] = sign * (1.0 + static_cast<double>(i) / static_cast<double>(std::max<size_t>(n - 1, 1)));
        }
        solve(alt);
        double alt_norm = 0.0;
        for (double v : alt)
        {
            alt_norm += std::fabs(v);
        }
        est = std::max(est, 2.0 * alt_norm / (3.0 * static_cast<double>(n)));
        return (est > 0.0) ? 1.0 / (m_anorm * est) : 0.0;
    }

private:

    double & at(ssize_t i, ssize_t j) { return m_lu[static_cast<size_t>(i * m_n + j)]; }
    double at(ssize_t i, ssize_t j) const { return m_lu[static_cast<size_t>(i * m_n + j)]; }

    void factorize()
    {
        for (ssize_t k = 0; k < m_n; ++k)
        {
            ssize_t p = k;
            for (ssize_t i = k + 1; i < m_n; ++i)
            {
                if (std::fabs(at(i, k)) > std::fabs(at(p, k)))
                {
                    p = i;
                }
            }
            m_piv[static_cast<size_t>(k)] = p;
            if (at(p, k) == 0.0)
            {
                m_singular = true;
                continue;
            }
            if (p != k)
            {
                for (ssize_t j = 0; j < m_n; ++j)
                {
                    std::swap(at(k, j), at(p, j));
                }
            }
            for (ssize_t i = k + 1; i < m_n; ++i)
            {
                // The matrices of a mechanism are mostly zero: a zero
                // multiplier leaves the row as it is.
                if (at(i, k) == 0.0)
                {
                    continue;
                }
                double const l = at(i, k) / at(k, k);
                at(i, k) = l;
                for (ssize_t j = k + 1; j < m_n; ++j)
                {
                    at(i, j) -= l * at(k, j);
                }
            }
        }
    }

    ssize_t m_n;
    std::vector<double> m_lu;
    std::vector<ssize_t> m_piv;
    double m_anorm = 0.0;
    bool m_singular = false;

}; /* end class DenseLu */

/// |R(k-1, k-1)| / |R(0, 0)| of a column-pivoted QR, the rank-revealing ratio.
double qr_rcond(SimpleArray<double> const & r)
{
    ssize_t const k = std::min(r.shape(0), r.shape(1));
    if (k == 0 || r(0, 0) == 0.0)
    {
        return 0.0;
    }
    return std::fabs(r(k - 1, k - 1)) / std::fabs(r(0, 0));
}

} /* end namespace */

namespace
{

/// Pivot ratio above which solve_by_lu() with quick set skips the condition
/// estimate.
constexpr double quick_pivot_ratio = 1e-13;

/// lu_solve().  With quick set, a pivot ratio of at least quick_pivot_ratio
/// accepts the factorization without the condition estimate (several more
/// triangular solves); the caller must then check the residual.
bool solve_by_lu(SimpleArray<double> const & a, SimpleArray<double> const & b, SimpleArray<double> & x, bool quick)
{
    if (nrow(a) != nrow(b))
    {
        throw std::logic_error("lu_solve: a and b differ in the number of rows");
    }
    if (a.size() == 0 || b.size() == 0)
    {
        x = zero_solution(a, b);
        return true;
    }
    DenseLu const lu(a);
    if (!(quick && lu.pivot_ratio() >= quick_pivot_ratio) && !(lu.rcond() >= machine_eps))
    {
        return false;
    }
    ssize_t const n = nrow(a), p = ncol(b);
    x = zero_solution(a, b);
    std::vector<double> col(static_cast<size_t>(n));
    for (ssize_t j = 0; j < p; ++j)
    {
        for (ssize_t i = 0; i < n; ++i)
        {
            col[static_cast<size_t>(i)] = rhs(b, i, j);
        }
        lu.solve(col);
        for (ssize_t i = 0; i < n; ++i)
        {
            set_out(x, i, j, col[static_cast<size_t>(i)]);
        }
    }
    return true;
}

} /* end namespace */

bool lu_solve(SimpleArray<double> const & a, SimpleArray<double> const & b, SimpleArray<double> & x)
{
    return solve_by_lu(a, b, x, false);
}

bool least_squares(SimpleArray<double> const & a, SimpleArray<double> const & b, SimpleArray<double> & x)
{
    if (nrow(a) != nrow(b))
    {
        throw std::logic_error("least_squares: a and b differ in the number of rows");
    }
    if (a.size() == 0 || b.size() == 0)
    {
        x = zero_solution(a, b);
        return true;
    }
    ssize_t const m = nrow(a), n = ncol(a), p = ncol(b);
    x = zero_solution(a, b);
    if (m >= n)
    {
        // A P = Q R: R z = (Q^T b)(0:n), x(perm(j)) = z(j).
        QrPivotFactorization<double> const qr(a);
        SimpleArray<double> const r = qr.r();
        if (!(qr_rcond(r) >= machine_eps))
        {
            return false;
        }
        SimpleArray<double> const qtb = qr.apply_qt(b);
        auto const & perm = qr.perm();
        std::vector<double> z(static_cast<size_t>(n));
        for (ssize_t j = 0; j < p; ++j)
        {
            for (ssize_t i = n - 1; i >= 0; --i)
            {
                double sum = rhs(qtb, i, j);
                for (ssize_t c = i + 1; c < n; ++c)
                {
                    sum -= r(i, c) * z[static_cast<size_t>(c)];
                }
                z[static_cast<size_t>(i)] = sum / r(i, i);
            }
            for (ssize_t i = 0; i < n; ++i)
            {
                set_out(x, perm[i], j, z[static_cast<size_t>(i)]);
            }
        }
        return true;
    }
    // Minimum norm: A^T P = Q R, so R^T y = P^T b and x = Q y.
    QrPivotFactorization<double> const qr(transposed(a));
    SimpleArray<double> const r = qr.r();
    if (!(qr_rcond(r) >= machine_eps))
    {
        return false;
    }
    SimpleArray<double> const q = qr.q();
    auto const & perm = qr.perm();
    std::vector<double> y(static_cast<size_t>(m));
    for (ssize_t j = 0; j < p; ++j)
    {
        for (ssize_t i = 0; i < m; ++i)
        {
            double sum = rhs(b, perm[i], j);
            for (ssize_t c = 0; c < i; ++c)
            {
                sum -= r(c, i) * y[static_cast<size_t>(c)];
            }
            y[static_cast<size_t>(i)] = sum / r(i, i);
        }
        for (ssize_t row = 0; row < n; ++row)
        {
            double sum = 0.0;
            for (ssize_t c = 0; c < m; ++c)
            {
                sum += q(row, c) * y[static_cast<size_t>(c)];
            }
            set_out(x, row, j, sum);
        }
    }
    return true;
}

namespace
{

/// A non-finite system means the state diverged; a zero or least-squares
/// answer would hide it (the motion would coast on instead of stopping).
void require_finite(SimpleArray<double> const & a, SimpleArray<double> const & b, char const * who)
{
    if (!all_finite(a) || !all_finite(b))
    {
        throw std::runtime_error(std::string("[MBS] ") + who + ": the system has a non-finite entry (the simulation diverged)");
    }
}

} /* end namespace */

SimpleArray<double> min_norm_solve(SimpleArray<double> const & a, SimpleArray<double> const & b, double rel_tol)
{
    ssize_t const m = nrow(a), n = ncol(a);
    require_finite(a, b, "min_norm_solve");
    SimpleArray<double> x = zero_vector(n);
    if (a.size() == 0)
    {
        return x;
    }
    // Complete orthogonal decomposition: A P = Q R with the rank r from the
    // R diagonal, then the minimum-norm z of R(0:r, :) z = (Q^T b)(0:r)
    // from a QR of its transpose, and x = P z.  Two Householder QRs instead
    // of an SVD; the minimum-norm solution of a consistent system is unique,
    // so the answer is the SVD one.
    QrPivotFactorization<double> const qr(a);
    SimpleArray<double> const r = qr.r();
    ssize_t const k = std::min(m, n);
    double const r00 = (k > 0) ? std::fabs(r(0, 0)) : 0.0;
    double const tol = static_cast<double>(std::max(m, n)) * r00 * rel_tol;
    ssize_t rank = 0;
    while (rank < k && std::fabs(r(rank, rank)) > tol)
    {
        ++rank;
    }
    if (rank == 0)
    {
        return x;
    }
    SimpleArray<double> const qtb = qr.apply_qt(b);
    SimpleArray<double> r1t = zero_matrix(n, rank);
    for (ssize_t i = 0; i < rank; ++i)
    {
        for (ssize_t j = i; j < n; ++j)
        {
            r1t(j, i) = r(i, j);
        }
    }
    // R1^T P2 = Q2 R2, so R1 = P2 R2^T Q2^T: R2^T y = P2^T c, z = Q2 y.
    QrPivotFactorization<double> const qr2(r1t);
    SimpleArray<double> const r2 = qr2.r();
    SimpleArray<double> const q2 = qr2.q();
    auto const & perm2 = qr2.perm();
    std::vector<double> y(static_cast<size_t>(rank));
    for (ssize_t i = 0; i < rank; ++i)
    {
        double sum = qtb[perm2[i]];
        for (ssize_t c = 0; c < i; ++c)
        {
            sum -= r2(c, i) * y[static_cast<size_t>(c)];
        }
        y[static_cast<size_t>(i)] = sum / r2(i, i);
    }
    auto const & perm = qr.perm();
    for (ssize_t row = 0; row < n; ++row)
    {
        double sum = 0.0;
        for (ssize_t c = 0; c < rank; ++c)
        {
            sum += q2(row, c) * y[static_cast<size_t>(c)];
        }
        x[perm[row]] = sum;
    }
    return x;
}

SimpleArray<double> robust_solve(SimpleArray<double> const & a, SimpleArray<double> const & b)
{
    require_finite(a, b, "robust_solve");
    SimpleArray<double> x;
    // The residual check below stands in for the condition estimate.
    bool const ok = (nrow(a) == ncol(a)) ? solve_by_lu(a, b, x, true) : least_squares(a, b, x);
    if (ok)
    {
        double const b_norm = norm_inf(b);
        if (!(b_norm > 0))
        {
            return x;
        }
        SimpleArray<double> res = matvec(a, x);
        for (ssize_t i = 0; i < nrow(b); ++i)
        {
            res[i] -= b[i];
        }
        if (norm_inf(res) / b_norm < 1e-8)
        {
            return x;
        }
    }
    return min_norm_solve(a, b, 1e-12);
}

SimpleArray<double> singular_values(SimpleArray<double> const & a)
{
    if (!all_finite(a))
    {
        throw std::runtime_error("singular_values: the matrix has a non-finite element");
    }
    return SvdFactorization<double>(a).s();
}

SimpleArray<double> null_space(SimpleArray<double> const & a)
{
    if (a.size() == 0)
    {
        return zero_matrix(0, 0);
    }
    return SvdFactorization<double>(a).null_space();
}

SimpleArray<double> null_space(SimpleArray<double> const & a, double rel_tol)
{
    if (a.size() == 0)
    {
        return zero_matrix(0, 0);
    }
    return SvdFactorization<double>(a).null_space(rel_tol);
}

ssize_t matrix_rank(SimpleArray<double> const & a, double rel_tol)
{
    if (a.size() == 0)
    {
        return 0;
    }
    return SvdFactorization<double>(a).rank(rel_tol);
}

SimpleArray<double> pseudo_inverse(SimpleArray<double> const & a) { return SvdFactorization<double>(a).pinv(); }

SimpleArray<double> pseudo_inverse(SimpleArray<double> const & a, double rel_tol)
{
    return SvdFactorization<double>(a).pinv(rel_tol);
}

bool inverse(SimpleArray<double> const & a, SimpleArray<double> & out)
{
    if (nrow(a) != ncol(a))
    {
        throw std::logic_error("inverse: matrix must be square");
    }
    if (a.size() == 0)
    {
        out = zero_matrix(0, 0);
        return true;
    }
    return lu_solve(a, identity_matrix(nrow(a)), out);
}

EigenDecomposition eigen_general(SimpleArray<double> const & a)
{
    ::solvcon::EigenDecomposition<double> const eig(a, /* compute_vectors */ true, /* balance */ true);
    auto const & values = eig.eigenvalues();
    auto const & vectors = eig.eigenvectors();
    ssize_t const n = nrow(a);
    EigenDecomposition out{zero_vector(n), zero_vector(n), zero_matrix(n, n)};
    // The packed real form of LAPACK geev: a complex pair (k, k + 1) with
    // wi(k) > 0 keeps the real and the imaginary part of vector k.
    for (ssize_t k = 0; k < n; ++k)
    {
        out.wr[k] = values[k].real();
        out.wi[k] = values[k].imag();
    }
    for (ssize_t k = 0; k < n; ++k)
    {
        if (out.wi[k] > 0.0 && k + 1 < n)
        {
            for (ssize_t i = 0; i < n; ++i)
            {
                out.vr(i, k) = vectors(i, k).real();
                out.vr(i, k + 1) = vectors(i, k).imag();
            }
            ++k;
            continue;
        }
        for (ssize_t i = 0; i < n; ++i)
        {
            out.vr(i, k) = vectors(i, k).real();
        }
    }
    return out;
}

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
