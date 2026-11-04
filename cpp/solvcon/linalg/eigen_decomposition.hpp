#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Eigenvalues and right eigenvectors of a real nonsymmetric matrix:
 * balancing, Householder reduction to upper Hessenberg form, Francis
 * double-shift QR iteration to the real Schur form, and back substitution
 * for the eigenvectors (the EISPACK balanc, orthes, ortran, hqr2, and balbak
 * family). Written natively, so it needs no vendor LAPACK.
 *
 * @ingroup group_numerics
 */

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>
#include <utility>

#include <solvcon/buffer/buffer.hpp>
#include <solvcon/math/Complex.hpp>
#include <solvcon/linalg/dense_util.hpp>

namespace solvcon
{

/**
 * Eigen decomposition A V = V diag(w) of a real n-by-n matrix.
 *
 * The eigenvalues w are complex; complex eigenvalues come in conjugate
 * pairs, the one with the positive imaginary part first. Column j of the
 * complex n-by-n V is the right eigenvector of w[j], normalized to unit
 * 2-norm with its largest component real (the convention of LAPACK geev and
 * numpy.linalg.eig). The eigenvalues are in the order the QR iteration
 * deflates them along the Schur diagonal, not sorted.
 *
 * Balancing scales rows and columns by powers of 2 (exact in floating
 * point) to equalize their norms, which improves the accuracy for badly
 * scaled matrices.
 *
 * Supported element types: float, double.
 *
 * @ingroup group_numerics
 */
template <typename T>
class EigenDecomposition
{

    static_assert(std::is_floating_point_v<T>, "EigenDecomposition<T> requires a real floating-point T");

public:

    using value_type = T;
    using complex_type = Complex<value_type>;
    using array_type = SimpleArray<value_type>;
    using complex_array_type = SimpleArray<complex_type>;

    /**
     * Compute the eigenvalues and, if compute_vectors, the right
     * eigenvectors.
     *
     * @throws std::invalid_argument if a is not square 2D or has a non-finite
     *         entry.
     * @throws std::runtime_error if the QR iteration does not converge.
     */
    explicit EigenDecomposition(array_type const & a, bool compute_vectors = true, bool balance = true);

    EigenDecomposition() = delete;
    EigenDecomposition(EigenDecomposition const &) = default;
    EigenDecomposition(EigenDecomposition &&) = default;
    EigenDecomposition & operator=(EigenDecomposition const &) = default;
    EigenDecomposition & operator=(EigenDecomposition &&) = default;
    ~EigenDecomposition() = default;

    ssize_t n() const { return m_n; }
    bool has_vectors() const { return m_has_vectors; }

    /// Total number of double-shift QR steps.
    ssize_t iterations() const { return m_iterations; }

    complex_array_type const & eigenvalues() const { return m_values; }

    /// @throws std::runtime_error if constructed without compute_vectors.
    complex_array_type const & eigenvectors() const
    {
        if (!m_has_vectors)
        {
            throw std::runtime_error("EigenDecomposition: eigenvectors were not computed (compute_vectors=false)");
        }
        return m_vectors;
    }

private:

    void balance_matrix();
    void reduce_hessenberg();
    void iterate_schur();
    void back_substitute();
    void form_eigenvectors();

    ssize_t m_n = 0;
    bool m_has_vectors = true;
    ssize_t m_iterations = 0;
    // The working Hessenberg / Schur matrix, the accumulated transformation
    // (then the real eigenvector storage of hqr2), the real and imaginary
    // parts of the eigenvalues, and the balancing scale.
    array_type m_h;
    array_type m_vr;
    array_type m_wr;
    array_type m_wi;
    array_type m_scale;
    complex_array_type m_values;
    complex_array_type m_vectors;

}; /* end class EigenDecomposition */

template <typename T>
EigenDecomposition<T>::EigenDecomposition(array_type const & a, bool compute_vectors, bool balance)
    : m_has_vectors(compute_vectors)
{
    detail::dense_check_matrix(a, "EigenDecomposition");
    if (a.shape(0) != a.shape(1))
    {
        throw std::invalid_argument(std::format(
            "EigenDecomposition: a must be a square 2D SimpleArray, but got shape {}",
            detail::format_shape(a.shape())));
    }
    m_n = a.shape(0);
    small_vector<ssize_t> const square{m_n, m_n};
    small_vector<ssize_t> const line{m_n};
    m_h = array_type(square);
    for (ssize_t it = 0; it < m_n; ++it)
    {
        for (ssize_t jt = 0; jt < m_n; ++jt)
        {
            m_h(it, jt) = a(it, jt);
        }
    }
    m_wr = array_type(line, T(0));
    m_wi = array_type(line, T(0));
    m_scale = array_type(line, T(1));
    if (m_has_vectors)
    {
        m_vr = array_type(square, T(0));
    }

    if (balance)
    {
        balance_matrix();
    }
    reduce_hessenberg();
    iterate_schur();

    m_values = complex_array_type(line);
    for (ssize_t it = 0; it < m_n; ++it)
    {
        m_values[it] = complex_type(m_wr[it], m_wi[it]);
    }
    if (m_has_vectors)
    {
        back_substitute();
        form_eigenvectors();
    }
}

template <typename T>
void EigenDecomposition<T>::balance_matrix()
{
    // EISPACK balanc without the permutation step: B = D^-1 A D with D a
    // diagonal of powers of 2, so the eigenvectors of A are D times those of
    // B. The pass limit only guards against a pathological cycle.
    T const radix = 2;
    T const sqrdx = radix * radix;
    bool done = false;
    for (ssize_t pass = 0; pass < 1000 && !done; ++pass)
    {
        done = true;
        for (ssize_t it = 0; it < m_n; ++it)
        {
            T col = 0;
            T row = 0;
            for (ssize_t jt = 0; jt < m_n; ++jt)
            {
                if (jt != it)
                {
                    col += std::abs(m_h(jt, it));
                    row += std::abs(m_h(it, jt));
                }
            }
            if (col == T(0) || row == T(0))
            {
                continue;
            }
            T const total = col + row;
            T factor = 1;
            T bound = row / radix;
            while (col < bound)
            {
                factor *= radix;
                col *= sqrdx;
            }
            bound = row * radix;
            while (col > bound)
            {
                factor /= radix;
                col /= sqrdx;
            }
            if ((col + row) / factor < T(0.95) * total)
            {
                done = false;
                m_scale[it] *= factor;
                T const inverse = T(1) / factor;
                for (ssize_t jt = 0; jt < m_n; ++jt)
                {
                    m_h(it, jt) *= inverse;
                }
                for (ssize_t jt = 0; jt < m_n; ++jt)
                {
                    m_h(jt, it) *= factor;
                }
            }
        }
    }
}

template <typename T>
void EigenDecomposition<T>::reduce_hessenberg()
{
    // EISPACK orthes and ortran: Householder similarity transformations
    // reduce H to upper Hessenberg form, and their product is accumulated in
    // m_vr. Column im - 1 below the subdiagonal keeps the Householder vector
    // until the accumulation reads it.
    ssize_t const high = m_n - 1;
    array_type ort(small_vector<ssize_t>{m_n}, T(0));
    for (ssize_t im = 1; im <= high - 1; ++im)
    {
        T scale = 0;
        for (ssize_t it = im; it <= high; ++it)
        {
            scale += std::abs(m_h(it, im - 1));
        }
        if (scale == T(0))
        {
            continue;
        }

        T hsum = 0;
        for (ssize_t it = high; it >= im; --it)
        {
            ort[it] = m_h(it, im - 1) / scale;
            hsum += ort[it] * ort[it];
        }
        T gv = std::sqrt(hsum);
        if (ort[im] > T(0))
        {
            gv = -gv;
        }
        hsum -= ort[im] * gv;
        ort[im] -= gv;

        // H = (I - u u^T / h) H (I - u u^T / h)
        for (ssize_t jt = im; jt < m_n; ++jt)
        {
            T fv = 0;
            for (ssize_t it = high; it >= im; --it)
            {
                fv += ort[it] * m_h(it, jt);
            }
            fv /= hsum;
            for (ssize_t it = im; it <= high; ++it)
            {
                m_h(it, jt) -= fv * ort[it];
            }
        }
        for (ssize_t it = 0; it <= high; ++it)
        {
            T fv = 0;
            for (ssize_t jt = high; jt >= im; --jt)
            {
                fv += ort[jt] * m_h(it, jt);
            }
            fv /= hsum;
            for (ssize_t jt = im; jt <= high; ++jt)
            {
                m_h(it, jt) -= fv * ort[jt];
            }
        }
        ort[im] *= scale;
        m_h(im, im - 1) = scale * gv;
    }

    if (m_has_vectors)
    {
        for (ssize_t it = 0; it < m_n; ++it)
        {
            m_vr(it, it) = T(1);
        }
        for (ssize_t im = high - 1; im >= 1; --im)
        {
            if (m_h(im, im - 1) == T(0))
            {
                continue;
            }
            for (ssize_t it = im + 1; it <= high; ++it)
            {
                ort[it] = m_h(it, im - 1);
            }
            for (ssize_t jt = im; jt <= high; ++jt)
            {
                T gv = 0;
                for (ssize_t it = im; it <= high; ++it)
                {
                    gv += ort[it] * m_vr(it, jt);
                }
                // Double division avoids possible underflow.
                gv = (gv / ort[im]) / m_h(im, im - 1);
                for (ssize_t it = im; it <= high; ++it)
                {
                    m_vr(it, jt) += gv * ort[it];
                }
            }
        }
    }

    for (ssize_t it = 2; it < m_n; ++it)
    {
        for (ssize_t jt = 0; jt < it - 1; ++jt)
        {
            m_h(it, jt) = T(0);
        }
    }
}

template <typename T>
void EigenDecomposition<T>::iterate_schur()
{
    // EISPACK hqr2 (the iteration part), as in the public-domain JAMA
    // translation. The scalars pv, qv, rv, sv, tv, wv, xv, yv, zv are p, q,
    // r, s, t, w, x, y, z there; iend is "en", the last row of the active
    // block, ilow its first row, and imid the start of the bulge.
    ssize_t const nsize = m_n;
    ssize_t const low = 0;
    ssize_t const high = nsize - 1;
    T const eps = std::numeric_limits<T>::epsilon();
    T const tiny = std::numeric_limits<T>::min();
    // LAPACK dlahqr allows 30 * max(10, n) QR steps for one deflation.
    ssize_t const itmax = 30 * std::max<ssize_t>(10, nsize);

    T exshift = 0;
    T pv = 0;
    T qv = 0;
    T rv = 0;
    T sv = 0;
    T wv = 0;
    T xv = 0;
    T yv = 0;
    T zv = 0;

    T norm = 0;
    for (ssize_t it = 0; it < nsize; ++it)
    {
        for (ssize_t jt = std::max<ssize_t>(it - 1, 0); jt < nsize; ++jt)
        {
            norm += std::abs(m_h(it, jt));
        }
    }

    ssize_t iend = nsize - 1;
    ssize_t iter = 0;
    while (iend >= low)
    {
        // Look for a single small subdiagonal element.
        ssize_t ilow = iend;
        while (ilow > low)
        {
            sv = std::abs(m_h(ilow - 1, ilow - 1)) + std::abs(m_h(ilow, ilow));
            if (sv == T(0))
            {
                sv = norm;
            }
            T const sub = std::abs(m_h(ilow, ilow - 1));
            if (sub <= tiny || sub < eps * sv)
            {
                break;
            }
            --ilow;
        }

        if (ilow == iend)
        {
            // One root found.
            m_h(iend, iend) += exshift;
            m_wr[iend] = m_h(iend, iend);
            m_wi[iend] = T(0);
            --iend;
            iter = 0;
        }
        else if (ilow == iend - 1)
        {
            // Two roots found.
            wv = m_h(iend, iend - 1) * m_h(iend - 1, iend);
            pv = (m_h(iend - 1, iend - 1) - m_h(iend, iend)) / T(2);
            qv = pv * pv + wv;
            zv = std::sqrt(std::abs(qv));
            m_h(iend, iend) += exshift;
            m_h(iend - 1, iend - 1) += exshift;
            xv = m_h(iend, iend);

            if (qv >= T(0))
            {
                // Real pair: triangularize the 2-by-2 block by a rotation.
                zv = (pv >= T(0)) ? pv + zv : pv - zv;
                m_wr[iend - 1] = xv + zv;
                m_wr[iend] = m_wr[iend - 1];
                if (zv != T(0))
                {
                    m_wr[iend] = xv - wv / zv;
                }
                m_wi[iend - 1] = T(0);
                m_wi[iend] = T(0);
                xv = m_h(iend, iend - 1);
                sv = std::abs(xv) + std::abs(zv);
                pv = xv / sv;
                qv = zv / sv;
                rv = std::sqrt(pv * pv + qv * qv);
                pv /= rv;
                qv /= rv;

                for (ssize_t jt = iend - 1; jt < nsize; ++jt)
                {
                    zv = m_h(iend - 1, jt);
                    m_h(iend - 1, jt) = qv * zv + pv * m_h(iend, jt);
                    m_h(iend, jt) = qv * m_h(iend, jt) - pv * zv;
                }
                for (ssize_t it = 0; it <= iend; ++it)
                {
                    zv = m_h(it, iend - 1);
                    m_h(it, iend - 1) = qv * zv + pv * m_h(it, iend);
                    m_h(it, iend) = qv * m_h(it, iend) - pv * zv;
                }
                if (m_has_vectors)
                {
                    for (ssize_t it = low; it <= high; ++it)
                    {
                        zv = m_vr(it, iend - 1);
                        m_vr(it, iend - 1) = qv * zv + pv * m_vr(it, iend);
                        m_vr(it, iend) = qv * m_vr(it, iend) - pv * zv;
                    }
                }
            }
            else
            {
                // Complex pair.
                m_wr[iend - 1] = xv + pv;
                m_wr[iend] = xv + pv;
                m_wi[iend - 1] = zv;
                m_wi[iend] = -zv;
            }
            iend -= 2;
            iter = 0;
        }
        else
        {
            if (iter >= itmax)
            {
                throw std::runtime_error(std::format(
                    "EigenDecomposition: the QR iteration did not converge for eigenvalue {} after {} steps",
                    iend,
                    iter));
            }

            // Form the shift.
            xv = m_h(iend, iend);
            yv = T(0);
            wv = T(0);
            if (ilow < iend)
            {
                yv = m_h(iend - 1, iend - 1);
                wv = m_h(iend, iend - 1) * m_h(iend - 1, iend);
            }

            // Exceptional shifts, every 10 steps without deflation:
            // Wilkinson's original ad hoc shift, then the one of MATLAB.
            if (iter > 0 && iter % 20 == 10)
            {
                exshift += xv;
                for (ssize_t it = low; it <= iend; ++it)
                {
                    m_h(it, it) -= xv;
                }
                sv = std::abs(m_h(iend, iend - 1)) + std::abs(m_h(iend - 1, iend - 2));
                xv = T(0.75) * sv;
                yv = xv;
                wv = T(-0.4375) * sv * sv;
            }
            if (iter > 0 && iter % 20 == 0)
            {
                sv = (yv - xv) / T(2);
                sv = sv * sv + wv;
                if (sv > T(0))
                {
                    sv = std::sqrt(sv);
                    if (yv < xv)
                    {
                        sv = -sv;
                    }
                    sv = xv - wv / ((yv - xv) / T(2) + sv);
                    for (ssize_t it = low; it <= iend; ++it)
                    {
                        m_h(it, it) -= sv;
                    }
                    exshift += sv;
                    xv = T(0.964);
                    yv = xv;
                    wv = xv;
                }
            }

            ++iter;
            ++m_iterations;

            // Look for two consecutive small subdiagonal elements.
            ssize_t imid = iend - 2;
            while (imid >= ilow)
            {
                zv = m_h(imid, imid);
                rv = xv - zv;
                sv = yv - zv;
                pv = (rv * sv - wv) / m_h(imid + 1, imid) + m_h(imid, imid + 1);
                qv = m_h(imid + 1, imid + 1) - zv - rv - sv;
                rv = m_h(imid + 2, imid + 1);
                sv = std::abs(pv) + std::abs(qv) + std::abs(rv);
                pv /= sv;
                qv /= sv;
                rv /= sv;
                if (imid == ilow)
                {
                    break;
                }
                T const lhs = std::abs(m_h(imid, imid - 1)) * (std::abs(qv) + std::abs(rv));
                T const rhs = eps * (std::abs(pv) * (std::abs(m_h(imid - 1, imid - 1)) + std::abs(zv) + std::abs(m_h(imid + 1, imid + 1))));
                if (lhs < rhs)
                {
                    break;
                }
                --imid;
            }

            for (ssize_t it = imid + 2; it <= iend; ++it)
            {
                m_h(it, it - 2) = T(0);
                if (it > imid + 2)
                {
                    m_h(it, it - 3) = T(0);
                }
            }

            // Double QR step on rows ilow..iend and columns imid..iend.
            for (ssize_t kt = imid; kt <= iend - 1; ++kt)
            {
                bool const notlast = (kt != iend - 1);
                if (kt != imid)
                {
                    pv = m_h(kt, kt - 1);
                    qv = m_h(kt + 1, kt - 1);
                    rv = notlast ? m_h(kt + 2, kt - 1) : T(0);
                    xv = std::abs(pv) + std::abs(qv) + std::abs(rv);
                    if (xv == T(0))
                    {
                        continue;
                    }
                    pv /= xv;
                    qv /= xv;
                    rv /= xv;
                }

                sv = std::sqrt(pv * pv + qv * qv + rv * rv);
                if (pv < T(0))
                {
                    sv = -sv;
                }
                if (sv == T(0))
                {
                    continue;
                }
                if (kt != imid)
                {
                    m_h(kt, kt - 1) = -sv * xv;
                }
                else if (ilow != imid)
                {
                    m_h(kt, kt - 1) = -m_h(kt, kt - 1);
                }
                pv += sv;
                xv = pv / sv;
                yv = qv / sv;
                zv = rv / sv;
                qv /= pv;
                rv /= pv;

                // Row modification.
                for (ssize_t jt = kt; jt < nsize; ++jt)
                {
                    pv = m_h(kt, jt) + qv * m_h(kt + 1, jt);
                    if (notlast)
                    {
                        pv += rv * m_h(kt + 2, jt);
                        m_h(kt + 2, jt) -= pv * zv;
                    }
                    m_h(kt, jt) -= pv * xv;
                    m_h(kt + 1, jt) -= pv * yv;
                }

                // Column modification.
                for (ssize_t it = 0; it <= std::min(iend, kt + 3); ++it)
                {
                    pv = xv * m_h(it, kt) + yv * m_h(it, kt + 1);
                    if (notlast)
                    {
                        pv += zv * m_h(it, kt + 2);
                        m_h(it, kt + 2) -= pv * rv;
                    }
                    m_h(it, kt) -= pv;
                    m_h(it, kt + 1) -= pv * qv;
                }

                if (m_has_vectors)
                {
                    for (ssize_t it = low; it <= high; ++it)
                    {
                        pv = xv * m_vr(it, kt) + yv * m_vr(it, kt + 1);
                        if (notlast)
                        {
                            pv += zv * m_vr(it, kt + 2);
                            m_vr(it, kt + 2) -= pv * rv;
                        }
                        m_vr(it, kt) -= pv;
                        m_vr(it, kt + 1) -= pv * qv;
                    }
                }
            }
        }
    }
}

template <typename T>
void EigenDecomposition<T>::back_substitute()
{
    // EISPACK hqr2 (the eigenvector part): solve the quasi-triangular Schur
    // form for its eigenvectors, then multiply by the accumulated Schur
    // vectors. Variable names follow iterate_schur().
    ssize_t const nsize = m_n;
    ssize_t const low = 0;
    ssize_t const high = nsize - 1;
    T const eps = std::numeric_limits<T>::epsilon();

    T norm = 0;
    for (ssize_t it = 0; it < nsize; ++it)
    {
        for (ssize_t jt = std::max<ssize_t>(it - 1, 0); jt < nsize; ++jt)
        {
            norm += std::abs(m_h(it, jt));
        }
    }
    if (norm == T(0))
    {
        // The zero matrix: the accumulated transformation (identity) holds
        // the eigenvectors already.
        return;
    }

    // Complex scalar division (xr + i xi) / (yr + i yi) without overflow.
    auto cdiv = [](T xr, T xi, T yr, T yi)
    {
        if (std::abs(yr) > std::abs(yi))
        {
            T const ratio = yi / yr;
            T const den = yr + ratio * yi;
            return std::pair<T, T>((xr + ratio * xi) / den, (xi - ratio * xr) / den);
        }
        T const ratio = yr / yi;
        T const den = yi + ratio * yr;
        return std::pair<T, T>((ratio * xr + xi) / den, (ratio * xi - xr) / den);
    };

    T pv = 0;
    T qv = 0;
    T rv = 0;
    T sv = 0;
    T tv = 0;
    T wv = 0;
    T xv = 0;
    T yv = 0;
    T zv = 0;

    for (ssize_t iend = nsize - 1; iend >= 0; --iend)
    {
        pv = m_wr[iend];
        qv = m_wi[iend];

        if (qv == T(0))
        {
            // Real vector.
            ssize_t lpos = iend;
            m_h(iend, iend) = T(1);
            for (ssize_t it = iend - 1; it >= 0; --it)
            {
                wv = m_h(it, it) - pv;
                rv = T(0);
                for (ssize_t jt = lpos; jt <= iend; ++jt)
                {
                    rv += m_h(it, jt) * m_h(jt, iend);
                }
                if (m_wi[it] < T(0))
                {
                    zv = wv;
                    sv = rv;
                    continue;
                }
                lpos = it;
                if (m_wi[it] == T(0))
                {
                    m_h(it, iend) = (wv != T(0)) ? -rv / wv : -rv / (eps * norm);
                }
                else
                {
                    // Solve the real 2-by-2 equations.
                    xv = m_h(it, it + 1);
                    yv = m_h(it + 1, it);
                    qv = (m_wr[it] - pv) * (m_wr[it] - pv) + m_wi[it] * m_wi[it];
                    tv = (xv * sv - zv * rv) / qv;
                    m_h(it, iend) = tv;
                    m_h(it + 1, iend) = (std::abs(xv) > std::abs(zv)) ? (-rv - wv * tv) / xv : (-sv - yv * tv) / zv;
                }

                // Overflow control.
                tv = std::abs(m_h(it, iend));
                if ((eps * tv) * tv > T(1))
                {
                    for (ssize_t jt = it; jt <= iend; ++jt)
                    {
                        m_h(jt, iend) /= tv;
                    }
                }
            }
        }
        else if (qv < T(0))
        {
            // Complex vector of the pair (iend - 1, iend): the real part in
            // column iend - 1, the imaginary part in column iend.
            ssize_t lpos = iend - 1;

            // The last vector component is imaginary, so the matrix is
            // triangular.
            if (std::abs(m_h(iend, iend - 1)) > std::abs(m_h(iend - 1, iend)))
            {
                m_h(iend - 1, iend - 1) = qv / m_h(iend, iend - 1);
                m_h(iend - 1, iend) = -(m_h(iend, iend) - pv) / m_h(iend, iend - 1);
            }
            else
            {
                auto const [cr, ci] = cdiv(T(0), -m_h(iend - 1, iend), m_h(iend - 1, iend - 1) - pv, qv);
                m_h(iend - 1, iend - 1) = cr;
                m_h(iend - 1, iend) = ci;
            }
            m_h(iend, iend - 1) = T(0);
            m_h(iend, iend) = T(1);
            for (ssize_t it = iend - 2; it >= 0; --it)
            {
                T ra = 0;
                T sa = 0;
                for (ssize_t jt = lpos; jt <= iend; ++jt)
                {
                    ra += m_h(it, jt) * m_h(jt, iend - 1);
                    sa += m_h(it, jt) * m_h(jt, iend);
                }
                wv = m_h(it, it) - pv;

                if (m_wi[it] < T(0))
                {
                    zv = wv;
                    rv = ra;
                    sv = sa;
                    continue;
                }
                lpos = it;
                if (m_wi[it] == T(0))
                {
                    auto const [cr, ci] = cdiv(-ra, -sa, wv, qv);
                    m_h(it, iend - 1) = cr;
                    m_h(it, iend) = ci;
                }
                else
                {
                    // Solve the complex 2-by-2 equations.
                    xv = m_h(it, it + 1);
                    yv = m_h(it + 1, it);
                    T vr = (m_wr[it] - pv) * (m_wr[it] - pv) + m_wi[it] * m_wi[it] - qv * qv;
                    T const vi = (m_wr[it] - pv) * T(2) * qv;
                    if (vr == T(0) && vi == T(0))
                    {
                        vr = eps * norm * (std::abs(wv) + std::abs(qv) + std::abs(xv) + std::abs(yv) + std::abs(zv));
                    }
                    auto const [cr, ci] = cdiv(xv * rv - zv * ra + qv * sa, xv * sv - zv * sa - qv * ra, vr, vi);
                    m_h(it, iend - 1) = cr;
                    m_h(it, iend) = ci;
                    if (std::abs(xv) > (std::abs(zv) + std::abs(qv)))
                    {
                        m_h(it + 1, iend - 1) = (-ra - wv * m_h(it, iend - 1) + qv * m_h(it, iend)) / xv;
                        m_h(it + 1, iend) = (-sa - wv * m_h(it, iend) - qv * m_h(it, iend - 1)) / xv;
                    }
                    else
                    {
                        auto const [dr, di] = cdiv(-rv - yv * m_h(it, iend - 1), -sv - yv * m_h(it, iend), zv, qv);
                        m_h(it + 1, iend - 1) = dr;
                        m_h(it + 1, iend) = di;
                    }
                }

                // Overflow control.
                tv = std::max(std::abs(m_h(it, iend - 1)), std::abs(m_h(it, iend)));
                if ((eps * tv) * tv > T(1))
                {
                    for (ssize_t jt = it; jt <= iend; ++jt)
                    {
                        m_h(jt, iend - 1) /= tv;
                        m_h(jt, iend) /= tv;
                    }
                }
            }
        }
    }

    // Back transformation to the eigenvectors of the (balanced) matrix.
    for (ssize_t jt = nsize - 1; jt >= low; --jt)
    {
        for (ssize_t it = low; it <= high; ++it)
        {
            zv = T(0);
            for (ssize_t kt = low; kt <= std::min(jt, high); ++kt)
            {
                zv += m_vr(it, kt) * m_h(kt, jt);
            }
            m_vr(it, jt) = zv;
        }
    }
}

template <typename T>
void EigenDecomposition<T>::form_eigenvectors()
{
    // Undo the balancing (EISPACK balbak), pack the real and imaginary
    // parts, and normalize like LAPACK geev.
    for (ssize_t it = 0; it < m_n; ++it)
    {
        for (ssize_t jt = 0; jt < m_n; ++jt)
        {
            m_vr(it, jt) *= m_scale[it];
        }
    }

    m_vectors = complex_array_type(small_vector<ssize_t>{m_n, m_n});
    for (ssize_t jt = 0; jt < m_n; ++jt)
    {
        if (m_wi[jt] == T(0))
        {
            T scale = 0;
            for (ssize_t it = 0; it < m_n; ++it)
            {
                scale = std::max(scale, std::abs(m_vr(it, jt)));
            }
            T sum = 0;
            for (ssize_t it = 0; it < m_n; ++it)
            {
                T const ratio = m_vr(it, jt) / scale;
                sum += ratio * ratio;
            }
            T const norm = scale * std::sqrt(sum);
            for (ssize_t it = 0; it < m_n; ++it)
            {
                m_vectors(it, jt) = complex_type(m_vr(it, jt) / norm, T(0));
            }
        }
        else if (m_wi[jt] > T(0))
        {
            // Columns jt (real part) and jt + 1 (imaginary part) give the
            // vector of w[jt]; w[jt + 1] has its conjugate.
            T scale = 0;
            for (ssize_t it = 0; it < m_n; ++it)
            {
                scale = std::max({scale, std::abs(m_vr(it, jt)), std::abs(m_vr(it, jt + 1))});
            }
            T sum = 0;
            ssize_t ilarge = 0;
            T large = -1;
            for (ssize_t it = 0; it < m_n; ++it)
            {
                T const re = m_vr(it, jt) / scale;
                T const im = m_vr(it, jt + 1) / scale;
                T const mag2 = re * re + im * im;
                sum += mag2;
                if (mag2 > large)
                {
                    large = mag2;
                    ilarge = it;
                }
            }
            T const norm = scale * std::sqrt(sum);
            // Rotate by the phase of the largest component to make it real
            // and positive.
            T const lre = m_vr(ilarge, jt);
            T const lim = m_vr(ilarge, jt + 1);
            T const lmag = std::hypot(lre, lim);
            T const cre = lre / (lmag * norm);
            T const cim = -lim / (lmag * norm);
            for (ssize_t it = 0; it < m_n; ++it)
            {
                T const re = m_vr(it, jt);
                T const im = m_vr(it, jt + 1);
                T const outre = re * cre - im * cim;
                T const outim = (it == ilarge) ? T(0) : re * cim + im * cre;
                m_vectors(it, jt) = complex_type(outre, outim);
                m_vectors(it, jt + 1) = complex_type(outre, -outim);
            }
            ++jt;
        }
    }
}

/**
 * Free function wrapper for the eigen decomposition.
 *
 * @return (w, V): eigenvalues and unit right eigenvectors (columns).
 */
template <typename T>
std::pair<SimpleArray<Complex<T>>, SimpleArray<Complex<T>>> eig(SimpleArray<T> const & a, bool balance = true)
{
    EigenDecomposition<T> const f(a, /* compute_vectors */ true, balance);
    return {f.eigenvalues(), f.eigenvectors()};
}

/// Eigenvalues only; see EigenDecomposition.
template <typename T>
SimpleArray<Complex<T>> eigvals(SimpleArray<T> const & a, bool balance = true)
{
    return EigenDecomposition<T>(a, /* compute_vectors */ false, balance).eigenvalues();
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
