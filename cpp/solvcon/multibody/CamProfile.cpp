/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/multibody/CamProfile.hpp>
#include <solvcon/multibody/linear.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

namespace solvcon
{

namespace multibody
{

CamProfile::CamProfile(double a0, std::vector<double> a, std::vector<double> b)
    : m_a0(a0)
    , m_a(std::move(a))
    , m_b(std::move(b))
{
    if (m_a.size() != m_b.size())
    {
        throw std::invalid_argument("CamProfile: cosine and sine coefficient lists differ in length");
    }
}

CamProfile CamProfile::fit(const std::vector<double> & phi, const std::vector<double> & r, int n_harmonics)
{
    if (phi.size() != r.size())
    {
        throw std::invalid_argument("CamProfile::fit: phi and r differ in length");
    }
    std::size_t nterm = 1 + 2 * static_cast<std::size_t>(n_harmonics);
    if (n_harmonics < 0 || phi.size() < nterm)
    {
        throw std::invalid_argument("CamProfile::fit: need at least " + std::to_string(nterm) +
                                    " samples for " + std::to_string(n_harmonics) + " harmonics");
    }
    auto const nsample = static_cast<ssize_t>(phi.size());
    SimpleArray<double> a(small_vector<ssize_t>{nsample, static_cast<ssize_t>(nterm)});
    SimpleArray<double> b(small_vector<ssize_t>{nsample});
    for (ssize_t i = 0; i < nsample; i++)
    {
        a(i, 0) = 1.0;
        for (int k = 1; k <= n_harmonics; k++)
        {
            a(i, 2 * k - 1) = std::cos(k * phi[i]);
            a(i, 2 * k) = std::sin(k * phi[i]);
        }
        b[i] = r[i];
    }
    SimpleArray<double> c;
    bool const ok = (nrow(a) == ncol(a)) ? lu_solve(a, b, c) : least_squares(a, b, c);
    if (!ok)
    {
        c = matvec(pseudo_inverse(a), b);
    }
    std::vector<double> cos_terms(n_harmonics), sin_terms(n_harmonics);
    for (int k = 1; k <= n_harmonics; k++)
    {
        cos_terms[k - 1] = c[2 * k - 1];
        sin_terms[k - 1] = c[2 * k];
    }
    return CamProfile(c[0], cos_terms, sin_terms);
}

void CamProfile::eval(double phi, double & r, double & dr, double & ddr) const
{
    r = m_a0;
    dr = 0.0;
    ddr = 0.0;
    for (std::size_t i = 0; i < m_a.size(); i++)
    {
        double k = static_cast<double>(i + 1);
        double c = std::cos(k * phi), s = std::sin(k * phi);
        r += m_a[i] * c + m_b[i] * s;
        dr += k * (-m_a[i] * s + m_b[i] * c);
        ddr += -k * k * (m_a[i] * c + m_b[i] * s);
    }
}

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
