#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <vector>

namespace solvcon
{

namespace multibody
{

/**
 * @brief Periodic profile R(phi) = a0 + sum_k (a_k cos(k phi) + b_k sin(k phi)).
 *
 * Used as the polar radius of a cam (point and roller followers) or as the
 * support function h(psi) of a convex cam (flat-face followers, Haug Sec. 3.4.2).
 */
class CamProfile
{
public:
    CamProfile() = default;
    CamProfile(double a0, std::vector<double> a, std::vector<double> b);

    /// Least-squares Fourier fit of samples (phi_k, r_k) with n_harmonics terms.
    static CamProfile fit(const std::vector<double> & phi, const std::vector<double> & r, int n_harmonics);

    /// Value and first two derivatives with respect to phi.
    void eval(double phi, double & r, double & dr, double & ddr) const;

    bool empty() const { return m_a0 == 0.0 && m_a.empty() && m_b.empty(); }

private:
    double m_a0 = 0.0;
    std::vector<double> m_a, m_b;
};

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
