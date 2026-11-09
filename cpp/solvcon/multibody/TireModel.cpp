/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/multibody/TireModel.hpp>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace solvcon
{

namespace multibody
{

namespace
{

void check_curve(MagicFormulaTire::Curve const & curve, char const * name)
{
    bool const valid = curve.mu > 0.0 && curve.load_exponent > 0.0 && curve.stiffness > 0.0 && curve.shape > 0.0 && curve.shape < 2.0 && curve.curvature <= 1.0 && std::isfinite(curve.mu) && std::isfinite(curve.load_exponent) && std::isfinite(curve.stiffness) && std::isfinite(curve.curvature);
    if (!valid)
    {
        throw std::invalid_argument(std::string("MagicFormulaTire: the ") + name + " curve needs mu > 0, load_exponent > 0, stiffness > 0, 0 < shape < 2, and curvature <= 1");
    }
}

} /* end namespace */

MagicFormulaTire::MagicFormulaTire(double fz0, Curve const & longitudinal, Curve const & lateral)
    : m_fz0(fz0)
    , m_x(longitudinal)
    , m_y(lateral)
{
    if (!(fz0 > 0.0) || !std::isfinite(fz0))
    {
        throw std::invalid_argument("MagicFormulaTire: fz0 must be positive");
    }
    check_curve(m_x, "longitudinal");
    check_curve(m_y, "lateral");
}

double MagicFormulaTire::peak(Curve const & curve, double fz) const
{
    if (fz <= 0.0)
    {
        return 0.0;
    }
    return curve.mu * fz * std::pow(fz / m_fz0, curve.load_exponent - 1.0);
}

double MagicFormulaTire::friction(Curve const & curve, double fz) const
{
    return fz > 0.0 ? peak(curve, fz) / fz : 0.0;
}

double MagicFormulaTire::slip_stiffness(Curve const & curve, double fz) const
{
    return curve.stiffness * curve.shape * peak(curve, fz);
}

double MagicFormulaTire::pure(Curve const & curve, double d, double sigma) const
{
    double const x = curve.stiffness * sigma;
    return d * std::sin(curve.shape * std::atan(x - curve.curvature * (x - std::atan(x))));
}

double MagicFormulaTire::sliding(Curve const & curve, double d) const
{
    // B s - E (B s - atan(B s)) grows without bound for E < 1 and tends to
    // pi / 2 for E = 1.
    double const half_pi = 0.5 * std::numbers::pi;
    double const angle = curve.curvature < 1.0 ? half_pi : std::atan(half_pi);
    return d * std::sin(curve.shape * angle);
}

MagicFormulaTire::Force MagicFormulaTire::force(double fz, double slip_ratio, double slip_angle) const
{
    Force out;
    out.slip_ratio = slip_ratio;
    if (fz <= 0.0)
    {
        return out;
    }
    double const tan_alpha = std::tan(slip_angle);
    double const slip = std::hypot(slip_ratio, tan_alpha);
    if (slip == 0.0)
    {
        return out;
    }
    double const rolling = 1.0 + slip_ratio;
    // A locked or reversed wheel slides on the whole patch: the curves at
    // infinite slip.
    auto value = [&](Curve const & curve)
    { return rolling > 0.0 ? pure(curve, peak(curve, fz), slip / rolling) : sliding(curve, peak(curve, fz)); };
    // sigma_x / sigma = kappa / slip and sigma_y / sigma = tan(alpha) / slip:
    // the factor 1 + kappa cancels.
    out.fx = slip_ratio / slip * value(m_x);
    out.fy = -tan_alpha / slip * value(m_y);
    return out;
}

double MagicFormulaTire::peak_slip_ratio(double fz, double slip_angle, double sign) const
{
    // Golden-section search of the largest |F_x| on [0, 1] (driving) or
    // [-1, 0] (braking); the curve rises to one peak and then falls.
    double const ratio = 0.5 * (std::sqrt(5.0) - 1.0);
    double lo = 0.0;
    double hi = sign > 0.0 ? max_drive_slip : -1.0;
    auto value = [&](double kappa)
    { return sign * force(fz, kappa, slip_angle).fx; };
    double a = hi - ratio * (hi - lo);
    double b = lo + ratio * (hi - lo);
    double fa = value(a);
    double fb = value(b);
    for (int it = 0; it < 80; ++it)
    {
        if (fa < fb)
        {
            lo = a;
            a = b;
            fa = fb;
            b = lo + ratio * (hi - lo);
            fb = value(b);
        }
        else
        {
            hi = b;
            b = a;
            fb = fa;
            a = hi - ratio * (hi - lo);
            fa = value(a);
        }
    }
    return 0.5 * (lo + hi);
}

MagicFormulaTire::Force MagicFormulaTire::force_for_demand(double fz, double demand, double slip_angle) const
{
    if (fz <= 0.0 || demand == 0.0)
    {
        return force(fz, 0.0, slip_angle);
    }
    double const sign = demand > 0.0 ? 1.0 : -1.0;
    double const top = peak_slip_ratio(fz, slip_angle, sign);
    if (sign * force(fz, top, slip_angle).fx <= sign * demand)
    {
        // Beyond the peak the wheel locks or spins up.
        return force(fz, sign > 0.0 ? max_drive_slip : -1.0, slip_angle);
    }
    double lo = 0.0;
    double hi = top;
    for (int it = 0; it < 100; ++it)
    {
        double const mid = 0.5 * (lo + hi);
        if (sign * force(fz, mid, slip_angle).fx < sign * demand)
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    return force(fz, 0.5 * (lo + hi), slip_angle);
}

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
