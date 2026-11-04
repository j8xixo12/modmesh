/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/multibody/DriveFunction.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace solvcon
{

namespace multibody
{

DriveFunction::DriveFunction()
    : m_kind(Kind::Polynomial)
    , m_coeffs{0.0}
{
}

DriveFunction DriveFunction::polynomial(const std::vector<double> & coeffs)
{
    if (coeffs.empty())
    {
        throw std::invalid_argument("DriveFunction::polynomial: coefficient list is empty");
    }
    DriveFunction f;
    f.m_kind = Kind::Polynomial;
    f.m_coeffs = coeffs;
    return f;
}

DriveFunction DriveFunction::sine(double amplitude, double omega, double phase, double offset)
{
    DriveFunction f;
    f.m_kind = Kind::Sine;
    f.m_amplitude = amplitude;
    f.m_omega = omega;
    f.m_phase = phase;
    f.m_offset = offset;
    return f;
}

void DriveFunction::check_table(const std::vector<double> & times, const std::vector<double> & values, std::size_t min_points)
{
    if (times.size() != values.size())
    {
        throw std::invalid_argument("DriveFunction: times and values differ in length (" +
                                    std::to_string(times.size()) + " vs " +
                                    std::to_string(values.size()) + ")");
    }
    if (times.size() < min_points)
    {
        throw std::invalid_argument("DriveFunction: need at least " + std::to_string(min_points) +
                                    " points, got " + std::to_string(times.size()));
    }
    for (std::size_t k = 1; k < times.size(); k++)
    {
        if (!(times[k] > times[k - 1]))
        {
            throw std::invalid_argument("DriveFunction: times must increase strictly (index " +
                                        std::to_string(k) + ")");
        }
    }
}

DriveFunction DriveFunction::table(const std::vector<double> & times,
                                   const std::vector<double> & values)
{
    check_table(times, values, 2);
    DriveFunction f;
    f.m_kind = Kind::Table;
    f.m_times = times;
    f.m_values = values;
    return f;
}

DriveFunction DriveFunction::spline(const std::vector<double> & times,
                                    const std::vector<double> & values)
{
    check_table(times, values, 2);
    DriveFunction f;
    f.m_kind = Kind::Spline;
    f.m_times = times;
    f.m_values = values;

    // Natural cubic spline: solve the tridiagonal system for the second
    // derivatives M_k with M_0 = M_{n-1} = 0 (Thomas algorithm).
    std::size_t n = times.size();
    f.m_second.assign(n, 0.0);
    if (n > 2)
    {
        std::vector<double> diag(n, 0.0), upper(n, 0.0), rhs(n, 0.0);
        for (std::size_t k = 1; k + 1 < n; k++)
        {
            double h0 = times[k] - times[k - 1];
            double h1 = times[k + 1] - times[k];
            diag[k] = 2.0 * (h0 + h1);
            upper[k] = h1;
            rhs[k] = 6.0 * ((values[k + 1] - values[k]) / h1 - (values[k] - values[k - 1]) / h0);
        }
        // Forward elimination over interior rows 1..n-2 (lower coefficient is h0).
        for (std::size_t k = 2; k + 1 < n; k++)
        {
            double lower = times[k] - times[k - 1];
            double w = lower / diag[k - 1];
            diag[k] -= w * upper[k - 1];
            rhs[k] -= w * rhs[k - 1];
        }
        for (std::size_t k = n - 2; k >= 1; k--)
        {
            double next = (k + 2 < n) ? f.m_second[k + 1] : 0.0;
            f.m_second[k] = (rhs[k] - upper[k] * next) / diag[k];
            if (k == 1)
            {
                break;
            }
        }
    }
    return f;
}

DriveFunction DriveFunction::parameter(double value)
{
    DriveFunction f;
    f.m_kind = Kind::Parameter;
    f.m_offset = value;
    return f;
}

void DriveFunction::set_value(double value)
{
    if (m_kind != Kind::Parameter)
    {
        throw std::logic_error("DriveFunction::set_value: only valid for the Parameter kind");
    }
    m_offset = value;
}

std::size_t DriveFunction::find_segment(double t) const
{
    if (t < m_times.front() || t > m_times.back())
    {
        throw std::out_of_range("DriveFunction: t=" + std::to_string(t) +
                                " is outside the table range [" + std::to_string(m_times.front()) +
                                ", " + std::to_string(m_times.back()) + "]");
    }
    auto it = std::upper_bound(m_times.begin(), m_times.end(), t);
    std::size_t k = static_cast<std::size_t>(it - m_times.begin());
    if (k == 0)
    {
        k = 1;
    }
    if (k >= m_times.size())
    {
        k = m_times.size() - 1;
    }
    return k - 1; // segment [k-1, k] holds t
}

void DriveFunction::eval(double t, double & c, double & cd, double & cdd) const
{
    switch (m_kind)
    {
    case Kind::Polynomial:
    {
        // Horner scheme for the value and both derivatives.
        c = cd = cdd = 0.0;
        for (std::size_t k = m_coeffs.size(); k-- > 0;)
        {
            cdd = cdd * t + 2.0 * cd;
            cd = cd * t + c;
            c = c * t + m_coeffs[k];
        }
        return;
    }
    case Kind::Sine:
    {
        double arg = m_omega * t + m_phase;
        c = m_offset + m_amplitude * std::sin(arg);
        cd = m_amplitude * m_omega * std::cos(arg);
        cdd = -m_amplitude * m_omega * m_omega * std::sin(arg);
        return;
    }
    case Kind::Table:
    {
        std::size_t k = find_segment(t);
        double h = m_times[k + 1] - m_times[k];
        double slope = (m_values[k + 1] - m_values[k]) / h;
        c = m_values[k] + slope * (t - m_times[k]);
        cd = slope;
        cdd = 0.0;
        return;
    }
    case Kind::Spline:
    {
        std::size_t k = find_segment(t);
        double h = m_times[k + 1] - m_times[k];
        double a = (m_times[k + 1] - t) / h;
        double b = (t - m_times[k]) / h;
        double m0 = m_second[k];
        double m1 = m_second[k + 1];
        double y0 = m_values[k];
        double y1 = m_values[k + 1];
        c = a * y0 + b * y1 + ((a * a * a - a) * m0 + (b * b * b - b) * m1) * h * h / 6.0;
        cd = (y1 - y0) / h - (3.0 * a * a - 1.0) * h * m0 / 6.0 +
             (3.0 * b * b - 1.0) * h * m1 / 6.0;
        cdd = a * m0 + b * m1;
        return;
    }
    case Kind::Parameter:
        c = m_offset;
        cd = 0.0;
        cdd = 0.0;
        return;
    }
    throw std::logic_error("DriveFunction::eval: unknown kind");
}

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
