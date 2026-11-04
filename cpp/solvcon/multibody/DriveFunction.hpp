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
 * @brief Time function C(t) for driving constraints (Haug Sec. 9.5).
 *
 * Every kind returns C(t), C'(t), and C''(t). Tabulated kinds reject time
 * values outside the table range instead of extrapolating.
 */
class DriveFunction
{
public:
    enum class Kind
    {
        Polynomial,
        Sine,
        Table,
        Spline,
        Parameter
    };

    /// Zero polynomial: C(t) = 0.
    DriveFunction();

    /// C(t) = sum_k coeffs[k] * t^k.
    static DriveFunction polynomial(const std::vector<double> & coeffs);
    /// C(t) = offset + amplitude * sin(omega * t + phase).
    static DriveFunction sine(double amplitude, double omega, double phase, double offset);
    /// Piecewise-linear interpolation. Times must increase strictly.
    static DriveFunction table(const std::vector<double> & times, const std::vector<double> & values);
    /// Natural cubic spline (C'' = 0 at both ends). Times must increase strictly.
    static DriveFunction spline(const std::vector<double> & times,
                                const std::vector<double> & values);
    /// Constant set by the caller (quasi-static sweeps). C' = C'' = 0.
    static DriveFunction parameter(double value);

    /**
     * @brief Evaluate the function and its first two time derivatives.
     * @param t time
     * @param c output C(t)
     * @param cd output C'(t)
     * @param cdd output C''(t)
     * @throws std::out_of_range if t is outside the range of a tabulated kind
     */
    void eval(double t, double & c, double & cd, double & cdd) const;

    /// Set the value of a Parameter kind. Throws std::logic_error for other kinds.
    void set_value(double value);

    Kind kind() const { return m_kind; }

private:
    Kind m_kind;
    std::vector<double> m_coeffs; // polynomial coefficients
    double m_amplitude = 0.0;
    double m_omega = 0.0;
    double m_phase = 0.0;
    double m_offset = 0.0; // also the value of the Parameter kind
    std::vector<double> m_times;
    std::vector<double> m_values;
    std::vector<double> m_second; // spline second derivatives at knots

    static void check_table(const std::vector<double> & times, const std::vector<double> & values, std::size_t min_points);
    std::size_t find_segment(double t) const;
};

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
