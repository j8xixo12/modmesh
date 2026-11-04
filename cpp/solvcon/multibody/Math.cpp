/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/multibody/Math.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>

namespace solvcon
{

namespace multibody
{

namespace
{

double sign_of(double v) { return (v < 0.0) ? -1.0 : 1.0; }

} /* end namespace */

Matrix3 tbi_from_euler_angles(double psi, double tht, double phi)
{
    double const spsi = std::sin(psi), cpsi = std::cos(psi);
    double const stht = std::sin(tht), ctht = std::cos(tht);
    double const sphi = std::sin(phi), cphi = std::cos(phi);
    return Matrix3(cpsi * ctht,
                   spsi * ctht,
                   -stht,
                   cpsi * stht * sphi - spsi * cphi,
                   spsi * stht * sphi + cpsi * cphi,
                   ctht * sphi,
                   cpsi * stht * cphi + spsi * sphi,
                   spsi * stht * cphi - cpsi * sphi,
                   ctht * cphi);
}

Quaternion quaternion_from_tbi(Matrix3 const & tbi)
{
    Matrix3 const a = tbi.transpose();
    double const q_square[4] = {// NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
                                std::fabs(1.0 + a(0, 0) + a(1, 1) + a(2, 2)),
                                std::fabs(1.0 + a(0, 0) - a(1, 1) - a(2, 2)),
                                std::fabs(1.0 - a(0, 0) + a(1, 1) - a(2, 2)),
                                std::fabs(1.0 - a(0, 0) - a(1, 1) + a(2, 2))};
    // The largest of the four squares gives the best-conditioned division.
    auto const j = static_cast<size_t>(std::max_element(q_square, q_square + 4) - q_square);

    Quaternion p;
    p[j] = 0.5 * std::sqrt(q_square[j]);
    switch (j)
    {
    case 0:
        p[1] = 0.25 * (a(2, 1) - a(1, 2)) / p[0];
        p[2] = 0.25 * (a(0, 2) - a(2, 0)) / p[0];
        p[3] = 0.25 * (a(1, 0) - a(0, 1)) / p[0];
        break;
    case 1:
        p[0] = 0.25 * (a(2, 1) - a(1, 2)) / p[1];
        p[2] = 0.25 * (a(1, 0) + a(0, 1)) / p[1];
        p[3] = 0.25 * (a(0, 2) + a(2, 0)) / p[1];
        break;
    case 2:
        p[0] = 0.25 * (a(0, 2) - a(2, 0)) / p[2];
        p[1] = 0.25 * (a(1, 0) + a(0, 1)) / p[2];
        p[3] = 0.25 * (a(2, 1) + a(1, 2)) / p[2];
        break;
    default:
        p[0] = 0.25 * (a(1, 0) - a(0, 1)) / p[3];
        p[1] = 0.25 * (a(2, 0) + a(0, 2)) / p[3];
        p[2] = 0.25 * (a(2, 1) + a(1, 2)) / p[3];
        break;
    }
    return p;
}

Matrix3 tbi_from_quaternion(Quaternion const & p)
{
    return Matrix3(2. * (p[0] * p[0] + p[1] * p[1]) - 1.,
                   2. * (p[1] * p[2] + p[0] * p[3]),
                   2. * (p[1] * p[3] - p[0] * p[2]),
                   2. * (p[1] * p[2] - p[0] * p[3]),
                   2. * (p[0] * p[0] + p[2] * p[2]) - 1.,
                   2. * (p[2] * p[3] + p[0] * p[1]),
                   2. * (p[1] * p[3] + p[0] * p[2]),
                   2. * (p[2] * p[3] - p[0] * p[1]),
                   2. * (p[0] * p[0] + p[3] * p[3]) - 1.);
}

Vector3 euler_angles(Matrix3 const & tbi)
{
    // Near |pitch| = 90 deg, cos(pitch) is replaced by a small positive number
    // so that the yaw and roll divisions stay finite.
    constexpr double cos_floor = 1.e-10;
    double const tbd13 = tbi(0, 2);
    double tht = 0.0, ctht = 0.0;
    if (std::fabs(tbd13) < 1)
    {
        tht = std::asin(-tbd13);
        ctht = std::cos(tht);
    }
    else
    {
        tht = std::numbers::pi / 2 * sign_of(-tbd13);
        ctht = cos_floor;
    }
    double const cpsi = std::clamp(tbi(0, 0) / ctht, -1.0, 1.0);
    double const psi = std::acos(cpsi) * sign_of(tbi(0, 1));
    double const cphi = std::clamp(tbi(2, 2) / ctht, -1.0, 1.0);
    double const phi = std::acos(cphi) * sign_of(tbi(1, 2));
    return Vector3(phi, tht, psi);
}

Matrix34 build_G(Quaternion const & p)
{
    double const e0 = p[0], e1 = p[1], e2 = p[2], e3 = p[3];
    Matrix34 g;
    g(0, 0) = -e1;
    g(0, 1) = e0;
    g(0, 2) = e3;
    g(0, 3) = -e2;
    g(1, 0) = -e2;
    g(1, 1) = -e3;
    g(1, 2) = e0;
    g(1, 3) = e1;
    g(2, 0) = -e3;
    g(2, 1) = e2;
    g(2, 2) = -e1;
    g(2, 3) = e0;
    return g;
}

Matrix34 build_E(Quaternion const & p)
{
    double const e0 = p[0], e1 = p[1], e2 = p[2], e3 = p[3];
    Matrix34 e;
    e(0, 0) = -e1;
    e(0, 1) = e0;
    e(0, 2) = -e3;
    e(0, 3) = e2;
    e(1, 0) = -e2;
    e(1, 1) = e3;
    e(1, 2) = e0;
    e(1, 3) = -e1;
    e(2, 0) = -e3;
    e(2, 1) = -e2;
    e(2, 2) = e1;
    e(2, 3) = e0;
    return e;
}

Matrix34 build_Abar(Quaternion const & p, Vector3 const & s)
{
    // A s' = (2 e0^2 - 1) s' + 2 (e . s') e + 2 e0 (e x s'), e = [e1, e2, e3]:
    // d/de0 = 4 e0 s' + 2 e x s', d/de = 2 ((e . s') I + e s'^T - e0 [s']x).
    double const e0 = p[0];
    Vector3 const e = p.vector_part();
    Matrix34 out;
    out.set_column(0, 4.0 * e0 * s + 2.0 * cross(e, s));
    Matrix3 const dade = 2.0 * (dot(e, s) * Matrix3::identity() + outer(e, s) - e0 * skew(s));
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            out(i, j + 1) = dade(i, j);
        }
    }
    return out;
}

void perpendicular_frame(Vector3 const & axis, Vector3 & f, Vector3 & g)
{
    Vector3 ref(1.0, 0.0, 0.0);
    if (std::fabs(dot(axis, ref)) > 0.9)
    {
        ref = Vector3(0.0, 1.0, 0.0);
    }
    f = normalized(cross(axis, ref));
    g = cross(axis, f);
}

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
