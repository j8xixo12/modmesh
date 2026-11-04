#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Rotation parameterizations and the Euler-parameter matrices of Haug Sec. 9.3.
 *
 * TBI is the inertial-to-body rotation matrix, the transpose of Haug's A
 * (body-to-global).  p = [e0, e1, e2, e3] are the Euler parameters of A.
 *
 * @ingroup group_numerics
 */

#include <solvcon/multibody/kinematics.hpp>

namespace solvcon
{

namespace multibody
{

/// TBI from the yaw psi, pitch tht, and roll phi angles [rad] (z-y-x sequence).
Matrix3 tbi_from_euler_angles(double psi, double tht, double phi);

/// Euler parameters of A = TBI^T (Haug Eq. 9.3.12-9.3.16).
Quaternion quaternion_from_tbi(Matrix3 const & tbi);

/// TBI = A(p)^T for unit Euler parameters p.
Matrix3 tbi_from_quaternion(Quaternion const & p);

/// Roll, pitch, and yaw angles [rad] of TBI, in this order.
Vector3 euler_angles(Matrix3 const & tbi);

/**
 * G = [-e, -e~ + e0 I] (Haug Eq. 9.3.19); the body-frame angular velocity is
 * omega' = 2 G p_dot.  G is linear in p, so G(p_dot) is its time derivative.
 */
Matrix34 build_G(Quaternion const & p);

/**
 * E = [-e, e~ + e0 I]; the global angular velocity is omega = 2 E p_dot and
 * A = E G^T.  For p^T p_dot = 0, p_dot = 0.5 E^T omega.  E is linear in p, so
 * E(p_dot) is its time derivative.  The relation d(A s')/dt = (2 E p_dot) x
 * (A s') is checked numerically by the finite-difference tests.
 */
Matrix34 build_E(Quaternion const & p);

/**
 * d(A(p) s')/dp (Haug Eq. 9.3.30), the full derivative used by the
 * position-level Newton iteration (not restricted to the tangent space of
 * p^T p = 1).
 */
Matrix34 build_Abar(Quaternion const & p, Vector3 const & s);

/**
 * Unit vectors f and g with (axis, f, g) right-handed and orthonormal for a
 * unit axis: f = normalized(axis x ref), g = axis x f, where ref is the x axis,
 * or the y axis if |axis . x| > 0.9.
 */
void perpendicular_frame(Vector3 const & axis, Vector3 & f, Vector3 & g);

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
