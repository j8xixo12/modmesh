/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/multibody/Body.hpp>

#include <stdexcept>
#include <string>

namespace solvcon
{

namespace multibody
{

void Body::set_tbi(Matrix3 const & tbi)
{
    m_tbi = tbi;
    m_quaternion = quaternion_from_tbi(m_tbi);
}

void Body::set_applied_load(Vector3 const & force, Vector3 const & torque)
{
    m_force = force;
    m_applied_torque = m_tbi * torque;
}

Ground::Ground(uint32_t num)
    : Body(num)
{
    m_tbi = Matrix3::identity();
    m_quaternion = Quaternion::identity();
}

MobilizedBody::MobilizedBody(uint32_t num, Vector3 const & position, Vector3 const & velocity, Vector3 const & acceleration, Vector3 const & angles, Vector3 const & angular_velocity, Vector3 const & angular_acceleration, double mass, Matrix3 const & inertia, Vector3 const & force, Vector3 const & torque)
    : Body(num)
{
    m_position = position;
    m_velocity = velocity;
    m_acceleration = acceleration;
    m_angles = angles;
    m_angular_velocity = angular_velocity;
    m_angular_acceleration = angular_acceleration;
    m_force = force;
    m_applied_torque = torque;
    m_inertia = inertia;
    m_mass = mass;

    m_tbi = tbi_from_euler_angles(m_angles[2], m_angles[1], m_angles[0]);
    m_quaternion = quaternion_from_tbi(m_tbi);
    m_p_dot = 0.5 * transpose_mul(build_E(m_quaternion), angular_velocity);
    m_rotational_mass = congruence(build_G(m_quaternion), m_inertia);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            m_rotational_mass(i, j) *= 4.0;
        }
    }
}

Matrix3 MobilizedBody::inertia_tensor(std::span<double const> values)
{
    if (values.size() == 3)
    {
        return Matrix3::diagonal(Vector3(values[0], values[1], values[2]));
    }
    if (values.size() == 6)
    {
        return Matrix3(values[0], values[3], values[4], values[3], values[1], values[5], values[4], values[5], values[2]);
    }
    throw std::invalid_argument("MobilizedBody: inertia needs 3 or 6 values, got " + std::to_string(values.size()));
}

void MobilizedBody::update(Vector3 const & r, Vector3 const & v, Quaternion const & p, Quaternion const & p_dot)
{
    m_tbi = tbi_from_quaternion(p);
    m_quaternion = p;
    m_p_dot = p_dot;
    m_position = r;
    m_velocity = v;
    m_angles = euler_angles(m_tbi);
    m_angular_velocity = 2.0 * (build_E(p) * p_dot);

    // The mass matrix depends on p through G (Haug Eq. 11.1.7).
    Matrix34 const g = build_G(p);
    m_rotational_mass = congruence(g, m_inertia);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            m_rotational_mass(i, j) *= 4.0;
        }
    }

    // Q_p = 2 G^T n' with n' = n_applied - omega' x (J' omega') (Haug Eq. 11.1.14).
    Vector3 const omega_body = 2.0 * (g * p_dot);
    Vector3 const n_prime = m_applied_torque - cross(omega_body, m_inertia * omega_body);
    m_torque = 2.0 * transpose_mul(g, n_prime);
}

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
