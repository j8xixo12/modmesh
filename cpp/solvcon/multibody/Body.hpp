#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Rigid bodies in the Euler-parameter formulation (Haug Sec. 11.1): 7
 * generalized coordinates [r, p] per body.
 *
 * @ingroup group_numerics
 */

#include <solvcon/multibody/Math.hpp>
#include <solvcon/multibody/kinematics.hpp>

#include <cstdint>
#include <memory>
#include <span>

namespace solvcon
{

namespace multibody
{

/**
 * State and mass properties of one body.  Vectors are in the global frame
 * unless stated otherwise.
 */
class Body
    : public std::enable_shared_from_this<Body>
{

public:

    Body(Body const &) = delete;
    Body(Body &&) = delete;
    Body & operator=(Body const &) = delete;
    Body & operator=(Body &&) = delete;
    virtual ~Body() = default;

    /// Body number; it is the block index of the body in the system arrays.
    uint32_t num() const { return m_num; }

    Vector3 const & position() const { return m_position; }
    Vector3 const & velocity() const { return m_velocity; }
    Vector3 const & acceleration() const { return m_acceleration; }
    /// Roll, pitch, and yaw [rad].
    Vector3 const & angles() const { return m_angles; }
    Vector3 const & angular_velocity() const { return m_angular_velocity; }
    Vector3 const & angular_acceleration() const { return m_angular_acceleration; }
    /// Applied force at the center of mass.
    Vector3 const & force() const { return m_force; }
    /// Generalized force conjugate to p: 2 G^T n' (Haug Eq. 11.1.14).
    Quaternion const & torque() const { return m_torque; }
    /// Inertial-to-body rotation matrix (Haug's A^T).
    Matrix3 const & tbi() const { return m_tbi; }
    /// Euler parameters p.
    Quaternion const & quaternion() const { return m_quaternion; }
    Quaternion const & p_dot() const { return m_p_dot; }
    /// Translational mass (1 for the ground).
    double mass() const { return m_mass; }
    /// Rotational mass block 4 G^T J' G of the 7-by-7 mass matrix (Haug Eq. 11.1.7).
    Matrix4 const & rotational_mass() const { return m_rotational_mass; }

    void set_position(Vector3 const & v) { m_position = v; }
    void set_acceleration(Vector3 const & v) { m_acceleration = v; }
    void set_angular_acceleration(Vector3 const & v) { m_angular_acceleration = v; }
    void set_p_dot(Quaternion const & v) { m_p_dot = v; }
    /// Set the rotation and the Euler parameters derived from it.
    void set_tbi(Matrix3 const & tbi);
    /// Applied load: force at the center of mass and torque, both global.
    void set_applied_load(Vector3 const & force, Vector3 const & torque);

    /// Set the state; the angular velocity and the mass matrix follow from p and p_dot.
    virtual void update(Vector3 const & r, Vector3 const & v, Quaternion const & p, Quaternion const & p_dot) = 0;

protected:

    explicit Body(uint32_t num)
        : m_num(num)
    {
    }

    uint32_t m_num = 0;
    Vector3 m_position;
    Vector3 m_velocity;
    Vector3 m_acceleration;
    Vector3 m_angles;
    Vector3 m_angular_velocity;
    Vector3 m_angular_acceleration;
    Vector3 m_force;
    Quaternion m_torque;
    Vector3 m_applied_torque; ///< Body frame.
    Matrix3 m_tbi;
    Quaternion m_quaternion;
    Quaternion m_p_dot;
    double m_mass = 1.0;
    Matrix4 m_rotational_mass = Matrix4::identity();
    Matrix3 m_inertia; ///< Inertia tensor about the center of mass, body frame (Haug Sec. 11.2).

}; /* end class Body */

/// The fixed body: r = 0 and p = [1, 0, 0, 0] are imposed by constraints.
class Ground
    : public Body
{

public:

    explicit Ground(uint32_t num);

    void update(Vector3 const &, Vector3 const &, Quaternion const &, Quaternion const &) override {}

}; /* end class Ground */

class MobilizedBody
    : public Body
{

public:

    /**
     * @param num  Body number.
     * @param position  Center of mass.
     * @param velocity  Velocity of the center of mass.
     * @param acceleration  Initial acceleration.
     * @param angles  Roll, pitch, and yaw [rad].
     * @param angular_velocity  Global angular velocity.
     * @param angular_acceleration  Initial angular acceleration.
     * @param mass  Mass.
     * @param inertia  Inertia tensor about the center of mass in the body frame.
     * @param force  Applied force at the center of mass (global).
     * @param torque  Applied torque (body frame).
     */
    MobilizedBody(uint32_t num, Vector3 const & position, Vector3 const & velocity, Vector3 const & acceleration, Vector3 const & angles, Vector3 const & angular_velocity, Vector3 const & angular_acceleration, double mass, Matrix3 const & inertia, Vector3 const & force, Vector3 const & torque);

    /**
     * Inertia tensor from [Ixx, Iyy, Izz] (principal axes) or [Ixx, Iyy, Izz,
     * Ixy, Ixz, Iyz].
     *
     * @throws std::invalid_argument for another number of values.
     */
    static Matrix3 inertia_tensor(std::span<double const> values);

    void update(Vector3 const & r, Vector3 const & v, Quaternion const & p, Quaternion const & p_dot) override;

}; /* end class MobilizedBody */

using BodyPtr = std::shared_ptr<Body>;

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
