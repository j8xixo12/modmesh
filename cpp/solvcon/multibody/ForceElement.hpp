#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Force elements between two bodies (Haug Sec. 11.4, 12.5).  Every element
 * reports a global force at the center of mass and a global torque about the
 * center of mass of each body.
 *
 * @ingroup group_numerics
 */

#include <solvcon/multibody/Body.hpp>
#include <solvcon/multibody/DriveFunction.hpp>
#include <solvcon/multibody/kinematics.hpp>

#include <solvcon/buffer/buffer.hpp>

#include <memory>
#include <vector>

namespace solvcon
{

namespace multibody
{

/**
 * Translational spring-damper-actuator (Haug Sec. 11.4.1).  The tension
 * f = k (L - L0) + c dL/dt + F_actuator pulls the two points together for
 * f > 0.
 */
class ForceElement
{

public:

    /**
     * @param body_i, body_j  The connected bodies.
     * @param point_i, point_j  Attachment points in the body frames.
     * @param k  Stiffness.
     * @param c  Damping coefficient.
     * @param free_length  L0.
     * @param actuator_force  Constant actuator tension.
     */
    ForceElement(BodyPtr body_i, BodyPtr body_j, Vector3 const & point_i, Vector3 const & point_j, double k, double c, double free_length, double actuator_force = 0.0);
    ForceElement(ForceElement const &) = delete;
    ForceElement(ForceElement &&) = delete;
    ForceElement & operator=(ForceElement const &) = delete;
    ForceElement & operator=(ForceElement &&) = delete;
    virtual ~ForceElement() = default;

    /// Compute the loads from the current body states.
    virtual void compute();
    /// Time for the actuator functions; the system sets it before compute().
    void set_time(double t) { m_time = t; }

    Vector3 const & force_i() const { return m_force_i; }
    Vector3 const & force_j() const { return m_force_j; }
    Vector3 const & torque_i() const { return m_torque_i; }
    Vector3 const & torque_j() const { return m_torque_j; }
    /// Global attachment points (for force arrows in a viewer).
    virtual Vector3 point_i() const;
    virtual Vector3 point_j() const;
    BodyPtr const & body_i() const { return m_body_i; }
    BodyPtr const & body_j() const { return m_body_j; }

protected:

    double m_time = 0.0;
    BodyPtr m_body_i;
    BodyPtr m_body_j;
    Vector3 m_point_i; ///< Body frame.
    Vector3 m_point_j; ///< Body frame.
    double m_k;
    double m_c;
    double m_free_length;
    double m_actuator_force;
    Vector3 m_force_i;
    Vector3 m_force_j;
    Vector3 m_torque_i;
    Vector3 m_torque_j;

}; /* end class ForceElement */

using ForceElementPtr = std::shared_ptr<ForceElement>;

/**
 * Rotational spring-damper-actuator (Haug Sec. 11.4.2, Eq. 11.4.12-11.4.16).
 * theta is the cumulative angle from f_i (perpendicular to h_i on body i) to
 * f_j (perpendicular to h_j on body j) about h_i, theta0 its value at
 * construction.  n = k (theta - theta0) + c theta_dot - T(t), and the virtual
 * work is -n d theta.
 */
class RotationalSDA
    : public ForceElement
{

public:

    /**
     * @param axis_i, axis_j  Rotation axes h_i and h_j in the body frames.
     * @param torque  Actuator torque T(t) on body j.
     */
    RotationalSDA(BodyPtr body_i, BodyPtr body_j, Vector3 const & axis_i, Vector3 const & axis_j, double k, double c, DriveFunction torque);
    void compute() override;
    double angle() const { return m_theta - m_theta0; }

private:

    void frames(Vector3 & fi, Vector3 & gi, Vector3 & fj) const;

    Vector3 m_fi; ///< Body frame of body i.
    Vector3 m_gi; ///< Body frame of body i.
    Vector3 m_fj; ///< Body frame of body j.
    DriveFunction m_torque_fn;
    double m_theta = 0.0;
    double m_theta0 = 0.0;

}; /* end class RotationalSDA */

/**
 * Linear 6-by-6 bushing (or massless beam) between body i and body j.
 *
 * The bushing frame B sits on body i at point_i with the body-frame
 * orientation R_B.  The deformation in frame B is zero at the construction
 * pose:
 *   delta = R_B^T A_i^T (r_j^P - r_i^P) - delta0,
 *   theta = log(R0^T R_B^T A_i^T A_j) (rotation vector of the relative rotation).
 * The load on body j at P_j in frame B is [F; T] = -K [delta; theta] - C
 * [delta_dot; omega_rel], and body i takes the equal and opposite load.  The
 * rotational part is exactly conservative for an isotropic rotational K;
 * otherwise it is the small-angle bushing.
 */
class Bushing
    : public ForceElement
{

public:

    /**
     * @param stiffness  6-by-6 K on [dx, dy, dz, rx, ry, rz].
     * @param damping  6-by-6 C on the deformation rates.
     * @param frame  R_B, the bushing frame in the body frame of body i.
     * @throws std::invalid_argument if K or C is not 6-by-6.
     */
    Bushing(BodyPtr body_i, BodyPtr body_j, Vector3 const & point_i, Vector3 const & point_j, SimpleArray<double> const & stiffness, SimpleArray<double> const & damping, Matrix3 const & frame);
    void compute() override;
    /// Translational deformation delta in frame B.
    Vector3 const & translation() const { return m_delta; }
    /// Rotational deformation theta in frame B.
    Vector3 const & rotation() const { return m_rotation; }

private:

    void relative(Vector3 & delta, Vector3 & theta, Matrix3 & r_rel) const;

    SimpleArray<double> m_stiffness;
    SimpleArray<double> m_damping;
    Matrix3 m_frame;
    Matrix3 m_r0 = Matrix3::identity();
    Vector3 m_delta0;
    Vector3 m_delta;
    Vector3 m_rotation;

}; /* end class Bushing */

/**
 * Spring-damper whose damper force follows a force-velocity curve (a
 * measured or specified damper characteristic).  With L the length and v =
 * dL/dt (v > 0 extends the damper: rebound), the tension is
 *
 *   f = k (L - L0) + F(v),
 *
 * where F is the piecewise-linear curve through the table points (v_n, F_n)
 * and extends past the first and the last point with the slope of the end
 * segment.  F(v) > 0 is tension (rebound), F(v) < 0 compression (bump).
 * The table needs at least two points, strictly increasing velocities, and a
 * passive curve: F(v) v >= 0 at every point.
 */
class CurveDamper
    : public ForceElement
{

public:

    /**
     * @param velocities  Table velocities v_n (m/s), strictly increasing.
     * @param forces  Table forces F_n (N), tension positive.
     * @throws std::invalid_argument for a bad table.
     */
    CurveDamper(BodyPtr body_i, BodyPtr body_j, Vector3 const & point_i, Vector3 const & point_j, double k, double free_length, std::vector<double> velocities, std::vector<double> forces);
    void compute() override;
    /// The damper force F(v) of the curve (N, tension positive).
    double damper_force(double velocity) const;
    /// Length L (m) and rate dL/dt (m/s) at the last evaluation.
    double length() const { return m_length; }
    double length_rate() const { return m_length_rate; }
    /// Damper force (N) at the last evaluation.
    double last_damper_force() const { return m_damper_force; }
    std::vector<double> const & velocities() const { return m_velocities; }
    std::vector<double> const & forces() const { return m_forces; }

private:

    std::vector<double> m_velocities;
    std::vector<double> m_forces;
    double m_length = 0.0;
    double m_length_rate = 0.0;
    double m_damper_force = 0.0;

}; /* end class CurveDamper */

/**
 * One-sided travel stop along the line between two points (a bump stop and
 * a droop stop on the damper line).  With L the length and d the
 * penetration past a limit (d = L_min - L below L_min, d = L - L_max above
 * L_max), the stop force is f = max(0, k d + k2 d^2 + c dd/dt): it pushes
 * the points apart below L_min and pulls them together above L_max, and it
 * is zero between the limits.  A limit that is not finite is disabled.
 */
class LengthStop
    : public ForceElement
{

public:

    /**
     * @param min_length  L_min (bump stop contact); -inf disables it.
     * @param max_length  L_max (droop stop contact); +inf disables it.
     * @param k  Linear rate (N/m).
     * @param k2  Quadratic rate (N/m^2) for a progressive stop.
     * @param c  Damping (N.s/m) while engaged.
     * @throws std::invalid_argument if a rate is negative or min_length >= max_length.
     */
    LengthStop(BodyPtr body_i, BodyPtr body_j, Vector3 const & point_i, Vector3 const & point_j, double min_length, double max_length, double k, double k2, double c);
    void compute() override;
    /// Current length L (m).
    double length() const { return m_length; }
    /// Penetration past the engaged limit (m); 0 between the limits.
    double penetration() const { return m_penetration; }
    /// Stop force magnitude (N) at the last evaluation.
    double stop_force() const { return m_stop_force; }
    /// -1 at the bump stop (L < L_min), +1 at the droop stop (L > L_max), else 0.
    int engaged() const { return m_engaged; }
    double min_length() const { return m_min_length; }
    double max_length() const { return m_max_length; }

private:

    double m_min_length;
    double m_max_length;
    double m_k2;
    double m_length = 0.0;
    double m_penetration = 0.0;
    double m_stop_force = 0.0;
    int m_engaged = 0;

}; /* end class LengthStop */

/**
 * Point-contact tire on a flat road (Haug Sec. 12.5, Eqs. 12.5.1-12.5.3).
 *
 * The wheel center r_w is the origin of body j (the road, body i, is unused);
 * the spin axis is s = A_j s'.  The contact point is c = r_w + R d with d the
 * unit vector in the wheel plane that points down.  The penetration is delta
 * = road_z - c_z (> 0 in contact).  The vertical force F_z = max(0, k delta -
 * c dz/dt) acts along +z at c.  The lateral force along l has the slip angle
 * alpha = atan2(v_c . l, |v_c . h|), F_lat = -C_alpha alpha, |F_lat| <= mu F_z.
 *
 * Options (set_slip_model(), all off by default):
 *
 * - Load sensitivity with the reference load F_z0 > 0 and the exponent n:
 *   C_alpha(F_z) = C_alpha (F_z/F_z0)^n, mu(F_z) = mu (F_z/F_z0)^(n-1), the
 *   AC tyre form; c_alpha and mu are the values at F_z0.
 * - The brush (Fiala) side force (Pacejka, Tire and Vehicle Dynamics, 2012,
 *   Sec. 3.2.2) in place of the clipped line: with s = C_alpha tan(alpha)
 *   and D = mu_y F_z, F = -(s - s|s|/(3D) + s^3/(27D^2)) up to
 *   |s| = 3D, then -D sign(s).
 * - The relaxation length sigma > 0: the slip angle of the force lags the
 *   kinematic one, d(alpha')/dx = (alpha - alpha')/sigma over the rolled
 *   distance x.  The lag is exact for a constant alpha over one step from
 *   the value that commit() stores after each accepted step.
 * - A longitudinal force demand ratio r (set_longitudinal()): F_x = r F_z,
 *   at most mu F_z, along the rolling direction; the side force keeps
 *   mu_y F_z = sqrt((mu F_z)^2 - F_x^2) (the friction circle).
 */
class TireForce
    : public ForceElement
{

public:

    /// spin_axis is s' in the wheel frame.
    TireForce(BodyPtr road, BodyPtr wheel, Vector3 const & spin_axis, double radius, double k, double c, double c_alpha, double mu, double road_z);
    /**
     * The road is the plane through (x_w, y_w, h(t)) with the normal
     * (-sin b, 0, cos b), b(t) the forward slope: an effective road plane
     * (enveloping models) that moves under the wheel.  The force acts along
     * the plane normal, only in compression.
     */
    void set_road(DriveFunction height, DriveFunction slope);
    /// Load sensitivity (fz0 > 0 turns it on), brush curve, relaxation.
    void set_slip_model(double fz0, double load_exponent, bool brush, double relaxation_length);
    /// Longitudinal force demand F_x / F_z (+ drive, - brake).
    void set_longitudinal(double ratio) { m_fx_ratio = ratio; }
    /// Store the lagged slip angle at the end of an accepted step.
    void commit();
    void compute() override;
    Vector3 point_j() const override { return m_contact; }
    double vertical_force() const { return m_fz; }
    double slip_angle() const { return m_alpha; }
    /// The lagged slip angle of the side force (rad).
    double force_slip_angle() const { return m_alpha_force; }
    double lateral_force() const { return m_fy; }
    double longitudinal_force() const { return m_fx; }
    /// Normal deflection (m); negative when the tire is off the road.
    double deflection() const { return m_deflection; }
    bool in_contact() const { return m_fz > 0.0; }

private:

    Vector3 m_spin; ///< Wheel frame, unit.
    double m_radius;
    double m_c_alpha;
    double m_mu;
    double m_road_z;
    Vector3 m_contact;
    double m_fz = 0.0;
    double m_alpha = 0.0;
    double m_alpha_force = 0.0;
    double m_fy = 0.0;
    double m_fx = 0.0;
    double m_vh = 0.0;
    double m_deflection = 0.0;
    double m_fz0 = 0.0;
    double m_load_exponent = 1.0;
    bool m_brush = false;
    double m_relaxation = 0.0;
    double m_fx_ratio = 0.0;
    bool m_committed = false;
    double m_alpha_committed = 0.0;
    double m_time_committed = 0.0;
    bool m_moving_road = false;
    DriveFunction m_road_height = DriveFunction::parameter(0.0);
    DriveFunction m_road_slope = DriveFunction::parameter(0.0);

}; /* end class TireForce */

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
