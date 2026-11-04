#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Kinematic joints, drivers, and cam followers between two bodies (Haug Ch. 9
 * and Sec. 3.4).
 *
 * @ingroup group_numerics
 */

#include <solvcon/multibody/Body.hpp>
#include <solvcon/multibody/CamProfile.hpp>
#include <solvcon/multibody/DriveFunction.hpp>
#include <solvcon/multibody/Math.hpp>
#include <solvcon/multibody/kinematics.hpp>

#include <solvcon/buffer/buffer.hpp>

#include <cstdint>
#include <memory>

namespace solvcon
{

namespace multibody
{

/**
 * Constraint rows Phi(q, t) = 0 between body i and body j.
 *
 * Basic joints (number of rows in parentheses): 0 spherical (3), 1 revolute
 * (5), 2 prismatic (5), 3 rotational driving (1), 4 universal (4), 5
 * cylindrical (4), 6 distance (1), 7 revolute-cylindrical (3).
 *
 * Drivers (Haug Sec. 9.5), one row each, with C(t) from the drive function:
 * - 8 absolute coordinate: u . (r_i^P - r_j^P) - C(t), u = a_i (global, fixed);
 * - 9 distance: 0.5 (d . d - C(t)^2);
 * - 10 relative translation: h_i . (r_j^P - r_i^P) - C(t), h_i = A_i a_i;
 * - 11 relative rotation: wrap(theta - C(t)), theta from f_i to f_j about h_i.
 *
 * Composite joints (Haug Sec. 9.4.4-9.4.5) from basic constraint rows: 12
 * screw (5, cylindrical plus h_i . d_ij - s0 - pitch (theta - theta0), pitch =
 * drive_omega), 13 revolute-spherical (2), 14 revolute-revolute parallel (4),
 * 15 revolute-revolute orthogonal (4), 16 revolute-translational (4, f_j =
 * q_j), 17 strut (2).  The coupler length is the distance argument, or the
 * construction pose if it is not positive.
 *
 * Cam followers (Haug Sec. 3.4.2-3.4.3) in the plane of cam body i (normal
 * a_i, center p_i): 18 point follower (2), point p_j of body j on the polar
 * profile R(phi) of body i; 19 flat-face follower (1), the face of body j
 * (point p_j, normal a_j toward the cam) tangent to the convex cam with the
 * support function h(psi).
 *
 * The Jacobian blocks are nc-by-7 on [r, p] of each body.  The velocity form
 * is [Phi_r, 2 Phi_pi E] with the global virtual rotation pi (Haug Sec. 9.6);
 * the position form is the full derivative dPhi/d[r, p] for Newton-Raphson.
 */
class Joint
    : public std::enable_shared_from_this<Joint>
{

public:

    /**
     * @param type  Joint type (see the class description).
     * @param pi, pj  Joint points in the body frames of body i and body j.
     * @param qi, qj  Additional points or directions in the body frames (q_j
     *      is f_j of the revolute-translational joint).
     * @param ai, aj  Joint axes in the body frames.
     * @param drive_omega  Angular velocity of type 3, or the screw pitch.
     * @param distance  Length of the distance and composite joints.
     * @param drive  C(t) of the drivers.
     * @param profile  Cam profile of the followers.
     */
    Joint(uint32_t type, Vector3 const & pi, Vector3 const & pj, Vector3 const & qi, Vector3 const & qj, Vector3 const & ai, Vector3 const & aj, BodyPtr body_i, BodyPtr body_j, double drive_omega = 0.0, double distance = 0.0, DriveFunction drive = DriveFunction(), CamProfile profile = CamProfile());
    Joint(Joint const &) = delete;
    Joint(Joint &&) = delete;
    Joint & operator=(Joint const &) = delete;
    Joint & operator=(Joint &&) = delete;
    ~Joint() = default;

    /**
     * Natural coordinate g of the joint and its derivatives (for gear couplers).
     * Kind 0: relative angle from f_i to f_j about h_i (wrapped to (-pi, pi]);
     * kind 1: translation h_i . (r_j^P - r_i^P).
     */
    struct Measure
    {
        double g = 0.0;
        /// Velocity form: dg = ri . dr_i + wi . dpi_i + rj . dr_j + wj . dpi_j.
        Vector3 ri, wi, rj, wj;
        /// Position form: dg/dp_i and dg/dp_j.
        Quaternion pi_pos, pj_pos;
        /// g_ddot = J q_ddot + gamma_part.
        double gamma_part = 0.0;
    }; /* end struct Measure */

    Measure measure(int kind) const;

    /// Evaluate the rows at the current body states and time.
    void update();
    void set_time(double t) { m_time = t; }

    uint32_t type() const { return m_type; }
    uint32_t nconstraints() const { return m_nc; }
    double drive_omega() const { return m_drive_omega; }
    bool is_driver() const { return m_type >= 8 && m_type <= 11; }
    /// dPhi/dC of the drivers 8-11 (for sweep sensitivities).  Throws for other types.
    double drive_sensitivity() const;
    DriveFunction & drive() { return m_drive; }
    BodyPtr const & body_i() const { return m_body_i; }
    BodyPtr const & body_j() const { return m_body_j; }

    /// Joint points in the body frames.
    Vector3 const & point_i() const { return m_pi; }
    Vector3 const & point_j() const { return m_pj; }
    /// A_i p_i and A_j p_j: global arms from the centers of mass to the joint points.
    Vector3 const & arm_i() const { return m_arm_i; }
    Vector3 const & arm_j() const { return m_arm_j; }

    /// @name Rows at the last update(); vectors have nc elements.
    ///@{
    SimpleArray<double> const & constraint() const { return m_constraint; }
    SimpleArray<double> const & gamma() const { return m_gamma; }
    /// Explicit time derivative Phi_t.
    SimpleArray<double> const & ct() const { return m_ct; }
    /// Velocity-form Jacobian blocks (nc-by-7).
    SimpleArray<double> const & jacobian_i() const { return m_cq_i; }
    SimpleArray<double> const & jacobian_j() const { return m_cq_j; }
    /// Angular Jacobian blocks in the global virtual rotation (nc-by-3).
    SimpleArray<double> const & jacobian_omega_i() const { return m_cq_omega_i; }
    SimpleArray<double> const & jacobian_omega_j() const { return m_cq_omega_j; }
    /// Position-form Jacobian blocks (nc-by-7), computed by the call.
    SimpleArray<double> const & position_jacobian_i();
    SimpleArray<double> const & position_jacobian_j();
    ///@}

private:

    // One basic constraint row of a composite joint (types 12-17).
    struct Prim
    {
        enum class Kind
        {
            Dot1,
            Dot2,
            Dist,
            Screw
        };
        Kind kind = Kind::Dot1;
        Vector3 x_l; ///< Body-fixed vector in the frame of body xb.
        int32_t xb = 0; ///< 0 for body i, 1 for body j.
        Vector3 y_l; ///< Second vector of Dot1, in the frame of body yb.
        int32_t yb = 0;
    }; /* end struct Prim */

    void build_constraint();
    void build_jacobian();
    void build_gamma();
    void build_ct();

    void build_composite();
    void composite_init();
    void composite_constraint();
    void composite_jacobian();
    void composite_gamma(Vector3 const & d_dot, Vector3 const & gamma_pos);
    void composite_position(int32_t body, SimpleArray<double> & out) const;

    void follower_constraint();
    void follower_jacobian();
    void follower_gamma(Vector3 const & d_dot, Vector3 const & gamma_pos);
    void follower_position(int32_t body, SimpleArray<double> & out) const;

    /**
     * Relative angle theta = atan2(g_i . f_j, f_i . f_j) from f_i to f_j about
     * h_i.  d theta = (dpi_i - dpi_j) . w, and the acceleration part is
     * gamma_part = -(w_i - w_j) . w_dot.
     */
    void rel_angle(double & theta, Vector3 & w, double & gamma_part) const;
    void rel_angle_pos(Quaternion & dth_dpi, Quaternion & dth_dpj) const;

    Vector3 global_i(Vector3 const & local) const { return transpose_mul(m_tbi_i, local); }
    Vector3 global_j(Vector3 const & local) const { return transpose_mul(m_tbi_j, local); }
    Vector3 global_of(int32_t body, Vector3 const & local) const { return (body == 0) ? global_i(local) : global_j(local); }

    uint32_t m_type;
    uint32_t m_nc;
    BodyPtr m_body_i;
    BodyPtr m_body_j;

    // Construction data in the body frames.
    Vector3 m_pi;
    Vector3 m_pj;
    Vector3 m_qi;
    Vector3 m_qj;
    Vector3 m_ai;
    Vector3 m_aj;
    // Unit vectors perpendicular to a_i and a_j; they rotate with their bodies.
    Vector3 m_fi_local;
    Vector3 m_gi_local;
    Vector3 m_fj_local;
    Vector3 m_gj_local;
    // Prismatic rotation lock: g axis of a body-j frame equal to the frame of
    // body i at construction, so f_i . g_lock = 0 at the initial relative rotation.
    Vector3 m_gj_lock_local;

    // Global quantities at the last update().
    Matrix3 m_tbi_i;
    Matrix3 m_tbi_j;
    Vector3 m_ri; ///< Position of body i.
    Vector3 m_rj; ///< Position of body j.
    Vector3 m_wi; ///< Angular velocity of body i.
    Vector3 m_wj; ///< Angular velocity of body j.
    Vector3 m_arm_i;
    Vector3 m_arm_j;
    Vector3 m_d; ///< d = r_i^P - r_j^P.
    Vector3 m_hi; ///< A_i a_i.
    Vector3 m_hj; ///< A_j a_j.
    Vector3 m_fi;
    Vector3 m_gi;
    Vector3 m_fj;
    Vector3 m_gj;
    Vector3 m_gj_lock;

    // Rotational driving (type 3).
    double m_drive_omega;
    double m_drive_angle_offset = 0.0;
    double m_time = 0.0;
    // Distance and coupler length.
    double m_dist_length;
    // C(t) of the drivers 8-11, evaluated at m_time in update().
    DriveFunction m_drive;
    double m_drive_c = 0.0;
    double m_drive_cd = 0.0;
    double m_drive_cdd = 0.0;
    Vector3 m_drive_u; ///< Type 8: fixed global direction.

    small_vector<Prim, 5> m_prims;
    CamProfile m_profile;
    bool m_composite_init_pending = false;
    double m_screw_pitch = 0.0;
    double m_screw_s0 = 0.0;
    double m_screw_theta0 = 0.0;
    double m_screw_theta = 0.0;

    SimpleArray<double> m_constraint;
    SimpleArray<double> m_gamma;
    SimpleArray<double> m_ct;
    SimpleArray<double> m_cq_i;
    SimpleArray<double> m_cq_j;
    SimpleArray<double> m_cq_omega_i;
    SimpleArray<double> m_cq_omega_j;
    SimpleArray<double> m_cq_pos_i;
    SimpleArray<double> m_cq_pos_j;

}; /* end class Joint */

using JointPtr = std::shared_ptr<Joint>;

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
