/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/multibody/Joint.hpp>
#include <solvcon/multibody/linear.hpp>

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

constexpr double two_pi = 2.0 * std::numbers::pi;

void set_row(SimpleArray<double> & a, ssize_t row, ssize_t col, Vector3 const & v)
{
    double * p = a.vptr(row, col);
    p[0] = v[0];
    p[1] = v[1];
    p[2] = v[2];
}

void add_row(SimpleArray<double> & a, ssize_t row, ssize_t col, Vector3 const & v)
{
    double * p = a.vptr(row, col);
    p[0] += v[0];
    p[1] += v[1];
    p[2] += v[2];
}

void sub_row(SimpleArray<double> & a, ssize_t row, ssize_t col, Vector3 const & v)
{
    double * p = a.vptr(row, col);
    p[0] -= v[0];
    p[1] -= v[1];
    p[2] -= v[2];
}

void set_row(SimpleArray<double> & a, ssize_t row, ssize_t col, Quaternion const & v)
{
    double * p = a.vptr(row, col);
    p[0] = v[0];
    p[1] = v[1];
    p[2] = v[2];
    p[3] = v[3];
}

Vector3 row3(SimpleArray<double> const & a, ssize_t row)
{
    double const * p = a.vptr(row, 0);
    return Vector3(p[0], p[1], p[2]);
}

void set_block(SimpleArray<double> & a, ssize_t row, ssize_t col, Matrix3 const & m)
{
    for (size_t i = 0; i < 3; ++i)
    {
        set_row(a, row + static_cast<ssize_t>(i), col, Vector3(m(i, 0), m(i, 1), m(i, 2)));
    }
}

// Rows 0-2 of the angular columns 3-6 of a position-form block.
void set_block(SimpleArray<double> & a, Matrix34 const & m, double scale)
{
    for (size_t i = 0; i < 3; ++i)
    {
        set_row(a, static_cast<ssize_t>(i), 3, Quaternion(scale * m(i, 0), scale * m(i, 1), scale * m(i, 2), scale * m(i, 3)));
    }
}

// Map an angle difference to (-pi, pi].  This chooses the revolution count n
// of Haug Eq. 9.5.4 so that theta + 2 n pi is the closest value to C(t).
double wrap_to_pi(double a)
{
    a = std::fmod(a + std::numbers::pi, two_pi);
    if (a <= 0.0)
    {
        a += two_pi;
    }
    return a - std::numbers::pi;
}

// Second-derivative terms of the basic constraints (Haug Table 9.4.1) with
// global vectors.  For a body-fixed vector X with the global angular velocity
// w, X_dot = w x X, and the non-acceleration part of X_ddot is w x (w x X).

// dot-1: Phi = X . Y, X on one body (angular velocity wx), Y on the other (wy).
double gamma_dot1(Vector3 const & x, Vector3 const & wx, Vector3 const & y, Vector3 const & wy)
{
    Vector3 const x_dot = cross(wx, x);
    Vector3 const y_dot = cross(wy, y);
    return -dot(cross(wx, x_dot), y) - 2.0 * dot(x_dot, y_dot) - dot(x, cross(wy, y_dot));
}

// dot-2: Phi = X . d, d = r_i^P - r_j^P.  gamma_pos is the gamma of d itself
// (the spherical rows), so the non-acceleration part of d_ddot is -gamma_pos.
double gamma_dot2(Vector3 const & x, Vector3 const & wx, Vector3 const & d, Vector3 const & d_dot, Vector3 const & gamma_pos)
{
    Vector3 const x_dot = cross(wx, x);
    return -dot(cross(wx, x_dot), d) - 2.0 * dot(x_dot, d_dot) + dot(x, gamma_pos);
}

// X = V . e with V on body i: the value, the rate, and the non-acceleration
// part of X_ddot.
struct DotE
{
    double x, xd, xdd_na;
}; /* end struct DotE */

DotE dot_e(Vector3 const & v, Vector3 const & wi, Vector3 const & e, Vector3 const & e_dot, Vector3 const & gamma_pos)
{
    Vector3 const v_dot = cross(wi, v);
    return {dot(v, e), dot(v_dot, e) + dot(v, e_dot), dot(cross(wi, v_dot), e) + 2.0 * dot(v_dot, e_dot) + dot(v, gamma_pos)};
}

uint32_t rows_of_type(uint32_t type)
{
    switch (type)
    {
    case 0:
    case 7: return 3;
    case 3:
    case 6:
    case 8:
    case 9:
    case 10:
    case 11:
    case 19: return 1;
    case 4:
    case 5:
    case 14:
    case 15:
    case 16: return 4;
    case 13:
    case 17:
    case 18: return 2;
    default: return 5; // revolute, prismatic, screw
    }
}

} /* end namespace */

Joint::Joint(uint32_t type, Vector3 const & pi, Vector3 const & pj, Vector3 const & qi, Vector3 const & qj, Vector3 const & ai, Vector3 const & aj, BodyPtr body_i, BodyPtr body_j, double drive_omega, double distance, DriveFunction drive, CamProfile profile)
    : m_type(type)
    , m_nc(rows_of_type(type))
    , m_body_i(std::move(body_i))
    , m_body_j(std::move(body_j))
    , m_pi(pi)
    , m_pj(pj)
    , m_qi(qi)
    , m_qj(qj)
    , m_ai(ai)
    , m_aj(aj)
    , m_tbi_i(Matrix3::identity())
    , m_tbi_j(Matrix3::identity())
    , m_drive_omega(drive_omega)
    , m_dist_length(distance)
    , m_drive(std::move(drive))
    , m_profile(std::move(profile))
{
    auto const nc = static_cast<ssize_t>(m_nc);
    m_constraint = zero_vector(nc);
    m_gamma = zero_vector(nc);
    m_ct = zero_vector(nc);
    m_cq_i = zero_matrix(nc, 7);
    m_cq_j = zero_matrix(nc, 7);
    m_cq_omega_i = zero_matrix(nc, 3);
    m_cq_omega_j = zero_matrix(nc, 3);
    m_cq_pos_i = zero_matrix(nc, 7);
    m_cq_pos_j = zero_matrix(nc, 7);

    perpendicular_frame(normalized(m_ai), m_fi_local, m_gi_local);
    perpendicular_frame(normalized(m_aj), m_fj_local, m_gj_local);
    m_drive_u = normalized(m_ai);
    if (m_type >= 12 && m_type <= 17)
    {
        if (m_type == 12)
        {
            m_screw_pitch = drive_omega;
        }
        build_composite();
    }

    // Prismatic rotation lock: map the f axis of body i into body j at the
    // construction pose and remove its component along a_j.
    if (m_type == 2)
    {
        Vector3 const aj_l = normalized(m_aj);
        Vector3 f_in_j = m_body_j->tbi() * transpose_mul(m_body_i->tbi(), m_fi_local);
        f_in_j -= dot(f_in_j, aj_l) * aj_l;
        if (norm(f_in_j) < 1e-9)
        {
            throw std::invalid_argument("Joint: prismatic axes of body i and body j are not "
                                        "parallel at construction");
        }
        f_in_j = normalized(f_in_j);
        m_gj_lock_local = cross(aj_l, f_in_j);
    }

    update();

    if (m_type == 3)
    {
        Vector3 fl, gl;
        perpendicular_frame(normalized(m_aj), fl, gl);
        Vector3 const f_in = transpose_mul(m_body_j->tbi(), fl);
        m_drive_angle_offset = std::atan2(dot(gl, f_in), dot(fl, f_in));
    }
    if (m_type == 6 && m_dist_length <= 0.0)
    {
        m_dist_length = norm(m_d);
    }
}

void Joint::update()
{
    m_tbi_i = m_body_i->tbi();
    m_tbi_j = m_body_j->tbi();
    m_wi = m_body_i->angular_velocity();
    m_wj = m_body_j->angular_velocity();
    m_arm_i = global_i(m_pi);
    m_arm_j = global_j(m_pj);
    m_ri = m_body_i->position();
    m_rj = m_body_j->position();
    m_d = m_ri + m_arm_i - m_rj - m_arm_j;
    m_hi = global_i(m_ai);
    m_hj = global_j(m_aj);
    m_fi = global_i(m_fi_local);
    m_gi = global_i(m_gi_local);
    m_fj = global_j(m_fj_local);
    m_gj = global_j(m_gj_local);
    m_gj_lock = global_j(m_gj_lock_local);

    if (is_driver())
    {
        m_drive.eval(m_time, m_drive_c, m_drive_cd, m_drive_cdd);
    }
    if (m_composite_init_pending)
    {
        composite_init();
    }

    build_constraint();
    build_jacobian();
    build_gamma();
    build_ct();
}

void Joint::build_constraint()
{
    if (!m_prims.empty())
    {
        composite_constraint();
        return;
    }
    if (m_type == 18 || m_type == 19)
    {
        follower_constraint();
        return;
    }
    Vector3 const & d = m_d;
    SimpleArray<double> & c = m_constraint;
    switch (m_type)
    {
    case 0:
        c[0] = d[0];
        c[1] = d[1];
        c[2] = d[2];
        break;
    case 1:
        c[0] = d[0];
        c[1] = d[1];
        c[2] = d[2];
        c[3] = dot(m_hi, m_fj);
        c[4] = dot(m_hi, m_gj);
        break;
    case 2:
        c[0] = dot(m_fi, d);
        c[1] = dot(m_gi, d);
        c[2] = dot(m_hi, m_fj);
        c[3] = dot(m_hi, m_gj);
        c[4] = dot(m_fi, m_gj_lock); // rotation about the slide axis
        break;
    case 3:
    {
        Vector3 f_local, g_local;
        perpendicular_frame(normalized(m_aj), f_local, g_local);
        Vector3 const f_inertial = global_j(f_local);
        double const cos_theta = dot(f_local, f_inertial);
        double const sin_theta = dot(g_local, f_inertial);
        double const target = m_drive_angle_offset + m_drive_omega * m_time;
        c[0] = sin_theta * std::cos(target) - cos_theta * std::sin(target);
        break;
    }
    case 4:
        c[0] = d[0];
        c[1] = d[1];
        c[2] = d[2];
        c[3] = dot(m_hi, m_hj);
        break;
    case 5:
        c[0] = dot(m_fj, d);
        c[1] = dot(m_gj, d);
        c[2] = dot(m_hi, m_fj);
        c[3] = dot(m_hi, m_gj);
        break;
    case 6: c[0] = 0.5 * (dot(d, d) - m_dist_length * m_dist_length); break;
    case 7:
        c[0] = dot(m_fj, d);
        c[1] = dot(m_gj, d);
        c[2] = dot(m_hi, m_hj);
        break;
    case 8: c[0] = dot(m_drive_u, d) - m_drive_c; break;
    case 9: c[0] = 0.5 * (dot(d, d) - m_drive_c * m_drive_c); break;
    // d = r_i^P - r_j^P, so h_i . (r_j^P - r_i^P) = -h_i . d.
    case 10: c[0] = -dot(m_hi, d) - m_drive_c; break;
    case 11: c[0] = wrap_to_pi(std::atan2(dot(m_gi, m_fj), dot(m_fi, m_fj)) - m_drive_c); break;
    default: break;
    }
}

void Joint::build_jacobian()
{
    m_cq_i.fill(0.0);
    m_cq_j.fill(0.0);
    m_cq_omega_i.fill(0.0);
    m_cq_omega_j.fill(0.0);

    Vector3 const & d = m_d;
    if (m_type == 0 || m_type == 1 || m_type == 4)
    {
        set_block(m_cq_i, 0, 0, Matrix3::identity());
        set_block(m_cq_j, 0, 0, -1.0 * Matrix3::identity());
        set_block(m_cq_omega_i, 0, 0, -1.0 * skew(m_arm_i));
        set_block(m_cq_omega_j, 0, 0, skew(m_arm_j));
    }
    switch (m_type)
    {
    case 1:
    {
        Vector3 const hi_x_fj = cross(m_hi, m_fj);
        Vector3 const hi_x_gj = cross(m_hi, m_gj);
        set_row(m_cq_omega_i, 3, 0, hi_x_fj);
        set_row(m_cq_omega_j, 3, 0, -hi_x_fj);
        set_row(m_cq_omega_i, 4, 0, hi_x_gj);
        set_row(m_cq_omega_j, 4, 0, -hi_x_gj);
        break;
    }
    case 2:
    {
        set_row(m_cq_i, 0, 0, m_fi);
        set_row(m_cq_i, 1, 0, m_gi);
        set_row(m_cq_j, 0, 0, -m_fi);
        set_row(m_cq_j, 1, 0, -m_gi);
        set_row(m_cq_omega_i, 0, 0, cross(m_arm_i, m_fi) + cross(m_fi, d));
        set_row(m_cq_omega_j, 0, 0, -cross(m_arm_j, m_fi));
        set_row(m_cq_omega_i, 1, 0, cross(m_arm_i, m_gi) + cross(m_gi, d));
        set_row(m_cq_omega_j, 1, 0, -cross(m_arm_j, m_gi));
        Vector3 const hi_x_fj = cross(m_hi, m_fj);
        Vector3 const hi_x_gj = cross(m_hi, m_gj);
        Vector3 const fi_x_gl = cross(m_fi, m_gj_lock);
        set_row(m_cq_omega_i, 2, 0, hi_x_fj);
        set_row(m_cq_omega_j, 2, 0, -hi_x_fj);
        set_row(m_cq_omega_i, 3, 0, hi_x_gj);
        set_row(m_cq_omega_j, 3, 0, -hi_x_gj);
        set_row(m_cq_omega_i, 4, 0, fi_x_gl);
        set_row(m_cq_omega_j, 4, 0, -fi_x_gl);
        break;
    }
    case 3:
    {
        // Phi = F . u with F = A_j f_l, so d Phi = d pi_j . (F x u).
        Vector3 f_l, g_l;
        perpendicular_frame(normalized(m_aj), f_l, g_l);
        Vector3 const f = global_j(f_l);
        double const t = m_drive_angle_offset + m_drive_omega * m_time;
        Vector3 const u = std::cos(t) * g_l - std::sin(t) * f_l;
        set_row(m_cq_omega_j, 0, 0, cross(f, u));
        break;
    }
    case 4:
    {
        Vector3 const hi_x_hj = cross(m_hi, m_hj);
        set_row(m_cq_omega_i, 3, 0, hi_x_hj);
        set_row(m_cq_omega_j, 3, 0, -hi_x_hj);
        break;
    }
    case 5:
    case 7:
    {
        set_row(m_cq_i, 0, 0, m_fj);
        set_row(m_cq_j, 0, 0, -m_fj);
        set_row(m_cq_omega_i, 0, 0, cross(m_arm_i, m_fj));
        set_row(m_cq_omega_j, 0, 0, cross(m_fj, d) - cross(m_arm_j, m_fj));
        set_row(m_cq_i, 1, 0, m_gj);
        set_row(m_cq_j, 1, 0, -m_gj);
        set_row(m_cq_omega_i, 1, 0, cross(m_arm_i, m_gj));
        set_row(m_cq_omega_j, 1, 0, cross(m_gj, d) - cross(m_arm_j, m_gj));
        if (m_type == 5)
        {
            Vector3 const hi_x_fj = cross(m_hi, m_fj);
            Vector3 const hi_x_gj = cross(m_hi, m_gj);
            set_row(m_cq_omega_i, 2, 0, hi_x_fj);
            set_row(m_cq_omega_j, 2, 0, -hi_x_fj);
            set_row(m_cq_omega_i, 3, 0, hi_x_gj);
            set_row(m_cq_omega_j, 3, 0, -hi_x_gj);
        }
        else
        {
            Vector3 const hi_x_hj = cross(m_hi, m_hj);
            set_row(m_cq_omega_i, 2, 0, hi_x_hj);
            set_row(m_cq_omega_j, 2, 0, -hi_x_hj);
        }
        break;
    }
    case 6:
    case 9:
        set_row(m_cq_i, 0, 0, d);
        set_row(m_cq_j, 0, 0, -d);
        set_row(m_cq_omega_i, 0, 0, transpose_mul(skew(m_arm_i), -d));
        set_row(m_cq_omega_j, 0, 0, transpose_mul(skew(m_arm_j), d));
        break;
    case 8:
        set_row(m_cq_i, 0, 0, m_drive_u);
        set_row(m_cq_j, 0, 0, -m_drive_u);
        set_row(m_cq_omega_i, 0, 0, cross(m_arm_i, m_drive_u));
        set_row(m_cq_omega_j, 0, 0, -cross(m_arm_j, m_drive_u));
        break;
    case 10:
        set_row(m_cq_i, 0, 0, -m_hi);
        set_row(m_cq_j, 0, 0, m_hi);
        set_row(m_cq_omega_i, 0, 0, cross(m_hi, m_arm_i - d));
        set_row(m_cq_omega_j, 0, 0, cross(m_arm_j, m_hi));
        break;
    case 11:
    {
        // theta = atan2(s, c), c = f_i . f_j, s = g_i . f_j; d theta = (dpi_i - dpi_j) . w.
        double const c = dot(m_fi, m_fj);
        double const s = dot(m_gi, m_fj);
        Vector3 const w = (c * cross(m_gi, m_fj) - s * cross(m_fi, m_fj)) / (c * c + s * s);
        set_row(m_cq_omega_i, 0, 0, w);
        set_row(m_cq_omega_j, 0, 0, -w);
        break;
    }
    default: break;
    }
    if (!m_prims.empty())
    {
        composite_jacobian();
    }
    if (m_type == 18 || m_type == 19)
    {
        follower_jacobian();
    }

    // The rows use global virtual rotations, so Phi_p = 2 Phi_pi E (Haug Sec. 9.6).
    Matrix34 const e_i = build_E(m_body_i->quaternion());
    Matrix34 const e_j = build_E(m_body_j->quaternion());
    for (ssize_t k = 0; k < static_cast<ssize_t>(m_nc); ++k)
    {
        set_row(m_cq_i, k, 3, 2.0 * transpose_mul(e_i, row3(m_cq_omega_i, k)));
        set_row(m_cq_j, k, 3, 2.0 * transpose_mul(e_j, row3(m_cq_omega_j, k)));
    }
}

void Joint::build_gamma()
{
    // Gamma in the angular-velocity form first, then the Euler-parameter
    // correction gamma_p = gamma_omega - 2 Phi_pi E(p_dot) p_dot per body.
    m_gamma.fill(0.0);
    SimpleArray<double> & g = m_gamma;

    Vector3 const & wi = m_wi;
    Vector3 const & wj = m_wj;
    Vector3 const wi_x_arm_i = cross(wi, m_arm_i);
    Vector3 const wj_x_arm_j = cross(wj, m_arm_j);
    Vector3 const gamma_pos = -cross(wi, wi_x_arm_i) + cross(wj, wj_x_arm_j);
    Vector3 const & d = m_d;
    Vector3 const d_dot = m_body_i->velocity() + wi_x_arm_i - m_body_j->velocity() - wj_x_arm_j;

    switch (m_type)
    {
    case 0:
        g[0] = gamma_pos[0];
        g[1] = gamma_pos[1];
        g[2] = gamma_pos[2];
        break;
    case 1:
        g[0] = gamma_pos[0];
        g[1] = gamma_pos[1];
        g[2] = gamma_pos[2];
        g[3] = gamma_dot1(m_hi, wi, m_fj, wj);
        g[4] = gamma_dot1(m_hi, wi, m_gj, wj);
        break;
    case 2:
        g[0] = gamma_dot2(m_fi, wi, d, d_dot, gamma_pos);
        g[1] = gamma_dot2(m_gi, wi, d, d_dot, gamma_pos);
        g[2] = gamma_dot1(m_hi, wi, m_fj, wj);
        g[3] = gamma_dot1(m_hi, wi, m_gj, wj);
        g[4] = gamma_dot1(m_fi, wi, m_gj_lock, wj);
        break;
    case 3:
    {
        // Phi = F . u, F = A_j f_l, u = cos(T) g_l - sin(T) f_l, T = T0 + w_d t:
        // u_dot = -w_d v with v = sin(T) g_l + cos(T) f_l, u_ddot = -w_d^2 u.
        Vector3 f_l, g_l;
        perpendicular_frame(normalized(m_aj), f_l, g_l);
        Vector3 const f = global_j(f_l);
        double const t = m_drive_angle_offset + m_drive_omega * m_time;
        Vector3 const u = std::cos(t) * g_l - std::sin(t) * f_l;
        Vector3 const v = std::sin(t) * g_l + std::cos(t) * f_l;
        Vector3 const f_dot = cross(wj, f);
        g[0] = -dot(cross(wj, f_dot), u) + 2.0 * m_drive_omega * dot(f_dot, v) + m_drive_omega * m_drive_omega * dot(f, u);
        break;
    }
    case 4:
        g[0] = gamma_pos[0];
        g[1] = gamma_pos[1];
        g[2] = gamma_pos[2];
        g[3] = gamma_dot1(m_hi, wi, m_hj, wj);
        break;
    case 5:
        g[0] = gamma_dot2(m_fj, wj, d, d_dot, gamma_pos);
        g[1] = gamma_dot2(m_gj, wj, d, d_dot, gamma_pos);
        g[2] = gamma_dot1(m_hi, wi, m_fj, wj);
        g[3] = gamma_dot1(m_hi, wi, m_gj, wj);
        break;
    case 6: g[0] = -dot(d_dot, d_dot) + dot(d, gamma_pos); break;
    case 7:
        g[0] = gamma_dot2(m_fj, wj, d, d_dot, gamma_pos);
        g[1] = gamma_dot2(m_gj, wj, d, d_dot, gamma_pos);
        g[2] = gamma_dot1(m_hi, wi, m_hj, wj);
        break;
    case 8: g[0] = dot(m_drive_u, gamma_pos) + m_drive_cdd; break;
    case 9: g[0] = -dot(d_dot, d_dot) + dot(d, gamma_pos) + m_drive_cd * m_drive_cd + m_drive_c * m_drive_cdd; break;
    case 10:
    {
        Vector3 const hi_dot = cross(wi, m_hi);
        Vector3 const hi_dd = cross(wi, hi_dot);
        g[0] = dot(hi_dd, d) + 2.0 * dot(hi_dot, d_dot) - dot(m_hi, gamma_pos) + m_drive_cdd;
        break;
    }
    case 11:
    {
        // theta_dot = (w_i - w_j) . w(q); gamma = -(w_i - w_j) . w_dot + C''.
        double theta, gamma_part;
        Vector3 w;
        rel_angle(theta, w, gamma_part);
        g[0] = gamma_part + m_drive_cdd;
        break;
    }
    default: break;
    }
    if (!m_prims.empty())
    {
        composite_gamma(d_dot, gamma_pos);
    }
    if (m_type == 18 || m_type == 19)
    {
        follower_gamma(d_dot, gamma_pos);
    }

    // E is linear in p, so E_dot = E(p_dot).
    Quaternion const & pdot_i = m_body_i->p_dot();
    Quaternion const & pdot_j = m_body_j->p_dot();
    Matrix34 const edot_i = build_E(pdot_i);
    Matrix34 const edot_j = build_E(pdot_j);
    for (ssize_t k = 0; k < static_cast<ssize_t>(m_nc); ++k)
    {
        g[k] -= 2.0 * dot(transpose_mul(edot_i, row3(m_cq_omega_i, k)), pdot_i);
    }
    for (ssize_t k = 0; k < static_cast<ssize_t>(m_nc); ++k)
    {
        g[k] -= 2.0 * dot(transpose_mul(edot_j, row3(m_cq_omega_j, k)), pdot_j);
    }
}

void Joint::build_ct()
{
    m_ct.fill(0.0);
    if (m_type == 3)
    {
        Vector3 f_local, g_local;
        perpendicular_frame(normalized(m_aj), f_local, g_local);
        Vector3 const f_inertial = global_j(f_local);
        double const cos_theta = dot(f_local, f_inertial);
        double const sin_theta = dot(g_local, f_inertial);
        double const target = m_drive_angle_offset + m_drive_omega * m_time;
        double const cos_error = cos_theta * std::cos(target) + sin_theta * std::sin(target);
        m_ct[0] = -cos_error * m_drive_omega;
    }
    else if (m_type == 8 || m_type == 10 || m_type == 11)
    {
        m_ct[0] = -m_drive_cd;
    }
    else if (m_type == 9)
    {
        m_ct[0] = -m_drive_c * m_drive_cd;
    }
}

double Joint::drive_sensitivity() const
{
    if (m_type == 8 || m_type == 10 || m_type == 11)
    {
        return -1.0;
    }
    if (m_type == 9)
    {
        return -m_drive_c;
    }
    throw std::logic_error("Joint::drive_sensitivity: joint type " + std::to_string(m_type) + " is not a driver");
}

SimpleArray<double> const & Joint::position_jacobian_i()
{
    // The translational columns are the same as the velocity form.
    std::copy(m_cq_i.begin(), m_cq_i.end(), m_cq_pos_i.begin());
    SimpleArray<double> & out = m_cq_pos_i;
    if (!m_prims.empty())
    {
        composite_position(0, out);
        return out;
    }
    if (m_type == 18 || m_type == 19)
    {
        follower_position(0, out);
        return out;
    }
    for (ssize_t k = 0; k < static_cast<ssize_t>(m_nc); ++k)
    {
        set_row(out, k, 3, Quaternion());
    }

    Quaternion const & p = m_body_i->quaternion();
    Vector3 const & d = m_d;
    switch (m_type)
    {
    case 0: set_block(out, build_Abar(p, m_pi), 1.0); break;
    case 1:
    {
        set_block(out, build_Abar(p, m_pi), 1.0);
        Matrix34 const ab = build_Abar(p, m_ai);
        set_row(out, 3, 3, transpose_mul(ab, m_fj));
        set_row(out, 4, 3, transpose_mul(ab, m_gj));
        break;
    }
    case 2:
    {
        Matrix34 const ab_pi = build_Abar(p, m_pi);
        Matrix34 const ab_fi = build_Abar(p, m_fi_local);
        Matrix34 const ab_gi = build_Abar(p, m_gi_local);
        Matrix34 const ab_ai = build_Abar(p, m_ai);
        set_row(out, 0, 3, transpose_mul(ab_fi, d) + transpose_mul(ab_pi, m_fi));
        set_row(out, 1, 3, transpose_mul(ab_gi, d) + transpose_mul(ab_pi, m_gi));
        set_row(out, 2, 3, transpose_mul(ab_ai, m_fj));
        set_row(out, 3, 3, transpose_mul(ab_ai, m_gj));
        set_row(out, 4, 3, transpose_mul(ab_fi, m_gj_lock));
        break;
    }
    case 4:
        set_block(out, build_Abar(p, m_pi), 1.0);
        set_row(out, 3, 3, transpose_mul(build_Abar(p, m_ai), m_hj));
        break;
    case 5:
    {
        Matrix34 const ab_pi = build_Abar(p, m_pi);
        Matrix34 const ab_ai = build_Abar(p, m_ai);
        set_row(out, 0, 3, transpose_mul(ab_pi, m_fj));
        set_row(out, 1, 3, transpose_mul(ab_pi, m_gj));
        set_row(out, 2, 3, transpose_mul(ab_ai, m_fj));
        set_row(out, 3, 3, transpose_mul(ab_ai, m_gj));
        break;
    }
    case 6:
    case 9: set_row(out, 0, 3, transpose_mul(build_Abar(p, m_pi), d)); break;
    case 7:
    {
        Matrix34 const ab_pi = build_Abar(p, m_pi);
        set_row(out, 0, 3, transpose_mul(ab_pi, m_fj));
        set_row(out, 1, 3, transpose_mul(ab_pi, m_gj));
        set_row(out, 2, 3, transpose_mul(build_Abar(p, m_ai), m_hj));
        break;
    }
    case 8: set_row(out, 0, 3, transpose_mul(build_Abar(p, m_pi), m_drive_u)); break;
    case 10: set_row(out, 0, 3, -(transpose_mul(build_Abar(p, m_ai), d) + transpose_mul(build_Abar(p, m_pi), m_hi))); break;
    case 11:
    {
        double const c = dot(m_fi, m_fj);
        double const s = dot(m_gi, m_fj);
        Quaternion const dc = transpose_mul(build_Abar(p, m_fi_local), m_fj);
        Quaternion const ds = transpose_mul(build_Abar(p, m_gi_local), m_fj);
        set_row(out, 0, 3, (c * ds - s * dc) / (c * c + s * s));
        break;
    }
    default: break; // The rotational driver does not depend on p_i.
    }
    return out;
}

SimpleArray<double> const & Joint::position_jacobian_j()
{
    std::copy(m_cq_j.begin(), m_cq_j.end(), m_cq_pos_j.begin());
    SimpleArray<double> & out = m_cq_pos_j;
    if (!m_prims.empty())
    {
        composite_position(1, out);
        return out;
    }
    if (m_type == 18 || m_type == 19)
    {
        follower_position(1, out);
        return out;
    }
    for (ssize_t k = 0; k < static_cast<ssize_t>(m_nc); ++k)
    {
        set_row(out, k, 3, Quaternion());
    }

    Quaternion const & p = m_body_j->quaternion();
    Vector3 const & d = m_d;
    switch (m_type)
    {
    case 0: set_block(out, build_Abar(p, m_pj), -1.0); break;
    case 1:
        set_block(out, build_Abar(p, m_pj), -1.0);
        set_row(out, 3, 3, transpose_mul(build_Abar(p, m_fj_local), m_hi));
        set_row(out, 4, 3, transpose_mul(build_Abar(p, m_gj_local), m_hi));
        break;
    case 2:
    {
        Matrix34 const ab_pj = build_Abar(p, m_pj);
        set_row(out, 0, 3, transpose_mul(ab_pj, -m_fi));
        set_row(out, 1, 3, transpose_mul(ab_pj, -m_gi));
        set_row(out, 2, 3, transpose_mul(build_Abar(p, m_fj_local), m_hi));
        set_row(out, 3, 3, transpose_mul(build_Abar(p, m_gj_local), m_hi));
        set_row(out, 4, 3, transpose_mul(build_Abar(p, m_gj_lock_local), m_fi));
        break;
    }
    case 3:
    {
        Vector3 f_local, g_local;
        perpendicular_frame(normalized(m_aj), f_local, g_local);
        Matrix34 const ab_fl = build_Abar(p, f_local);
        double const target = m_drive_angle_offset + m_drive_omega * m_time;
        Quaternion const dsin = transpose_mul(ab_fl, g_local);
        Quaternion const dcos = transpose_mul(ab_fl, f_local);
        set_row(out, 0, 3, std::cos(target) * dsin - std::sin(target) * dcos);
        break;
    }
    case 4:
        set_block(out, build_Abar(p, m_pj), -1.0);
        set_row(out, 3, 3, transpose_mul(build_Abar(p, m_aj), m_hi));
        break;
    case 5:
    {
        Matrix34 const ab_pj = build_Abar(p, m_pj);
        set_row(out, 0, 3, transpose_mul(build_Abar(p, m_fj_local), d) - transpose_mul(ab_pj, m_fj));
        set_row(out, 1, 3, transpose_mul(build_Abar(p, m_gj_local), d) - transpose_mul(ab_pj, m_gj));
        set_row(out, 2, 3, transpose_mul(build_Abar(p, m_fj_local), m_hi));
        set_row(out, 3, 3, transpose_mul(build_Abar(p, m_gj_local), m_hi));
        break;
    }
    case 6:
    case 9: set_row(out, 0, 3, transpose_mul(build_Abar(p, m_pj), -d)); break;
    case 7:
    {
        Matrix34 const ab_pj = build_Abar(p, m_pj);
        set_row(out, 0, 3, transpose_mul(build_Abar(p, m_fj_local), d) - transpose_mul(ab_pj, m_fj));
        set_row(out, 1, 3, transpose_mul(build_Abar(p, m_gj_local), d) - transpose_mul(ab_pj, m_gj));
        set_row(out, 2, 3, transpose_mul(build_Abar(p, m_aj), m_hi));
        break;
    }
    case 8: set_row(out, 0, 3, transpose_mul(build_Abar(p, m_pj), -m_drive_u)); break;
    case 10: set_row(out, 0, 3, transpose_mul(build_Abar(p, m_pj), m_hi)); break;
    case 11:
    {
        double const c = dot(m_fi, m_fj);
        double const s = dot(m_gi, m_fj);
        Quaternion const dc = transpose_mul(build_Abar(p, m_fj_local), m_fi);
        Quaternion const ds = transpose_mul(build_Abar(p, m_fj_local), m_gi);
        set_row(out, 0, 3, (c * ds - s * dc) / (c * c + s * s));
        break;
    }
    default: break;
    }
    return out;
}

void Joint::build_composite()
{
    using Kind = Prim::Kind;
    Vector3 const z;
    Vector3 const a_i = normalized(m_ai);
    auto dot1 = [&](Vector3 const & x, int32_t xb, Vector3 const & y, int32_t yb)
    { m_prims.push_back({Kind::Dot1, x, xb, y, yb}); };
    auto dot2 = [&](Vector3 const & x, int32_t xb)
    { m_prims.push_back({Kind::Dot2, x, xb, z, 0}); };
    auto dist = [&]()
    { m_prims.push_back({Kind::Dist, z, 0, z, 0}); };

    if (m_type == 12)
    {
        // Screw: cylindrical (Eq. 9.4.23) plus the screw row (Eq. 9.4.26).
        dot2(m_fj_local, 1);
        dot2(m_gj_local, 1);
        dot1(a_i, 0, m_fj_local, 1);
        dot1(a_i, 0, m_gj_local, 1);
        m_prims.push_back({Kind::Screw, a_i, 0, z, 0});
    }
    else if (m_type == 13)
    {
        // Revolute-spherical (Eq. 9.4.29).
        dist();
        dot2(a_i, 0);
    }
    else if (m_type == 14)
    {
        // Revolute-revolute, parallel axes (Eq. 9.4.30).
        dot1(a_i, 0, m_fj_local, 1);
        dot1(a_i, 0, m_gj_local, 1);
        dot2(a_i, 0);
        dist();
    }
    else if (m_type == 15)
    {
        // Revolute-revolute, orthogonal intersecting axes (Eq. 9.4.31).
        dot2(a_i, 0);
        dot2(m_fj_local, 1);
        dot2(m_gj_local, 1);
        dist();
    }
    else if (m_type == 16)
    {
        // Revolute-translational (Eq. 9.4.33), f_j given by q_j.
        if (norm(m_qj) < 1e-12)
        {
            throw std::invalid_argument("revolute-translational joint: f_j (qj) is zero");
        }
        Vector3 const f_j = normalized(m_qj);
        if (std::fabs(dot(f_j, normalized(m_aj))) > 1e-9)
        {
            throw std::invalid_argument("revolute-translational joint: f_j must be "
                                        "perpendicular to the translation axis aj");
        }
        Vector3 u1, u2;
        perpendicular_frame(f_j, u1, u2);
        dot1(a_i, 0, u1, 1);
        dot1(a_i, 0, u2, 1);
        dot2(m_fj_local, 1);
        dot2(m_gj_local, 1);
    }
    else if (m_type == 17)
    {
        // Strut (Eq. 9.4.34).
        dot2(m_fi_local, 0);
        dot2(m_gi_local, 0);
    }
    m_composite_init_pending = true;
}

void Joint::composite_init()
{
    m_composite_init_pending = false;
    if ((m_type == 13 || m_type == 14 || m_type == 15) && m_dist_length <= 0.0)
    {
        m_dist_length = norm(m_d);
    }
    if (m_type == 12)
    {
        double g_unused;
        Vector3 w_unused;
        rel_angle(m_screw_theta, w_unused, g_unused);
        m_screw_theta0 = m_screw_theta;
        m_screw_s0 = -dot(m_hi, m_d);
    }
}

void Joint::rel_angle(double & theta, Vector3 & w, double & gamma_part) const
{
    double const c = dot(m_fi, m_fj);
    double const s = dot(m_gi, m_fj);
    double const n = c * c + s * s;
    theta = std::atan2(s, c);
    Vector3 const a = cross(m_gi, m_fj);
    Vector3 const b = cross(m_fi, m_fj);
    w = (c * a - s * b) / n;
    Vector3 const rel = m_wi - m_wj;
    double const c_dot = dot(rel, b);
    double const s_dot = dot(rel, a);
    Vector3 const fi_dot = cross(m_wi, m_fi);
    Vector3 const gi_dot = cross(m_wi, m_gi);
    Vector3 const fj_dot = cross(m_wj, m_fj);
    Vector3 const a_dot = cross(gi_dot, m_fj) + cross(m_gi, fj_dot);
    Vector3 const b_dot = cross(fi_dot, m_fj) + cross(m_fi, fj_dot);
    double const n_dot = 2.0 * c * c_dot + 2.0 * s * s_dot;
    Vector3 const w_dot = (c_dot * a + c * a_dot - s_dot * b - s * b_dot) / n - w * (n_dot / n);
    gamma_part = -dot(rel, w_dot);
}

void Joint::rel_angle_pos(Quaternion & dth_dpi, Quaternion & dth_dpj) const
{
    Quaternion const & pi_q = m_body_i->quaternion();
    Quaternion const & pj_q = m_body_j->quaternion();
    double const c = dot(m_fi, m_fj);
    double const s = dot(m_gi, m_fj);
    double const n = c * c + s * s;
    Quaternion const dc_i = transpose_mul(build_Abar(pi_q, m_fi_local), m_fj);
    Quaternion const ds_i = transpose_mul(build_Abar(pi_q, m_gi_local), m_fj);
    Quaternion const dc_j = transpose_mul(build_Abar(pj_q, m_fj_local), m_fi);
    Quaternion const ds_j = transpose_mul(build_Abar(pj_q, m_fj_local), m_gi);
    dth_dpi = (c * ds_i - s * dc_i) / n;
    dth_dpj = (c * ds_j - s * dc_j) / n;
}

void Joint::composite_constraint()
{
    Vector3 const & d = m_d;
    for (size_t k = 0; k < m_prims.size(); ++k)
    {
        Prim const & pr = m_prims[k];
        Vector3 const x = global_of(pr.xb, pr.x_l);
        auto const row = static_cast<ssize_t>(k);
        switch (pr.kind)
        {
        case Prim::Kind::Dot1: m_constraint[row] = dot(x, global_of(pr.yb, pr.y_l)); break;
        case Prim::Kind::Dot2: m_constraint[row] = dot(x, d); break;
        case Prim::Kind::Dist: m_constraint[row] = 0.5 * (dot(d, d) - m_dist_length * m_dist_length); break;
        case Prim::Kind::Screw:
        {
            // Continuous angle: unwrap against the last evaluated value.
            double raw, g_unused;
            Vector3 w_unused;
            rel_angle(raw, w_unused, g_unused);
            m_screw_theta = raw + two_pi * std::round((m_screw_theta - raw) / two_pi);
            m_constraint[row] = -dot(x, d) - m_screw_s0 - m_screw_pitch * (m_screw_theta - m_screw_theta0);
            break;
        }
        }
    }
}

void Joint::composite_jacobian()
{
    Vector3 const & d = m_d;
    for (size_t k = 0; k < m_prims.size(); ++k)
    {
        Prim const & pr = m_prims[k];
        Vector3 const x = global_of(pr.xb, pr.x_l);
        auto const row = static_cast<ssize_t>(k);
        auto add_w = [&](int32_t body, Vector3 const & v)
        { add_row((body == 0) ? m_cq_omega_i : m_cq_omega_j, row, 0, v); };
        switch (pr.kind)
        {
        case Prim::Kind::Dot1:
        {
            Vector3 const y = global_of(pr.yb, pr.y_l);
            add_w(pr.xb, cross(x, y));
            add_w(pr.yb, cross(y, x));
            break;
        }
        case Prim::Kind::Dot2:
            set_row(m_cq_i, row, 0, x);
            set_row(m_cq_j, row, 0, -x);
            add_w(0, cross(m_arm_i, x));
            add_w(1, -cross(m_arm_j, x));
            add_w(pr.xb, cross(x, d));
            break;
        case Prim::Kind::Dist:
            set_row(m_cq_i, row, 0, d);
            set_row(m_cq_j, row, 0, -d);
            add_w(0, cross(m_arm_i, d));
            add_w(1, -cross(m_arm_j, d));
            break;
        case Prim::Kind::Screw:
        {
            // The -X . d part (X = h_i on body i), then the -pitch theta part.
            set_row(m_cq_i, row, 0, -x);
            set_row(m_cq_j, row, 0, x);
            add_w(0, -(cross(m_arm_i, x) + cross(x, d)));
            add_w(1, cross(m_arm_j, x));
            double th, g_unused;
            Vector3 w;
            rel_angle(th, w, g_unused);
            add_w(0, -m_screw_pitch * w);
            add_w(1, m_screw_pitch * w);
            break;
        }
        }
    }
}

void Joint::composite_gamma(Vector3 const & d_dot, Vector3 const & gamma_pos)
{
    Vector3 const & d = m_d;
    for (size_t k = 0; k < m_prims.size(); ++k)
    {
        Prim const & pr = m_prims[k];
        Vector3 const x = global_of(pr.xb, pr.x_l);
        Vector3 const & wx = (pr.xb == 0) ? m_wi : m_wj;
        auto const row = static_cast<ssize_t>(k);
        switch (pr.kind)
        {
        case Prim::Kind::Dot1: m_gamma[row] = gamma_dot1(x, wx, global_of(pr.yb, pr.y_l), (pr.yb == 0) ? m_wi : m_wj); break;
        case Prim::Kind::Dot2: m_gamma[row] = gamma_dot2(x, wx, d, d_dot, gamma_pos); break;
        case Prim::Kind::Dist: m_gamma[row] = -dot(d_dot, d_dot) + dot(d, gamma_pos); break;
        case Prim::Kind::Screw:
        {
            double th, g_theta;
            Vector3 w;
            rel_angle(th, w, g_theta);
            m_gamma[row] = -gamma_dot2(x, wx, d, d_dot, gamma_pos) - m_screw_pitch * g_theta;
            break;
        }
        }
    }
}

void Joint::composite_position(int32_t body, SimpleArray<double> & out) const
{
    Vector3 const & d = m_d;
    Quaternion const & p_b = ((body == 0) ? m_body_i : m_body_j)->quaternion();
    Vector3 const & pt = (body == 0) ? m_pi : m_pj;
    // d = r_i^P - r_j^P.
    double const sign = (body == 0) ? 1.0 : -1.0;
    for (size_t k = 0; k < m_prims.size(); ++k)
    {
        Prim const & pr = m_prims[k];
        Vector3 const x = global_of(pr.xb, pr.x_l);
        Quaternion row;
        switch (pr.kind)
        {
        case Prim::Kind::Dot1:
        {
            Vector3 const y = global_of(pr.yb, pr.y_l);
            if (pr.xb == body)
            {
                row += transpose_mul(build_Abar(p_b, pr.x_l), y);
            }
            if (pr.yb == body)
            {
                row += transpose_mul(build_Abar(p_b, pr.y_l), x);
            }
            break;
        }
        case Prim::Kind::Dot2:
            row = transpose_mul(build_Abar(p_b, pt), sign * x);
            if (pr.xb == body)
            {
                row += transpose_mul(build_Abar(p_b, pr.x_l), d);
            }
            break;
        case Prim::Kind::Dist: row = transpose_mul(build_Abar(p_b, pt), sign * d); break;
        case Prim::Kind::Screw:
        {
            row = transpose_mul(build_Abar(p_b, pt), -sign * x);
            if (pr.xb == body)
            {
                row -= transpose_mul(build_Abar(p_b, pr.x_l), d);
            }
            Quaternion dpi, dpj;
            rel_angle_pos(dpi, dpj);
            row -= m_screw_pitch * ((body == 0) ? dpi : dpj);
            break;
        }
        }
        set_row(out, static_cast<ssize_t>(k), 3, row);
    }
}

Joint::Measure Joint::measure(int kind) const
{
    Measure m;
    if (kind == 0)
    {
        Vector3 w;
        rel_angle(m.g, w, m.gamma_part);
        m.wi = w;
        m.wj = -w;
        rel_angle_pos(m.pi_pos, m.pj_pos);
        return m;
    }
    if (kind != 1)
    {
        throw std::invalid_argument("Joint::measure: kind must be 0 (rotation) or 1 (translation)");
    }
    Vector3 const & d = m_d;
    Vector3 const d_dot = m_body_i->velocity() + cross(m_wi, m_arm_i) - m_body_j->velocity() - cross(m_wj, m_arm_j);
    Vector3 const gamma_pos = -cross(m_wi, cross(m_wi, m_arm_i)) + cross(m_wj, cross(m_wj, m_arm_j));
    m.g = -dot(m_hi, d);
    m.ri = -m_hi;
    m.rj = m_hi;
    m.wi = cross(m_hi, m_arm_i - d);
    m.wj = cross(m_arm_j, m_hi);
    Quaternion const & pi_q = m_body_i->quaternion();
    Quaternion const & pj_q = m_body_j->quaternion();
    m.pi_pos = -(transpose_mul(build_Abar(pi_q, m_ai), d) + transpose_mul(build_Abar(pi_q, m_pi), m_hi));
    m.pj_pos = transpose_mul(build_Abar(pj_q, m_pj), m_hi);
    Vector3 const hi_dot = cross(m_wi, m_hi);
    m.gamma_part = dot(cross(m_wi, hi_dot), d) + 2.0 * dot(hi_dot, d_dot) - dot(m_hi, gamma_pos);
    return m;
}

// Cam followers (types 18 and 19), with e = r_j^P - r_i^P = -d.
// Type 18 rows: [h_i . e, rho(X, Y) - R(phi(X, Y))], X = f_i . e, Y = g_i . e.
// Type 19 row: u . e - h(psi), u = -A_j n_j, psi = atan2(g_i . u, f_i . u).

void Joint::follower_constraint()
{
    Vector3 const e = -m_d;
    if (m_type == 18)
    {
        double const x = dot(m_fi, e), y = dot(m_gi, e);
        double r, dr, ddr;
        m_profile.eval(std::atan2(y, x), r, dr, ddr);
        m_constraint[0] = dot(m_hi, e);
        m_constraint[1] = std::sqrt(x * x + y * y) - r;
    }
    else
    {
        Vector3 const u = -global_j(normalized(m_aj));
        double h, dh, ddh;
        m_profile.eval(std::atan2(dot(m_gi, u), dot(m_fi, u)), h, dh, ddh);
        m_constraint[0] = dot(u, e) - h;
    }
}

void Joint::follower_jacobian()
{
    Vector3 const e = -m_d;
    // Velocity-form rows of X = V . e with V on body i: r_i -V, r_j V,
    // w_i V x (e + arm_i), w_j arm_j x V.
    auto add_dot_e = [&](ssize_t row, Vector3 const & v, double coef)
    {
        add_row(m_cq_i, row, 0, -coef * v);
        add_row(m_cq_j, row, 0, coef * v);
        add_row(m_cq_omega_i, row, 0, coef * cross(v, e + m_arm_i));
        add_row(m_cq_omega_j, row, 0, coef * cross(m_arm_j, v));
    };
    if (m_type == 18)
    {
        add_dot_e(0, m_hi, 1.0);
        double const x = dot(m_fi, e), y = dot(m_gi, e);
        double const rho2 = x * x + y * y, rho = std::sqrt(rho2);
        double r, dr, ddr;
        m_profile.eval(std::atan2(y, x), r, dr, ddr);
        double const fx = x / rho + dr * y / rho2;
        double const fy = y / rho - dr * x / rho2;
        add_dot_e(1, m_fi, fx);
        add_dot_e(1, m_gi, fy);
    }
    else
    {
        Vector3 const u = -global_j(normalized(m_aj));
        double const c = dot(m_fi, u), s = dot(m_gi, u), n = c * c + s * s;
        double h, dh, ddh;
        m_profile.eval(std::atan2(s, c), h, dh, ddh);
        // a = u . e
        set_row(m_cq_i, 0, 0, -u);
        set_row(m_cq_j, 0, 0, u);
        set_row(m_cq_omega_i, 0, 0, -cross(m_arm_i, u));
        set_row(m_cq_omega_j, 0, 0, cross(u, e) + cross(m_arm_j, u));
        // -h' d psi with d psi = (c ds - s dc) / n and dc = dpi_i . (f_i x u) + dpi_j . (u x f_i).
        Vector3 const dpsi_i = (c * cross(m_gi, u) - s * cross(m_fi, u)) / n;
        Vector3 const dpsi_j = (c * cross(u, m_gi) - s * cross(u, m_fi)) / n;
        sub_row(m_cq_omega_i, 0, 0, dh * dpsi_i);
        sub_row(m_cq_omega_j, 0, 0, dh * dpsi_j);
    }
}

void Joint::follower_gamma(Vector3 const & d_dot, Vector3 const & gamma_pos)
{
    Vector3 const e = -m_d, e_dot = -d_dot;
    Vector3 const & wi = m_wi;
    Vector3 const & wj = m_wj;
    if (m_type == 18)
    {
        DotE const z = dot_e(m_hi, wi, e, e_dot, gamma_pos);
        m_gamma[0] = -z.xdd_na;
        DotE const x = dot_e(m_fi, wi, e, e_dot, gamma_pos);
        DotE const y = dot_e(m_gi, wi, e, e_dot, gamma_pos);
        double const rho2 = x.x * x.x + y.x * y.x, rho = std::sqrt(rho2), rho3 = rho2 * rho;
        double r, dr, ddr;
        m_profile.eval(std::atan2(y.x, x.x), r, dr, ddr);
        double const px = -y.x / rho2, py = x.x / rho2;
        double const pxx = 2.0 * x.x * y.x / (rho2 * rho2), pyy = -pxx;
        double const pxy = (y.x * y.x - x.x * x.x) / (rho2 * rho2);
        double const fx = x.x / rho - dr * px, fy = y.x / rho - dr * py;
        double const fxx = y.x * y.x / rho3 - ddr * px * px - dr * pxx;
        double const fyy = x.x * x.x / rho3 - ddr * py * py - dr * pyy;
        double const fxy = -x.x * y.x / rho3 - ddr * px * py - dr * pxy;
        double const quad = fxx * x.xd * x.xd + 2.0 * fxy * x.xd * y.xd + fyy * y.xd * y.xd;
        m_gamma[1] = -(fx * x.xdd_na + fy * y.xdd_na + quad);
    }
    else
    {
        Vector3 const u = -global_j(normalized(m_aj));
        Vector3 const u_dot = cross(wj, u);
        // a = u . e with u on body j: the dot_e structure with the rotation of body j.
        double const a_dd_na = dot(cross(wj, u_dot), e) + 2.0 * dot(u_dot, e_dot) + dot(u, gamma_pos);
        double const c = dot(m_fi, u), s = dot(m_gi, u), n = c * c + s * s;
        Vector3 const fi_dot = cross(wi, m_fi), gi_dot = cross(wi, m_gi);
        double const c_dot = dot(fi_dot, u) + dot(m_fi, u_dot);
        double const s_dot = dot(gi_dot, u) + dot(m_gi, u_dot);
        // Non-acceleration parts of c_ddot and s_ddot (dot-1 rows).
        double const c_dd_na = -gamma_dot1(m_fi, wi, u, wj);
        double const s_dd_na = -gamma_dot1(m_gi, wi, u, wj);
        double const psi_dot = (c * s_dot - s * c_dot) / n;
        double const n_dot = 2.0 * (c * c_dot + s * s_dot);
        double const psi_dd_na = (c * s_dd_na - s * c_dd_na) / n - psi_dot * n_dot / n;
        double h, dh, ddh;
        m_profile.eval(std::atan2(s, c), h, dh, ddh);
        m_gamma[0] = -(a_dd_na - dh * psi_dd_na - ddh * psi_dot * psi_dot);
    }
}

void Joint::follower_position(int32_t body, SimpleArray<double> & out) const
{
    Vector3 const e = -m_d;
    Quaternion const & pi_q = m_body_i->quaternion();
    Quaternion const & pj_q = m_body_j->quaternion();
    // d(V . e)/dp with V = A_i v_l on body i.
    auto dot_e_pos = [&](Vector3 const & v, Vector3 const & v_l) -> Quaternion
    {
        if (body == 0)
        {
            return transpose_mul(build_Abar(pi_q, v_l), e) - transpose_mul(build_Abar(pi_q, m_pi), v);
        }
        return transpose_mul(build_Abar(pj_q, m_pj), v);
    };
    if (m_type == 18)
    {
        set_row(out, 0, 3, dot_e_pos(m_hi, normalized(m_ai)));
        double const x = dot(m_fi, e), y = dot(m_gi, e);
        double const rho2 = x * x + y * y, rho = std::sqrt(rho2);
        double r, dr, ddr;
        m_profile.eval(std::atan2(y, x), r, dr, ddr);
        double const fx = x / rho + dr * y / rho2;
        double const fy = y / rho - dr * x / rho2;
        set_row(out, 1, 3, fx * dot_e_pos(m_fi, m_fi_local) + fy * dot_e_pos(m_gi, m_gi_local));
    }
    else
    {
        Vector3 const n_l = -normalized(m_aj); // u = A_j n_l
        Vector3 const u = global_j(n_l);
        double const c = dot(m_fi, u), s = dot(m_gi, u), n = c * c + s * s;
        double h, dh, ddh;
        m_profile.eval(std::atan2(s, c), h, dh, ddh);
        Quaternion da, dc, ds;
        if (body == 0)
        {
            da = transpose_mul(build_Abar(pi_q, m_pi), -u);
            dc = transpose_mul(build_Abar(pi_q, m_fi_local), u);
            ds = transpose_mul(build_Abar(pi_q, m_gi_local), u);
        }
        else
        {
            da = transpose_mul(build_Abar(pj_q, n_l), e) + transpose_mul(build_Abar(pj_q, m_pj), u);
            dc = transpose_mul(build_Abar(pj_q, n_l), m_fi);
            ds = transpose_mul(build_Abar(pj_q, n_l), m_gi);
        }
        set_row(out, 0, 3, da - dh * (c * ds - s * dc) / n);
    }
}

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
