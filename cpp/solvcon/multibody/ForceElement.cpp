/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/multibody/ForceElement.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace solvcon
{

namespace multibody
{

ForceElement::ForceElement(BodyPtr body_i, BodyPtr body_j, Vector3 const & point_i, Vector3 const & point_j, double k, double c, double free_length, double actuator_force)
    : m_body_i(std::move(body_i))
    , m_body_j(std::move(body_j))
    , m_point_i(point_i)
    , m_point_j(point_j)
    , m_k(k)
    , m_c(c)
    , m_free_length(free_length)
    , m_actuator_force(actuator_force)
{
}

void ForceElement::compute()
{
    Vector3 const arm_i = transpose_mul(m_body_i->tbi(), m_point_i);
    Vector3 const arm_j = transpose_mul(m_body_j->tbi(), m_point_j);
    Vector3 const d = (m_body_j->position() + arm_j) - (m_body_i->position() + arm_i);
    double const length = norm(d);
    if (length < 1e-12)
    {
        m_force_i = m_force_j = m_torque_i = m_torque_j = Vector3();
        return;
    }
    Vector3 const n = d / length;
    Vector3 const vi = m_body_i->velocity() + cross(m_body_i->angular_velocity(), arm_i);
    Vector3 const vj = m_body_j->velocity() + cross(m_body_j->angular_velocity(), arm_j);
    double const length_dot = dot(vj - vi, n);

    // A stretched spring (L > L0) and an extending damper are in tension.
    double const tension = m_k * (length - m_free_length) + m_c * length_dot + m_actuator_force;
    m_force_i = tension * n;
    m_force_j = -tension * n;
    m_torque_i = cross(arm_i, m_force_i);
    m_torque_j = cross(arm_j, m_force_j);
}

Vector3 ForceElement::point_i() const { return m_body_i->position() + transpose_mul(m_body_i->tbi(), m_point_i); }

Vector3 ForceElement::point_j() const { return m_body_j->position() + transpose_mul(m_body_j->tbi(), m_point_j); }

RotationalSDA::RotationalSDA(BodyPtr body_i, BodyPtr body_j, Vector3 const & axis_i, Vector3 const & axis_j, double k, double c, DriveFunction torque)
    : ForceElement(std::move(body_i), std::move(body_j), Vector3(), Vector3(), k, c, 0.0, 0.0)
    , m_torque_fn(std::move(torque))
{
    Vector3 gj_unused;
    perpendicular_frame(normalized(axis_i), m_fi, m_gi);
    perpendicular_frame(normalized(axis_j), m_fj, gj_unused);
    Vector3 fi, gi, fj;
    frames(fi, gi, fj);
    m_theta = std::atan2(dot(gi, fj), dot(fi, fj));
    m_theta0 = m_theta;
}

void RotationalSDA::frames(Vector3 & fi, Vector3 & gi, Vector3 & fj) const
{
    fi = transpose_mul(m_body_i->tbi(), m_fi);
    gi = transpose_mul(m_body_i->tbi(), m_gi);
    fj = transpose_mul(m_body_j->tbi(), m_fj);
}

void RotationalSDA::compute()
{
    constexpr double two_pi = 2.0 * std::numbers::pi;
    Vector3 fi, gi, fj;
    frames(fi, gi, fj);
    double const c = dot(fi, fj);
    double const s = dot(gi, fj);
    double const raw = std::atan2(s, c);
    m_theta = raw + two_pi * std::round((m_theta - raw) / two_pi);
    // d theta = (d pi_i - d pi_j) . w, the same derivation as the rotational driver.
    Vector3 const w = (c * cross(gi, fj) - s * cross(fi, fj)) / (c * c + s * s);
    double const theta_dot = dot(m_body_i->angular_velocity() - m_body_j->angular_velocity(), w);
    double tc = 0.0, tcd = 0.0, tcdd = 0.0;
    m_torque_fn.eval(m_time, tc, tcd, tcdd);
    // The actuator drives body j (Haug Eq. 11.4.12): dW = -n d theta, so the
    // global torques are -n w on body i and n w on body j.
    double const n = m_k * (m_theta - m_theta0) + m_c * theta_dot - tc;
    m_force_i = m_force_j = Vector3();
    m_torque_i = -n * w;
    m_torque_j = n * w;
}

namespace
{

// Rotation vector of a rotation matrix (log map), valid for angles below pi.
Vector3 rotation_vector(Matrix3 const & r)
{
    double const c = std::max(-1.0, std::min(1.0, 0.5 * (r.trace() - 1.0)));
    double const angle = std::acos(c);
    Vector3 const w(r(2, 1) - r(1, 2), r(0, 2) - r(2, 0), r(1, 0) - r(0, 1));
    if (angle < 1e-8)
    {
        // First order: R = I + [theta]x.
        return 0.5 * w;
    }
    if (std::numbers::pi - angle < 1e-6)
    {
        throw std::runtime_error("Bushing: relative rotation reached pi; the linear bushing "
                                 "model is not valid there");
    }
    return angle / (2.0 * std::sin(angle)) * w;
}

SimpleArray<double> checked_6x6(SimpleArray<double> const & a)
{
    if (a.ndim() != 2 || a.shape(0) != 6 || a.shape(1) != 6)
    {
        throw std::invalid_argument("Bushing: K and C must be 6x6");
    }
    return a.to_row_major();
}

} /* end namespace */

Bushing::Bushing(BodyPtr body_i, BodyPtr body_j, Vector3 const & point_i, Vector3 const & point_j, SimpleArray<double> const & stiffness, SimpleArray<double> const & damping, Matrix3 const & frame)
    : ForceElement(std::move(body_i), std::move(body_j), point_i, point_j, 0.0, 0.0, 0.0, 0.0)
    , m_stiffness(checked_6x6(stiffness))
    , m_damping(checked_6x6(damping))
    , m_frame(frame)
{
    Vector3 d, th;
    Matrix3 r_rel;
    relative(d, th, r_rel);
    m_delta0 = d;
    m_r0 = r_rel;
}

void Bushing::relative(Vector3 & delta, Vector3 & theta, Matrix3 & r_rel) const
{
    Matrix3 const & tbi_i = m_body_i->tbi();
    Matrix3 const & tbi_j = m_body_j->tbi();
    Matrix3 const ab = transpose_mul(tbi_i, m_frame);
    Vector3 const rpi = m_body_i->position() + transpose_mul(tbi_i, m_point_i);
    Vector3 const rpj = m_body_j->position() + transpose_mul(tbi_j, m_point_j);
    delta = transpose_mul(ab, rpj - rpi) - m_delta0;
    r_rel = transpose_mul(ab, tbi_j.transpose());
    theta = rotation_vector(transpose_mul(m_r0, r_rel));
}

void Bushing::compute()
{
    Matrix3 const & tbi_i = m_body_i->tbi();
    Matrix3 const & tbi_j = m_body_j->tbi();
    Matrix3 const ab = transpose_mul(tbi_i, m_frame);
    Vector3 const arm_i = transpose_mul(tbi_i, m_point_i);
    Vector3 const arm_j = transpose_mul(tbi_j, m_point_j);
    Vector3 const rpi = m_body_i->position() + arm_i;
    Vector3 const rpj = m_body_j->position() + arm_j;
    Vector3 const & wi = m_body_i->angular_velocity();
    Vector3 const & wj = m_body_j->angular_velocity();
    Vector3 delta, theta;
    Matrix3 r_rel;
    relative(delta, theta, r_rel);
    // Rates in frame B: delta_dot = AB^T (v_Pj - v_Pi - w_i x (r_Pj - r_Pi)), w_rel = AB^T (w_j - w_i).
    Vector3 const vpi = m_body_i->velocity() + cross(wi, arm_i);
    Vector3 const vpj = m_body_j->velocity() + cross(wj, arm_j);
    Vector3 const delta_dot = transpose_mul(ab, vpj - vpi - cross(wi, rpj - rpi));
    Vector3 const w_rel = transpose_mul(ab, wj - wi);

    // [F; T] on body j in frame B.
    double q6[6] = {delta[0], delta[1], delta[2], theta[0], theta[1], theta[2]}; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    double v6[6] = {delta_dot[0], delta_dot[1], delta_dot[2], w_rel[0], w_rel[1], w_rel[2]}; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    double load[6]; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    for (ssize_t r = 0; r < 6; ++r)
    {
        double kq = 0.0, cv = 0.0;
        for (ssize_t p = 0; p < 6; ++p)
        {
            kq += m_stiffness(r, p) * q6[p];
            cv += m_damping(r, p) * v6[p];
        }
        load[r] = -kq - cv;
    }
    m_delta = delta;
    m_rotation = theta;

    Vector3 const fj = ab * Vector3(load[0], load[1], load[2]);
    Vector3 const tj = ab * Vector3(load[3], load[4], load[5]);
    Vector3 const fi = -fj;
    Vector3 const ti = -tj - cross(rpj - rpi, fj);
    m_force_i = fi;
    m_force_j = fj;
    m_torque_i = ti + cross(arm_i, fi);
    m_torque_j = tj + cross(arm_j, fj);
}

CurveDamper::CurveDamper(BodyPtr body_i, BodyPtr body_j, Vector3 const & point_i, Vector3 const & point_j, double k, double free_length, std::vector<double> velocities, std::vector<double> forces)
    : ForceElement(std::move(body_i), std::move(body_j), point_i, point_j, k, 0.0, free_length, 0.0)
    , m_velocities(std::move(velocities))
    , m_forces(std::move(forces))
{
    if (m_velocities.size() != m_forces.size() || m_velocities.size() < 2)
    {
        throw std::invalid_argument("CurveDamper: the table needs at least two (velocity, force) points");
    }
    for (size_t it = 0; it < m_velocities.size(); ++it)
    {
        if (!std::isfinite(m_velocities[it]) || !std::isfinite(m_forces[it]))
        {
            throw std::invalid_argument("CurveDamper: the table values must be finite");
        }
        if (it > 0 && !(m_velocities[it] > m_velocities[it - 1]))
        {
            throw std::invalid_argument("CurveDamper: the table velocities must increase strictly");
        }
        if (m_forces[it] * m_velocities[it] < 0.0)
        {
            throw std::invalid_argument("CurveDamper: the curve must be passive (F(v) v >= 0)");
        }
    }
}

double CurveDamper::damper_force(double velocity) const
{
    auto const & vs = m_velocities;
    auto const & fs = m_forces;
    // The segment that holds the velocity; the end segments extend outward.
    size_t hi = static_cast<size_t>(std::upper_bound(vs.begin(), vs.end(), velocity) - vs.begin());
    hi = std::clamp<size_t>(hi, 1, vs.size() - 1);
    size_t const lo = hi - 1;
    double const slope = (fs[hi] - fs[lo]) / (vs[hi] - vs[lo]);
    return fs[lo] + slope * (velocity - vs[lo]);
}

void CurveDamper::compute()
{
    Vector3 const arm_i = transpose_mul(m_body_i->tbi(), m_point_i);
    Vector3 const arm_j = transpose_mul(m_body_j->tbi(), m_point_j);
    Vector3 const d = (m_body_j->position() + arm_j) - (m_body_i->position() + arm_i);
    m_length = norm(d);
    if (m_length < 1e-12)
    {
        m_force_i = m_force_j = m_torque_i = m_torque_j = Vector3();
        m_length_rate = m_damper_force = 0.0;
        return;
    }
    Vector3 const n = d / m_length;
    Vector3 const vi = m_body_i->velocity() + cross(m_body_i->angular_velocity(), arm_i);
    Vector3 const vj = m_body_j->velocity() + cross(m_body_j->angular_velocity(), arm_j);
    m_length_rate = dot(vj - vi, n);
    m_damper_force = damper_force(m_length_rate);
    double const tension = m_k * (m_length - m_free_length) + m_damper_force;
    m_force_i = tension * n;
    m_force_j = -tension * n;
    m_torque_i = cross(arm_i, m_force_i);
    m_torque_j = cross(arm_j, m_force_j);
}

LengthStop::LengthStop(BodyPtr body_i, BodyPtr body_j, Vector3 const & point_i, Vector3 const & point_j, double min_length, double max_length, double k, double k2, double c)
    : ForceElement(std::move(body_i), std::move(body_j), point_i, point_j, k, c, 0.0, 0.0)
    , m_min_length(min_length)
    , m_max_length(max_length)
    , m_k2(k2)
{
    if (k < 0.0 || k2 < 0.0 || c < 0.0)
    {
        throw std::invalid_argument("LengthStop: k, k2, and c must not be negative");
    }
    if (!(min_length < max_length))
    {
        throw std::invalid_argument("LengthStop: min_length must be less than max_length");
    }
}

void LengthStop::compute()
{
    Vector3 const arm_i = transpose_mul(m_body_i->tbi(), m_point_i);
    Vector3 const arm_j = transpose_mul(m_body_j->tbi(), m_point_j);
    Vector3 const d = (m_body_j->position() + arm_j) - (m_body_i->position() + arm_i);
    m_length = norm(d);
    m_force_i = m_force_j = m_torque_i = m_torque_j = Vector3();
    m_penetration = m_stop_force = 0.0;
    m_engaged = 0;
    if (m_length < 1e-12)
    {
        return;
    }
    Vector3 const n = d / m_length;
    Vector3 const vi = m_body_i->velocity() + cross(m_body_i->angular_velocity(), arm_i);
    Vector3 const vj = m_body_j->velocity() + cross(m_body_j->angular_velocity(), arm_j);
    double const length_dot = dot(vj - vi, n);
    // sign = +1 pushes the points apart (bump stop), -1 pulls them together.
    double sign = 0.0;
    double rate = 0.0;
    if (m_length < m_min_length)
    {
        m_engaged = -1;
        sign = 1.0;
        m_penetration = m_min_length - m_length;
        rate = -length_dot;
    }
    else if (m_length > m_max_length)
    {
        m_engaged = 1;
        sign = -1.0;
        m_penetration = m_length - m_max_length;
        rate = length_dot;
    }
    else
    {
        return;
    }
    double const pen = m_penetration;
    // A stop only pushes: the damping term never makes it pull.
    m_stop_force = std::max(0.0, m_k * pen + m_k2 * pen * pen + m_c * rate);
    // Tension (pulls together) is -sign * force, as in ForceElement::compute().
    double const tension = -sign * m_stop_force;
    m_force_i = tension * n;
    m_force_j = -tension * n;
    m_torque_i = cross(arm_i, m_force_i);
    m_torque_j = cross(arm_j, m_force_j);
}

TireForce::TireForce(BodyPtr road, BodyPtr wheel, Vector3 const & spin_axis, double radius, double k, double c, double c_alpha, double mu, double road_z)
    : ForceElement(std::move(road), std::move(wheel), Vector3(), Vector3(), k, c, 0.0, 0.0)
    , m_spin(normalized(spin_axis))
    , m_radius(radius)
    , m_c_alpha(c_alpha)
    , m_mu(mu)
    , m_road_z(road_z)
{
    if (m_radius <= 0.0)
    {
        throw std::invalid_argument("TireForce: radius must be positive");
    }
}

void TireForce::set_road(DriveFunction height, DriveFunction slope)
{
    m_road_height = std::move(height);
    m_road_slope = std::move(slope);
    m_moving_road = true;
}

void TireForce::set_slip_model(double fz0, double load_exponent, bool brush, double relaxation_length)
{
    if (fz0 < 0.0 || relaxation_length < 0.0)
    {
        throw std::invalid_argument("TireForce: fz0 and relaxation_length must not be negative");
    }
    m_fz0 = fz0;
    m_load_exponent = load_exponent;
    m_brush = brush;
    m_relaxation = relaxation_length;
}

void TireForce::commit()
{
    m_alpha_committed = m_alpha_force;
    m_time_committed = m_time;
    m_committed = true;
}

void TireForce::compute()
{
    Vector3 const & rw = m_body_j->position();
    Vector3 const & wj = m_body_j->angular_velocity();
    Vector3 const s = transpose_mul(m_body_j->tbi(), m_spin);
    // The road plane: height h and its rate, forward slope b (about y).
    double h = m_road_z, hd = 0.0, b = 0.0;
    if (m_moving_road)
    {
        double hdd = 0.0, bd = 0.0, bdd = 0.0;
        m_road_height.eval(m_time, h, hd, hdd);
        m_road_slope.eval(m_time, b, bd, bdd);
    }
    Vector3 const n(-std::sin(b), 0.0, std::cos(b));
    Vector3 down = -n - dot(-n, s) * s;
    double const dn = norm(down);
    m_force_i = m_force_j = m_torque_i = m_torque_j = Vector3();
    m_fz = 0.0;
    m_alpha = 0.0;
    m_fy = 0.0;
    m_fx = 0.0;
    if (dn < 1e-9)
    {
        // A wheel plane parallel to the road has no defined contact point.
        m_contact = rw;
        m_deflection = 0.0;
        return;
    }
    down /= dn;
    m_contact = rw + m_radius * down;
    Vector3 const p0(rw[0], rw[1], h);
    m_deflection = dot(p0 - m_contact, n);
    if (m_deflection <= 0.0)
    {
        m_alpha_force = m_committed ? m_alpha_committed : 0.0;
        return;
    }
    Vector3 const arm = m_contact - rw;
    Vector3 const vc = m_body_j->velocity() + cross(wj, arm);
    // Normal approach rate relative to the moving plane (it rises at hd).
    double const vn = dot(vc, n) - hd * n[2];
    m_fz = std::max(0.0, m_k * m_deflection - m_c * vn);
    // Rolling direction in the road plane and the lateral direction.
    Vector3 hdir = cross(s, n);
    double const hn = norm(hdir);
    Vector3 force = m_fz * n;
    if (hn > 1e-9 && m_c_alpha > 0.0)
    {
        hdir /= hn;
        Vector3 const l = cross(n, hdir);
        double const vh = dot(vc, hdir), vl = dot(vc, l);
        m_vh = vh;
        if (std::fabs(vh) > 1e-6 || std::fabs(vl) > 1e-6)
        {
            m_alpha = std::atan2(vl, std::fabs(vh));
        }
        // The slip angle of the force: the first-order lag over the
        // distance rolled since the last commit.
        m_alpha_force = m_alpha;
        if (m_relaxation > 0.0 && m_committed)
        {
            double const x = std::fabs(vh) * std::max(0.0, m_time - m_time_committed);
            m_alpha_force = m_alpha + (m_alpha_committed - m_alpha) * std::exp(-x / m_relaxation);
        }
        double c_alpha = m_c_alpha;
        double mu = m_mu;
        if (m_fz0 > 0.0)
        {
            double const ratio = m_fz / m_fz0;
            c_alpha *= std::pow(ratio, m_load_exponent);
            mu *= std::pow(ratio, m_load_exponent - 1.0);
        }
        double const cap = mu * m_fz;
        m_fx = std::max(-cap, std::min(cap, m_fx_ratio * m_fz));
        double const cap_y = std::sqrt(std::max(0.0, cap * cap - m_fx * m_fx));
        if (m_brush)
        {
            double const sl = c_alpha * std::tan(m_alpha_force);
            if (cap_y <= 0.0)
            {
                m_fy = 0.0;
            }
            else if (std::fabs(sl) >= 3.0 * cap_y)
            {
                m_fy = -std::copysign(cap_y, sl);
            }
            else
            {
                m_fy = -(sl - sl * std::fabs(sl) / (3.0 * cap_y) + sl * sl * sl / (27.0 * cap_y * cap_y));
            }
        }
        else
        {
            m_fy = std::max(-cap_y, std::min(cap_y, -c_alpha * m_alpha_force));
        }
        force += m_fy * l + m_fx * hdir;
    }
    m_force_j = force;
    m_torque_j = cross(arm, force);
    // The road reaction; the solver ignores loads on the ground.
    m_force_i = -force;
}

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
