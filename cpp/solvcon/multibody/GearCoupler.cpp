/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/multibody/GearCoupler.hpp>
#include <solvcon/multibody/linear.hpp>

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace solvcon
{

namespace multibody
{

GearCoupler::GearCoupler(JointPtr joint_a, int32_t kind_a, JointPtr joint_b, int32_t kind_b, double ratio)
    : m_joint_a(std::move(joint_a))
    , m_joint_b(std::move(joint_b))
    , m_kind_a(kind_a)
    , m_kind_b(kind_b)
    , m_ratio(ratio)
    , m_jacobian(zero_matrix(4, 7))
    , m_position_jacobian(zero_matrix(4, 7))
{
    if ((m_kind_a != 0 && m_kind_a != 1) || (m_kind_b != 0 && m_kind_b != 1))
    {
        throw std::invalid_argument("GearCoupler: kind must be 0 (rotation) or 1 (translation)");
    }
    m_ga0 = m_ga = m_joint_a->measure(m_kind_a).g;
    m_gb0 = m_gb = m_joint_b->measure(m_kind_b).g;
}

double GearCoupler::unwrap(double prev, double raw)
{
    constexpr double two_pi = 2.0 * std::numbers::pi;
    return raw + two_pi * std::round((prev - raw) / two_pi);
}

void GearCoupler::add_block(uint32_t body, Vector3 const & r, Vector3 const & w, Quaternion const & p_pos, Quaternion const & p, double coef)
{
    // Phi_p = 2 Phi_pi E in the velocity form.
    Vector3 const v_r = coef * r;
    Quaternion const v_p = transpose_mul(build_E(p), coef * 2.0 * w);
    Quaternion const q_p = coef * p_pos;
    ssize_t k = 0;
    while (k < static_cast<ssize_t>(m_bodies.size()) && m_bodies[k] != body)
    {
        ++k;
    }
    if (k == static_cast<ssize_t>(m_bodies.size()))
    {
        m_bodies.push_back(body);
    }
    for (ssize_t c = 0; c < 3; ++c)
    {
        m_jacobian(k, c) += v_r[c];
        m_position_jacobian(k, c) += v_r[c];
    }
    for (ssize_t c = 0; c < 4; ++c)
    {
        m_jacobian(k, 3 + c) += v_p[c];
        m_position_jacobian(k, 3 + c) += q_p[c];
    }
}

void GearCoupler::update()
{
    Joint::Measure const ma = m_joint_a->measure(m_kind_a);
    Joint::Measure const mb = m_joint_b->measure(m_kind_b);
    m_ga = (m_kind_a == 0) ? unwrap(m_ga, ma.g) : ma.g;
    m_gb = (m_kind_b == 0) ? unwrap(m_gb, mb.g) : mb.g;
    m_constraint = (m_ga - m_ga0) - m_ratio * (m_gb - m_gb0);
    m_gamma = ma.gamma_part - m_ratio * mb.gamma_part;
    m_bodies.clear();
    m_jacobian.fill(0.0);
    m_position_jacobian.fill(0.0);
    auto add = [&](JointPtr const & j, Joint::Measure const & m, double coef)
    {
        BodyPtr const & bi = j->body_i();
        BodyPtr const & bj = j->body_j();
        add_block(bi->num(), m.ri, m.wi, m.pi_pos, bi->quaternion(), coef);
        add_block(bj->num(), m.rj, m.wj, m.pj_pos, bj->quaternion(), coef);
    };
    add(m_joint_a, ma, 1.0);
    add(m_joint_b, mb, -m_ratio);
}

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
