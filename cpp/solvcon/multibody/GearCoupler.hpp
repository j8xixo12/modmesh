#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Linear coupling of the natural coordinates of two joints (Haug Sec. 3.4.1).
 *
 * @ingroup group_numerics
 */

#include <solvcon/multibody/Joint.hpp>

#include <solvcon/buffer/buffer.hpp>

#include <cstdint>
#include <memory>

namespace solvcon
{

namespace multibody
{

/**
 * One constraint row Phi = (g_A - g_A0) - ratio (g_B - g_B0), where g is the
 * relative rotation angle (kind 0) or the relative translation (kind 1) of a
 * joint and g_0 its value at construction.  Rotation with rotation is a gear
 * pair (ratio = -R_B / R_A for external gears); rotation with translation is
 * a rack and pinion.
 */
class GearCoupler
{

public:

    GearCoupler(JointPtr joint_a, int32_t kind_a, JointPtr joint_b, int32_t kind_b, double ratio);
    GearCoupler(GearCoupler const &) = delete;
    GearCoupler(GearCoupler &&) = delete;
    GearCoupler & operator=(GearCoupler const &) = delete;
    GearCoupler & operator=(GearCoupler &&) = delete;
    ~GearCoupler() = default;

    /// Evaluate the row at the current joint states (the joints must be updated).
    void update();

    double constraint() const { return m_constraint; }
    double gamma() const { return m_gamma; }
    /// Numbers of the bodies that the row depends on (up to 4).
    small_vector<uint32_t, 4> const & bodies() const { return m_bodies; }
    /// Velocity-form Jacobian: row k (7 elements) belongs to bodies()[k].
    SimpleArray<double> const & jacobian() const { return m_jacobian; }
    /// Position-form Jacobian: row k (7 elements) belongs to bodies()[k].
    SimpleArray<double> const & position_jacobian() const { return m_position_jacobian; }

private:

    static double unwrap(double prev, double raw);
    void add_block(uint32_t body, Vector3 const & r, Vector3 const & w, Quaternion const & p_pos, Quaternion const & p, double coef);

    JointPtr m_joint_a;
    JointPtr m_joint_b;
    int32_t m_kind_a;
    int32_t m_kind_b;
    double m_ratio;
    // Continuous (unwrapped) coordinate values.
    double m_ga0 = 0.0;
    double m_gb0 = 0.0;
    double m_ga = 0.0;
    double m_gb = 0.0;
    double m_constraint = 0.0;
    double m_gamma = 0.0;
    small_vector<uint32_t, 4> m_bodies;
    SimpleArray<double> m_jacobian;
    SimpleArray<double> m_position_jacobian;

}; /* end class GearCoupler */

using GearCouplerPtr = std::shared_ptr<GearCoupler>;

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
