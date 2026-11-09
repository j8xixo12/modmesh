#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * The steady-state slip model of a tire: the Magic Formula pure-slip
 * curves combined by the theoretical slip (Pacejka, Tire and Vehicle
 * Dynamics, 3rd ed., 2012, Sec. 4.3.1 and 4.3.2).
 *
 * @ingroup group_numerics
 */

namespace solvcon
{

namespace multibody
{

/**
 * A reduced Magic Formula tire.
 *
 * The pure-slip curve of each direction is
 *
 *     F0(sigma) = D sin(C atan(B sigma - E (B sigma - atan(B sigma))))
 *
 * with the peak D = mu F_z (F_z / F_z0)^(n - 1).  B, C, and E do not
 * depend on the load, so the slip at the peak is fixed and the slip
 * stiffness B C D grows with F_z^n.
 *
 * The combined slip uses the theoretical slip quantities
 *
 *     sigma_x = kappa / (1 + kappa), sigma_y = tan(alpha) / (1 + kappa),
 *     sigma = |(sigma_x, sigma_y)|,
 *     F_x = (sigma_x / sigma) F_x0(sigma), F_y = -(sigma_y / sigma) F_y0(sigma).
 *
 * A locked wheel (kappa = -1) slides: |F_y| / |F_x| = |tan(alpha)| F_ys /
 * F_xs with the sliding forces F_s = D sin(C pi / 2) (E < 1), so with equal
 * curves the force points against the slip velocity.
 */
class MagicFormulaTire
{

public:

    /// The parameters of one direction (longitudinal or lateral).
    struct Curve
    {
        double mu = 1.0; ///< Friction coefficient at F_z0.
        double load_exponent = 1.0; ///< n of D ~ F_z^n.
        double stiffness = 10.0; ///< B (per unit theoretical slip).
        double shape = 1.3; ///< C, in (0, 2).
        double curvature = 0.0; ///< E, at most 1.
    };

    /// The force of a combined slip.
    struct Force
    {
        double fx = 0.0; ///< Longitudinal force (N), + driving.
        double fy = 0.0; ///< Lateral force (N), against + slip angle.
        double slip_ratio = 0.0; ///< kappa of the force.
    };

    /// The largest slip ratio of a spinning wheel that force_for_demand uses.
    static constexpr double max_drive_slip = 1.0;

    MagicFormulaTire(double fz0, Curve const & longitudinal, Curve const & lateral);

    double fz0() const { return m_fz0; }
    Curve const & longitudinal() const { return m_x; }
    Curve const & lateral() const { return m_y; }

    /// The peak force D (N) of a direction at the load fz.
    double peak(Curve const & curve, double fz) const;
    /// The friction coefficient D / F_z of a direction at the load fz.
    double friction(Curve const & curve, double fz) const;
    /// The slip stiffness B C D (N per unit slip) of a direction at fz.
    double slip_stiffness(Curve const & curve, double fz) const;

    /// The force at the load fz (N), slip ratio kappa, and slip angle (rad).
    Force force(double fz, double slip_ratio, double slip_angle) const;
    /**
     * The force that delivers the longitudinal force demand at the slip
     * angle.  The slip ratio solves F_x(kappa) = demand on the stable branch
     * (between zero and the peak).  A demand beyond the peak locks the wheel
     * (kappa = -1, braking) or spins it (kappa = max_drive_slip, driving).
     */
    Force force_for_demand(double fz, double demand, double slip_angle) const;

private:

    double pure(Curve const & curve, double d, double sigma) const;
    /// The force of the curve at infinite slip.
    double sliding(Curve const & curve, double d) const;
    double peak_slip_ratio(double fz, double slip_angle, double sign) const;

    double m_fz0;
    Curve m_x;
    Curve m_y;

}; /* end class MagicFormulaTire */

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
