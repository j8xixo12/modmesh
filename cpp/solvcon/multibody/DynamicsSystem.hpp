#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * The constrained multibody system in Euler parameters: assembly, kinematic
 * and dynamic analysis, quasi-static sweeps, and static equilibrium (Haug
 * Ch. 9, 11).
 *
 * @ingroup group_numerics
 */

#include <solvcon/multibody/Body.hpp>
#include <solvcon/multibody/ForceElement.hpp>
#include <solvcon/multibody/GearCoupler.hpp>
#include <solvcon/multibody/Joint.hpp>
#include <solvcon/multibody/kinematics.hpp>

#include <solvcon/buffer/buffer.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace solvcon
{

namespace multibody
{

/**
 * Full constraint set at one state (Haug Eq. 9.6.5), without stabilization
 * terms.  The row layout is 7 ground rows, the joint rows in insertion order,
 * the gear coupler rows, and one Euler-parameter normalization row per mobile
 * body (Eq. 9.6.3).  The columns are [r, p] of each body (7 nbody).
 */
struct ConstraintEval
{
    SimpleArray<double> c; ///< Constraint values.
    SimpleArray<double> cq; ///< Velocity-form Jacobian [Phi_r, 2 Phi_pi E].
    SimpleArray<double> cq_pos; ///< Position-form Jacobian dPhi/d[r, p] for Newton-Raphson.
    SimpleArray<double> ct; ///< Explicit time derivative Phi_t.
    SimpleArray<double> gamma; ///< Acceleration right-hand side: Phi_q q'' = gamma.
}; /* end struct ConstraintEval */

/// Solver message for a warning panel.
struct Diagnostic
{
    std::string level; ///< "info", "warning", or "error".
    std::string code; ///< For example "redundant".
    int32_t joint; ///< Joint index in insertion order, or -1.
    int32_t row; ///< Row inside the joint, or -1.
    std::string message;
}; /* end struct Diagnostic */

/// Result of a quasi-static sweep; n is the number of converged values.
struct SweepResult
{
    SimpleArray<double> values; ///< (n) converged driver values.
    SimpleArray<double> positions; ///< (n, nbody, 3).
    SimpleArray<double> quaternions; ///< (n, nbody, 4).
    SimpleArray<double> dr_ds; ///< (n, nbody, 3), d r / d s.
    SimpleArray<double> domega_ds; ///< (n, nbody, 3), global rotation rate per unit s.
    SimpleArray<int32_t> iterations; ///< (n) Newton iterations.
    SimpleArray<double> residual; ///< (n) final |C|_inf.
    SimpleArray<double> rcond; ///< (n) sigma_min / sigma_max of the independent Jacobian rows.
    SimpleArray<int32_t> driver_joints; ///< (ndriver) joint indices of all drivers.
    SimpleArray<double> efforts; ///< (n, ndriver) quasi-static effort of each driver.
    bool completed = true;
    std::string message;
}; /* end struct SweepResult */

/**
 * Reaction force and torque of one joint (Haug Eq. 11.5.6, global frame).
 * torque_i acts about point_i and torque_j about point_j.  For a driver,
 * effort is the actuator force (translation) or torque (rotation) along the
 * driven coordinate.
 */
struct JointReaction
{
    int32_t body_i = 0;
    int32_t body_j = 0;
    Vector3 point_i, point_j;
    Vector3 force_i, torque_i, force_j, torque_j;
    bool is_driver = false;
    double effort = 0.0;
    SimpleArray<double> lambda; ///< (nc) multipliers of the joint rows.
}; /* end struct JointReaction */

struct EquilibriumResult
{
    bool converged = false;
    int32_t iterations = 0;
    double residual = 0.0;
    std::string message;
}; /* end struct EquilibriumResult */

class DynamicsSystem
{

public:

    /// Column offsets in a row of the state array (nbody, 14): [r, v, p, p_dot].
    static constexpr ssize_t STATE_R = 0;
    static constexpr ssize_t STATE_V = 3;
    static constexpr ssize_t STATE_P = 6;
    static constexpr ssize_t STATE_PDOT = 10;
    static constexpr ssize_t STATE_WIDTH = 14;

    /// dt is the output time step [s].
    explicit DynamicsSystem(double dt);
    DynamicsSystem(DynamicsSystem const &) = delete;
    DynamicsSystem(DynamicsSystem &&) = delete;
    DynamicsSystem & operator=(DynamicsSystem const &) = delete;
    DynamicsSystem & operator=(DynamicsSystem &&) = delete;
    ~DynamicsSystem() = default;

    /// Body numbers must follow the insertion order; body 0 is the ground.
    void add(BodyPtr body) { m_bodies.push_back(std::move(body)); }
    void add(JointPtr joint) { m_joints.push_back(std::move(joint)); }
    void add(ForceElementPtr element) { m_force_elements.push_back(std::move(element)); }
    void add(GearCouplerPtr coupler) { m_couplers.push_back(std::move(coupler)); }

    /// Forward positioning of tree topologies through the point-coincidence joints.
    void assemble();
    /// Assemble the configuration and velocities, check redundancy, and compute accelerations.
    void init();
    /// Advance one output step dt.
    void solve();

    /// Build the augmented system [M Cq^T; Cq 0]; stabilize = false omits the Baumgarte terms.
    void build_system(bool stabilize = true);

    /// Time integrator: "rk4" (default), "rk45", "abm4", "partition", "hybrid", "newmark", "hht", "houbolt".
    void set_integrator(std::string const & name);
    std::string integrator() const;
    /// Newton iterations of the last implicit step (0 for explicit methods).
    int32_t last_newton_iterations() const { return m_last_newton_iterations; }
    /// Accepted and rejected internal steps of the last rk45 solve().
    int32_t last_substeps() const { return m_rk45_substeps; }
    int32_t last_rejected() const { return m_rk45_rejected; }
    /// Error tolerances of the adaptive rk45 integrator.
    void set_tolerances(double rtol, double atol)
    {
        m_rk45_rtol = rtol;
        m_rk45_atol = atol;
    }
    /// HHT-alpha parameter, -1/3 <= alpha <= 0 (0 is the trapezoidal Newmark).
    void set_hht_alpha(double alpha);

    uint32_t nbody() const { return static_cast<uint32_t>(m_bodies.size()); }
    uint32_t njoint() const { return static_cast<uint32_t>(m_joints.size()); }
    /// Total number of constraint rows.
    uint32_t total_nconstraints() const;

    /// @name Body states, one row per body.
    ///@{
    SimpleArray<double> body_positions() const;
    SimpleArray<double> body_angles() const;
    SimpleArray<double> body_quaternions() const;
    SimpleArray<double> body_velocities() const;
    SimpleArray<double> body_angular_velocities() const;
    SimpleArray<double> body_accelerations() const;
    SimpleArray<double> body_angular_accelerations() const;
    ///@}

    /// Rank-based DOF from the last redundancy check (7 nbody - rank(Cq)).
    int32_t dof() const { return m_dof; }
    std::vector<Diagnostic> const & diagnostics() const { return m_diagnostics; }
    /// Joint reactions at the current state (needs init()).
    std::vector<JointReaction> joint_reactions() const;
    /**
     * Static equilibrium (Haug Sec. 6.5, 7.5): Phi(q) = 0 and Q(q) - Phi_q^T
     * lambda = 0 with zero velocities, by Newton iteration on a
     * finite-difference Jacobian.  The state keeps the start pose if the
     * iteration does not converge.
     */
    EquilibriumResult find_equilibrium(double tol = 1e-9, int32_t max_iter = 50);
    /**
     * Velocities of a state at rest except for the driven motion: the
     * minimum-norm solution of Phi_q q_dot = -Phi_t.  Zero when no driver moves
     * at the current time; a driver that moves at t (a sine road) gets its
     * velocity, so the motion starts consistent instead of with a Baumgarte
     * transient.
     */
    void set_rest_velocities();
    /// Quasi-static sweep: set the Parameter driver to each value and solve the positions.
    SweepResult sweep(JointPtr const & driver, std::vector<double> const & values);
    /**
     * Sweep several Parameter drivers together.  values[k][d] is driver d at
     * step k, rates[d] = dC_d/ds is used for the sensitivities, and the
     * reported values are values[k][0].
     */
    SweepResult sweep_group(std::vector<JointPtr> const & drivers, std::vector<std::vector<double>> const & values, std::vector<double> const & rates);

    /// Update the joints at time t and assemble all constraints at the current body states.
    ConstraintEval evaluate_constraints(double t);
    /// Overwrite the state of body i (and the state array after init()).
    void set_body_state(uint32_t i, Vector3 const & r, Vector3 const & v, Quaternion const & p, Quaternion const & p_dot);
    /// Applied load of mobile body i: global force at the center of mass and global torque.
    void set_applied_load(uint32_t i, Vector3 const & force, Vector3 const & torque);

    /// @name The augmented system of the last build_system().
    ///@{
    SimpleArray<double> const & jacobian() const { return m_cq; }
    SimpleArray<double> const & mass_matrix() const { return m_mass; }
    SimpleArray<double> const & augmented_matrix() const { return m_matrix; }
    SimpleArray<double> const & augmented_rhs() const { return m_rhs; }
    ///@}

    std::vector<BodyPtr> const & bodies() const { return m_bodies; }
    std::vector<JointPtr> const & joints() const { return m_joints; }
    std::vector<ForceElementPtr> const & force_elements() const { return m_force_elements; }
    /// State array (nbody, STATE_WIDTH); empty before init().
    SimpleArray<double> const & state() const { return m_q; }
    void set_state(SimpleArray<double> const & q);
    double sim_time() const { return m_sim_time; }

private:

    enum class IntegratorKind
    {
        RK4,
        ABM4,
        Partition,
        Hybrid,
        Newmark,
        Houbolt,
        RK45,
        HHT
    };

    bool has_state() const { return m_q.ndim() == 2 && m_q.shape(0) == static_cast<ssize_t>(nbody()); }
    ConstraintEval assemble_constraints() const;
    void check_redundancy();
    void compute_accelerations();
    double independent_rcond(SimpleArray<double> const & cq) const;
    void owner_of_row(uint32_t row, int32_t & joint, int32_t & local_row) const;
    void update_bodies_from_state();
    void update_joints(double t);
    int32_t solve_position(double t, int32_t max_iter, double tol);
    void solve_kinematic();
    void solve_dynamic();
    SimpleArray<double> equilibrium_residual(SimpleArray<double> const & x);

    // Integrators (Integrators.cpp).  The packed state is x = [r_i; p_i] and
    // xd = [v_i; p_dot_i] per body (7 nbody each).
    void reset_integrator_state();
    void step_rk4();
    void step_abm4();
    void step_partition();
    void step_hybrid();
    void step_newmark();
    void step_houbolt();
    void step_rk45();
    void after_step();
    void pack_state(SimpleArray<double> & x, SimpleArray<double> & xd) const;
    void unpack_state(SimpleArray<double> const & x, SimpleArray<double> const & xd);
    SimpleArray<double> accel_at(SimpleArray<double> const & x, SimpleArray<double> const & xd, double t);
    void select_partition(small_vector<ssize_t> & indep, small_vector<ssize_t> & dep);
    void project_dependent(SimpleArray<double> & x, SimpleArray<double> & xd, double t, small_vector<ssize_t> const & indep, small_vector<ssize_t> const & dep);
    void implicit_step(bool houbolt, double h, double alpha = 0.0);
    void newmark_substeps(double h, int32_t depth, double alpha = 0.0);
    /// Derivative of the state array q_in at time t (the stabilized equations of motion).
    void dynamic_function(SimpleArray<double> const & q_in, SimpleArray<double> & qd_out, double t);

    double m_dt;
    double m_sim_time = 0.0;
    uint32_t m_step_count = 0; ///< Dynamic steps taken, for periodic divergence checks.
    double m_eval_time = 0.0; ///< Time of the last update_joints() (the force elements use it).

    IntegratorKind m_integrator = IntegratorKind::RK4;
    int32_t m_last_newton_iterations = 0;
    double m_rk45_rtol = 1e-9;
    double m_rk45_atol = 1e-12;
    double m_hht_alpha = -0.05;
    double m_rk45_h = 0.0; ///< Last accepted internal step of rk45.
    int32_t m_rk45_rejected = 0;
    int32_t m_rk45_substeps = 0;
    std::vector<SimpleArray<double>> m_abm_history; ///< f_{n-3}, f_{n-2}, f_{n-1}.
    std::vector<SimpleArray<double>> m_past_positions; ///< Houbolt: x_{n-2}, x_{n-1}.
    SimpleArray<double> m_newmark_acc; ///< Newmark: a_n.

    SimpleArray<double> m_q; ///< (nbody, STATE_WIDTH).
    SimpleArray<double> m_cq;
    SimpleArray<double> m_mass;
    SimpleArray<double> m_matrix;
    SimpleArray<double> m_rhs;
    SimpleArray<double> m_lambda; ///< Lagrange multipliers at the current state.

    int32_t m_dof = 0;
    SimpleCollector<uint32_t> m_independent_rows; ///< Rows of Cq kept by the redundancy check.
    std::vector<Diagnostic> m_diagnostics;

    std::vector<BodyPtr> m_bodies;
    std::vector<JointPtr> m_joints;
    std::vector<ForceElementPtr> m_force_elements;
    std::vector<GearCouplerPtr> m_couplers; ///< Their rows follow the joint rows.

}; /* end class DynamicsSystem */

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
