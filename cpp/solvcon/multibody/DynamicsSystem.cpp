/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/multibody/DynamicsSystem.hpp>
#include <solvcon/multibody/linear.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace solvcon
{

namespace multibody
{

namespace
{

Vector3 vector3_at(SimpleArray<double> const & a, ssize_t row, ssize_t col)
{
    double const * p = a.vptr(row, col);
    return Vector3(p[0], p[1], p[2]);
}

Quaternion quaternion_at(SimpleArray<double> const & a, ssize_t row, ssize_t col)
{
    double const * p = a.vptr(row, col);
    return Quaternion(p[0], p[1], p[2], p[3]);
}

template <typename V>
void put(SimpleArray<double> & a, ssize_t row, ssize_t col, V const & v)
{
    double * p = a.vptr(row, col);
    for (size_t k = 0; k < v.size(); ++k)
    {
        p[k] = v[k];
    }
}

template <typename V>
V segment(SimpleArray<double> const & a, ssize_t begin)
{
    V out;
    for (size_t k = 0; k < out.size(); ++k)
    {
        out[k] = a[begin + static_cast<ssize_t>(k)];
    }
    return out;
}

template <typename V>
void put_segment(SimpleArray<double> & a, ssize_t begin, V const & v)
{
    for (size_t k = 0; k < v.size(); ++k)
    {
        a[begin + static_cast<ssize_t>(k)] = v[k];
    }
}

template <typename V>
void add_segment(SimpleArray<double> & a, ssize_t begin, V const & v)
{
    for (size_t k = 0; k < v.size(); ++k)
    {
        a[begin + static_cast<ssize_t>(k)] += v[k];
    }
}

// Elements [begin, begin + n) of a vector.
SimpleArray<double> slice(SimpleArray<double> const & v, ssize_t begin, ssize_t n)
{
    SimpleArray<double> out(small_vector<ssize_t>{n});
    std::copy_n(v.begin() + begin, n, out.begin());
    return out;
}

SimpleArray<double> negated(SimpleArray<double> const & v)
{
    SimpleArray<double> out(v);
    for (double & x : out)
    {
        x = -x;
    }
    return out;
}

// a - b for vectors of equal length.
SimpleArray<double> difference(SimpleArray<double> const & a, SimpleArray<double> const & b)
{
    SimpleArray<double> out(a);
    for (ssize_t k = 0; k < static_cast<ssize_t>(out.size()); ++k)
    {
        out[k] -= b[k];
    }
    return out;
}

// Add a block b at (row, col) of a.
void add_block(SimpleArray<double> & a, ssize_t row, ssize_t col, SimpleArray<double> const & b)
{
    for (ssize_t i = 0; i < b.shape(0); ++i)
    {
        double * dst = a.vptr(row + i, col);
        double const * src = b.vptr(i, 0);
        for (ssize_t j = 0; j < b.shape(1); ++j)
        {
            dst[j] += src[j];
        }
    }
}

// Copy the collected elements into a new array of the given shape.
template <typename T>
SimpleArray<T> collected(SimpleCollector<T> const & c, small_vector<ssize_t> const & shape)
{
    SimpleArray<T> out(shape);
    for (size_t k = 0; k < c.size(); ++k)
    {
        out[k] = c[k];
    }
    return out;
}

} /* end namespace */

DynamicsSystem::DynamicsSystem(double dt)
    : m_dt(dt)
{
}

ConstraintEval DynamicsSystem::assemble_constraints() const
{
    ssize_t const ndof = static_cast<ssize_t>(nbody()) * 7;
    ssize_t const total_nc = total_nconstraints();

    ConstraintEval ev;
    ev.c = zero_vector(total_nc);
    ev.ct = zero_vector(total_nc);
    ev.gamma = zero_vector(total_nc);
    ev.cq = zero_matrix(total_nc, ndof);
    ev.cq_pos = zero_matrix(total_nc, ndof);

    // Ground: r = 0 and p = [1, 0, 0, 0].
    for (ssize_t k = 0; k < 7; ++k)
    {
        ev.cq(k, k) = 1.0;
        ev.cq_pos(k, k) = 1.0;
    }
    put_segment(ev.c, 0, m_bodies[0]->position());
    put_segment(ev.c, 3, m_bodies[0]->quaternion() - Quaternion::identity());

    ssize_t row = 7;
    for (auto const & jt : m_joints)
    {
        auto const jnc = static_cast<ssize_t>(jt->nconstraints());
        ssize_t const ci = static_cast<ssize_t>(jt->body_i()->num()) * 7;
        ssize_t const cj = static_cast<ssize_t>(jt->body_j()->num()) * 7;
        std::copy_n(jt->constraint().begin(), jnc, ev.c.begin() + row);
        std::copy_n(jt->ct().begin(), jnc, ev.ct.begin() + row);
        std::copy_n(jt->gamma().begin(), jnc, ev.gamma.begin() + row);
        // Add so that a joint between a body and itself cannot overwrite a block.
        add_block(ev.cq, row, ci, jt->jacobian_i());
        add_block(ev.cq, row, cj, jt->jacobian_j());
        add_block(ev.cq_pos, row, ci, jt->position_jacobian_i());
        add_block(ev.cq_pos, row, cj, jt->position_jacobian_j());
        row += jnc;
    }

    for (auto const & gc : m_couplers)
    {
        gc->update();
        ev.c[row] = gc->constraint();
        ev.gamma[row] = gc->gamma();
        for (ssize_t k = 0; k < static_cast<ssize_t>(gc->bodies().size()); ++k)
        {
            ssize_t const col = static_cast<ssize_t>(gc->bodies()[k]) * 7;
            for (ssize_t c = 0; c < 7; ++c)
            {
                ev.cq(row, col + c) += gc->jacobian()(k, c);
                ev.cq_pos(row, col + c) += gc->position_jacobian()(k, c);
            }
        }
        ++row;
    }

    for (ssize_t i = 1; i < static_cast<ssize_t>(nbody()); ++i)
    {
        Quaternion const & p = m_bodies[i]->quaternion();
        Quaternion const & pdot = m_bodies[i]->p_dot();
        ev.c[row] = dot(p, p) - 1.0;
        put(ev.cq, row, i * 7 + 3, 2.0 * p);
        put(ev.cq_pos, row, i * 7 + 3, 2.0 * p);
        ev.gamma[row] = -2.0 * dot(pdot, pdot);
        ++row;
    }
    return ev;
}

void DynamicsSystem::update_bodies_from_state()
{
    for (ssize_t i = 0; i < static_cast<ssize_t>(nbody()); ++i)
    {
        m_bodies[i]->update(vector3_at(m_q, i, STATE_R), vector3_at(m_q, i, STATE_V), quaternion_at(m_q, i, STATE_P), quaternion_at(m_q, i, STATE_PDOT));
    }
}

void DynamicsSystem::update_joints(double t)
{
    m_eval_time = t;
    for (auto & jt : m_joints)
    {
        jt->set_time(t);
        jt->update();
    }
}

ConstraintEval DynamicsSystem::evaluate_constraints(double t)
{
    update_joints(t);
    return assemble_constraints();
}

void DynamicsSystem::set_body_state(uint32_t i, Vector3 const & r, Vector3 const & v, Quaternion const & p, Quaternion const & p_dot)
{
    if (i >= nbody())
    {
        throw std::out_of_range("set_body_state: body index " + std::to_string(i) + " >= nbody " + std::to_string(nbody()));
    }
    if (has_state())
    {
        auto const row = static_cast<ssize_t>(i);
        put(m_q, row, STATE_R, r);
        put(m_q, row, STATE_V, v);
        put(m_q, row, STATE_P, p);
        put(m_q, row, STATE_PDOT, p_dot);
    }
    m_bodies[i]->update(r, v, p, p_dot);
}

void DynamicsSystem::set_state(SimpleArray<double> const & q)
{
    m_q = SimpleArray<double>(q);
}

// Newton-Raphson position analysis at time t (Haug Sec. 4.5, 9.6.1).  Returns
// the number of iterations; throws if the residual does not drop below tol.
int32_t DynamicsSystem::solve_position(double t, int32_t max_iter, double tol)
{
    double c_err = 0.0;
    for (int32_t iter = 0; iter <= max_iter; ++iter)
    {
        update_bodies_from_state();
        ConstraintEval ev = evaluate_constraints(t);
        c_err = norm_inf(ev.c);
        if (c_err < tol)
        {
            return iter;
        }
        if (iter == max_iter)
        {
            break;
        }

        SimpleArray<double> const dq = robust_solve(ev.cq_pos, negated(ev.c));
        if (!all_finite(dq))
        {
            throw std::runtime_error("[MBS] Position analysis: non-finite Newton step at t=" + std::to_string(t));
        }
        // Limit the step so that one bad linearization cannot throw bodies far away.
        double step = 1.0;
        double const dq_norm = norm_inf(dq);
        if (dq_norm > 1.0)
        {
            step = 1.0 / dq_norm;
        }
        for (ssize_t i = 1; i < static_cast<ssize_t>(nbody()); ++i)
        {
            for (ssize_t k = 0; k < 3; ++k)
            {
                m_q(i, STATE_R + k) += step * dq[i * 7 + k];
            }
            for (ssize_t k = 0; k < 4; ++k)
            {
                m_q(i, STATE_P + k) += step * dq[i * 7 + 3 + k];
            }
            put(m_q, i, STATE_P, quaternion_at(m_q, i, STATE_P) / norm(quaternion_at(m_q, i, STATE_P)));
        }
    }
    throw std::runtime_error("[MBS] Position analysis did not converge at t=" + std::to_string(t) + " (residual=" + std::to_string(c_err) + ", tol=" + std::to_string(tol) + ")");
}

// Redundant constraint detection (Haug Sec. 4.6).  Greedy Gram-Schmidt over
// the rows of Cq in insertion order: a row whose component outside the span
// of the earlier kept rows is tiny is redundant.  The rank gives the actual
// DOF used to choose kinematic or dynamic analysis.
void DynamicsSystem::owner_of_row(uint32_t row, int32_t & joint, int32_t & local_row) const
{
    joint = -1;
    local_row = -1;
    if (row < 7)
    {
        local_row = static_cast<int32_t>(row);
        return;
    }
    uint32_t r = 7;
    for (uint32_t k = 0; k < njoint(); ++k)
    {
        uint32_t const jnc = m_joints[k]->nconstraints();
        if (row < r + jnc)
        {
            joint = static_cast<int32_t>(k);
            local_row = static_cast<int32_t>(row - r);
            return;
        }
        r += jnc;
    }
    if (row < r + m_couplers.size())
    {
        // Gear coupler k is reported as joint njoint + k.
        joint = static_cast<int32_t>(njoint() + (row - r));
        local_row = 0;
        return;
    }
    r += static_cast<uint32_t>(m_couplers.size());
    // The normalization row of body local_row + 1.
    local_row = static_cast<int32_t>(row - r);
}

void DynamicsSystem::check_redundancy()
{
    update_bodies_from_state();
    ConstraintEval ev = evaluate_constraints(m_sim_time);
    SimpleArray<double> const & a = ev.cq;
    ssize_t const nrows = a.shape(0), ncols = a.shape(1);

    std::erase_if(m_diagnostics, [](Diagnostic const & d)
                  { return d.code == "redundant"; });
    m_independent_rows.clear();

    constexpr double rel_tol = 1e-8;
    std::vector<SimpleArray<double>> basis;
    for (ssize_t r = 0; r < nrows; ++r)
    {
        SimpleArray<double> v(small_vector<ssize_t>{ncols});
        std::copy_n(a.vptr(r, 0), ncols, v.begin());
        double const row_norm = norm2(v);
        if (row_norm == 0.0)
        {
            int32_t j, lr;
            owner_of_row(static_cast<uint32_t>(r), j, lr);
            m_diagnostics.push_back({"warning", "redundant", j, lr, "constraint row " + std::to_string(r) + " has a zero Jacobian"});
            continue;
        }
        // The second pass restores the orthogonality lost to rounding.
        for (int32_t pass = 0; pass < 2; ++pass)
        {
            for (auto const & b : basis)
            {
                double d = 0.0;
                for (ssize_t k = 0; k < ncols; ++k)
                {
                    d += b[k] * v[k];
                }
                for (ssize_t k = 0; k < ncols; ++k)
                {
                    v[k] -= d * b[k];
                }
            }
        }
        double const resid = norm2(v);
        if (resid > rel_tol * row_norm)
        {
            for (double & x : v)
            {
                x /= resid;
            }
            basis.push_back(std::move(v));
            m_independent_rows.push_back(static_cast<uint32_t>(r));
        }
        else
        {
            int32_t j, lr;
            owner_of_row(static_cast<uint32_t>(r), j, lr);
            m_diagnostics.push_back({"warning", "redundant", j, lr, "joint " + std::to_string(j) + " row " + std::to_string(lr) + " is redundant: it depends linearly on earlier constraints"});
        }
    }
    m_dof = static_cast<int32_t>(ncols) - static_cast<int32_t>(m_independent_rows.size());
}

double DynamicsSystem::independent_rcond(SimpleArray<double> const & cq) const
{
    auto const nrows = static_cast<ssize_t>(m_independent_rows.size());
    ssize_t const ncols = cq.shape(1);
    SimpleArray<double> rows(small_vector<ssize_t>{nrows, ncols});
    for (ssize_t k = 0; k < nrows; ++k)
    {
        std::copy_n(cq.vptr(static_cast<ssize_t>(m_independent_rows[k]), 0), ncols, rows.vptr(k, 0));
    }
    SimpleArray<double> const s = singular_values(rows);
    if (s.size() == 0)
    {
        return 0.0;
    }
    double const smax = *std::max_element(s.begin(), s.end());
    if (smax == 0.0)
    {
        return 0.0;
    }
    return *std::min_element(s.begin(), s.end()) / smax;
}

// Accelerations at the current state.  Kinematic mode: Cq q'' = gamma (Haug
// Sec. 9.6.3), then the inverse dynamics Cq^T lambda = Q - M q'' (Sec. 6.4,
// 11.5).  Dynamic mode: [M Cq^T; Cq 0][q''; lambda] = [Q; gamma_stab].  The
// global angular acceleration is alpha = 2 E p'' (E_dot p_dot = 0).
void DynamicsSystem::compute_accelerations()
{
    ssize_t const ndof = static_cast<ssize_t>(nbody()) * 7;
    update_bodies_from_state();
    update_joints(m_sim_time);
    SimpleArray<double> qdd;
    if (m_dof <= 0)
    {
        ConstraintEval ev = assemble_constraints();
        qdd = robust_solve(ev.cq, ev.gamma);
        build_system();
        m_lambda = robust_solve(transposed(m_cq), difference(slice(m_rhs, 0, ndof), matvec(m_mass, qdd)));
    }
    else
    {
        build_system();
        SimpleArray<double> const ans = robust_solve(m_matrix, m_rhs);
        // Redundant constraints keep the augmented system consistent, so the
        // minimum-norm answer satisfies it.  A large residual means the
        // system is singular for the loads (a diverged state makes the
        // constraint rows dwarf the mass rows); answering with the truncated
        // solution would drop the accelerations and hide the divergence.
        SimpleArray<double> const residual = difference(matvec(m_matrix, ans), m_rhs);
        double const scale = std::max(1.0, norm_inf(m_rhs));
        if (!(norm_inf(residual) <= 1e-6 * scale))
        {
            throw std::runtime_error("[MBS] acceleration solve at t=" + std::to_string(m_sim_time) + " has no consistent solution (residual " + std::to_string(norm_inf(residual) / scale) +
                                     " of the load); the system is singular or the simulation diverged");
        }
        qdd = slice(ans, 0, ndof);
        m_lambda = slice(ans, ndof, static_cast<ssize_t>(ans.size()) - ndof);
    }
    for (ssize_t i = 0; i < static_cast<ssize_t>(nbody()); ++i)
    {
        m_bodies[i]->set_acceleration(segment<Vector3>(qdd, i * 7));
        Quaternion const pdd = segment<Quaternion>(qdd, i * 7 + 3);
        m_bodies[i]->set_angular_acceleration(2.0 * (build_E(m_bodies[i]->quaternion()) * pdd));
    }
}

SweepResult DynamicsSystem::sweep(JointPtr const & driver, std::vector<double> const & values)
{
    std::vector<std::vector<double>> rows;
    rows.reserve(values.size());
    for (double v : values)
    {
        rows.push_back({v});
    }
    return sweep_group({driver}, rows, {1.0});
}

// Quasi-static sweep over a group of Parameter drivers (Haug Sec. 3.6.4 on a
// value grid).
SweepResult DynamicsSystem::sweep_group(std::vector<JointPtr> const & drivers, std::vector<std::vector<double>> const & values, std::vector<double> const & rates)
{
    if (!has_state())
    {
        throw std::logic_error("sweep: call init() before sweep()");
    }
    if (drivers.empty() || rates.size() != drivers.size())
    {
        throw std::invalid_argument("sweep: need one rate per driver");
    }
    small_vector<ssize_t> rows_of;
    for (auto const & driver : drivers)
    {
        if (!driver->is_driver())
        {
            throw std::invalid_argument("sweep: joint type " + std::to_string(driver->type()) + " is not a driver");
        }
        if (driver->drive().kind() != DriveFunction::Kind::Parameter)
        {
            throw std::invalid_argument("sweep: the driver must use DriveFunction.parameter()");
        }
        ssize_t r = 7;
        bool found = false;
        for (auto const & jt : m_joints)
        {
            if (jt == driver)
            {
                found = true;
                break;
            }
            r += jt->nconstraints();
        }
        if (!found)
        {
            throw std::invalid_argument("sweep: a driver is not part of this system");
        }
        rows_of.push_back(r);
    }
    for (auto const & row : values)
    {
        if (row.size() != drivers.size())
        {
            throw std::invalid_argument("sweep: each value row needs one value per driver");
        }
    }

    SweepResult res;
    SimpleCollector<int32_t> driver_joints;
    for (uint32_t k = 0; k < njoint(); ++k)
    {
        if (m_joints[k]->is_driver())
        {
            driver_joints.push_back(static_cast<int32_t>(k));
        }
    }
    auto const ndriver = static_cast<ssize_t>(driver_joints.size());
    auto const nb = static_cast<ssize_t>(nbody());
    SimpleCollector<double> out_values, out_positions, out_quaternions, out_dr_ds, out_domega_ds, out_residual, out_rcond, out_efforts;
    SimpleCollector<int32_t> out_iterations;

    std::vector<double> last_good(drivers.size());
    for (size_t d = 0; d < drivers.size(); ++d)
    {
        double cd, cdd;
        drivers[d]->drive().eval(m_sim_time, last_good[d], cd, cdd);
    }
    for (auto const & row_values : values)
    {
        SimpleArray<double> const backup(m_q);
        for (size_t d = 0; d < drivers.size(); ++d)
        {
            drivers[d]->drive().set_value(row_values[d]);
        }
        double const value = row_values[0];
        int32_t iters = 0;
        try
        {
            iters = solve_position(m_sim_time, 100, 1e-10);
        }
        catch (std::runtime_error const & e)
        {
            m_q = backup;
            // The state matches the last good row.
            for (size_t d = 0; d < drivers.size(); ++d)
            {
                drivers[d]->drive().set_value(last_good[d]);
            }
            update_bodies_from_state();
            update_joints(m_sim_time);
            res.completed = false;
            std::ostringstream msg;
            msg << "stopped at value " << value << ": " << e.what();
            res.message = msg.str();
            break;
        }
        update_bodies_from_state();
        ConstraintEval ev = evaluate_constraints(m_sim_time);

        // Sensitivity: Cq dq/ds = -dPhi/ds = -sum_d dPhi_d/dC_d rate_d.
        SimpleArray<double> rhs = zero_vector(static_cast<ssize_t>(ev.c.size()));
        for (size_t d = 0; d < drivers.size(); ++d)
        {
            rhs[rows_of[d]] = -drivers[d]->drive_sensitivity() * rates[d];
        }
        SimpleArray<double> const dq = robust_solve(ev.cq, rhs);

        for (ssize_t i = 0; i < nb; ++i)
        {
            for (ssize_t k = 0; k < 3; ++k)
            {
                out_positions.push_back(m_q(i, STATE_R + k));
                out_dr_ds.push_back(dq[i * 7 + k]);
            }
            for (ssize_t k = 0; k < 4; ++k)
            {
                out_quaternions.push_back(m_q(i, STATE_P + k));
            }
            Vector3 const dw = 2.0 * (build_E(m_bodies[i]->quaternion()) * segment<Quaternion>(dq, i * 7 + 3));
            for (size_t k = 0; k < 3; ++k)
            {
                out_domega_ds.push_back(dw[k]);
            }
        }
        last_good = row_values;
        out_values.push_back(value);
        out_iterations.push_back(iters);
        out_residual.push_back(norm_inf(ev.c));
        out_rcond.push_back(independent_rcond(ev.cq));

        // Quasi-static loads: zero velocities, then the static multipliers.  In a
        // fully driven system (DOF 0) this is the inverse dynamics with q'' = 0.
        for (ssize_t i = 0; i < nb; ++i)
        {
            put(m_q, i, STATE_V, Vector3());
            put(m_q, i, STATE_PDOT, Quaternion());
        }
        compute_accelerations();
        if (m_lambda.size() == total_nconstraints())
        {
            std::vector<JointReaction> const reac = joint_reactions();
            for (ssize_t k = 0; k < ndriver; ++k)
            {
                out_efforts.push_back(reac[driver_joints[k]].effort);
            }
        }
        else
        {
            for (ssize_t k = 0; k < ndriver; ++k)
            {
                out_efforts.push_back(0.0);
            }
        }
    }

    auto const n = static_cast<ssize_t>(out_values.size());
    res.values = collected(out_values, small_vector<ssize_t>{n});
    res.positions = collected(out_positions, small_vector<ssize_t>{n, nb, 3});
    res.quaternions = collected(out_quaternions, small_vector<ssize_t>{n, nb, 4});
    res.dr_ds = collected(out_dr_ds, small_vector<ssize_t>{n, nb, 3});
    res.domega_ds = collected(out_domega_ds, small_vector<ssize_t>{n, nb, 3});
    res.iterations = collected(out_iterations, small_vector<ssize_t>{n});
    res.residual = collected(out_residual, small_vector<ssize_t>{n});
    res.rcond = collected(out_rcond, small_vector<ssize_t>{n});
    res.driver_joints = collected(driver_joints, small_vector<ssize_t>{ndriver});
    res.efforts = collected(out_efforts, small_vector<ssize_t>{n, ndriver});
    return res;
}

// The augmented dynamic system (Haug Eq. 11.3.x in Euler parameters).
void DynamicsSystem::build_system(bool stabilize)
{
    auto const nb = static_cast<ssize_t>(nbody());
    ssize_t const ndof = nb * 7;
    ConstraintEval ev = assemble_constraints();
    auto const total_nc = static_cast<ssize_t>(ev.c.size());
    ssize_t const sys_size = ndof + total_nc;

    m_cq = std::move(ev.cq);
    m_mass = zero_matrix(ndof, ndof);
    m_matrix = zero_matrix(sys_size, sys_size);
    m_rhs = zero_vector(sys_size);

    // Block-diagonal 7-by-7 mass matrix and the applied loads.
    for (ssize_t i = 0; i < nb; ++i)
    {
        Body const & b = *m_bodies[i];
        for (ssize_t k = 0; k < 3; ++k)
        {
            m_mass(i * 7 + k, i * 7 + k) = b.mass();
        }
        Matrix4 const & mp = b.rotational_mass();
        for (size_t r = 0; r < 4; ++r)
        {
            for (size_t c = 0; c < 4; ++c)
            {
                m_mass(i * 7 + 3 + static_cast<ssize_t>(r), i * 7 + 3 + static_cast<ssize_t>(c)) = mp(r, c);
            }
        }
        put_segment(m_rhs, i * 7, b.force());
        put_segment(m_rhs, i * 7 + 3, b.torque());
    }

    // Force elements: all loads first, then the accumulation.  A global torque T
    // does the virtual work T . dpi = T . 2 E dp, so Q_p = 2 E^T T.
    for (auto & fe : m_force_elements)
    {
        fe->set_time(m_eval_time);
        fe->compute();
    }
    for (auto const & fe : m_force_elements)
    {
        ssize_t const ci = static_cast<ssize_t>(fe->body_i()->num()) * 7;
        ssize_t const cj = static_cast<ssize_t>(fe->body_j()->num()) * 7;
        add_segment(m_rhs, ci, fe->force_i());
        add_segment(m_rhs, cj, fe->force_j());
        add_segment(m_rhs, ci + 3, 2.0 * transpose_mul(build_E(fe->body_i()->quaternion()), fe->torque_i()));
        add_segment(m_rhs, cj + 3, 2.0 * transpose_mul(build_E(fe->body_j()->quaternion()), fe->torque_j()));
    }

    // Baumgarte constraint stabilization (Haug Sec. 7.3.4).
    constexpr double baumgarte_alpha = 10.0;
    constexpr double baumgarte_beta = 10.0;
    SimpleArray<double> gamma = std::move(ev.gamma);
    if (stabilize)
    {
        SimpleArray<double> qdot(small_vector<ssize_t>{ndof});
        for (ssize_t i = 0; i < nb; ++i)
        {
            put_segment(qdot, i * 7, m_bodies[i]->velocity());
            put_segment(qdot, i * 7 + 3, m_bodies[i]->p_dot());
        }
        SimpleArray<double> const cq_qdot = matvec(m_cq, qdot);
        for (ssize_t k = 0; k < total_nc; ++k)
        {
            gamma[k] += -2.0 * baumgarte_alpha * (cq_qdot[k] + ev.ct[k]) - baumgarte_beta * baumgarte_beta * ev.c[k];
        }
    }

    for (ssize_t i = 0; i < ndof; ++i)
    {
        std::copy_n(m_mass.vptr(i, 0), ndof, m_matrix.vptr(i, 0));
    }
    for (ssize_t r = 0; r < total_nc; ++r)
    {
        for (ssize_t c = 0; c < ndof; ++c)
        {
            m_matrix(c, ndof + r) = m_cq(r, c);
            m_matrix(ndof + r, c) = m_cq(r, c);
        }
        m_rhs[ndof + r] = gamma[r];
    }
}

// Forward positioning for tree topologies.  Only the point-coincidence joints
// (spherical, revolute, universal) place body j; other joints and drivers do
// not define a coincident point.  Velocities are left to init().
void DynamicsSystem::assemble()
{
    for (auto const & jt : m_joints)
    {
        uint32_t const type = jt->type();
        if (type != 0 && type != 1 && type != 4)
        {
            continue;
        }
        BodyPtr const & bi = jt->body_i();
        BodyPtr const & bj = jt->body_j();
        if (bj->num() == 0)
        {
            continue;
        }
        bj->set_position(bi->position() + transpose_mul(bi->tbi(), jt->point_i()) - transpose_mul(bj->tbi(), jt->point_j()));
    }
}

void DynamicsSystem::init()
{
    auto const nb = static_cast<ssize_t>(nbody());
    ssize_t const ndof = nb * 7;
    m_q = zero_matrix(nb, STATE_WIDTH);
    reset_integrator_state();
    for (ssize_t i = 0; i < nb; ++i)
    {
        Body const & b = *m_bodies[i];
        put(m_q, i, STATE_R, b.position());
        put(m_q, i, STATE_V, b.velocity());
        put(m_q, i, STATE_P, b.quaternion());
        put(m_q, i, STATE_PDOT, b.p_dot());
    }

    // DOF analysis of the 7-coordinate formulation.
    ssize_t const total_nc = total_nconstraints();
    auto const dof = static_cast<int32_t>(ndof - total_nc);
    std::cerr << "[MBS] DOF analysis: " << ndof << " DOF - " << total_nc << " constraints = " << dof << " free DOF" << std::endl;
    if (dof < 0)
    {
        std::cerr << "[MBS] WARNING: Over-constrained by " << (-dof) << ". SVD solver will handle redundancy." << std::endl;
    }
    else if (dof == 0)
    {
        std::cerr << "[MBS] INFO: 0 DOF (kinematic system)." << std::endl;
    }

    // Capture the user velocities now: the Newton loop below calls update(),
    // which recomputes the angular velocity from p_dot at the corrected p.
    std::vector<Vector3> user_vel(nb), user_omega(nb);
    for (ssize_t i = 0; i < nb; ++i)
    {
        user_vel[i] = m_bodies[i]->velocity();
        user_omega[i] = m_bodies[i]->angular_velocity();
    }

    // Newton-Raphson position correction (assembly, Haug Sec. 4.3).
    {
        update_bodies_from_state();
        double const c0 = norm_inf(evaluate_constraints(m_sim_time).c);
        std::cerr << "[MBS] Assembly: initial constraint error = " << c0 << std::endl;
        int32_t const iters = solve_position(m_sim_time, 500, 1e-10);
        std::cerr << "[MBS] Assembly converged in " << iters << " iterations" << std::endl;
    }

    check_redundancy();
    if (m_dof != dof)
    {
        std::cerr << "[MBS] Rank-based DOF = " << m_dof << " (" << (m_dof - dof) << " redundant constraints)" << std::endl;
    }

    // Consistent velocities: Cq [v; p_dot] = -Ct.
    {
        update_bodies_from_state();
        ConstraintEval ev = evaluate_constraints(m_sim_time);
        SimpleArray<double> const & cq = ev.cq;
        SimpleArray<double> const neg_ct = negated(ev.ct);

        // Prescribed velocities S qdot = s.  Each non-zero user vector is one
        // block of 3 rows: v_i, or omega_i = 2 E(p_i) p_dot_i (global).  A zero
        // vector means "not specified".
        SimpleCollector<double> s_rows, s_values;
        ssize_t ns = 0;
        auto add_rows = [&](ssize_t col, Matrix34 const * e, Vector3 const & value)
        {
            for (ssize_t r = 0; r < 3; ++r)
            {
                for (ssize_t c = 0; c < ndof; ++c)
                {
                    double v = 0.0;
                    if (e == nullptr)
                    {
                        v = (c == col + r) ? 1.0 : 0.0;
                    }
                    else if (c >= col && c < col + 4)
                    {
                        v = 2.0 * (*e)(static_cast<size_t>(r), static_cast<size_t>(c - col));
                    }
                    s_rows.push_back(v);
                }
                s_values.push_back(value[static_cast<size_t>(r)]);
            }
            ns += 3;
        };
        for (ssize_t i = 1; i < nb; ++i)
        {
            if (norm(user_vel[i]) > 0.0)
            {
                add_rows(i * 7, nullptr, user_vel[i]);
            }
            if (norm(user_omega[i]) > 0.0)
            {
                Matrix34 const e = build_E(m_bodies[i]->quaternion());
                add_rows(i * 7 + 3, &e, user_omega[i]);
            }
        }
        SimpleArray<double> const smat = collected(s_rows, small_vector<ssize_t>{ns, ndof});
        SimpleArray<double> const svec = collected(s_values, small_vector<ssize_t>{ns});

        // Every solution of Cq qdot = -Ct is qdot = qp + N z, where qp is the
        // minimum-norm particular solution and N spans null(Cq).  The joint
        // equations hold exactly.  Inside null(Cq), z is the minimum-norm
        // least-squares fit to S qdot = s, so unspecified motion stays as small
        // as possible and conflicting input is fitted, not forced.
        SimpleArray<double> const nmat = null_space(cq);
        ssize_t const nn = (nmat.ndim() == 2) ? nmat.shape(1) : 0;
        SimpleArray<double> qdot = robust_solve(cq, neg_ct);
        if (nn > 0)
        {
            SimpleArray<double> const proj = matvec(nmat, matvec_transpose(nmat, qdot));
            for (ssize_t k = 0; k < ndof; ++k)
            {
                qdot[k] -= proj[k];
            }
        }
        if (ns > 0 && nn > 0)
        {
            SimpleArray<double> const z = matvec(pseudo_inverse(matmul(smat, nmat)), difference(svec, matvec(smat, qdot)));
            SimpleArray<double> const nz = matvec(nmat, z);
            for (ssize_t k = 0; k < ndof; ++k)
            {
                qdot[k] += nz[k];
            }
        }
        if (ns > 0)
        {
            double const pres_err = norm_inf(difference(matvec(smat, qdot), svec));
            if (pres_err > 1e-8 * std::max(1.0, norm_inf(svec)))
            {
                std::cerr << "[MBS] WARNING: user initial velocities conflict with joints; "
                             "least-squares fit used (residual="
                          << pres_err << ")" << std::endl;
            }
        }

        for (ssize_t i = 1; i < nb; ++i)
        {
            put(m_q, i, STATE_V, segment<Vector3>(qdot, i * 7));
            put(m_q, i, STATE_PDOT, segment<Quaternion>(qdot, i * 7 + 3));
        }
        update_bodies_from_state();

        double const vel_err_after = norm_inf(difference(matvec(cq, qdot), neg_ct));
        std::cerr << "[MBS] Velocity solve: Cq*qdot-(-Ct) error=" << vel_err_after << std::endl;
    }

    update_joints(m_sim_time);
    build_system();
    compute_accelerations();
}

void DynamicsSystem::solve()
{
    // m_dof is the rank-based DOF from init(), so redundant constraints do not
    // turn a free mechanism into a kinematic one.
    if (m_dof <= 0)
    {
        solve_kinematic();
    }
    else
    {
        solve_dynamic();
    }
}

// Kinematic analysis: positions, then the velocities Cq [v; p_dot] = -Ct (Haug Sec. 9.6).
void DynamicsSystem::solve_kinematic()
{
    m_sim_time += m_dt;
    solve_position(m_sim_time, 200, 1e-10);

    update_bodies_from_state();
    ConstraintEval ev = evaluate_constraints(m_sim_time);
    SimpleArray<double> const qdot = robust_solve(ev.cq, negated(ev.ct));
    for (ssize_t i = 1; i < static_cast<ssize_t>(nbody()); ++i)
    {
        put(m_q, i, STATE_V, segment<Vector3>(qdot, i * 7));
        put(m_q, i, STATE_PDOT, segment<Quaternion>(qdot, i * 7 + 3));
    }
    update_bodies_from_state();
    compute_accelerations();
}

void DynamicsSystem::solve_dynamic()
{
    switch (m_integrator)
    {
    case IntegratorKind::RK4: step_rk4(); break;
    case IntegratorKind::ABM4: step_abm4(); break;
    case IntegratorKind::Partition: step_partition(); break;
    case IntegratorKind::Hybrid: step_hybrid(); break;
    case IntegratorKind::Newmark: step_newmark(); break;
    case IntegratorKind::Houbolt: step_houbolt(); break;
    case IntegratorKind::RK45: step_rk45(); break;
    case IntegratorKind::HHT: newmark_substeps(m_dt, 0, m_hht_alpha); break;
    }
    after_step();
}

void DynamicsSystem::after_step()
{
    compute_accelerations();
    // Divergence check every 100 steps.
    ++m_step_count;
    if (m_step_count % 100 == 0)
    {
        for (ssize_t i = 0; i < static_cast<ssize_t>(nbody()); ++i)
        {
            Vector3 const r = vector3_at(m_q, i, STATE_R);
            if (!std::isfinite(r[0]) || !std::isfinite(r[1]) || !std::isfinite(r[2]))
            {
                throw std::runtime_error("[MBS] ERROR: Body " + std::to_string(i) + " position is NaN/Inf at t=" + std::to_string(m_sim_time) + "s. Simulation diverged.");
            }
        }
    }
}

// The derivative of the state array: [v, a, p_dot, p_ddot] per body.
void DynamicsSystem::dynamic_function(SimpleArray<double> const & q_in, SimpleArray<double> & qd_out, double t)
{
    for (ssize_t i = 0; i < static_cast<ssize_t>(nbody()); ++i)
    {
        m_bodies[i]->update(vector3_at(q_in, i, STATE_R), vector3_at(q_in, i, STATE_V), quaternion_at(q_in, i, STATE_P), quaternion_at(q_in, i, STATE_PDOT));
    }
    for (auto & jt : m_joints)
    {
        jt->set_time(t);
        jt->update();
    }
    build_system();
    SimpleArray<double> const ans = robust_solve(m_matrix, m_rhs);
    for (ssize_t i = 0; i < static_cast<ssize_t>(nbody()); ++i)
    {
        put(qd_out, i, STATE_R, vector3_at(q_in, i, STATE_V));
        put(qd_out, i, STATE_V, segment<Vector3>(ans, i * 7));
        put(qd_out, i, STATE_P, quaternion_at(q_in, i, STATE_PDOT));
        put(qd_out, i, STATE_PDOT, segment<Quaternion>(ans, i * 7 + 3));
    }
}

namespace
{

template <typename V>
SimpleArray<double> body_rows(std::vector<BodyPtr> const & bodies, V const & (Body::*getter)() const)
{
    auto const nb = static_cast<ssize_t>(bodies.size());
    auto const width = static_cast<ssize_t>(V().size());
    SimpleArray<double> out(small_vector<ssize_t>{nb, width});
    for (ssize_t i = 0; i < nb; ++i)
    {
        put(out, i, 0, ((*bodies[i]).*getter)());
    }
    return out;
}

} /* end namespace */

SimpleArray<double> DynamicsSystem::body_positions() const { return body_rows(m_bodies, &Body::position); }

SimpleArray<double> DynamicsSystem::body_angles() const { return body_rows(m_bodies, &Body::angles); }

SimpleArray<double> DynamicsSystem::body_quaternions() const { return body_rows(m_bodies, &Body::quaternion); }

SimpleArray<double> DynamicsSystem::body_velocities() const { return body_rows(m_bodies, &Body::velocity); }

SimpleArray<double> DynamicsSystem::body_angular_velocities() const { return body_rows(m_bodies, &Body::angular_velocity); }

SimpleArray<double> DynamicsSystem::body_accelerations() const { return body_rows(m_bodies, &Body::acceleration); }

SimpleArray<double> DynamicsSystem::body_angular_accelerations() const { return body_rows(m_bodies, &Body::angular_acceleration); }

uint32_t DynamicsSystem::total_nconstraints() const
{
    // The ground rows, the joint rows, the coupler rows, and the normalization rows.
    uint32_t total_nc = 7;
    for (auto const & jt : m_joints)
    {
        total_nc += jt->nconstraints();
    }
    total_nc += static_cast<uint32_t>(m_couplers.size());
    total_nc += nbody() - 1;
    return total_nc;
}

// Joint reactions (Haug Eq. 11.5.6) in the global frame.  The generalized
// constraint force on body k is -Phi_qk^T lambda; in the angular-velocity form
// the force at the center of mass is F = -Phi_r^T lambda and the torque about
// the center of mass is N = -Phi_pi^T lambda.  About the joint point P the
// torque is T_P = N - (r_P - r) x F.
std::vector<JointReaction> DynamicsSystem::joint_reactions() const
{
    if (m_lambda.size() != total_nconstraints())
    {
        throw std::logic_error("get_joint_reactions: no multipliers; call init() first");
    }
    // -b^T lam for the first 3 columns of an nc-row block b.
    auto load = [](SimpleArray<double> const & b, SimpleArray<double> const & lam)
    {
        Vector3 out;
        for (ssize_t k = 0; k < static_cast<ssize_t>(lam.size()); ++k)
        {
            for (ssize_t c = 0; c < 3; ++c)
            {
                out[static_cast<size_t>(c)] += b(k, c) * lam[k];
            }
        }
        return -out;
    };
    std::vector<JointReaction> out;
    ssize_t row = 7;
    for (auto const & jt : m_joints)
    {
        auto const jnc = static_cast<ssize_t>(jt->nconstraints());
        JointReaction r;
        r.lambda = slice(m_lambda, row, jnc);
        row += jnc;
        r.body_i = static_cast<int32_t>(jt->body_i()->num());
        r.body_j = static_cast<int32_t>(jt->body_j()->num());
        r.point_i = jt->body_i()->position() + jt->arm_i();
        r.point_j = jt->body_j()->position() + jt->arm_j();
        r.force_i = load(jt->jacobian_i(), r.lambda);
        r.force_j = load(jt->jacobian_j(), r.lambda);
        r.torque_i = load(jt->jacobian_omega_i(), r.lambda) - cross(jt->arm_i(), r.force_i);
        r.torque_j = load(jt->jacobian_omega_j(), r.lambda) - cross(jt->arm_j(), r.force_j);
        if (jt->is_driver())
        {
            // Phi = g(q) - C: the actuator effort along g is -lambda; the distance
            // driver scales by C.
            r.is_driver = true;
            r.effort = r.lambda[0] * jt->drive_sensitivity();
        }
        out.push_back(std::move(r));
    }
    return out;
}

// Static equilibrium (Haug Sec. 6.5, 7.5).  x = [q (r, p per body); lambda]
// and R(x) = [Q(q) - Cq(q)^T lambda; Phi(q)] with zero velocities.
SimpleArray<double> DynamicsSystem::equilibrium_residual(SimpleArray<double> const & x)
{
    ssize_t const ndof = static_cast<ssize_t>(nbody()) * 7;
    for (ssize_t i = 0; i < static_cast<ssize_t>(nbody()); ++i)
    {
        put(m_q, i, STATE_R, segment<Vector3>(x, i * 7));
        put(m_q, i, STATE_V, Vector3());
        put(m_q, i, STATE_P, segment<Quaternion>(x, i * 7 + 3));
        put(m_q, i, STATE_PDOT, Quaternion());
    }
    update_bodies_from_state();
    update_joints(m_sim_time);
    build_system();
    ConstraintEval ev = assemble_constraints();
    auto const n = static_cast<ssize_t>(x.size());
    SimpleArray<double> r(small_vector<ssize_t>{n});
    SimpleArray<double> const cqt_lambda = matvec_transpose(m_cq, slice(x, ndof, n - ndof));
    for (ssize_t k = 0; k < ndof; ++k)
    {
        r[k] = m_rhs[k] - cqt_lambda[k];
    }
    std::copy_n(ev.c.begin(), n - ndof, r.begin() + ndof);
    return r;
}

EquilibriumResult DynamicsSystem::find_equilibrium(double tol, int32_t max_iter)
{
    if (!has_state())
    {
        throw std::logic_error("find_equilibrium: call init() first");
    }
    auto const nb = static_cast<ssize_t>(nbody());
    ssize_t const ndof = nb * 7;
    ssize_t const nc = total_nconstraints();
    ssize_t const n = ndof + nc;
    SimpleArray<double> const backup(m_q);

    SimpleArray<double> x = zero_vector(n);
    for (ssize_t i = 0; i < nb; ++i)
    {
        put_segment(x, i * 7, vector3_at(m_q, i, STATE_R));
        put_segment(x, i * 7 + 3, quaternion_at(m_q, i, STATE_P));
    }
    SimpleArray<double> r = equilibrium_residual(x);
    // Least-squares start for lambda at the start pose: Cq^T lambda = Q.
    SimpleArray<double> const lambda0 = robust_solve(transposed(m_cq), slice(m_rhs, 0, ndof));
    std::copy_n(lambda0.begin(), nc, x.begin() + ndof);
    r = equilibrium_residual(x);

    auto weighted = [](SimpleArray<double> const & w, SimpleArray<double> const & v)
    {
        SimpleArray<double> out(v);
        for (ssize_t k = 0; k < static_cast<ssize_t>(out.size()); ++k)
        {
            out[k] = w[k] * v[k];
        }
        return out;
    };

    EquilibriumResult res;
    double last_dq = 1.0;
    for (int32_t iter = 0; iter <= max_iter; ++iter)
    {
        res.iterations = iter;
        res.residual = norm_inf(r);
        // Converged if the residual is small, or if the last Newton update changed
        // the positions only at round-off level (stiff elements put a noise floor on R).
        if (res.residual < tol || (iter > 0 && last_dq < 1e-14 * std::max(1.0, norm_inf(slice(x, 0, ndof)))))
        {
            res.converged = true;
            break;
        }
        if (iter == max_iter)
        {
            break;
        }
        SimpleArray<double> jac = zero_matrix(n, n);
        for (ssize_t k = 0; k < n; ++k)
        {
            double const h = 1e-7 * std::max(1.0, std::fabs(x[k]));
            SimpleArray<double> xp(x), xm(x);
            xp[k] += h;
            xm[k] -= h;
            SimpleArray<double> const rp = equilibrium_residual(xp);
            SimpleArray<double> const rm = equilibrium_residual(xm);
            for (ssize_t i = 0; i < n; ++i)
            {
                jac(i, k) = (rp[i] - rm[i]) / (2.0 * h);
            }
        }
        // Equilibrate the rows and the columns: the system mixes forces,
        // positions, and multipliers of very different magnitudes.
        SimpleArray<double> rs(small_vector<ssize_t>{n}), cs(small_vector<ssize_t>{n});
        for (ssize_t i = 0; i < n; ++i)
        {
            double m = 0.0;
            for (ssize_t j = 0; j < n; ++j)
            {
                m = std::max(m, std::abs(jac(i, j)));
            }
            rs[i] = m > 0.0 ? 1.0 / m : 1.0;
        }
        for (ssize_t i = 0; i < n; ++i)
        {
            for (ssize_t j = 0; j < n; ++j)
            {
                jac(i, j) *= rs[i];
            }
        }
        for (ssize_t j = 0; j < n; ++j)
        {
            double m = 0.0;
            for (ssize_t i = 0; i < n; ++i)
            {
                m = std::max(m, std::abs(jac(i, j)));
            }
            cs[j] = m > 0.0 ? 1.0 / m : 1.0;
        }
        for (ssize_t i = 0; i < n; ++i)
        {
            for (ssize_t j = 0; j < n; ++j)
            {
                jac(i, j) *= cs[j];
            }
        }
        SimpleArray<double> const dx = weighted(cs, robust_solve(jac, negated(weighted(rs, r))));
        if (!all_finite(dx))
        {
            res.message = "non-finite Newton step";
            break;
        }
        // Backtracking: accept the first step size that reduces |R|.
        double step = 1.0;
        SimpleArray<double> x_new, r_new;
        double const merit0 = norm2(weighted(rs, r));
        for (int32_t ls = 0; ls < 20; ++ls)
        {
            x_new = SimpleArray<double>(x);
            for (ssize_t k = 0; k < n; ++k)
            {
                x_new[k] += step * dx[k];
            }
            r_new = equilibrium_residual(x_new);
            if (norm2(weighted(rs, r_new)) < merit0)
            {
                break;
            }
            step *= 0.5;
        }
        // The full Newton correction (not the line-search step): a stalled line
        // search must not look like convergence.
        last_dq = norm_inf(slice(dx, 0, ndof));
        x = std::move(x_new);
        r = std::move(r_new);
    }

    if (!res.converged)
    {
        m_q = backup;
        update_bodies_from_state();
        update_joints(m_sim_time);
        if (res.message.empty())
        {
            res.message = "no convergence after " + std::to_string(res.iterations) + " iterations (residual " + std::to_string(res.residual) + ")";
        }
        return res;
    }
    for (ssize_t i = 1; i < nb; ++i)
    {
        put(m_q, i, STATE_P, quaternion_at(m_q, i, STATE_P) / norm(quaternion_at(m_q, i, STATE_P)));
    }
    update_bodies_from_state();
    set_rest_velocities();
    update_joints(m_sim_time);
    build_system();
    compute_accelerations();
    return res;
}

void DynamicsSystem::set_rest_velocities()
{
    auto const nb = static_cast<ssize_t>(nbody());
    ssize_t const ndof = nb * 7;
    update_bodies_from_state();
    ConstraintEval const ev = evaluate_constraints(m_sim_time);
    SimpleArray<double> qdot = zero_vector(ndof);
    if (ev.cq.size() > 0 && norm_inf(ev.ct) > 0.0)
    {
        SimpleArray<double> neg_ct = ev.ct;
        for (ssize_t k = 0; k < nrow(neg_ct); ++k)
        {
            neg_ct[k] = -neg_ct[k];
        }
        qdot = robust_solve(ev.cq, neg_ct);
        SimpleArray<double> const nmat = null_space(ev.cq);
        if (nmat.ndim() == 2 && nmat.shape(1) > 0)
        {
            SimpleArray<double> const proj = matvec(nmat, matvec_transpose(nmat, qdot));
            for (ssize_t k = 0; k < ndof; ++k)
            {
                qdot[k] -= proj[k];
            }
        }
    }
    for (ssize_t i = 1; i < nb; ++i)
    {
        put(m_q, i, STATE_V, segment<Vector3>(qdot, i * 7));
        put(m_q, i, STATE_PDOT, segment<Quaternion>(qdot, i * 7 + 3));
    }
    update_bodies_from_state();
}

void DynamicsSystem::set_applied_load(uint32_t i, Vector3 const & force, Vector3 const & torque)
{
    if (i == 0 || i >= nbody())
    {
        throw std::out_of_range("set_applied_load: body index " + std::to_string(i) + " is not a mobile body");
    }
    m_bodies[i]->set_applied_load(force, torque);
    if (has_state())
    {
        update_bodies_from_state();
    }
}

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
