/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

// Time integrators for the constrained equations of motion.
//
//   rk4        direct integration of the Baumgarte-stabilized ODE (Haug 7.3.3-7.3.4)
//   abm4       Adams-Bashforth-Moulton 4th order PECE, RK4 startup (Haug 7.4)
//   partition  generalized coordinate partitioning with RK4 (Haug 7.3.2)
//   hybrid     rk4 step plus partitioning correction (Haug 7.3.5)
//   newmark    index-3 implicit Newmark-beta, beta = 1/4, gamma = 1/2 (Rui TMM 7.3.2.2).
//              No numerical damping: acceleration errors can grow on index-3 systems
//              with fast rotation.
//   hht        index-3 HHT-alpha (Hilber-Hughes-Taylor), alpha in [-1/3, 0],
//              beta = (1-alpha)^2/4, gamma = 1/2 - alpha: 2nd order, damps high frequencies
//   houbolt    index-3 implicit Houbolt backward differences (Rui TMM 7.3.2)
//   rk45       adaptive Dormand-Prince 5(4) with error control inside each output step
//              (variable-step integration, Haug 7.4.4)

#include <solvcon/multibody/DynamicsSystem.hpp>
#include <solvcon/multibody/linear.hpp>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

namespace solvcon
{

namespace multibody
{

namespace
{

using index_list = small_vector<ssize_t>;

// Least-squares or minimum-norm solve with the pseudo-inverse fallback (a may
// be rank deficient).
SimpleArray<double> lsq_solve(SimpleArray<double> const & a, SimpleArray<double> const & b)
{
    SimpleArray<double> x;
    if (nrow(a) == ncol(a) && lu_solve(a, b, x))
    {
        SimpleArray<double> res = matvec(a, x);
        for (ssize_t k = 0; k < static_cast<ssize_t>(res.size()); ++k)
        {
            res[k] -= b[k];
        }
        if (norm_inf(res) <= 1e-10 * std::max(1.0, norm_inf(b)))
        {
            return x;
        }
    }
    return matvec(pseudo_inverse(a), b);
}

// a + s b.
SimpleArray<double> axpy(SimpleArray<double> const & a, double s, SimpleArray<double> const & b)
{
    SimpleArray<double> out(a);
    for (ssize_t k = 0; k < static_cast<ssize_t>(out.size()); ++k)
    {
        out[k] += s * b[k];
    }
    return out;
}

// k1 + 2 k2 + 2 k3 + k4.
SimpleArray<double> rk_sum(SimpleArray<double> const & k1, SimpleArray<double> const & k2, SimpleArray<double> const & k3, SimpleArray<double> const & k4)
{
    SimpleArray<double> out(k1);
    for (ssize_t k = 0; k < static_cast<ssize_t>(out.size()); ++k)
    {
        out[k] = k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k];
    }
    return out;
}

SimpleArray<double> gather(SimpleArray<double> const & v, index_list const & idx)
{
    SimpleArray<double> out(small_vector<ssize_t>{static_cast<ssize_t>(idx.size())});
    for (size_t k = 0; k < idx.size(); ++k)
    {
        out[k] = v[idx[k]];
    }
    return out;
}

void scatter(SimpleArray<double> & v, index_list const & idx, SimpleArray<double> const & w)
{
    for (size_t k = 0; k < idx.size(); ++k)
    {
        v[idx[k]] = w[k];
    }
}

SimpleArray<double> head(SimpleArray<double> const & v, ssize_t n)
{
    SimpleArray<double> out(small_vector<ssize_t>{n});
    std::copy_n(v.begin(), n, out.begin());
    return out;
}

// The rows and columns of a at the given indices.
SimpleArray<double> submatrix(SimpleArray<double> const & a, index_list const & rows, index_list const & cols)
{
    SimpleArray<double> out(small_vector<ssize_t>{static_cast<ssize_t>(rows.size()), static_cast<ssize_t>(cols.size())});
    for (size_t i = 0; i < rows.size(); ++i)
    {
        for (size_t j = 0; j < cols.size(); ++j)
        {
            out(static_cast<ssize_t>(i), static_cast<ssize_t>(j)) = a(rows[i], cols[j]);
        }
    }
    return out;
}

} /* end namespace */

void DynamicsSystem::set_integrator(std::string const & name)
{
    if (name == "rk4")
    {
        m_integrator = IntegratorKind::RK4;
    }
    else if (name == "abm4")
    {
        m_integrator = IntegratorKind::ABM4;
    }
    else if (name == "partition")
    {
        m_integrator = IntegratorKind::Partition;
    }
    else if (name == "hybrid")
    {
        m_integrator = IntegratorKind::Hybrid;
    }
    else if (name == "newmark")
    {
        m_integrator = IntegratorKind::Newmark;
    }
    else if (name == "houbolt")
    {
        m_integrator = IntegratorKind::Houbolt;
    }
    else if (name == "rk45")
    {
        m_integrator = IntegratorKind::RK45;
    }
    else if (name == "hht")
    {
        m_integrator = IntegratorKind::HHT;
    }
    else
    {
        throw std::invalid_argument("set_integrator: unknown integrator '" + name + "' (rk4, rk45, abm4, partition, hybrid, newmark, hht, houbolt)");
    }
    reset_integrator_state();
}

std::string DynamicsSystem::integrator() const
{
    switch (m_integrator)
    {
    case IntegratorKind::RK4: return "rk4";
    case IntegratorKind::ABM4: return "abm4";
    case IntegratorKind::Partition: return "partition";
    case IntegratorKind::Hybrid: return "hybrid";
    case IntegratorKind::Newmark: return "newmark";
    case IntegratorKind::Houbolt: return "houbolt";
    case IntegratorKind::RK45: return "rk45";
    case IntegratorKind::HHT: return "hht";
    }
    return "unknown";
}

void DynamicsSystem::reset_integrator_state()
{
    m_abm_history.clear();
    m_past_positions.clear();
    m_newmark_acc = SimpleArray<double>();
    m_last_newton_iterations = 0;
    m_rk45_h = 0.0;
}

void DynamicsSystem::pack_state(SimpleArray<double> & x, SimpleArray<double> & xd) const
{
    auto const nb = static_cast<ssize_t>(nbody());
    x = SimpleArray<double>(small_vector<ssize_t>{nb * 7});
    xd = SimpleArray<double>(small_vector<ssize_t>{nb * 7});
    for (ssize_t i = 0; i < nb; ++i)
    {
        for (ssize_t k = 0; k < 3; ++k)
        {
            x[i * 7 + k] = m_q(i, STATE_R + k);
            xd[i * 7 + k] = m_q(i, STATE_V + k);
        }
        for (ssize_t k = 0; k < 4; ++k)
        {
            x[i * 7 + 3 + k] = m_q(i, STATE_P + k);
            xd[i * 7 + 3 + k] = m_q(i, STATE_PDOT + k);
        }
    }
}

void DynamicsSystem::unpack_state(SimpleArray<double> const & x, SimpleArray<double> const & xd)
{
    auto const nb = static_cast<ssize_t>(nbody());
    for (ssize_t i = 0; i < nb; ++i)
    {
        for (ssize_t k = 0; k < 3; ++k)
        {
            m_q(i, STATE_R + k) = x[i * 7 + k];
            m_q(i, STATE_V + k) = xd[i * 7 + k];
        }
        for (ssize_t k = 0; k < 4; ++k)
        {
            m_q(i, STATE_P + k) = x[i * 7 + 3 + k];
            m_q(i, STATE_PDOT + k) = xd[i * 7 + 3 + k];
        }
    }
    update_bodies_from_state();
}

// Accelerations of the unstabilized augmented system at (x, xd, t).
SimpleArray<double> DynamicsSystem::accel_at(SimpleArray<double> const & x, SimpleArray<double> const & xd, double t)
{
    unpack_state(x, xd);
    update_joints(t);
    build_system(false);
    return head(lsq_solve(m_matrix, m_rhs), static_cast<ssize_t>(nbody()) * 7);
}

namespace
{

// Divide the Euler parameters of every body by their norm.
void normalize_quaternions(SimpleArray<double> & q, double floor)
{
    for (ssize_t i = 0; i < q.shape(0); ++i)
    {
        double * p = q.vptr(i, DynamicsSystem::STATE_P);
        double const n = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3]);
        if (n > floor)
        {
            for (ssize_t k = 0; k < 4; ++k)
            {
                p[k] /= n;
            }
        }
    }
}

} /* end namespace */

void DynamicsSystem::step_rk4()
{
    SimpleArray<double> q_temp(m_q);
    SimpleArray<double> k1(m_q), k2(m_q), k3(m_q), k4(m_q);
    auto const n = static_cast<ssize_t>(m_q.size());
    double const dt = m_dt;

    dynamic_function(q_temp, k1, m_sim_time);
    for (ssize_t e = 0; e < n; ++e)
    {
        q_temp[e] = m_q[e] + 0.5 * dt * k1[e];
    }
    dynamic_function(q_temp, k2, m_sim_time + 0.5 * dt);
    for (ssize_t e = 0; e < n; ++e)
    {
        q_temp[e] = m_q[e] + 0.5 * dt * k2[e];
    }
    dynamic_function(q_temp, k3, m_sim_time + 0.5 * dt);
    for (ssize_t e = 0; e < n; ++e)
    {
        q_temp[e] = m_q[e] + dt * k3[e];
    }
    dynamic_function(q_temp, k4, m_sim_time + dt);
    for (ssize_t e = 0; e < n; ++e)
    {
        double const qd = (1.0 / 6.0) * (k1[e] + 2.0 * k2[e] + 2.0 * k3[e] + k4[e]);
        m_q[e] = m_q[e] + qd * dt;
    }
    m_sim_time += dt;
    normalize_quaternions(m_q, 1e-12);
    update_bodies_from_state();
}

// Choose the dependent coordinates by greedy column pivoting on the
// independent rows of Cq (pivoted Gram-Schmidt).  Haug Sec. 7.3.2 uses LU with
// full pivoting for the same purpose: the pivot columns are dependent.
void DynamicsSystem::select_partition(index_list & indep, index_list & dep)
{
    update_bodies_from_state();
    ConstraintEval ev = evaluate_constraints(m_sim_time);
    auto const r = static_cast<ssize_t>(m_independent_rows.size());
    ssize_t const n = ev.cq.shape(1);
    SimpleArray<double> a(small_vector<ssize_t>{r, n});
    for (ssize_t k = 0; k < r; ++k)
    {
        std::copy_n(ev.cq.vptr(static_cast<ssize_t>(m_independent_rows[k]), 0), n, a.vptr(k, 0));
    }
    auto col_norm = [&](ssize_t c)
    {
        double sum = 0.0;
        for (ssize_t i = 0; i < r; ++i)
        {
            sum += a(i, c) * a(i, c);
        }
        return std::sqrt(sum);
    };
    small_vector<bool> used(static_cast<size_t>(n), false);
    dep.clear();
    for (ssize_t k = 0; k < r; ++k)
    {
        double best = -1.0;
        ssize_t best_c = 0;
        for (ssize_t c = 0; c < n; ++c)
        {
            if (used[c])
            {
                continue;
            }
            double const nn = col_norm(c);
            if (nn > best)
            {
                best = nn;
                best_c = c;
            }
        }
        used[best_c] = true;
        dep.push_back(best_c);
        double const nb = col_norm(best_c);
        SimpleArray<double> u(small_vector<ssize_t>{r});
        for (ssize_t i = 0; i < r; ++i)
        {
            u[i] = a(i, best_c) / nb;
        }
        for (ssize_t c = 0; c < n; ++c)
        {
            if (used[c])
            {
                continue;
            }
            double d = 0.0;
            for (ssize_t i = 0; i < r; ++i)
            {
                d += u[i] * a(i, c);
            }
            for (ssize_t i = 0; i < r; ++i)
            {
                a(i, c) -= d * u[i];
            }
        }
    }
    indep.clear();
    for (ssize_t c = 0; c < n; ++c)
    {
        if (!used[c])
        {
            indep.push_back(c);
        }
    }
}

// Solve the dependent positions (Newton on Phi = 0) and the dependent
// velocities (Cq_D xd_D = -Ct - Cq_I xd_I) with the independent coordinates fixed.
void DynamicsSystem::project_dependent(SimpleArray<double> & x, SimpleArray<double> & xd, double t, index_list const & indep, index_list const & dep)
{
    index_list rows;
    for (size_t k = 0; k < m_independent_rows.size(); ++k)
    {
        rows.push_back(static_cast<ssize_t>(m_independent_rows[k]));
    }
    double err = 0.0;
    bool ok = false;
    for (int32_t it = 0; it < 50; ++it)
    {
        unpack_state(x, xd);
        ConstraintEval ev = evaluate_constraints(t);
        SimpleArray<double> neg_c = gather(ev.c, rows);
        err = norm_inf(neg_c);
        if (err < 1e-12)
        {
            ok = true;
            break;
        }
        for (double & v : neg_c)
        {
            v = -v;
        }
        SimpleArray<double> const delta = lsq_solve(submatrix(ev.cq_pos, rows, dep), neg_c);
        for (size_t k = 0; k < dep.size(); ++k)
        {
            x[dep[k]] += delta[k];
        }
    }
    if (!ok)
    {
        throw std::runtime_error("[MBS] partitioning: dependent position solve did not converge at t=" + std::to_string(t) + " (residual " + std::to_string(err) + ")");
    }
    unpack_state(x, xd);
    ConstraintEval ev = evaluate_constraints(t);
    SimpleArray<double> const cq_xi = matvec(submatrix(ev.cq, rows, indep), gather(xd, indep));
    SimpleArray<double> rhs(small_vector<ssize_t>{static_cast<ssize_t>(rows.size())});
    for (size_t k = 0; k < rows.size(); ++k)
    {
        rhs[k] = -ev.ct[rows[k]] - cq_xi[k];
    }
    scatter(xd, dep, lsq_solve(submatrix(ev.cq, rows, dep), rhs));
    unpack_state(x, xd);
}

void DynamicsSystem::step_partition()
{
    index_list ind, dep;
    select_partition(ind, dep);
    SimpleArray<double> x0, xd0;
    pack_state(x0, xd0);
    double const t0 = m_sim_time, h = m_dt;
    // The running guess for the dependent solves.
    SimpleArray<double> xg(x0), xdg(xd0);

    auto f = [&](SimpleArray<double> const & v, SimpleArray<double> const & vd, double t)
    {
        scatter(xg, ind, v);
        scatter(xdg, ind, vd);
        project_dependent(xg, xdg, t, ind, dep);
        return gather(accel_at(xg, xdg, t), ind);
    };
    SimpleArray<double> const v0 = gather(x0, ind), vd0 = gather(xd0, ind);
    SimpleArray<double> const k1v(vd0), k1a = f(v0, vd0, t0);
    SimpleArray<double> const k2v = axpy(vd0, 0.5 * h, k1a), k2a = f(axpy(v0, 0.5 * h, k1v), k2v, t0 + 0.5 * h);
    SimpleArray<double> const k3v = axpy(vd0, 0.5 * h, k2a), k3a = f(axpy(v0, 0.5 * h, k2v), k3v, t0 + 0.5 * h);
    SimpleArray<double> const k4v = axpy(vd0, h, k3a), k4a = f(axpy(v0, h, k3v), k4v, t0 + h);
    SimpleArray<double> const v1 = axpy(v0, h / 6.0, rk_sum(k1v, k2v, k3v, k4v));
    SimpleArray<double> const vd1 = axpy(vd0, h / 6.0, rk_sum(k1a, k2a, k3a, k4a));
    m_sim_time = t0 + h;
    scatter(xg, ind, v1);
    scatter(xdg, ind, vd1);
    project_dependent(xg, xdg, m_sim_time, ind, dep);
}

void DynamicsSystem::step_hybrid()
{
    step_rk4();
    index_list ind, dep;
    select_partition(ind, dep);
    SimpleArray<double> x, xd;
    pack_state(x, xd);
    project_dependent(x, xd, m_sim_time, ind, dep);
}

void DynamicsSystem::step_abm4()
{
    SimpleArray<double> fn(m_q);
    dynamic_function(m_q, fn, m_sim_time);
    if (m_abm_history.size() < 3)
    {
        m_abm_history.push_back(std::move(fn));
        step_rk4();
        return;
    }
    SimpleArray<double> const & f1 = m_abm_history[2]; // f_{n-1}
    SimpleArray<double> const & f2 = m_abm_history[1]; // f_{n-2}
    SimpleArray<double> const & f3 = m_abm_history[0]; // f_{n-3}
    double const h = m_dt;
    auto const n = static_cast<ssize_t>(m_q.size());
    SimpleArray<double> qp(m_q), fp(m_q);
    for (ssize_t e = 0; e < n; ++e)
    {
        qp[e] = m_q[e] + h / 24.0 * (55.0 * fn[e] - 59.0 * f1[e] + 37.0 * f2[e] - 9.0 * f3[e]);
    }
    dynamic_function(qp, fp, m_sim_time + h);
    for (ssize_t e = 0; e < n; ++e)
    {
        m_q[e] = m_q[e] + h / 24.0 * (9.0 * fp[e] + 19.0 * fn[e] - 5.0 * f1[e] + f2[e]);
    }
    m_abm_history.erase(m_abm_history.begin());
    m_abm_history.push_back(std::move(fn));
    m_sim_time += h;
    normalize_quaternions(m_q, 0.0);
    update_bodies_from_state();
}

void DynamicsSystem::set_hht_alpha(double alpha)
{
    if (alpha < -1.0 / 3.0 - 1e-15 || alpha > 0.0)
    {
        throw std::invalid_argument("set_hht_alpha: alpha must be in [-1/3, 0], got " + std::to_string(alpha));
    }
    m_hht_alpha = alpha;
}

// Index-3 implicit step.  The unknowns are the positions x_{n+1} (through a
// for Newmark) and the multipliers lambda, with the residual
//   R1 = M(x) a + Cq(x)^T lambda - Q(x, xd, t)     (equations of motion)
//   R2 = Phi(x, t) / s                               (position constraints)
// Newmark: x = x_n + h v_n + h^2 ((1/2 - beta) a_n + beta a), v = v_n + h ((1-g) a_n + g a).
// Houbolt: a = (2x - 5x_n + 4x_{n-1} - x_{n-2}) / h^2, v = (11x - 18x_n + 9x_{n-1} - 2x_{n-2}) / 6h.
// HHT form: M a + (1 + alpha) F_{n+1} - alpha F_n = 0 with F = Cq^T lambda - Q.
// At a converged step M_n a_n + F_n = 0, so -alpha F_n = alpha M_n a_n.
// Newton uses the standard approximate Jacobian with the derivatives of M and
// Q in q dropped from the constraint part.
void DynamicsSystem::implicit_step(bool houbolt, double h, double alpha)
{
    double const beta = 0.25 * (1.0 - alpha) * (1.0 - alpha);
    double const gam = 0.5 - alpha;
    ssize_t const ndof = static_cast<ssize_t>(nbody()) * 7;
    ssize_t const nc = total_nconstraints();
    double const t1 = m_sim_time + h;
    SimpleArray<double> xn, vn;
    pack_state(xn, vn);
    if (static_cast<ssize_t>(m_newmark_acc.size()) != ndof)
    {
        // a_n consistent with (x_n, v_n).
        m_newmark_acc = accel_at(xn, vn, m_sim_time);
    }
    SimpleArray<double> const an(m_newmark_acc);
    SimpleArray<double> mn_an;
    if (alpha != 0.0)
    {
        unpack_state(xn, vn);
        update_joints(m_sim_time);
        build_system(false);
        mn_an = matvec(m_mass, an);
    }

    // The unknown z is a (Newmark) or x (Houbolt), starting from the explicit prediction.
    SimpleArray<double> z(an);
    if (houbolt)
    {
        for (ssize_t k = 0; k < ndof; ++k)
        {
            z[k] = xn[k] + h * vn[k] + 0.5 * h * h * an[k];
        }
    }
    SimpleArray<double> lam = (static_cast<ssize_t>(m_lambda.size()) == nc) ? SimpleArray<double>(m_lambda) : zero_vector(nc);
    SimpleArray<double> const & xm1 = houbolt ? m_past_positions[1] : xn;
    SimpleArray<double> const & xm2 = houbolt ? m_past_positions[0] : xn;
    auto state_of = [&](SimpleArray<double> const & zz, SimpleArray<double> & x, SimpleArray<double> & v, SimpleArray<double> & a)
    {
        x = SimpleArray<double>(small_vector<ssize_t>{ndof});
        v = SimpleArray<double>(small_vector<ssize_t>{ndof});
        a = SimpleArray<double>(small_vector<ssize_t>{ndof});
        for (ssize_t k = 0; k < ndof; ++k)
        {
            if (houbolt)
            {
                x[k] = zz[k];
                a[k] = (2.0 * x[k] - 5.0 * xn[k] + 4.0 * xm1[k] - xm2[k]) / (h * h);
                v[k] = (11.0 * x[k] - 18.0 * xn[k] + 9.0 * xm1[k] - 2.0 * xm2[k]) / (6.0 * h);
            }
            else
            {
                a[k] = zz[k];
                x[k] = xn[k] + h * vn[k] + h * h * ((0.5 - beta) * an[k] + beta * a[k]);
                v[k] = vn[k] + h * ((1.0 - gam) * an[k] + gam * a[k]);
            }
        }
    };
    double const dadz = houbolt ? 2.0 / (h * h) : 1.0;
    // Chain factors dx/dz and dv/dz of the unknown z.
    double const dxdz = houbolt ? 1.0 : beta * h * h;
    double const dvdz = houbolt ? 11.0 / (6.0 * h) : gam * h;

    // R1(x, v, a, lambda) = M(x) a + Cq(x)^T lambda - Q(x, v, t1).
    auto r1_eval = [&](SimpleArray<double> const & x, SimpleArray<double> const & v, SimpleArray<double> const & a, SimpleArray<double> const & l)
    {
        unpack_state(x, v);
        update_joints(t1);
        build_system(false);
        SimpleArray<double> r = matvec(m_mass, a);
        SimpleArray<double> const cqt_l = matvec_transpose(m_cq, l);
        for (ssize_t k = 0; k < ndof; ++k)
        {
            r[k] = r[k] + (1.0 + alpha) * (cqt_l[k] - m_rhs[k]);
        }
        if (alpha != 0.0)
        {
            for (ssize_t k = 0; k < ndof; ++k)
            {
                r[k] += alpha * mn_an[k];
            }
        }
        return r;
    };
    // [R1; C / dxdz] of length ndof + nc.
    auto full_residual = [&](SimpleArray<double> const & r1, SimpleArray<double> const & c)
    {
        SimpleArray<double> r(small_vector<ssize_t>{ndof + nc});
        std::copy_n(r1.begin(), ndof, r.begin());
        for (ssize_t k = 0; k < nc; ++k)
        {
            r[ndof + k] = c[k] / dxdz;
        }
        return r;
    };

    SimpleArray<double> x, v, a;
    int32_t it = 0;
    double rnorm = 0.0;
    double last_dx = 1.0;
    for (; it < 30; ++it)
    {
        state_of(z, x, v, a);
        SimpleArray<double> const r1 = r1_eval(x, v, a, lam);
        SimpleArray<double> const mass(m_mass);
        SimpleArray<double> const cq(m_cq);
        double const ref = std::max(1.0, norm_inf(head(m_rhs, ndof)));
        ConstraintEval ev = assemble_constraints();
        SimpleArray<double> const r = full_residual(r1, ev.c);
        rnorm = norm_inf(r);
        // Converged if the residual is small, or if the last Newton update moved
        // the positions only at round-off level (stiff elements put a noise floor on R).
        if (it > 0 && (rnorm < 1e-10 * ref || last_dx < 1e-14 * std::max(1.0, norm_inf(x))))
        {
            break;
        }
        // dR1/dz = M da/dz + (dR1/dx) dx/dz + (dR1/dv) dv/dz; dR1/dx and dR1/dv by
        // central differences (spring stiffness, damping, gyroscopic and constraint terms).
        SimpleArray<double> jac = zero_matrix(ndof + nc, ndof + nc);
        for (ssize_t i = 0; i < ndof; ++i)
        {
            for (ssize_t j = 0; j < ndof; ++j)
            {
                jac(i, j) = mass(i, j) * dadz;
            }
        }
        for (ssize_t k = 0; k < ndof; ++k)
        {
            double const ex = 1e-7 * std::max(1.0, std::fabs(x[k]));
            double const ev_ = 1e-7 * std::max(1.0, std::fabs(v[k]));
            SimpleArray<double> xp(x), xm(x), vp(v), vm(v);
            xp[k] += ex;
            xm[k] -= ex;
            vp[k] += ev_;
            vm[k] -= ev_;
            SimpleArray<double> const rxp = r1_eval(xp, v, a, lam);
            SimpleArray<double> const rxm = r1_eval(xm, v, a, lam);
            SimpleArray<double> const rvp = r1_eval(x, vp, a, lam);
            SimpleArray<double> const rvm = r1_eval(x, vm, a, lam);
            for (ssize_t i = 0; i < ndof; ++i)
            {
                double const drdx = (rxp[i] - rxm[i]) / (2.0 * ex);
                double const drdv = (rvp[i] - rvm[i]) / (2.0 * ev_);
                jac(i, k) += drdx * dxdz + drdv * dvdz;
            }
        }
        for (ssize_t rr = 0; rr < nc; ++rr)
        {
            for (ssize_t c = 0; c < ndof; ++c)
            {
                jac(c, ndof + rr) = (1.0 + alpha) * cq(rr, c);
                // d(C / dxdz)/dz.
                jac(ndof + rr, c) = ev.cq_pos(rr, c);
            }
        }
        SimpleArray<double> neg_r(r);
        for (double & e : neg_r)
        {
            e = -e;
        }
        SimpleArray<double> const dz = lsq_solve(jac, neg_r);
        if (!all_finite(dz))
        {
            break;
        }
        // Backtracking line search on |R|_inf.
        double step = 1.0;
        for (int32_t ls = 0; ls < 10; ++ls)
        {
            SimpleArray<double> zt(z), lt(lam);
            for (ssize_t k = 0; k < ndof; ++k)
            {
                zt[k] += step * dz[k];
            }
            for (ssize_t k = 0; k < nc; ++k)
            {
                lt[k] += step * dz[ndof + k];
            }
            SimpleArray<double> xt, vt, at;
            state_of(zt, xt, vt, at);
            SimpleArray<double> const r1t = r1_eval(xt, vt, at, lt);
            SimpleArray<double> const rt = full_residual(r1t, assemble_constraints().c);
            if (norm_inf(rt) < rnorm || ls == 9)
            {
                break;
            }
            step *= 0.5;
        }
        for (ssize_t k = 0; k < ndof; ++k)
        {
            z[k] += step * dz[k];
        }
        for (ssize_t k = 0; k < nc; ++k)
        {
            lam[k] += step * dz[ndof + k];
        }
        // The full Newton correction.
        last_dx = norm_inf(head(dz, ndof)) * dxdz;
    }
    if (it >= 30)
    {
        throw std::runtime_error(std::string("[MBS] ") + (houbolt ? "Houbolt" : "Newmark") + ": Newton did not converge at t=" + std::to_string(t1) + " (residual " + std::to_string(rnorm) + ")");
    }
    m_last_newton_iterations = it;
    state_of(z, x, v, a);
    if (houbolt)
    {
        m_past_positions[0] = std::move(m_past_positions[1]);
        m_past_positions[1] = std::move(xn);
    }
    m_newmark_acc = std::move(a);
    m_sim_time = t1;
    unpack_state(x, v);
    m_lambda = std::move(lam);
}

// Newmark step with rejection: if Newton fails, restore the state and take
// two half steps (recursively, down to dt / 64).
void DynamicsSystem::newmark_substeps(double h, int32_t depth, double alpha)
{
    SimpleArray<double> q_backup(m_q);
    double const t_backup = m_sim_time;
    SimpleArray<double> acc_backup(m_newmark_acc), lam_backup(m_lambda);
    try
    {
        implicit_step(false, h, alpha);
    }
    catch (std::runtime_error const &)
    {
        if (depth >= 6)
        {
            throw;
        }
        m_q = std::move(q_backup);
        m_sim_time = t_backup;
        m_newmark_acc = std::move(acc_backup);
        m_lambda = std::move(lam_backup);
        update_bodies_from_state();
        newmark_substeps(0.5 * h, depth + 1, alpha);
        newmark_substeps(0.5 * h, depth + 1, alpha);
    }
}

void DynamicsSystem::step_newmark() { newmark_substeps(m_dt, 0); }

void DynamicsSystem::step_houbolt()
{
    // Startup: two Newmark steps provide x_{n-1} and x_{n-2}.
    if (m_past_positions.size() < 2)
    {
        SimpleArray<double> x, xd;
        pack_state(x, xd);
        m_past_positions.push_back(std::move(x));
        newmark_substeps(m_dt, 0);
        return;
    }
    SimpleArray<double> q_backup(m_q);
    double const t_backup = m_sim_time;
    SimpleArray<double> acc_backup(m_newmark_acc), lam_backup(m_lambda);
    try
    {
        implicit_step(true, m_dt);
    }
    catch (std::runtime_error const &)
    {
        // Houbolt needs an equally spaced history: restart from Newmark substeps.
        m_q = std::move(q_backup);
        m_sim_time = t_backup;
        m_newmark_acc = std::move(acc_backup);
        m_lambda = std::move(lam_backup);
        update_bodies_from_state();
        m_past_positions.clear();
        newmark_substeps(m_dt, 1);
    }
}

// Dormand-Prince 5(4) on the stabilized first-order system, with adaptive
// internal steps that end exactly at sim_time + dt.  The error norm is the
// maximum over the components of |err| / (atol + rtol max(|y_n|, |y_{n+1}|)).
void DynamicsSystem::step_rk45()
{
    static constexpr double c2 = 1.0 / 5, c3 = 3.0 / 10, c4 = 4.0 / 5, c5 = 8.0 / 9;
    static constexpr double a21 = 1.0 / 5;
    static constexpr double a31 = 3.0 / 40, a32 = 9.0 / 40;
    static constexpr double a41 = 44.0 / 45, a42 = -56.0 / 15, a43 = 32.0 / 9;
    static constexpr double a51 = 19372.0 / 6561, a52 = -25360.0 / 2187, a53 = 64448.0 / 6561, a54 = -212.0 / 729;
    static constexpr double a61 = 9017.0 / 3168, a62 = -355.0 / 33, a63 = 46732.0 / 5247, a64 = 49.0 / 176, a65 = -5103.0 / 18656;
    static constexpr double b1 = 35.0 / 384, b3 = 500.0 / 1113, b4 = 125.0 / 192, b5 = -2187.0 / 6784, b6 = 11.0 / 84;
    static constexpr double e1 = 71.0 / 57600, e3 = -71.0 / 16695, e4 = 71.0 / 1920, e5 = -17253.0 / 339200, e6 = 22.0 / 525, e7 = -1.0 / 40;

    auto const n = static_cast<ssize_t>(m_q.size());
    auto comb = [&](SimpleArray<double> const & y, double h, std::initializer_list<std::pair<double, SimpleArray<double> const *>> ks)
    {
        SimpleArray<double> out(y);
        for (auto const & kc : ks)
        {
            for (ssize_t e = 0; e < n; ++e)
            {
                out[e] += h * kc.first * (*kc.second)[e];
            }
        }
        return out;
    };
    double const t_end = m_sim_time + m_dt;
    double h = (m_rk45_h > 0.0) ? std::min(m_rk45_h, m_dt) : m_dt;
    m_rk45_substeps = 0;
    m_rk45_rejected = 0;
    SimpleArray<double> k1(m_q), k2(m_q), k3(m_q), k4(m_q), k5(m_q), k6(m_q), k7(m_q);
    dynamic_function(m_q, k1, m_sim_time);
    while (m_sim_time < t_end - 1e-14 * std::max(1.0, std::fabs(t_end)))
    {
        bool const last = (m_sim_time + h >= t_end);
        double const hs = last ? t_end - m_sim_time : h;
        double const t = m_sim_time;
        dynamic_function(comb(m_q, hs, {{a21, &k1}}), k2, t + c2 * hs);
        dynamic_function(comb(m_q, hs, {{a31, &k1}, {a32, &k2}}), k3, t + c3 * hs);
        dynamic_function(comb(m_q, hs, {{a41, &k1}, {a42, &k2}, {a43, &k3}}), k4, t + c4 * hs);
        dynamic_function(comb(m_q, hs, {{a51, &k1}, {a52, &k2}, {a53, &k3}, {a54, &k4}}), k5, t + c5 * hs);
        dynamic_function(comb(m_q, hs, {{a61, &k1}, {a62, &k2}, {a63, &k3}, {a64, &k4}, {a65, &k5}}), k6, t + hs);
        SimpleArray<double> y1 = comb(m_q, hs, {{b1, &k1}, {b3, &k3}, {b4, &k4}, {b5, &k5}, {b6, &k6}});
        dynamic_function(y1, k7, t + hs);
        double err = 0.0;
        for (ssize_t e = 0; e < n; ++e)
        {
            double const est = hs * (e1 * k1[e] + e3 * k3[e] + e4 * k4[e] + e5 * k5[e] + e6 * k6[e] + e7 * k7[e]);
            double const sc = m_rk45_atol + m_rk45_rtol * std::max(std::abs(m_q[e]), std::abs(y1[e]));
            err = std::max(err, std::abs(est) / sc);
        }
        if (!std::isfinite(err))
        {
            err = 1e10;
        }
        double const factor = (err == 0.0) ? 5.0 : std::min(5.0, std::max(0.2, 0.9 * std::pow(err, -0.2)));
        if (err <= 1.0)
        {
            m_q = std::move(y1);
            m_sim_time = last ? t_end : t + hs;
            normalize_quaternions(m_q, 0.0);
            // First same as last, after the renormalization.
            dynamic_function(m_q, k1, m_sim_time);
            ++m_rk45_substeps;
            if (!last)
            {
                h = hs * factor;
            }
            else
            {
                // Keep the controller step for the next call.
                m_rk45_h = std::max(h, hs);
            }
        }
        else
        {
            ++m_rk45_rejected;
            h = hs * factor;
            if (h < 1e-12 * m_dt)
            {
                throw std::runtime_error("[MBS] rk45: step size underflow at t=" + std::to_string(t));
            }
        }
    }
    if (m_rk45_h <= 0.0)
    {
        m_rk45_h = h;
    }
    update_bodies_from_state();
}

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
