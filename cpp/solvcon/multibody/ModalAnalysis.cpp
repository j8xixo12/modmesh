/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/multibody/ModalAnalysis.hpp>
#include <solvcon/multibody/linear.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace solvcon
{

namespace multibody
{

namespace
{

constexpr ssize_t STATE_R = DynamicsSystem::STATE_R;
constexpr ssize_t STATE_V = DynamicsSystem::STATE_V;
constexpr ssize_t STATE_P = DynamicsSystem::STATE_P;
constexpr ssize_t STATE_PDOT = DynamicsSystem::STATE_PDOT;

// Update the bodies from a state array, then the joints.
void apply_state(DynamicsSystem & sys, SimpleArray<double> const & q)
{
    auto const & bodies = sys.bodies();
    for (ssize_t i = 0; i < static_cast<ssize_t>(bodies.size()); ++i)
    {
        double const * row = q.vptr(i, 0);
        bodies[i]->update(Vector3(row[STATE_R], row[STATE_R + 1], row[STATE_R + 2]),
                          Vector3(row[STATE_V], row[STATE_V + 1], row[STATE_V + 2]),
                          Quaternion(row[STATE_P], row[STATE_P + 1], row[STATE_P + 2], row[STATE_P + 3]),
                          Quaternion(row[STATE_PDOT], row[STATE_PDOT + 1], row[STATE_PDOT + 2], row[STATE_PDOT + 3]));
    }
    for (auto const & jt : sys.joints())
    {
        jt->set_time(sys.sim_time());
        jt->update();
    }
}

// Accelerations (the first 7 nbody unknowns) of the augmented system.
SimpleArray<double> solve_accelerations(DynamicsSystem & sys)
{
    sys.build_system();
    auto const ndof = static_cast<ssize_t>(sys.nbody()) * 7;
    SimpleArray<double> const ans = robust_solve(sys.augmented_matrix(), sys.augmented_rhs());
    SimpleArray<double> out(small_vector<ssize_t>{ndof});
    std::copy_n(ans.begin(), ndof, out.begin());
    return out;
}

// d(acc_r)/d(xi_j) = R^T (acc_plus - acc_minus) / (2 eps), then column j of -M_r d(acc_r).
void set_column(SimpleArray<double> & out, ssize_t j, SimpleArray<double> const & m_r, SimpleArray<double> const & r, SimpleArray<double> const & acc_plus, SimpleArray<double> const & acc_minus, double eps)
{
    SimpleArray<double> diff(acc_plus);
    for (ssize_t k = 0; k < static_cast<ssize_t>(diff.size()); ++k)
    {
        diff[k] -= acc_minus[k];
    }
    SimpleArray<double> dacc_r = matvec_transpose(r, diff);
    for (double & v : dacc_r)
    {
        v /= 2.0 * eps;
    }
    SimpleArray<double> const col = matvec(m_r, dacc_r);
    for (ssize_t i = 0; i < static_cast<ssize_t>(col.size()); ++i)
    {
        out(i, j) = -col[i];
    }
}

} /* end namespace */

ModalResult modal_analysis(DynamicsSystem & sys)
{
    auto const nbody = static_cast<ssize_t>(sys.nbody());
    ssize_t const ndof = nbody * 7;

    SimpleArray<double> const q_orig(sys.state());
    apply_state(sys, q_orig);
    sys.build_system();
    SimpleArray<double> const m_star(sys.mass_matrix());
    SimpleArray<double> const cq_star(sys.jacobian());

    // Null space of the constraint Jacobian: the singular values above
    // max(m, n) * s_max * 1e-10 count toward the rank.
    if (!all_finite(cq_star))
    {
        throw std::runtime_error("modal_analysis: the constraint Jacobian has a non-finite entry");
    }
    double const rank_tol = static_cast<double>(std::max(cq_star.shape(0), cq_star.shape(1))) * 1e-10;
    ssize_t const rank = (cq_star.size() > 0) ? matrix_rank(cq_star, rank_tol) : 0;

    ssize_t const n = ndof - rank;
    if (n <= 0)
    {
        std::cerr << "[Modal] No independent DOF (rank=" << rank << ", ndof=" << ndof << ")" << std::endl;
        sys.set_state(q_orig);
        apply_state(sys, q_orig);
        ModalResult empty;
        empty.frequencies = zero_vector(0);
        empty.damping_ratios = zero_vector(0);
        empty.mode_shapes = zero_matrix(0, 0);
        return empty;
    }

    // R: an orthonormal basis of the null space (ndof-by-n).
    SimpleArray<double> const r = (cq_star.size() > 0) ? null_space(cq_star, rank_tol) : identity_matrix(ndof);
    SimpleArray<double> const m_r = matmul(matmul(transposed(r), m_star), r);

    // K_r and G_r from perturbations along the null-space directions:
    // K_r(:, j) = -M_r d(acc_r)/d(xi_j), G_r(:, j) = -M_r d(acc_r)/d(xi_dot_j),
    // with acc_r = R^T acc.
    constexpr double eps = 1e-7;
    SimpleArray<double> k_r = zero_matrix(n, n);
    SimpleArray<double> g_r = zero_matrix(n, n);
    SimpleArray<double> dq(small_vector<ssize_t>{ndof});

    // Stiffness: perturb the positions (and renormalize p).
    for (ssize_t j = 0; j < n; ++j)
    {
        for (ssize_t i = 0; i < ndof; ++i)
        {
            dq[i] = eps * r(i, j);
        }
        auto perturbed = [&](double sign)
        {
            SimpleArray<double> q(q_orig);
            for (ssize_t b = 0; b < nbody; ++b)
            {
                for (ssize_t k = 0; k < 3; ++k)
                {
                    q(b, STATE_R + k) += sign * dq[b * 7 + k];
                }
                for (ssize_t k = 0; k < 4; ++k)
                {
                    q(b, STATE_P + k) += sign * dq[b * 7 + 3 + k];
                }
                double * p = q.vptr(b, STATE_P);
                double const pnorm = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3]);
                if (pnorm > 1e-12)
                {
                    for (ssize_t k = 0; k < 4; ++k)
                    {
                        p[k] /= pnorm;
                    }
                }
            }
            return q;
        };
        apply_state(sys, perturbed(1.0));
        SimpleArray<double> const acc_plus = solve_accelerations(sys);
        apply_state(sys, perturbed(-1.0));
        SimpleArray<double> const acc_minus = solve_accelerations(sys);
        set_column(k_r, j, m_r, r, acc_plus, acc_minus, eps);
    }

    // Damping: perturb the velocities.
    for (ssize_t j = 0; j < n; ++j)
    {
        for (ssize_t i = 0; i < ndof; ++i)
        {
            dq[i] = eps * r(i, j);
        }
        auto perturbed = [&](double sign)
        {
            SimpleArray<double> q(q_orig);
            for (ssize_t b = 0; b < nbody; ++b)
            {
                for (ssize_t k = 0; k < 3; ++k)
                {
                    q(b, STATE_V + k) += sign * dq[b * 7 + k];
                }
                for (ssize_t k = 0; k < 4; ++k)
                {
                    q(b, STATE_PDOT + k) += sign * dq[b * 7 + 3 + k];
                }
            }
            return q;
        };
        apply_state(sys, perturbed(1.0));
        SimpleArray<double> const acc_plus = solve_accelerations(sys);
        apply_state(sys, perturbed(-1.0));
        SimpleArray<double> const acc_minus = solve_accelerations(sys);
        set_column(g_r, j, m_r, r, acc_plus, acc_minus, eps);
    }

    sys.set_state(q_orig);
    apply_state(sys, q_orig);

    // State-space matrix A = [-M_r^-1 G_r, -M_r^-1 K_r; I, 0].
    SimpleArray<double> m_r_inv;
    if (!inverse(m_r, m_r_inv))
    {
        m_r_inv = pseudo_inverse(m_r, static_cast<double>(n) * 1e-12);
    }
    SimpleArray<double> a = zero_matrix(2 * n, 2 * n);
    SimpleArray<double> const minv_g = matmul(m_r_inv, g_r);
    SimpleArray<double> const minv_k = matmul(m_r_inv, k_r);
    for (ssize_t i = 0; i < n; ++i)
    {
        for (ssize_t j = 0; j < n; ++j)
        {
            a(i, j) = -minv_g(i, j);
            a(i, n + j) = -minv_k(i, j);
        }
        a(n + i, i) = 1.0;
    }

    EigenDecomposition const eig = eigen_general(a);

    // Keep one eigenvalue of each complex pair (omega_d > 0).  Its eigenvector
    // real part is the column of the eigenvalue.
    std::vector<double> freqs, damps;
    std::vector<ssize_t> mode_indices;
    for (ssize_t i = 0; i < static_cast<ssize_t>(eig.wr.size()); ++i)
    {
        double const sigma = eig.wr[i];
        double const omega_d = eig.wi[i];
        if (omega_d <= 1e-10)
        {
            continue;
        }
        double const omega_n = std::sqrt(sigma * sigma + omega_d * omega_d);
        freqs.push_back(omega_n / (2.0 * std::numbers::pi));
        damps.push_back(-sigma / omega_n);
        mode_indices.push_back(i);
    }

    std::vector<size_t> order(freqs.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs)
              { return freqs[lhs] < freqs[rhs]; });

    auto const n_modes = static_cast<ssize_t>(order.size());
    ModalResult result;
    result.n_dof = static_cast<int32_t>(n);
    result.frequencies = zero_vector(n_modes);
    result.damping_ratios = zero_vector(n_modes);
    result.mode_shapes = zero_matrix(n_modes, n);
    for (ssize_t m = 0; m < n_modes; ++m)
    {
        size_t const si = order[static_cast<size_t>(m)];
        result.frequencies[m] = freqs[si];
        result.damping_ratios[m] = damps[si];
        for (ssize_t k = 0; k < n; ++k)
        {
            result.mode_shapes(m, k) = eig.vr(n + k, mode_indices[si]);
        }
    }

    std::cerr << "[Modal] Found " << n_modes << " modes with " << n << " independent DOF" << std::endl;
    return result;
}

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
