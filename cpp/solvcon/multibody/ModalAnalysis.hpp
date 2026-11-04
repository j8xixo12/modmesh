#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Linearized modes of a constrained multibody system about its current state.
 *
 * @ingroup group_numerics
 */

#include <solvcon/multibody/DynamicsSystem.hpp>

#include <solvcon/buffer/buffer.hpp>

#include <cstdint>

namespace solvcon
{

namespace multibody
{

struct ModalResult
{
    SimpleArray<double> frequencies; ///< (n_modes) natural frequencies [Hz], ascending.
    SimpleArray<double> damping_ratios; ///< (n_modes) damping ratios.
    SimpleArray<double> mode_shapes; ///< (n_modes, n_dof): row m is the shape of mode m in the independent coordinates.
    int32_t n_dof = 0; ///< Independent DOF after the constraint projection.
}; /* end struct ModalResult */

/**
 * Linearize the system around its current configuration and solve the
 * eigenproblem for the natural frequencies and the mode shapes:
 * 1. M*, G*, K* by finite differences of the force and constraint routines;
 * 2. the null space R of the constraint Jacobian Cq (from the SVD);
 * 3. the projection M_r = R^T M* R, G_r = R^T G* R, K_r = R^T K* R;
 * 4. the eigenvalues of the state-space matrix [-M_r^-1 G_r, -M_r^-1 K_r; I, 0].
 * The state of the system is restored.
 */
ModalResult modal_analysis(DynamicsSystem & sys);

} /* end namespace multibody */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
