#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Rigid multibody dynamics in Euler parameters (Haug, Computer-Aided
 * Kinematics and Dynamics of Mechanical Systems, Vol. I): bodies, joints and
 * drivers, force elements, the augmented equations of motion with several
 * time integrators, quasi-static sweeps, static equilibrium, and modal
 * analysis.  The dense kernels are solvcon's own (solvcon/linalg), so the engine
 * builds with or without a vendor LAPACK.
 */

#include <solvcon/multibody/kinematics.hpp>
#include <solvcon/multibody/linear.hpp>
#include <solvcon/multibody/Math.hpp>
#include <solvcon/multibody/DriveFunction.hpp>
#include <solvcon/multibody/CamProfile.hpp>
#include <solvcon/multibody/Body.hpp>
#include <solvcon/multibody/Joint.hpp>
#include <solvcon/multibody/TireModel.hpp>
#include <solvcon/multibody/ForceElement.hpp>
#include <solvcon/multibody/GearCoupler.hpp>
#include <solvcon/multibody/DynamicsSystem.hpp>
#include <solvcon/multibody/ModalAnalysis.hpp>

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
