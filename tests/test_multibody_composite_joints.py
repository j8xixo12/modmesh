# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for the composite joints of Haug Sec. 9.4.4-9.4.5.

Rules under test:

* A body attached to ground by a composite joint has the relative DOF that
  Haug states for that joint.
* A free body under gravity on each composite joint conserves energy up to
  RK4 truncation error: the drift falls with the RK4 order when dt halves.
* A driven screw joint translates by pitch * angle, also over several
  revolutions.
"""

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.tools.arrays import result_arrays

Z = [0.0, 0.0, 0.0]
AX = [1.0, 0.0, 0.0]
AZ = [0.0, 0.0, 1.0]
R0 = [0.0, 0.0, 0.8]  # body origin; body frame = global frame at start
OFF = [0.6, -0.0, -0.8]  # local point -> global (0.6, 0, 0), d = (-0.6, 0, 0)


@pytest.fixture
def mbs():
    """Import the compiled solver module or skip.

    :return: compiled solver module
    :rtype: module
    """
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    return multibody


def make(mbs, kind, g, b):
    """Create a composite joint between ground and body in a valid pose.

    :return: (joint, expected relative DOF)
    :rtype: tuple
    """
    if kind == "screw":
        return mbs.create_screw_joint(g, b, Z, Z, AZ, AZ, pitch=0.05), 1
    if kind == "rev_sph":
        return mbs.create_revolute_spherical_joint(g, b, Z, OFF, AZ), 4
    if kind == "rev_rev_parallel":
        return mbs.create_revolute_revolute_joint(g, b, Z, OFF, AZ, AZ), 2
    if kind == "rev_rev_orthogonal":
        return mbs.create_revolute_revolute_joint(g, b, Z, OFF, AZ, AX,
                                                  orthogonal=True), 2
    if kind == "rev_trans":
        return mbs.create_revolute_translational_joint(g, b, Z, OFF, AZ, AX,
                                                       f_j=AZ), 2
    if kind == "strut":
        return mbs.create_strut_joint(g, b, Z, Z, AZ), 4
    raise ValueError(kind)


KINDS = ["screw", "rev_sph", "rev_rev_parallel", "rev_rev_orthogonal",
         "rev_trans", "strut"]


def build(mbs, kind, omega=Z, gravity=Z, dt=1e-3):
    s = mbs.DynamicsSystem(dt)
    g = mbs.create_ground(0)
    b = mbs.create_mobilized_body_quat(1, R0, Z, Z, [1, 0, 0, 0], omega, Z,
                                       1.0, [0.02, 0.03, 0.04], gravity, Z)
    s.Add(g)
    s.Add(b)
    joint, dof = make(mbs, kind, g, b)
    s.Add(joint)
    return s, dof


@pytest.mark.parametrize("kind", KINDS)
def test_relative_dof_matches_haug(mbs, kind):
    s, dof = build(mbs, kind)
    s.init()
    assert s.get_dof() == dof
    assert not s.get_diagnostics()


def rot(q):
    e0, e1, e2, e3 = q
    e = np.array([e1, e2, e3])
    et = np.array([[0, -e3, e2], [e3, 0, -e1], [-e2, e1, 0]])
    return (2 * e0 * e0 - 1) * np.eye(3) + 2 * (np.outer(e, e) + e0 * et)


def energy_drift(mbs, kind, dt, t_end=1.0):
    """Max |E(t) - E(0)| and final |C| for a free body under gravity.

    :return: (max energy drift, final max |C|)
    :rtype: tuple
    """
    inertia = np.array([0.02, 0.03, 0.04])
    s, _ = build(mbs, kind, omega=[0.4, -0.7, 1.1], gravity=[0, -9.81, 0],
                 dt=dt)
    s.init()

    def energy():
        r = np.array(s.get_body_positions())[1]
        v = np.array(s.get_body_velocities())[1]
        w = np.array(s.get_body_angular_velocities())[1]
        wb = rot(np.asarray(s.get_body_quaternions())[1]).T @ w
        return 0.5 * v @ v + 0.5 * wb @ (inertia * wb) + 9.81 * r[1]

    e0 = energy()
    drift = 0.0
    for _ in range(int(round(t_end / dt))):
        s.solve()
        drift = max(drift, abs(energy() - e0))
    c = result_arrays(s.evaluate_constraints(s.get_sim_time()))["C"]
    return drift, np.max(np.abs(c))


@pytest.mark.parametrize("kind", KINDS)
def test_energy_drift_converges_with_rk4_order(mbs, kind):
    # Check type CV: energy is conserved by the model, so the drift is RK4
    # truncation error and must fall like dt^4. Measured order on these
    # cases is 3.1-4.0; the test requires at least 2.8.
    d1, c1 = energy_drift(mbs, kind, 1e-3)
    d2, c2 = energy_drift(mbs, kind, 5e-4)
    if d1 > 1e-10:
        assert np.log2(d1 / d2) > 2.8, (d1, d2)
    assert d2 < 1e-3
    assert c2 < 1e-6


def test_driven_screw_translates_by_pitch_times_angle(mbs):
    s = mbs.DynamicsSystem(1e-2)
    g = mbs.create_ground(0)
    b = mbs.create_mobilized_body_quat(1, R0, Z, Z, [1, 0, 0, 0], Z, Z, 1.0,
                                       [0.1, 0.1, 0.1], Z, Z)
    s.Add(g)
    s.Add(b)
    s.Add(mbs.create_screw_joint(g, b, Z, Z, AZ, AZ, pitch=0.05))
    f = mbs.DriveFunction.polynomial([0.0, 3.0, 1.0])
    s.Add(mbs.create_rotational_driver(body_i=g, body_j=b, axis_i=AZ,
                                       axis_j=AZ, func=f))
    s.init()
    for k in range(1, 401):  # C(4 s) = 28 rad: more than 4 revolutions
        s.solve()
        c, cd, cdd = f.eval(k * 1e-2)
        assert np.asarray(s.get_body_positions())[1][2] == pytest.approx(
            0.8 + 0.05 * c, abs=1e-10)
        assert np.asarray(s.get_body_velocities())[1][2] == pytest.approx(
            0.05 * cd, abs=1e-9)
        assert np.asarray(s.get_body_accelerations())[1][2] == pytest.approx(
            0.05 * cdd, abs=1e-9)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
