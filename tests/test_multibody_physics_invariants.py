# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Physics invariants for general 3D rotation (verification type IV).

Rules under test:

* A torque-free rigid body keeps its kinetic energy and its global angular
  momentum, also when the angular velocity is not along a principal axis.
* A spherical pendulum under gravity keeps its total energy, and the joint
  constraint does not drift.

Planar motion about one axis cannot detect a wrong angular velocity map
(G versus E), so every case here tumbles in 3D.
"""

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.tools.arrays import result_arrays

Z = [0.0, 0.0, 0.0]


@pytest.fixture
def mbs():
    """Import the compiled solver module or skip.

    :return: compiled solver module
    :rtype: module
    """
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    return multibody


def rot_body_to_global(q):
    """Haug A(p): body-to-global rotation matrix.

    :param q: Euler parameters [e0, e1, e2, e3]
    :type q: list
    :return: 3x3 matrix
    :rtype: numpy.ndarray
    """
    e0, e1, e2, e3 = q
    e = np.array([e1, e2, e3])
    et = np.array([[0, -e3, e2], [e3, 0, -e1], [-e2, e1, 0]])
    return (2 * e0 * e0 - 1) * np.eye(3) + 2 * (np.outer(e, e) + e0 * et)


def kinetic_energy(m, inertia, v, omega, q):
    """Translational plus rotational kinetic energy.

    :return: kinetic energy
    :rtype: float
    """
    a = rot_body_to_global(q)
    w_body = a.T @ np.asarray(omega)
    return 0.5 * m * np.dot(v, v) + 0.5 * w_body @ (np.diag(inertia) @ w_body)


def test_torque_free_asymmetric_body_conserves_energy_and_momentum(mbs):
    inertia = [0.3, 0.5, 0.9]
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    q0 = np.array([0.8, 0.2, -0.4, 0.4])
    q0 /= np.linalg.norm(q0)
    b = mbs.create_mobilized_body_quat(1, Z, Z, Z, list(q0), [1.0, 2.0, -1.5],
                                       Z, 2.0, inertia, Z, Z)
    s.Add(g)
    s.Add(b)
    s.init()

    def invariants():
        q = np.asarray(s.get_body_quaternions())[1]
        w = np.array(s.get_body_angular_velocities())[1]
        a = rot_body_to_global(q)
        h_global = a @ (np.diag(inertia) @ (a.T @ w))
        return kinetic_energy(2.0, inertia, Z, w, q), h_global

    e0, h0 = invariants()
    np.testing.assert_allclose(np.asarray(s.get_body_angular_velocities())[1],
                               [1.0, 2.0, -1.5], atol=1e-12)
    for _ in range(2000):
        s.solve()
    e1, h1 = invariants()
    assert abs(e1 - e0) / e0 < 1e-8
    np.testing.assert_allclose(h1, h0, rtol=1e-7, atol=1e-9)


def test_spherical_pendulum_3d_conserves_energy(mbs):
    m = 1.5
    inertia = [0.02, 0.05, 0.04]
    r_com = np.array([0.5, -0.8, 0.3])
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    b = mbs.create_mobilized_body_quat(1, list(r_com), Z, Z, [1, 0, 0, 0],
                                       [0.5, 1.0, -0.7], Z, m, inertia,
                                       [0.0, -9.81 * m, 0.0], Z)
    s.Add(g)
    s.Add(b)
    s.Add(mbs.create_joint(0, Z, list(-r_com), Z, Z, [0, 0, 1], [0, 0, 1],
                           g, b))
    s.init()

    def energy():
        r = np.array(s.get_body_positions())[1]
        v = np.array(s.get_body_velocities())[1]
        w = np.array(s.get_body_angular_velocities())[1]
        q = np.asarray(s.get_body_quaternions())[1]
        return kinetic_energy(m, inertia, v, w, q) + m * 9.81 * r[1]

    e0 = energy()
    worst_c = 0.0
    for _ in range(3000):
        s.solve()
        c = result_arrays(s.evaluate_constraints(s.get_sim_time()))["C"]
        worst_c = max(worst_c, np.max(np.abs(c)))
    assert abs(energy() - e0) < 1e-6 * abs(e0)
    assert worst_c < 1e-8

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
