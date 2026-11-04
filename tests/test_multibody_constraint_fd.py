# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Finite-difference verification of every joint and driver type.

For each type, the test puts two bodies in a generic state that does not
satisfy the joint, and moves both bodies along a rigid motion with constant
velocity v and constant inertial angular velocity omega. Along that path:

* Rule FD-1: ``Cq_pos`` equals the numeric derivative of C with respect to
  [r, p] (p perturbed without normalization).
* Rule FD-2: ``Cq @ qdot + Ct`` equals dC/dt.
* Rule FD-3: ``Cq @ qddot - gamma`` equals d2C/dt2. This is the identity
  that the acceleration equation Cq qddot = gamma relies on.
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


def unit(v):
    """Return v / |v|.

    :param v: vector
    :type v: list
    :return: normalized vector
    :rtype: numpy.ndarray
    """
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def g_matrix(p):
    """Haug G(p) as in ``build_G``: omega_global = 2 G(p) p_dot.

    :param p: Euler parameters [e0, e1, e2, e3]
    :type p: numpy.ndarray
    :return: 3x4 matrix
    :rtype: numpy.ndarray
    """
    e0, e1, e2, e3 = p
    return np.array([[-e1, e0, e3, -e2],
                     [-e2, -e3, e0, e1],
                     [-e3, e2, -e1, e0]])


def expm_series(a):
    """Matrix exponential by scaling and squaring of a Taylor series.

    :param a: square matrix
    :type a: numpy.ndarray
    :return: exp(a)
    :rtype: numpy.ndarray
    """
    k = max(0, int(np.ceil(np.log2(max(np.linalg.norm(a, 1), 1e-300)))) + 1)
    b = a / 2.0 ** k
    out = np.eye(a.shape[0])
    term = np.eye(a.shape[0])
    for n in range(1, 25):
        term = term @ b / n
        out = out + term
    for _ in range(k):
        out = out @ out
    return out


class RigidPath:
    """Rigid motion with constant v and constant global omega.

    p(t) solves p_dot = 0.5 G(p)^T omega, which is linear in p:
    p_dot = W p with W[:, k] = 0.5 G(e_k)^T omega.
    """

    def __init__(self, r0, v, p0, omega):
        self.r0 = np.asarray(r0, float)
        self.v = np.asarray(v, float)
        self.p0 = unit(p0)
        self.omega = np.asarray(omega, float)
        self.w = np.column_stack(
            [0.5 * g_matrix(np.eye(4)[k]).T @ self.omega for k in range(4)])

    def state(self, t):
        """Return (r, v, p, p_dot) at time t."""
        p = expm_series(self.w * t) @ self.p0
        return self.r0 + self.v * t, self.v, p, self.w @ p

    def accel(self, t):
        """Return (r_ddot, p_ddot) at time t."""
        p = expm_series(self.w * t) @ self.p0
        return np.zeros(3), self.w @ self.w @ p


PATHS = [
    RigidPath([0.1, 0.2, -0.3], [0.3, -0.2, 0.5],
              [0.9, 0.1, -0.3, 0.2], [0.7, -1.1, 0.4]),
    RigidPath([0.5, -0.1, 0.2], [-0.4, 0.1, 0.25],
              [0.8, -0.2, 0.4, 0.3], [-0.5, 0.6, 1.3]),
]

PI = [0.3, -0.2, 0.1]
PJ = [-0.1, 0.25, 0.05]
AI = list(unit([0.2, 0.3, 0.9]))
AJ = list(unit([0.1, -0.4, 0.8]))


def make_joint(mbs, kind, g, b1, b2):
    """Create the joint under test between b1 and b2 (or b1 and ground).

    :return: joint object
    :rtype: solvcon.multibody.Joint
    """
    sine = mbs.DriveFunction.sine(amplitude=0.3, omega=2.0, phase=0.4,
                                  offset=0.6)
    if kind in (0, 1, 2, 4, 5, 7):
        return mbs.create_joint(kind, PI, PJ, Z, Z, AI, AJ, b1, b2)
    if kind == 3:
        return mbs.create_joint_driving(3, Z, Z, Z, Z, AI, AJ, g, b1, 1.5)
    if kind == 6:
        return mbs.create_joint_distance(6, PI, PJ, Z, Z, AI, AJ, b1, b2, 0.7)
    if kind == 8:
        return mbs.create_absolute_driver(body=b1, ground=g, point=PI,
                                          direction=AI, func=sine)
    if kind == 9:
        return mbs.create_distance_driver(body_i=b1, body_j=b2, point_i=PI,
                                          point_j=PJ, func=sine)
    if kind == 10:
        return mbs.create_translational_driver(body_i=b1, body_j=b2,
                                               point_i=PI, point_j=PJ,
                                               axis_i=AI, func=sine)
    if kind == 11:
        return mbs.create_rotational_driver(body_i=b1, body_j=b2, axis_i=AI,
                                            axis_j=AJ, func=sine)
    if kind == 12:
        return mbs.create_screw_joint(b1, b2, PI, PJ, AI, AJ, pitch=0.05)
    if kind == 13:
        return mbs.create_revolute_spherical_joint(b1, b2, PI, PJ, AI,
                                                   distance=0.7)
    if kind in (14, 15):
        return mbs.create_revolute_revolute_joint(b1, b2, PI, PJ, AI, AJ,
                                                  orthogonal=(kind == 15),
                                                  distance=0.7)
    if kind == 16:
        f_j = np.cross(AJ, [1.0, 0.0, 0.0])
        return mbs.create_revolute_translational_joint(b1, b2, PI, PJ, AI, AJ,
                                                       list(unit(f_j)))
    if kind == 17:
        return mbs.create_strut_joint(b1, b2, PI, PJ, AI)
    raise ValueError(kind)


class Harness:
    """Two bodies on rigid paths plus one joint, evaluated through the API."""

    def __init__(self, mbs, kind):
        self.sys = mbs.DynamicsSystem(1e-3)
        g = mbs.create_ground(0)
        b1 = mbs.create_mobilized_body_quat(1, Z, Z, Z, [1, 0, 0, 0], Z, Z,
                                            1.0, [1, 1, 1], Z, Z)
        b2 = mbs.create_mobilized_body_quat(2, Z, Z, Z, [1, 0, 0, 0], Z, Z,
                                            1.0, [1, 1, 1], Z, Z)
        for b in (g, b1, b2):
            self.sys.Add(b)
        self.sys.Add(make_joint(mbs, kind, g, b1, b2))
        self.t0 = 0.3

    def set_path_state(self, t):
        for i, path in enumerate(PATHS, start=1):
            r, v, p, pd = path.state(t - self.t0)
            self.sys.set_body_state(i, list(r), list(v), list(p), list(pd))

    def eval_at(self, t):
        self.set_path_state(t)
        return result_arrays(self.sys.evaluate_constraints(t))

    def qdot(self, t):
        out = np.zeros(21)
        for i, path in enumerate(PATHS, start=1):
            _, v, _, pd = path.state(t - self.t0)
            out[7 * i:7 * i + 3] = v
            out[7 * i + 3:7 * i + 7] = pd
        return out

    def qddot(self, t):
        out = np.zeros(21)
        for i, path in enumerate(PATHS, start=1):
            ra, pa = path.accel(t - self.t0)
            out[7 * i:7 * i + 3] = ra
            out[7 * i + 3:7 * i + 7] = pa
        return out


ALL_KINDS = list(range(18))
JOINT_ROWS = slice(7, -2)  # skip 7 ground rows and 2 normalization rows


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_position_jacobian_matches_finite_difference(mbs, kind):
    h = 1e-6
    hz = Harness(mbs, kind)
    t = hz.t0
    ev = hz.eval_at(t)
    jac = ev["Cq_pos"]
    for i, path in enumerate(PATHS, start=1):
        r, v, p, pd = path.state(0.0)
        base = np.concatenate([r, p])
        for k in range(7):
            cols = []
            for sgn in (+1, -1):
                x = base.copy()
                x[k] += sgn * h
                hz.sys.set_body_state(i, list(x[:3]), list(v), list(x[3:]),
                                      list(pd))
                cols.append(result_arrays(hz.sys.evaluate_constraints(t))["C"])
            hz.set_path_state(t)
            fd = (cols[0] - cols[1]) / (2 * h)
            np.testing.assert_allclose(
                jac[JOINT_ROWS, 7 * i + k], fd[JOINT_ROWS], rtol=1e-6,
                atol=1e-8, err_msg=f"type {kind}, body {i}, coord {k}")


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_velocity_equation_matches_time_derivative(mbs, kind):
    h = 1e-6
    hz = Harness(mbs, kind)
    t = hz.t0
    ev = hz.eval_at(t)
    dcdt = (hz.eval_at(t + h)["C"] - hz.eval_at(t - h)["C"]) / (2 * h)
    pred = ev["Cq"] @ hz.qdot(t) + ev["Ct"]
    np.testing.assert_allclose(pred[JOINT_ROWS], dcdt[JOINT_ROWS],
                               rtol=1e-6, atol=1e-8, err_msg=f"type {kind}")


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_gamma_matches_second_time_derivative(mbs, kind):
    h = 1e-4
    hz = Harness(mbs, kind)
    t = hz.t0
    ev = hz.eval_at(t)
    c0 = ev["C"]
    d2 = (hz.eval_at(t + h)["C"] - 2 * c0 + hz.eval_at(t - h)["C"]) / h ** 2
    pred = ev["Cq"] @ hz.qddot(t) - ev["gamma"]
    # Central second difference error is O(h^2 * C'''') ~ 1e-8 here.
    np.testing.assert_allclose(pred[JOINT_ROWS], d2[JOINT_ROWS],
                               rtol=1e-5, atol=1e-6, err_msg=f"type {kind}")


def test_normalization_rows_satisfy_fd_rules(mbs):
    hz = Harness(mbs, 0)
    t = hz.t0
    h = 1e-4
    ev = hz.eval_at(t)
    d2 = (hz.eval_at(t + h)["C"] - 2 * ev["C"]
          + hz.eval_at(t - h)["C"]) / h ** 2
    pred = ev["Cq"] @ hz.qddot(t) - ev["gamma"]
    np.testing.assert_allclose(pred[-2:], d2[-2:], atol=1e-6)
    np.testing.assert_allclose(ev["Cq"][-2:] @ hz.qdot(t), 0.0, atol=1e-12)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
