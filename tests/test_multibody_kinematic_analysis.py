# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for acceleration analysis, quasi-static sweeps, redundant constraint
detection, and singular configuration monitoring (Haug 9.6.3, 3.6.4, 4.6, 3.7).

Rules under test:

* After every step, body accelerations satisfy Cq qddot = gamma. Driven
  coordinates have acceleration C''(t).
* A sweep over a parameter driver returns one pose per value and the pose
  sensitivity dq/ds. If the position analysis fails at a value, then the
  sweep stops at that value, returns only the converged values, and
  reports the failing value.
* If a mechanism has redundant constraints, then the solver reports each
  redundant row and uses the rank-based DOF to choose kinematic or dynamic
  analysis.
"""

import math

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.tools.arrays import result_arrays


Z = [0.0, 0.0, 0.0]
AX = [1.0, 0.0, 0.0]
AY = [0.0, 1.0, 0.0]
AZ = [0.0, 0.0, 1.0]


@pytest.fixture
def mbs():
    """Import the compiled solver module or skip.

    :return: compiled solver module
    :rtype: module
    """
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    return multibody


def slider(mbs, func, x0=0.3, dt=1e-2):
    """Body on a prismatic joint along x, driven by an absolute x driver.

    :return: (system, driver)
    :rtype: tuple
    """
    s = mbs.DynamicsSystem(dt)
    g = mbs.create_ground(0)
    b = mbs.create_mobilized_body_quat(1, [x0, 0, 0], Z, Z, [1, 0, 0, 0], Z,
                                       Z, 1.0, [0.1, 0.1, 0.1], Z, Z)
    s.Add(g)
    s.Add(b)
    s.Add(mbs.create_joint(2, Z, Z, Z, Z, AX, AX, g, b))
    drv = mbs.create_absolute_driver(body=b, ground=g, point=Z, direction=AX,
                                     func=func)
    s.Add(drv)
    return s, drv


def two_link_arm(mbs, func_x):
    """Planar two-link arm (lengths 1, 1) whose tip is driven to (x, 0).

    Revolute joints about z at the origin and at the elbow. The tip has an
    absolute y driver (y = 0) and an absolute x driver. The arm is fully
    stretched at x = 2, so x > 2 has no assembly.

    :return: (system, x driver)
    :rtype: tuple
    """
    s = mbs.DynamicsSystem(1e-2)
    g = mbs.create_ground(0)
    # Start at x = 1.2: elbow up, angles from geometry.
    x = 1.2
    a = math.acos(x / 2.0)
    q1 = [math.cos(a / 2), 0, 0, math.sin(a / 2)]
    q2 = [math.cos(-a / 2), 0, 0, math.sin(-a / 2)]
    c1 = [0.5 * math.cos(a), 0.5 * math.sin(a), 0]
    elbow = [math.cos(a), math.sin(a), 0]
    c2 = [elbow[0] + 0.5 * math.cos(a), elbow[1] - 0.5 * math.sin(a), 0]
    b1 = mbs.create_mobilized_body_quat(1, c1, Z, Z, q1, Z, Z, 1.0,
                                        [0.1, 0.1, 0.1], Z, Z)
    b2 = mbs.create_mobilized_body_quat(2, c2, Z, Z, q2, Z, Z, 1.0,
                                        [0.1, 0.1, 0.1], Z, Z)
    for b in (g, b1, b2):
        s.Add(b)
    s.Add(mbs.create_joint(1, Z, [-0.5, 0, 0], Z, Z, AZ, AZ, g, b1))
    s.Add(mbs.create_joint(1, [0.5, 0, 0], [-0.5, 0, 0], Z, Z, AZ, AZ, b1, b2))
    s.Add(mbs.create_absolute_driver(body=b2, ground=g, point=[0.5, 0, 0],
                                     direction=AY,
                                     func=mbs.DriveFunction.parameter(0.0)))
    drv = mbs.create_absolute_driver(body=b2, ground=g, point=[0.5, 0, 0],
                                     direction=AX, func=func_x)
    s.Add(drv)
    return s, drv


# ------------------------------------------------------------ accelerations

class TestAccelerationAnalysis:

    def test_driven_slider_acceleration_equals_c_ddot(self, mbs):
        f = mbs.DriveFunction.sine(amplitude=0.1, omega=2.0, offset=0.3)
        s, _ = slider(mbs, f)
        s.init()
        for k in range(1, 200):
            s.solve()
            ax = np.asarray(s.get_body_accelerations())[1][0]
            assert ax == pytest.approx(f.eval(k * 1e-2)[2], abs=1e-9)

    def test_driven_rotor_angular_acceleration_equals_c_ddot(self, mbs):
        s = mbs.DynamicsSystem(1e-2)
        g = mbs.create_ground(0)
        b = mbs.create_mobilized_body_quat(1, Z, Z, Z, [1, 0, 0, 0], Z, Z,
                                           1.0, [0.1, 0.1, 0.1], Z, Z)
        s.Add(g)
        s.Add(b)
        s.Add(mbs.create_joint(1, Z, Z, Z, Z, AZ, AZ, g, b))
        f = mbs.DriveFunction.sine(amplitude=1.0, omega=3.0)
        s.Add(mbs.create_rotational_driver(body_i=g, body_j=b, axis_i=AZ,
                                           axis_j=AZ, func=f))
        s.init()
        for k in range(1, 100):
            s.solve()
            alpha = np.asarray(s.get_body_angular_accelerations())[1]
            np.testing.assert_allclose(alpha, [0, 0, f.eval(k * 1e-2)[2]],
                                       atol=1e-9)

    def test_haug_slider_crank_acceleration_matches_fd_of_velocity(self, mbs):
        from solvcon.multibody.models.haug import (
            slider_crank_kinematics as haug)
        s, *_ = haug.build_slider_crank(use_new_driver=True)
        s.init()
        dt = 1e-4
        vel, acc = [], []
        for _ in range(400):
            s.solve()
            vel.append(np.array(s.get_body_velocities())[3])
            acc.append(np.array(s.get_body_accelerations())[3])
        vel = np.array(vel)
        acc = np.array(acc)
        fd = (vel[2:] - vel[:-2]) / (2 * dt)
        scale = np.max(np.abs(acc))
        np.testing.assert_allclose(acc[1:-1], fd, atol=1e-5 * scale)

    def test_free_body_acceleration_is_gravity(self, mbs):
        s = mbs.DynamicsSystem(1e-3)
        g = mbs.create_ground(0)
        b = mbs.create_mobilized_body_quat(1, Z, Z, Z, [1, 0, 0, 0], Z, Z,
                                           2.0, [0.1, 0.2, 0.3],
                                           [0, -19.62, 0], Z)
        s.Add(g)
        s.Add(b)
        s.init()
        s.solve()
        np.testing.assert_allclose(np.asarray(s.get_body_accelerations())[1],
                                   [0, -9.81, 0], atol=1e-10)


# ------------------------------------------------------------------- sweep

class TestSweep:

    def test_sweep_returns_pose_and_unit_sensitivity(self, mbs):
        s, drv = slider(mbs, mbs.DriveFunction.parameter(0.3))
        s.init()
        values = np.linspace(0.1, 0.7, 13)
        res = result_arrays(s.sweep(drv, list(values)))
        assert res["completed"]
        np.testing.assert_allclose(res["values"], values)
        np.testing.assert_allclose(res["positions"][:, 1, 0], values,
                                   atol=1e-10)
        np.testing.assert_allclose(res["dr_ds"][:, 1, 0], 1.0, atol=1e-10)
        assert np.all(res["residual"] < 1e-10)

    def test_sweep_sensitivity_matches_fd_on_haug_slider_crank(self, mbs):
        from solvcon.multibody.models.haug import (
            slider_crank_kinematics as haug)
        theta0 = 2.0 * math.atan2(0.71, math.sqrt(1 - 0.71 ** 2))
        s, *_ = haug.build_slider_crank(
            use_new_driver=True,
            crank_func=mbs.DriveFunction.parameter(theta0))
        s.init()
        drv = s.get_joints()[-1]
        angles = theta0 + np.linspace(0, 2 * math.pi, 73)
        res = result_arrays(s.sweep(drv, list(angles)))
        assert res["completed"]
        x = res["positions"][:, 3, 0]
        dxds = res["dr_ds"][:, 3, 0]
        fd = np.gradient(x, angles)
        # Central differences on a 5-degree grid: error ~ h^2 x''' / 6.
        np.testing.assert_allclose(dxds[1:-1], fd[1:-1], atol=2e-4)
        assert x.min() == pytest.approx(0.184952, abs=2e-5)
        assert x.max() == pytest.approx(0.290160, abs=2e-5)

    def test_sweep_stops_at_unreachable_value(self, mbs):
        s, drv = two_link_arm(mbs, mbs.DriveFunction.parameter(1.2))
        s.init()
        values = [1.2, 1.5, 1.8, 1.95, 1.99, 1.999, 1.9999, 2.05, 1.5]
        res = result_arrays(s.sweep(drv, values))
        assert not res["completed"]
        np.testing.assert_allclose(res["values"], values[:7])
        assert "2.05" in res["message"]
        # Approaching the stretched configuration, the Jacobian becomes
        # singular: rcond falls monotonically, like sqrt(2 - x).
        assert np.all(np.diff(res["rcond"]) < 0)
        assert res["rcond"][-1] < 0.1 * res["rcond"][0]
        # The system state stays at the last converged value.
        tip_x = np.asarray(s.get_body_positions())[2][0] + 0.5 * math.cos(
            2 * math.atan2(np.asarray(s.get_body_quaternions())[2][3],
                           np.asarray(s.get_body_quaternions())[2][0]))
        assert tip_x == pytest.approx(1.9999, abs=1e-9)
        # The driver is back at the last converged value: the next solve works.
        res2 = result_arrays(s.sweep(drv, [1.9]))
        assert res2["completed"]

    def test_sweep_rejects_non_parameter_driver(self, mbs):
        s, drv = slider(mbs, mbs.DriveFunction.sine(amplitude=0.1, omega=1.0,
                                                    offset=0.3))
        s.init()
        with pytest.raises(ValueError):
            s.sweep(drv, [0.3])


# -------------------------------------------------------------- redundancy

def planar_four_bar(mbs, gravity=True):
    """Planar four-bar with four revolute joints about z (3D model).

    In 3D, four parallel revolute joints have 3 redundant constraints.

    :return: system
    :rtype: solvcon.multibody.DynamicsSystem
    """
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    # Gravity along -x: both links stand vertical, so -y gravity would be a
    # dead-point equilibrium with zero acceleration.
    w = [-9.81, 0, 0] if gravity else Z
    crank = mbs.create_mobilized_body_quat(1, [0, 0.15, 0], Z, Z, [1, 0, 0, 0],
                                           Z, Z, 1.0, [0.01, 0.01, 0.01], w, Z)
    coupler = mbs.create_mobilized_body_quat(2, [0.5, 0.45, 0], Z, Z,
                                             [1, 0, 0, 0], Z, Z, 1.0,
                                             [0.01, 0.01, 0.01], w, Z)
    rocker = mbs.create_mobilized_body_quat(3, [1.0, 0.3, 0], Z, Z,
                                            [1, 0, 0, 0], Z, Z, 1.0,
                                            [0.01, 0.01, 0.01], w, Z)
    for b in (g, crank, coupler, rocker):
        s.Add(b)
    s.Add(mbs.create_joint(1, Z, [0, -0.15, 0], Z, Z, AZ, AZ, g, crank))
    s.Add(mbs.create_joint(1, [0, 0.15, 0], [-0.5, -0.15, 0], Z, Z, AZ, AZ,
                           crank, coupler))
    s.Add(mbs.create_joint(1, [0.5, 0.15, 0], [0, 0.3, 0], Z, Z, AZ, AZ,
                           coupler, rocker))
    s.Add(mbs.create_joint(1, [1.0, 0, 0], [0, -0.3, 0], Z, Z, AZ, AZ,
                           g, rocker))
    return s


class TestRedundancy:

    def test_planar_four_bar_reports_three_redundant_rows(self, mbs):
        s = planar_four_bar(mbs)
        s.init()
        diags = [d for d in s.get_diagnostics() if d["code"] == "redundant"]
        assert len(diags) == 3
        assert all(d["joint"] >= 0 for d in diags)
        assert s.get_dof() == 1

    def test_redundant_four_bar_runs_dynamics(self, mbs):
        s = planar_four_bar(mbs)
        s.init()
        x0 = np.asarray(s.get_body_positions())[2][0]
        for _ in range(300):
            s.solve()
        # Gravity along -x moves the free mechanism.
        assert np.asarray(s.get_body_positions())[2][0] < x0 - 1e-3
        c = result_arrays(s.evaluate_constraints(s.get_sim_time()))["C"]
        assert np.max(np.abs(c)) < 1e-8

    def test_mechanism_without_redundancy_reports_none(self, mbs):
        s, _ = slider(mbs, mbs.DriveFunction.parameter(0.3))
        s.init()
        assert not [d for d in s.get_diagnostics()
                    if d["code"] == "redundant"]
        assert s.get_dof() == 0

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
