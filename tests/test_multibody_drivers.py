# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for driver functions and driving constraints (Haug Sec. 9.5).

Rules under test:

* Every drive function returns C(t), C'(t), and C''(t), and the
  derivatives match finite differences of C.
* If a driver drives a zero-DOF mechanism, then the driven coordinate
  equals C(t) and its rate equals C'(t) at every kinematic step.
* If a driver acts in a mechanism with free DOF, then RK4 integration keeps
  the driving constraint satisfied. This checks the gamma term that
  contains C''(t).
"""

import math

import numpy as np
import pytest

from solvcon import multibody

Z = [0.0, 0.0, 0.0]
AX = [1.0, 0.0, 0.0]
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


def fd_check(func, t, h=1e-5):
    """Compare analytic derivatives of a drive function with central FD.

    :param func: drive function object
    :type func: solvcon.multibody.DriveFunction
    :param t: evaluation time
    :type t: float
    :param h: finite difference step
    :type h: float
    """
    c, cd, cdd = func.eval(t)
    cp, cdp, _ = func.eval(t + h)
    cm, cdm, _ = func.eval(t - h)
    assert cd == pytest.approx((cp - cm) / (2 * h), rel=1e-6, abs=1e-8)
    assert cdd == pytest.approx((cdp - cdm) / (2 * h), rel=1e-6, abs=1e-6)


# ---------------------------------------------------------------- functions

class TestDriveFunction:

    def test_polynomial_values(self, mbs):
        f = mbs.DriveFunction.polynomial([1.0, 2.0, 3.0])
        assert f.eval(2.0) == pytest.approx((17.0, 14.0, 6.0), abs=1e-14)

    def test_sine_values(self, mbs):
        f = mbs.DriveFunction.sine(amplitude=0.5, omega=3.0, phase=0.2,
                                   offset=1.0)
        t = 0.7
        arg = 3.0 * t + 0.2
        assert f.eval(t) == pytest.approx(
            (1.0 + 0.5 * math.sin(arg), 1.5 * math.cos(arg),
             -4.5 * math.sin(arg)), abs=1e-14)

    @pytest.mark.parametrize("t", [0.1, 0.9, 2.3])
    def test_derivatives_match_finite_difference(self, mbs, t):
        fd_check(mbs.DriveFunction.polynomial([0.3, -1.0, 0.5, 0.2]), t)
        fd_check(mbs.DriveFunction.sine(amplitude=2.0, omega=1.7), t)
        fd_check(mbs.DriveFunction.spline([0, 1, 2, 3], [0, 1, 0, 2]), t)

    def test_table_is_piecewise_linear(self, mbs):
        f = mbs.DriveFunction.table([0.0, 1.0, 3.0], [0.0, 2.0, 1.0])
        assert f.eval(0.5) == pytest.approx((1.0, 2.0, 0.0), abs=1e-14)
        assert f.eval(2.0) == pytest.approx((1.5, -0.5, 0.0), abs=1e-14)

    def test_spline_interpolates_knots_with_natural_ends(self, mbs):
        ts = [0.0, 0.5, 1.5, 2.0]
        vs = [1.0, -1.0, 2.0, 0.0]
        f = mbs.DriveFunction.spline(ts, vs)
        for t, v in zip(ts, vs):
            assert f.eval(t)[0] == pytest.approx(v, abs=1e-12)
        assert f.eval(0.0)[2] == pytest.approx(0.0, abs=1e-12)
        assert f.eval(2.0)[2] == pytest.approx(0.0, abs=1e-12)

    def test_spline_reproduces_linear_data(self, mbs):
        f = mbs.DriveFunction.spline([0, 1, 2, 4], [1, 3, 5, 9])
        assert f.eval(3.3) == pytest.approx((7.6, 2.0, 0.0), abs=1e-12)

    @pytest.mark.parametrize("kind", ["table", "spline"])
    def test_tabulated_function_rejects_time_outside_range(self, mbs, kind):
        f = getattr(mbs.DriveFunction, kind)([0.0, 1.0, 2.0], [0, 1, 0])
        with pytest.raises((ValueError, IndexError)):
            f.eval(2.5)

    def test_tabulated_function_rejects_bad_input(self, mbs):
        with pytest.raises(ValueError):
            mbs.DriveFunction.table([0.0, 0.0, 1.0], [0, 1, 2])
        with pytest.raises(ValueError):
            mbs.DriveFunction.spline([0.0, 1.0], [0.0])

    def test_parameter_has_settable_value_and_zero_rates(self, mbs):
        f = mbs.DriveFunction.parameter(0.25)
        assert f.eval(10.0) == (0.25, 0.0, 0.0)
        f.set_value(-0.1)
        assert f.eval(0.0) == (-0.1, 0.0, 0.0)


# ------------------------------------------------------------ mechanisms

def build_slider(mbs, x0=0.3, dt=1e-3):
    """Build a body on a prismatic joint along global x.

    :return: (system, ground, slider)
    :rtype: tuple
    """
    s = mbs.DynamicsSystem(dt)
    g = mbs.create_ground(0)
    s.Add(g)
    b = mbs.create_mobilized_body_quat(1, [x0, 0, 0], Z, Z, [1, 0, 0, 0], Z, Z,
                                       1.0, [0.1, 0.1, 0.1], Z, Z)
    s.Add(b)
    s.Add(mbs.create_joint(2, Z, Z, Z, Z, AX, AX, g, b))
    return s, g, b


def run_kinematic(s, n_steps, dt):
    """Init and step a zero-DOF system, recording t, x, vx, and omega_z.

    :return: arrays (t, x, vx, wz)
    :rtype: tuple
    """
    s.init()
    rows = []
    for k in range(n_steps + 1):
        if k > 0:
            s.solve()
        pos = np.asarray(s.get_body_positions())[1]
        vel = np.asarray(s.get_body_velocities())[1]
        w = np.asarray(s.get_body_angular_velocities())[1]
        rows.append((k * dt, pos[0], vel[0], w[2], pos[1], pos[2]))
    return np.array(rows)


class TestAbsoluteDriver:

    def test_slider_follows_absolute_x_driver(self, mbs):
        dt = 1e-2
        s, g, b = build_slider(mbs, dt=dt)
        f = mbs.DriveFunction.sine(amplitude=0.1, omega=2.0, offset=0.3)
        s.Add(mbs.create_absolute_driver(body=b, ground=g, point=Z,
                                         direction=AX, func=f))
        r = run_kinematic(s, 300, dt)
        expect = np.array([f.eval(t) for t in r[:, 0]])
        np.testing.assert_allclose(r[:, 1], expect[:, 0], atol=1e-10)
        np.testing.assert_allclose(r[:, 2], expect[:, 1], atol=1e-9)
        np.testing.assert_allclose(r[:, 4:6], 0.0, atol=1e-10)


class TestDistanceDriver:

    def test_slider_follows_distance_driver(self, mbs):
        dt = 1e-2
        s, g, b = build_slider(mbs, x0=0.5, dt=dt)
        f = mbs.DriveFunction.sine(amplitude=0.1, omega=3.0, offset=0.5)
        s.Add(mbs.create_distance_driver(body_i=g, body_j=b, point_i=Z,
                                         point_j=Z, func=f))
        r = run_kinematic(s, 300, dt)
        expect = np.array([f.eval(t) for t in r[:, 0]])
        np.testing.assert_allclose(r[:, 1], expect[:, 0], atol=1e-10)
        np.testing.assert_allclose(r[:, 2], expect[:, 1], atol=1e-9)


class TestTranslationalDriver:

    def test_slider_follows_translational_driver(self, mbs):
        dt = 1e-2
        s, g, b = build_slider(mbs, x0=0.0, dt=dt)
        # C(t) crosses zero, so the driver must allow negative values.
        f = mbs.DriveFunction.sine(amplitude=0.2, omega=2.0, offset=-0.05)
        s.Add(mbs.create_translational_driver(body_i=g, body_j=b, point_i=Z,
                                              point_j=Z, axis_i=AX, func=f))
        r = run_kinematic(s, 300, dt)
        expect = np.array([f.eval(t) for t in r[:, 0]])
        np.testing.assert_allclose(r[:, 1], expect[:, 0], atol=1e-10)
        np.testing.assert_allclose(r[:, 2], expect[:, 1], atol=1e-9)


def build_rotor(mbs, dt):
    """Build a body on a revolute joint about global z at the origin.

    :return: (system, ground, rotor)
    :rtype: tuple
    """
    s = mbs.DynamicsSystem(dt)
    g = mbs.create_ground(0)
    s.Add(g)
    b = mbs.create_mobilized_body_quat(1, Z, Z, Z, [1, 0, 0, 0], Z, Z,
                                       1.0, [0.1, 0.1, 0.1], Z, Z)
    s.Add(b)
    s.Add(mbs.create_joint(1, Z, Z, Z, Z, AZ, AZ, g, b))
    return s, g, b


def unwrap_z_angle(quats):
    """Return the continuous rotation angle about z from quaternions.

    :param quats: quaternions [e0, e1, e2, e3] of a body that rotates
        about z only
    :type quats: numpy.ndarray
    :return: unwrapped angles in radians
    :rtype: numpy.ndarray
    """
    return np.unwrap(2.0 * np.arctan2(quats[:, 3], quats[:, 0]))


class TestRotationalDriver:

    def test_rotor_follows_driver_over_many_revolutions(self, mbs):
        dt = 1e-2
        s, g, b = build_rotor(mbs, dt)
        f = mbs.DriveFunction.polynomial([0.5, 3.0, 0.5])
        s.Add(mbs.create_rotational_driver(body_i=g, body_j=b, axis_i=AZ,
                                           axis_j=AZ, func=f))
        s.init()
        times, quats, wz = [], [], []
        for k in range(501):  # t up to 5 s: about 4.5 revolutions
            if k > 0:
                s.solve()
            times.append(k * dt)
            quats.append(np.asarray(s.get_body_quaternions())[1])
            wz.append(np.asarray(s.get_body_angular_velocities())[1][2])
        expect = np.array([f.eval(t) for t in times])
        theta = unwrap_z_angle(np.array(quats))
        np.testing.assert_allclose(theta, expect[:, 0], atol=1e-9)
        np.testing.assert_allclose(wz, expect[:, 1], atol=1e-9)
        assert expect[-1, 0] > 8 * math.pi


class TestDriversInDynamics:
    """Driven body plus a free pendulum: the driver row must stay exact."""

    def _build(self, mbs, driver_kind, dt):
        s = mbs.DynamicsSystem(dt)
        g = mbs.create_ground(0)
        s.Add(g)
        b1 = mbs.create_mobilized_body_quat(1, Z, Z, Z, [1, 0, 0, 0], Z, Z,
                                            1.0, [0.1, 0.1, 0.1], Z, Z)
        s.Add(b1)
        # Pendulum hangs from b1 at its COM, revolute about z, gravity -y.
        b2 = mbs.create_mobilized_body_quat(2, [0.5, -0.5, 0], Z, Z,
                                            [1, 0, 0, 0], Z, Z, 1.0,
                                            [0.05, 0.05, 0.05],
                                            [0, -9.81, 0], Z)
        s.Add(b2)
        s.Add(mbs.create_joint(1, Z, [-0.5, 0.5, 0], Z, Z, AZ, AZ, b1, b2))
        if driver_kind == "rotational":
            s.Add(mbs.create_joint(1, Z, Z, Z, Z, AZ, AZ, g, b1))
            f = mbs.DriveFunction.polynomial([0.0, 1.0, 2.0])
            s.Add(mbs.create_rotational_driver(body_i=g, body_j=b1,
                                               axis_i=AZ, axis_j=AZ, func=f))
        else:
            s.Add(mbs.create_joint(2, Z, Z, Z, Z, AX, AX, g, b1))
            f = mbs.DriveFunction.sine(amplitude=0.2, omega=4.0)
            s.Add(mbs.create_translational_driver(body_i=g, body_j=b1,
                                                  point_i=Z, point_j=Z,
                                                  axis_i=AX, func=f))
        return s, f

    @pytest.mark.parametrize("driver_kind", ["rotational", "translational"])
    def test_driver_constraint_holds_during_rk4(self, mbs, driver_kind):
        dt = 1e-3
        s, f = self._build(mbs, driver_kind, dt)
        s.init()
        worst = 0.0
        quats = [np.asarray(s.get_body_quaternions())[1]]
        xs = [np.asarray(s.get_body_positions())[1][0]]
        for _ in range(2000):
            s.solve()
            quats.append(np.asarray(s.get_body_quaternions())[1])
            xs.append(np.asarray(s.get_body_positions())[1][0])
        times = np.arange(len(xs)) * dt
        expect = np.array([f.eval(t)[0] for t in times])
        if driver_kind == "rotational":
            actual = unwrap_z_angle(np.array(quats))
        else:
            actual = np.array(xs)
        worst = np.max(np.abs(actual - expect))
        # A missing or wrong C'' term in gamma gives an error near
        # C''/beta^2 ~ 1e-2. Correct gamma leaves only RK4 error.
        assert worst < 1e-7


class TestHaugSliderCrankWithRotationalDriver:
    """XM: Haug 10.2 with the new driver equals the legacy type-3 driver."""

    def _build(self, mbs, use_new_driver):
        from solvcon.multibody.models.haug import (
            slider_crank_kinematics as haug)

        s, g, crank, rod, slider, dt = haug.build_slider_crank(
            use_new_driver=use_new_driver)
        return s

    def test_slider_history_matches_legacy_driver(self, mbs):
        hist = []
        for use_new in (False, True):
            s = self._build(mbs, use_new)
            s.init()
            xs = []
            for _ in range(2000):
                s.solve()
                xs.append(np.asarray(s.get_body_positions())[3][0])
            hist.append(np.array(xs))
        np.testing.assert_allclose(hist[1], hist[0], atol=1e-9)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
