# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for gears, rack and pinion, and cam followers (Haug Sec. 3.4).

Rules under test:

* Follower rows and gear coupler rows satisfy the finite-difference rules
  FD-1, FD-2, FD-3 of tests/test_constraint_fd.py.
* A gear pair keeps theta_A = ratio * theta_B; a rack moves R * theta.
* A point follower on a line through the cam center sits at R(pi/2 - theta).
* A flat-face follower on an eccentric circular cam (support function
  h = r0 + e cos psi) sits at r0 + e cos(theta).
"""

import math

import numpy as np
import pytest

import test_multibody_constraint_fd as fd

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


# ------------------------------------------------------------- FD checks

def follower(mbs, kind, g, b1, b2):
    if kind == 18:
        prof = mbs.CamProfile(0.5, [0.05, 0.0], [0.0, 0.02])
        return mbs.create_point_follower(b1, b2, fd.PI, fd.AI, fd.PJ, prof)
    prof = mbs.CamProfile(0.4, [0.03], [0.01])
    return mbs.create_flat_follower(b1, b2, fd.PI, fd.AI, fd.PJ, fd.AJ, prof)


class FollowerHarness(fd.Harness):
    def __init__(self, mbs, kind):  # noqa: D107 - same layout as fd.Harness
        self.sys = mbs.DynamicsSystem(1e-3)
        g = mbs.create_ground(0)
        b1 = mbs.create_mobilized_body_quat(1, Z, Z, Z, [1, 0, 0, 0], Z, Z,
                                            1.0, [1, 1, 1], Z, Z)
        b2 = mbs.create_mobilized_body_quat(2, Z, Z, Z, [1, 0, 0, 0], Z, Z,
                                            1.0, [1, 1, 1], Z, Z)
        for b in (g, b1, b2):
            self.sys.Add(b)
        self.sys.Add(follower(mbs, kind, g, b1, b2))
        self.t0 = 0.3


class CouplerHarness(fd.Harness):
    def __init__(self, mbs, kind_b):  # noqa: D107 - same layout as fd.Harness
        self.sys = mbs.DynamicsSystem(1e-3)
        g = mbs.create_ground(0)
        b1 = mbs.create_mobilized_body_quat(1, Z, Z, Z, [1, 0, 0, 0], Z, Z,
                                            1.0, [1, 1, 1], Z, Z)
        b2 = mbs.create_mobilized_body_quat(2, Z, Z, Z, [1, 0, 0, 0], Z, Z,
                                            1.0, [1, 1, 1], Z, Z)
        for b in (g, b1, b2):
            self.sys.Add(b)
        ja = mbs.create_joint(1, fd.PI, fd.PJ, Z, Z, fd.AI, fd.AJ, b1, b2)
        jb = mbs.create_joint(2 if kind_b == "translation" else 1, fd.PJ,
                              fd.PI, Z, Z, fd.AJ, fd.AJ, g, b2)
        self.sys.Add(ja)
        self.sys.Add(jb)
        self.sys.Add(mbs.create_gear_coupler(ja, "rotation", jb, kind_b, -1.7))
        self.t0 = 0.3


HARNESSES = [("point", lambda m: FollowerHarness(m, 18)),
             ("flat", lambda m: FollowerHarness(m, 19)),
             ("gear", lambda m: CouplerHarness(m, "rotation")),
             ("rack", lambda m: CouplerHarness(m, "translation"))]


def rows_under_test(name):
    # Followers: their own rows. Couplers: only the coupler row (the two
    # joints are already covered by test_constraint_fd.py).
    return {"point": slice(7, 9), "flat": slice(7, 8), "gear": slice(17, 18),
            "rack": slice(17, 18)}[name]


@pytest.mark.parametrize("name,make", HARNESSES, ids=[h[0] for h in HARNESSES])
def test_fd_rules(mbs, name, make):
    hz = make(mbs)
    rows = rows_under_test(name)
    t, h1, h2 = hz.t0, 1e-6, 1e-4
    ev = hz.eval_at(t)
    # FD-1 position Jacobian
    for i, path in enumerate(fd.PATHS, start=1):
        r, v, p, pd = path.state(0.0)
        base = np.concatenate([r, p])
        for k in range(7):
            c = []
            for sgn in (+1, -1):
                x = base.copy()
                x[k] += sgn * h1
                hz.sys.set_body_state(i, list(x[:3]), list(v), list(x[3:]),
                                      list(pd))
                c.append(result_arrays(hz.sys.evaluate_constraints(t))["C"])
            hz.set_path_state(t)
            np.testing.assert_allclose(ev["Cq_pos"][rows, 7 * i + k],
                                       ((c[0] - c[1]) / (2 * h1))[rows],
                                       rtol=1e-6, atol=1e-8)
    # FD-2 velocity equation
    ev = hz.eval_at(t)
    dcdt = (hz.eval_at(t + h1)["C"] - hz.eval_at(t - h1)["C"]) / (2 * h1)
    np.testing.assert_allclose((ev["Cq"] @ hz.qdot(t) + ev["Ct"])[rows],
                               dcdt[rows], rtol=1e-6, atol=1e-8)
    # FD-3 gamma
    ev = hz.eval_at(t)
    d2 = (hz.eval_at(t + h2)["C"] - 2 * ev["C"]
          + hz.eval_at(t - h2)["C"]) / h2 ** 2
    np.testing.assert_allclose((ev["Cq"] @ hz.qddot(t) - ev["gamma"])[rows],
                               d2[rows], rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------ analytic

def rotor(mbs, s, g, num, center):
    b = mbs.create_mobilized_body_quat(num, center, Z, Z, [1, 0, 0, 0], Z, Z,
                                       1.0, [0.1, 0.1, 0.1], Z, Z)
    s.Add(b)
    j = mbs.create_joint(1, center, Z, Z, Z, AZ, AZ, g, b)
    s.Add(j)
    return b, j


def z_angle(q):
    return 2.0 * math.atan2(q[3], q[0])


def test_gear_pair_ratio(mbs):
    s = mbs.DynamicsSystem(1e-2)
    g = mbs.create_ground(0)
    s.Add(g)
    _, ja = rotor(mbs, s, g, 1, [0, 0, 0])
    bb, jb = rotor(mbs, s, g, 2, [0.3, 0, 0])
    s.Add(mbs.create_gear_coupler(ja, "rotation", jb, "rotation", -2.0))
    f = mbs.DriveFunction.polynomial([0.0, 1.5, 0.4])
    s.Add(mbs.create_rotational_driver(body_i=g, body_j=bb, axis_i=AZ,
                                       axis_j=AZ, func=f))
    s.init()
    angle_a = 0.0
    for k in range(1, 401):  # C(4) = 12.4 rad: several revolutions
        s.solve()
        raw = z_angle(np.asarray(s.get_body_quaternions())[1])
        angle_a = raw + 2 * math.pi * round((angle_a - raw) / (2 * math.pi))
        assert angle_a == pytest.approx(-2.0 * f.eval(k * 1e-2)[0], abs=1e-9)


def test_rack_and_pinion(mbs):
    s = mbs.DynamicsSystem(1e-2)
    g = mbs.create_ground(0)
    s.Add(g)
    pinion, jp = rotor(mbs, s, g, 1, [0, 0, 0])
    rack = mbs.create_mobilized_body_quat(2, [0, -0.05, 0], Z, Z, [1, 0, 0, 0],
                                          Z, Z, 1.0, [0.1, 0.1, 0.1], Z, Z)
    s.Add(rack)
    jr = mbs.create_joint(2, [0, -0.05, 0], Z, Z, Z, AX, AX, g, rack)
    s.Add(jr)
    s.Add(mbs.create_gear_coupler(jr, "translation", jp, "rotation", 0.05))
    f = mbs.DriveFunction.sine(amplitude=2.0, omega=3.0)
    s.Add(mbs.create_rotational_driver(body_i=g, body_j=pinion, axis_i=AZ,
                                       axis_j=AZ, func=f))
    s.init()
    for k in range(1, 201):
        s.solve()
        c, cd, _ = f.eval(k * 1e-2)
        assert np.asarray(s.get_body_positions())[2][0] == pytest.approx(
            0.05 * c, abs=1e-10)
        assert np.asarray(s.get_body_velocities())[2][0] == pytest.approx(
            0.05 * cd, abs=1e-9)


def cam_with_y_follower(mbs, make_contact, y0):
    s = mbs.DynamicsSystem(1e-2)
    g = mbs.create_ground(0)
    s.Add(g)
    cam, _ = rotor(mbs, s, g, 1, [0, 0, 0])
    fol = mbs.create_mobilized_body_quat(2, [0, y0, 0], Z, Z, [1, 0, 0, 0], Z,
                                         Z, 1.0, [0.1, 0.1, 0.1], Z, Z)
    s.Add(fol)
    s.Add(mbs.create_joint(2, Z, Z, Z, Z, AY, AY, g, fol))
    s.Add(make_contact(cam, fol))
    f = mbs.DriveFunction.polynomial([0.0, 2.0])
    s.Add(mbs.create_rotational_driver(body_i=g, body_j=cam, axis_i=AZ,
                                       axis_j=AZ, func=f))
    s.init()
    return s, f


def test_point_follower_tracks_polar_profile(mbs):
    prof = mbs.CamProfile(0.5, [0.05, 0.0], [0.0, 0.02])
    # Cam plane frame from axis z: f = (0, 1, 0), g = (-1, 0, 0), so the
    # follower on +y sees the profile at phi = -theta (cam rotated by theta).
    s, f = cam_with_y_follower(
        mbs,
        lambda cam, fol: mbs.create_point_follower(cam, fol, Z, AZ, Z, prof),
        prof.eval(0.0)[0])
    for k in range(1, 301):
        s.solve()
        theta = f.eval(k * 1e-2)[0]
        assert np.asarray(s.get_body_positions())[2][1] == pytest.approx(
            prof.eval(-theta)[0], abs=1e-10)


def test_flat_follower_on_eccentric_circle(mbs):
    r0, e = 0.4, 0.06
    support = mbs.CamProfile(r0, [e], [0.0])  # circle centered at e along f
    s, f = cam_with_y_follower(
        mbs, lambda cam, fol: mbs.create_flat_follower(
            cam, fol, Z, AZ, Z, [0, -1, 0], support), r0 + e)
    for k in range(1, 301):
        s.solve()
        theta = f.eval(k * 1e-2)[0]
        assert np.asarray(s.get_body_positions())[2][1] == pytest.approx(
            r0 + e * math.cos(theta), abs=1e-10)


def test_cam_profile_fit_recovers_coefficients(mbs):
    phi = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    r = 0.5 + 0.05 * np.cos(phi) - 0.02 * np.sin(2 * phi)
    prof = mbs.CamProfile.fit(list(phi), list(r), 3)
    for x in (0.1, 1.3, 4.0):
        assert prof.eval(x)[0] == pytest.approx(
            0.5 + 0.05 * math.cos(x) - 0.02 * math.sin(2 * x), abs=1e-12)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
