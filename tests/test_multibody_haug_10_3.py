# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Haug Section 10.3: kinematic analysis of a spatial four-bar (BK, AN, XM).

Book data: ``tests/data/multibody/haug_10_3.json`` (Tables 10.3.1 to 10.3.6,
Figs. 10.3.3 to 10.3.5, pp. 402-408). Model:
``solvcon/multibody/models/haug/spatial_four_bar.py``.

Rules under test:

* If the bodies start at the estimates of Table 10.3.5, then ``init()``
  assembles model 1 with zero DOF and a constraint residual below 1e-10.
* Every body origin of the assembled configuration equals Table 10.3.6
  within the precision of the printed table.
* Every digitized point of Figs. 10.3.3 to 10.3.5 lies within the read
  uncertainty of the simulated curve.
* If the coupler is replaced by a spherical-spherical constraint (model 2),
  then the rocker motion is identical to model 1.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.tools.arrays import result_arrays

DATA_DIR = Path(__file__).parent / "data" / "multibody"
DATA = json.loads((DATA_DIR / "haug_10_3.json").read_text())
T_END = 2.5
DT = 0.002


@pytest.fixture(scope="module")
def fb():
    """Import the four-bar builder, or skip if the solver is not built.

    :return: builder module ``spatial_four_bar``
    :rtype: module
    """
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    from solvcon.multibody.models.haug import spatial_four_bar
    return spatial_four_bar


@pytest.fixture(scope="module")
def model1(fb):
    """Assemble model 1 and run the kinematic analysis to 2.5 s.

    :return: (assembled positions, assembled quaternions, dof, max |C|,
        history)
    :rtype: tuple
    """
    s, _ = fb.build_model_1(dt=DT)
    s.init()
    pos = [np.array(p) for p in np.asarray(s.get_body_positions())]
    quat = [np.array(q) for q in np.asarray(s.get_body_quaternions())]
    c0 = float(np.abs(result_arrays(s.evaluate_constraints(0.0))["C"]).max())
    hist = {k: np.array(v) for k, v in fb.simulate(s, T_END).items()}
    return pos, quat, s.get_dof(), c0, hist


def rot(e):
    """Return Haug's A(p) for unnormalized e (Eq. 6.1.13).

    :param e: Euler parameters [e0, e1, e2, e3]
    :type e: list
    :return: 3x3 matrix
    :rtype: numpy.ndarray
    """
    e0, e1, e2, e3 = e
    ev = np.array([e1, e2, e3])
    sk = np.array([[0, -e3, e2], [e3, 0, -e1], [-e2, e1, 0]])
    return (2 * e0 * e0 - 1) * np.eye(3) + 2 * (np.outer(ev, ev) + e0 * sk)


def canonical(e):
    """Return the Euler parameters with e0 >= 0 (p and -p are one rotation).

    :param e: Euler parameters
    :type e: numpy.ndarray
    :return: Euler parameters with e0 >= 0
    :rtype: numpy.ndarray
    """
    e = np.asarray(e, dtype=float)
    return e if e[0] >= 0 else -e


def book_table():
    """Return Table 10.3.6 as solver-numbered (position, e) pairs.

    :return: dict body number -> (r, e)
    :rtype: dict
    """
    t = DATA["results"]["table_10_3_6_assembled_model_1"]
    out = {}
    for key, num in (("link_1", 1), ("link_2", 2), ("link_3", 3)):
        row = t[key]
        out[num] = (np.array(row[:3]), np.array(row[3:]))
    return out


def table_violation():
    """Largest violation of the book's own constraints by Table 10.3.6.

    The joint points of Tables 10.3.1 to 10.3.3 are evaluated at the printed
    configuration. Revolute A and D give point mismatches, spherical C a
    point mismatch, and the rocker's Euler parameters a length error: A(p)
    with e.e != 1 scales the 7.4 m distance between C and D.

    :return: largest violation [m]
    :rtype: float
    """
    b = book_table()

    def pt(num, local):
        r, e = b[num]
        return r + rot(e) @ np.array(local)

    viol = [
        np.linalg.norm(pt(1, [0, 0, 0])),                                   # A
        np.linalg.norm(pt(3, [0, 3.7, 0]) - np.array([-4.0, -8.5, 0.0])),   # D
        np.linalg.norm(pt(2, [0, -6.1, 0]) - pt(3, [0, -3.7, 0])),          # C
        np.linalg.norm(pt(1, [0, 0, 2]) - pt(2, [0, 6.1, 0])),              # B
        # e.e of link 3
        abs(np.dot(b[3][1], b[3][1]) - 1.0) * 7.4,
    ]
    return max(viol)


# The tolerances for Table 10.3.6 follow from the printed table itself.
# TOL_POS is about twice the largest constraint violation of the printed
# values (5.2e-3 m, checked in test_book_table_precision). Rotation errors are
# a position error over the shortest lever: rocker half-length 3.7 m for the
# rocker Euler parameters (Euler parameters carry half the angle), coupler
# half-length 6.1 m for the coupler axis.
TOL_POS = 1.0e-2
TOL_E = TOL_POS / 3.7 / 2.0
TOL_AXIS = TOL_POS / 6.1


class TestAssembly:

    def test_zero_dof_and_residual(self, model1):
        _, _, dof, c0, _ = model1
        assert dof == 0
        # Newton tolerance of init() is 1e-10.
        assert c0 < 1e-10

    def test_book_table_precision(self):
        """Table 10.3.6 is consistent with its own joints only to 5e-3."""
        v = table_violation()
        assert 4.0e-3 < v < 6.0e-3

    @pytest.mark.parametrize("num", [1, 2, 3])
    def test_table_10_3_6_positions(self, model1, num):
        pos = model1[0]
        r_book, _ = book_table()[num]
        np.testing.assert_allclose(pos[num], r_book, rtol=0, atol=TOL_POS)

    @pytest.mark.parametrize("num", [1, 3])
    def test_table_10_3_6_euler_parameters(self, model1, num):
        quat = model1[1]
        _, e_book = book_table()[num]
        np.testing.assert_allclose(canonical(quat[num]), canonical(e_book),
                                   rtol=0, atol=TOL_E)

    def test_coupler_axis(self, model1):
        """The coupler y' axis (line BC) matches the book.

        The book's coupler Euler parameters are not compared: they violate
        the universal joint condition h1 . h2 = 0 of Table 10.3.2 (the
        printed value gives -0.68). The printed and the computed coupler
        frames differ only by a spin about the coupler axis.
        """
        quat = model1[1]
        _, e_book = book_table()[2]
        a_ours = rot(quat[2])
        a_book = rot(e_book / np.linalg.norm(e_book))
        np.testing.assert_allclose(a_ours[:, 1], a_book[:, 1], rtol=0,
                                   atol=TOL_AXIS)
        h1 = rot(quat[1]) @ np.array([0.750, -0.662, 0.0])
        assert abs(h1 @ a_ours[:, 2]) < 1e-9
        assert abs(h1 @ a_book[:, 2]) > 0.5
        rel = a_book.T @ a_ours
        w, v = np.linalg.eig(rel)
        axis = np.real(v[:, np.argmin(np.abs(w - 1.0))])
        assert abs(abs(axis[1]) - 1.0) < 1e-2


def pixel_distance(points, t, y, upp):
    """Distance in pixels from each book point to a simulated curve.

    The simulated curve is linear between samples. The distance uses the
    pixel scale of the figure in both axes.

    :param points: book points [[t, y], ...]
    :type points: list
    :param t: simulated times
    :type t: numpy.ndarray
    :param y: simulated values
    :type y: numpy.ndarray
    :param upp: data units per pixel [t, y]
    :type upp: list
    :return: distance of each point [px]
    :rtype: numpy.ndarray
    """
    tt = np.linspace(t[0], t[-1], 20 * len(t))
    yy = np.interp(tt, t, y)
    return np.array([np.min(np.hypot((tt - a) / upp[0], (yy - b) / upp[1]))
                     for a, b in points])


class TestFigures:

    @pytest.mark.parametrize("fig,key", [
        ("fig_10_3_3", "z_B"),
        ("fig_10_3_4", "z_rocker"),
        ("fig_10_3_5", "zdot_rocker"),
    ])
    def test_curve_within_read_uncertainty(self, model1, fig, key):
        hist = model1[4]
        f = DATA["results"][fig]
        # Tolerance: the read uncertainty of the figure in pixels
        # (``read_uncertainty_basis`` in the JSON file).
        d = pixel_distance(f["points_t_y"], hist["t"], hist[key],
                           f["units_per_px"])
        assert d.max() <= f["read_uncertainty_px"], (fig, d.max())

    def test_point_b_analytic(self, model1):
        """z_B = 2 cos(pi t): revolute A about x and theta = pi t (AN)."""
        hist = model1[4]
        # Position Newton tolerance 1e-10; 1e-9 allows the 2 m lever.
        np.testing.assert_allclose(hist["z_B"],
                                   2.0 * np.cos(math.pi * hist["t"]), rtol=0,
                                   atol=1e-9)

    def test_rocker_extremes(self, model1):
        """Rocker z passes through 3.7 m (vertical rocker, Fig. 10.3.4)."""
        hist = model1[4]
        f = DATA["results"]["fig_10_3_4"]
        tol = f["read_uncertainty_px"] * f["units_per_px"][1]
        for key in ("max_1", "max_2"):
            tb, zb = f["extremes_read"][key]
            m = np.abs(hist["t"] - tb) < 0.1
            assert hist["z_rocker"][m].max() == pytest.approx(zb, abs=tol)
            # The exact maximum is the rocker half-length 3.7 m (Table 10.3.1).
            assert hist["z_rocker"][m].max() == pytest.approx(3.7, abs=1e-5)


class TestModel2:

    def test_model_2_equals_model_1(self, fb, model1):
        """Model 2 gives identical results (p. 408) (XM)."""
        s, _ = fb.build_model_2(dt=DT)
        s.init()
        assert s.get_dof() == 0
        h2 = fb.simulate(s, T_END, rocker=2)
        h1 = model1[4]
        # Both models solve the same rocker equations to the Newton
        # tolerance 1e-10; 1e-8 allows the velocity solve.
        np.testing.assert_allclose(h2["z_rocker"], h1["z_rocker"], rtol=0,
                                   atol=1e-8)
        np.testing.assert_allclose(h2["zdot_rocker"], h1["zdot_rocker"],
                                   rtol=0, atol=1e-8)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
