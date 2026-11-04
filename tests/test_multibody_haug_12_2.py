# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Book reproduction (BK): Haug Section 12.2, spatial slider-crank dynamics.

Reference data: ``tests/data/multibody/haug_12_2.json`` (Table 12.2.1, Figs.
12.2.1 to 12.2.4, pp. 455-458). Model:
``solvcon/multibody/models/haug/slider_crank_dynamics.py``.

Rules under test:

* If the crank turns at 2 pi rad/s, then the driver torque of the inverse
  dynamic analysis matches Fig. 12.2.1 for l = 0.3, 0.27, and 0.24 m.
* If the crank starts at the rate that gives the plotted initial slider
  velocity, then the slider position, velocity, and acceleration spike
  times match Figs. 12.2.2 to 12.2.4.
* The stated initial crank speeds (2 pi and 8 rad/s) cannot give the
  plotted initial slider velocities (book inconsistency, documented).

Figure comparisons. A book point (t_b, y_b) passes if the simulated curve
enters the read-uncertainty ellipse ((t - t_b) / u_t)^2 + ((y - y_b) / u_y)^2
<= 1. u_t and u_y are 1 % of the axis spans (JSON
``figure_read_uncertainty``): the tick fit residual is <= 0.12 % of span,
the line width is 0.3-0.5 % of span. The ellipse treats a flat extremum
(time poorly defined) and a steep flank (value poorly defined) alike.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from solvcon import multibody

DATA_DIR = Path(__file__).parent / "data" / "multibody"
if not multibody.HAS_MULTIBODY:
    pytest.skip("solvcon is built without the multibody engine",
                allow_module_level=True)

from solvcon.multibody.models.haug import (  # noqa: E402
    slider_crank_dynamics as sc)

DATA = json.loads((DATA_DIR / "haug_12_2.json").read_text())
INV = DATA["inverse_dynamics"]
DYN = DATA["dynamics"]

#: Read uncertainty: 1 % of the axis span (see module docstring).
READ_FRACTION = 0.01
#: Read uncertainty of the plotted initial slider velocity [m/s] (JSON).
U_XDOT0 = 0.007
#: Number of crank rates that sample the initial-velocity uncertainty band.
N_RATES = 5


def _spans(axes):
    """Return (u_t, u_y) from the axis ranges of a figure.

    :param axes: {"time": [t0, t1], "value": [y0, y1]}
    :type axes: dict
    :return: read uncertainties in time and value
    :rtype: tuple
    """
    return (READ_FRACTION * (axes["time"][1] - axes["time"][0]),
            READ_FRACTION * (axes["value"][1] - axes["value"][0]))


def _ellipse_distance(t, y, tb, yb, ut, uy):
    """Return the smallest normalized distance from a book point to a curve.

    :param t: simulated times
    :type t: numpy.ndarray
    :param y: simulated values
    :type y: numpy.ndarray
    :param tb: book time
    :type tb: float
    :param yb: book value
    :type yb: float
    :param ut: time read uncertainty
    :type ut: float
    :param uy: value read uncertainty
    :type uy: float
    :return: min of sqrt(((t - tb) / ut)^2 + ((y - yb) / uy)^2)
    :rtype: float
    """
    return float(np.min(np.hypot((t - tb) / ut, (y - yb) / uy)))


@pytest.fixture(scope="module")
def torque_histories():
    """Driver torque of the inverse dynamic analysis for the three rods.

    :return: {rod length: (t, torque)}
    :rtype: dict
    """
    out = {}
    for rod in (0.3, 0.27, 0.24):
        s, _ = sc.build(rod_length=rod, crank_rate=-2.0 * math.pi, dt=0.001)
        s.init()
        assert s.get_dof() == 0
        res = sc.simulate(s, 1.0, True)
        out[rod] = (np.array(res["t"]), np.array(res["torque"]))
    return out


@pytest.mark.parametrize("rod", [0.3, 0.27, 0.24])
def test_driving_torque_matches_fig_12_2_1(torque_histories, rod):
    """Fig. 12.2.1 (p. 456): driving torque extremes for each rod length."""
    ut, uy = _spans(INV["axes"])
    t, tau = torque_histories[rod]
    bad = []
    for p in INV["points"]:
        if p["curve"] != rod:
            continue
        d = _ellipse_distance(t, tau, p["t"], p["y"], ut, uy)
        if d > 1.0:
            bad.append((p["kind"], p["t"], p["y"], round(d, 2)))
    assert not bad, f"l = {rod}: points outside the read uncertainty: {bad}"


def test_torque_needs_gravity():
    """Without gravity the torque at the l = 0.3 minimum misses the book.

    This check fixes the force model (the book prints no force data): the
    book minimum -0.295 N m at t = 0.773 s is outside the read uncertainty
    of the model without gravity.
    """
    ut, uy = _spans(INV["axes"])
    s, _ = sc.build(rod_length=0.3, crank_rate=-2.0 * math.pi, gravity=0.0)
    s.init()
    res = sc.simulate(s, 1.0, True)
    p = [q for q in INV["points"] if q["curve"] == 0.3 and q["t"] > 0.7][0]
    d = _ellipse_distance(np.array(res["t"]), np.array(res["torque"]), p["t"],
                          p["y"], ut, uy)
    assert d > 1.0


def _crank_rate(xdot_read):
    """Crank rate that gives the plotted slider velocity at the read time.

    The read is at t = 0.005 s. One correction step scales the rate from
    the t = 0 value to the read time.

    :param xdot_read: plotted slider velocity at t = 0.005 s [m/s]
    :type xdot_read: float
    :return: crank rate about +x [rad/s]
    :rtype: float
    """
    t_read = DYN["initial_slider_velocity_read"]["t"]
    rate = sc.crank_rate_from_slider_velocity(xdot_read)
    s, _ = sc.build(crank_rate=rate, driven=False, dt=t_read)
    s.set_integrator("rk45")
    s.init()
    s.solve()
    return rate * xdot_read / float(np.asarray(s.get_body_velocities())[3][0])


@pytest.fixture(scope="module")
def free_runs():
    """Free motion for each run over the initial-velocity uncertainty band.

    :return: {curve: list of result dicts, one per crank rate}
    :rtype: dict
    """
    out = {}
    for curve in ("w=6", "w=2pi", "w=8"):
        v0 = DYN["initial_slider_velocity_read"][curve]
        runs = []
        for v in np.linspace(v0 - U_XDOT0, v0 + U_XDOT0, N_RATES):
            s, _ = sc.build(crank_rate=_crank_rate(float(v)), driven=False,
                            dt=0.001)
            s.set_integrator("rk45")
            s.init()
            assert s.get_dof() == 1
            res = sc.simulate(s, 1.0, False)
            runs.append({k: np.array(val) for k, val in res.items()})
        out[curve] = runs
    return out


def _check_points(runs, points, key, axes, curve):
    """Return the book points that no run of the band reaches.

    :param runs: simulated results over the initial-velocity band
    :type runs: list
    :param points: book points
    :type points: list
    :param key: result key ("x" or "xdot")
    :type key: str
    :param axes: figure axes
    :type axes: dict
    :param curve: run label
    :type curve: str
    :return: failing points with their smallest normalized distance
    :rtype: list
    """
    ut, uy = _spans(axes)
    bad = []
    for p in points:
        if p["curve"] != curve:
            continue
        d = min(_ellipse_distance(r["t"], r[key], p["t"], p["y"], ut, uy)
                for r in runs)
        if d > 1.0:
            bad.append((p["kind"], p["t"], p["y"], round(d, 2)))
    return bad


@pytest.mark.parametrize("curve", ["w=6", "w=2pi", "w=8"])
def test_slider_position_matches_fig_12_2_2(free_runs, curve):
    """Fig. 12.2.2 (p. 457): extremes of x and their times (the period)."""
    bad = _check_points(free_runs[curve], DYN["x_points"], "x",
                        DYN["axes"]["x"], curve)
    assert not bad, f"{curve}: points outside the read uncertainty: {bad}"


@pytest.mark.parametrize("curve", ["w=6", "w=2pi", "w=8"])
def test_slider_velocity_matches_fig_12_2_3(free_runs, curve):
    """Fig. 12.2.3 (p. 457): extremes of xdot before and after reversal."""
    bad = _check_points(free_runs[curve], DYN["xdot_points"], "xdot",
                        DYN["axes"]["xdot"], curve)
    assert not bad, f"{curve}: points outside the read uncertainty: {bad}"


@pytest.mark.parametrize("curve", ["w=6", "w=2pi", "w=8"])
def test_slider_acceleration_spike_times_match_fig_12_2_4(free_runs, curve):
    """Fig. 12.2.4 (p. 458): times of the positive acceleration spikes.

    The spike time passes if one run of the band has its spike within
    u_t = 0.01 s (1 % of the time axis) of the book time.
    """
    ut, _ = _spans(DYN["axes"]["xddot"])
    bad = []
    for p in DYN["xddot_spike_times"]["points"]:
        if p["curve"] != curve:
            continue
        errs = []
        for r in free_runs[curve]:
            w = (r["t"] > p["t"] - 0.05) & (r["t"] < p["t"] + 0.05)
            errs.append(abs(r["t"][w][np.argmax(r["xddot"][w])] - p["t"]))
        if min(errs) > ut:
            bad.append((p["t"], round(min(errs), 4)))
    assert not bad, f"{curve}: spike times outside the read uncertainty: {bad}"


def test_stated_crank_speeds_cannot_give_plotted_slider_velocity():
    """Book inconsistency (p. 458 vs Fig. 12.2.3), see JSON dynamics note.

    The velocity analysis gives xdot = (dx/dtheta) omega. Over a full crank
    turn |dx/dtheta| has a maximum. If the stated speed times this maximum
    is less than the plotted xdot(0) minus its read uncertainty, then no
    crank angle reproduces the plot with the stated speed.
    """
    s, _ = sc.build(crank_rate=1.0, dt=0.01)
    s.init()
    res = sc.simulate(s, 2.0 * math.pi, True)
    max_slope = float(np.max(np.abs(res["xdot"])))  # m/rad at 1 rad/s
    speeds = dict(zip(("w=6", "w=2pi", "w=8"),
                      DYN["stated_initial_crank_speed"]["values_rad_s"]))
    for curve in ("w=2pi", "w=8"):
        plotted = DYN["initial_slider_velocity_read"][curve]
        assert speeds[curve] * max_slope < plotted - U_XDOT0

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
