# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Book reproduction (BK): Haug Section 12.3, spatial four-bar dynamics.

Reference data: ``tests/data/multibody/haug_12_3.json`` (Tables 12.3.1 to
12.3.3, Figs. 12.3.1 and 12.3.2, pp. 458-462). Model:
``solvcon/multibody/models/haug/four_bar_dynamics.py``.

Rules under test:

* The static equilibrium under gravity is a stationary minimum of the
  potential energy, and it matches Table 12.3.3 (model 2).
* The configurations of Tables 12.3.2 and 12.3.3 are closed
  configurations of the model at the tabulated crank angle.
* If link 1 turns at pi rad/s, then the driver torque and the x reaction
  at A match Fig. 12.3.1 for both models.
* If a torque of 10 N m acts on link 1 from rest, then the rocker centroid
  height and velocity match Fig. 12.3.2 for both models.

Figure comparisons. A book point (t_b, y_b) passes if the simulated curve
enters the read-uncertainty ellipse ((t - t_b) / u_t)^2 + ((y - y_b) / u_y)^2
<= 1. u_t and u_y are 1 % of the axis spans (JSON
``figure_read_uncertainty``): tick fit residual <= 0.15 % of span, line
width 0.3-0.5 % of span.

Table comparisons use half a unit of the last printed digit (5e-4).
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.tools.arrays import result_arrays

DATA_DIR = Path(__file__).parent / "data" / "multibody"
if not multibody.HAS_MULTIBODY:
    pytest.skip("solvcon is built without the multibody engine",
                allow_module_level=True)

from solvcon.multibody.models.haug import four_bar_dynamics as fb  # noqa: E402

DATA = json.loads((DATA_DIR / "haug_12_3.json").read_text())
EQ = DATA["equilibrium"]

#: Read uncertainty: 1 % of the axis span (see module docstring).
READ_FRACTION = 0.01
#: Half a unit of the last printed digit of Tables 12.3.2 and 12.3.3.
TABLE_TOL = 5e-4
#: One unit of the last printed digit of the table heights [m]. A
#: minimization stops inside its energy tolerance; the book claims this
#: precision for the equilibrium height.
HEIGHT_TOL = 1e-3


def _theta(e):
    """Crank angle about +x from Euler parameters (e0, e1, 0, 0).

    :param e: Euler parameters
    :type e: list
    :return: angle in (-pi, pi] [rad]
    :rtype: float
    """
    a = 2.0 * math.atan2(e[1], e[0])
    return math.atan2(math.sin(a), math.cos(a))


def _same_quat(a, b):
    """Return the largest difference of two Euler parameter sets (sign-free).

    :param a: Euler parameters
    :type a: list
    :param b: Euler parameters
    :type b: list
    :return: max abs difference after the sign choice q ~ -q
    :rtype: float
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(min(np.max(np.abs(a - b)), np.max(np.abs(a + b))))


def _potential(model, positions):
    """Potential energy divided by g: sum of m z over the moving bodies.

    :param model: 1 or 2
    :type model: int
    :param positions: body positions in solver order
    :type positions: list
    :return: sum of m z [kg m]
    :rtype: float
    """
    names = ["link1", "link2", "link3"] if model == 1 else ["link1", "link3"]
    return sum(fb.INERTIA[n][0] * positions[i][2] for i, n
               in enumerate(names, start=1))


def _sweep(model, thetas):
    """Closed configurations at the given crank angles.

    :param model: 1 or 2
    :type model: int
    :param thetas: crank angles [rad], sorted by distance from 0
    :type thetas: list
    :return: sweep result dict of the solver
    :rtype: dict
    """
    s, parts = fb.build(model, crank="parameter")
    s.init()
    res = result_arrays(s.sweep(parts["driver"], list(thetas)))
    assert res["completed"], res["message"]
    return res


def _approach(theta, step=math.radians(2.0)):
    """Crank angles from 0 to theta in small steps (assembly branch).

    :param theta: target angle [rad]
    :type theta: float
    :param step: step size [rad]
    :type step: float
    :return: list of angles ending at theta
    :rtype: list
    """
    n = max(2, int(abs(theta) / step) + 1)
    return [theta * (k + 1) / n for k in range(n)]


@pytest.fixture(scope="module")
def equilibria():
    """Equilibrium from a start at the tabulated crank angle of each model.

    :return: {model: (result, positions, quaternions)}
    :rtype: dict
    """
    out = {}
    for model, key, body in ((1, "model_1_table_12_3_2", "link1"),
                             (2, "model_2_table_12_3_3", "link1")):
        theta_book = _theta(EQ[key][body][3:])
        s, _ = fb.build(model, state=fb.assembled_state(model, theta_book))
        s.init()
        res = s.find_equilibrium(1e-10, 100)
        out[model] = (res,
                      [list(p) for p in np.asarray(s.get_body_positions())],
                      [list(q) for q in np.asarray(s.get_body_quaternions())])
    return out


@pytest.mark.parametrize("model", [1, 2])
def test_equilibrium_is_a_potential_minimum(equilibria, model):
    """find_equilibrium converges to a stationary minimum (IV + sweep).

    Independent check: a quasi-static sweep of the crank angle in 0.1 deg
    steps around the equilibrium. The potential energy is smallest within
    one step of the Newton result.
    """
    res, pos, quat = equilibria[model]
    assert res["converged"], res["message"]
    theta_eq = _theta(quat[1])
    grid = [theta_eq + math.radians(0.1 * k) for k in range(-30, 31)]
    sweep = _sweep(model, _approach(grid[0]) + grid)
    pe = [_potential(model, p) for p in sweep["positions"][-len(grid):]]
    k_min = int(np.argmin(pe))
    assert abs(k_min - 30) <= 1
    assert pe[0] > pe[30] and pe[-1] > pe[30]


@pytest.mark.parametrize("model, key",
                         [(1, "model_1_table_12_3_2"),
                          (2, "model_2_table_12_3_3")])
def test_table_configuration_is_closed_at_table_crank_angle(model, key):
    """Tables 12.3.2 and 12.3.3: positions and rocker Euler parameters.

    At the tabulated crank angle the model reproduces every tabulated
    position and the rocker orientation to half a unit of the last digit.
    The link 2 orientation of Table 12.3.2 is not compared: it violates
    the universal joint of Table 10.3.2 (z'_2 . u_1 = -0.048, JSON note).
    """
    table = EQ[key]
    theta_book = _theta(table["link1"][3:])
    sweep = _sweep(model, _approach(theta_book))
    pos = sweep["positions"][-1]
    quat = sweep["quaternions"][-1]
    rows = ([("link1", 1), ("link2", 2), ("link3", 3)] if model == 1
            else [("link1", 1), ("rocker", 2)])
    for name, i in rows:
        assert (np.max(np.abs(np.array(pos[i]) - table[name][:3]))
                <= TABLE_TOL), name
    rocker = "link3" if model == 1 else "rocker"
    assert _same_quat(quat[rows[-1][1]], table[rocker][3:]) <= TABLE_TOL


def test_model_2_equilibrium_matches_table_12_3_3(equilibria):
    """Table 12.3.3 (p. 460): rocker height and crank angle of model 2.

    The rocker height is stationary at the equilibrium, so it is compared
    to one unit of the last printed digit. The tabulated crank angle passes
    if it lies in the band where the height is within that unit of the
    minimum (the minimization of the book stops on the energy).
    """
    _, pos, _ = equilibria[2]
    table = EQ["model_2_table_12_3_3"]
    assert abs(pos[2][2] - table["rocker"][2]) <= HEIGHT_TOL
    sweep = _sweep(2, _approach(_theta(table["link1"][3:])))
    assert sweep["positions"][-1][2][2] - pos[2][2] <= HEIGHT_TOL


def test_table_12_3_2_is_not_stationary_for_the_documented_model():
    """Book inconsistency (Table 12.3.2), see JSON consistency note.

    At the tabulated crank angle the slope of the potential energy is
    d(sum m z)/d(theta) = 0.81 kg m/rad, that is 8 N m of gravity torque on
    link 1. A stationary point has zero slope. The check uses only the
    kinematic sweep and the masses of Table 12.3.1.
    """
    theta_book = _theta(EQ["model_1_table_12_3_2"]["link1"][3:])
    h = 1e-3
    sweep = _sweep(1, _approach(theta_book - h) + [theta_book + h])
    pe_hi = _potential(1, sweep["positions"][-1])
    pe_lo = _potential(1, sweep["positions"][-2])
    slope = (pe_hi - pe_lo) / (2 * h)
    assert abs(slope) > 0.5


#: Model 1 figure comparisons disagree with the book because the book's
#: model 1 data are inconsistent (see JSON consistency notes).
MODEL_1_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="Book model 1 inconsistent: Table 10.3.6 (p. 406) link 2 violates "
           "the universal joint of Table 10.3.2 (p. 404), z'2.u1 = -0.68; "
           "Table 12.3.2 (p. 459) pose is not an equilibrium (PE slope 0.81 "
           "kg m/rad, documented-model minimum at -135.9 deg); model 2 "
           "through the same solver paths matches Figs. 12.3.1-12.3.2. "
           "Strict: a pass means re-check.")
MODELS_FIG = [pytest.param(1, marks=MODEL_1_XFAIL), 2]


def _spans(axes):
    """Return (u_t, u_y) from the axis ranges of a figure.

    :param axes: {"time": [t0, t1], "value": [y0, y1]}
    :type axes: dict
    :return: read uncertainties in time and value
    :rtype: tuple
    """
    return (READ_FRACTION * (axes["time"][1] - axes["time"][0]),
            READ_FRACTION * (axes["value"][1] - axes["value"][0]))


def _bad_points(t, y, fig, model):
    """Return the book points of one curve outside the read uncertainty.

    :param t: simulated times
    :type t: numpy.ndarray
    :param y: simulated values
    :type y: numpy.ndarray
    :param fig: figure entry of the JSON data
    :type fig: dict
    :param model: curve label (model number)
    :type model: int
    :return: list of (kind, t, y, normalized distance)
    :rtype: list
    """
    ut, uy = _spans(fig["axes"])
    bad = []
    for p in fig["points"]:
        if p["curve"] != model:
            continue
        d = float(np.min(np.hypot((t - p["t"]) / ut, (y - p["y"]) / uy)))
        if d > 1.0:
            bad.append((p["kind"], p["t"], p["y"], round(d, 2)))
    return bad


@pytest.fixture(scope="module")
def inverse_runs():
    """Inverse dynamics, theta = pi t from Table 10.3.6 (Sec. 12.3.3).

    :return: {model: dict of arrays t, torque, fx_a}
    :rtype: dict
    """
    out = {}
    for model in (1, 2):
        s, parts = fb.build(model, crank=fb.CRANK_SPEED,
                            state=fb.assembled_state(model))
        s.init()
        assert s.get_dof() == 0
        out[model] = {k: np.array(v) for k, v
                      in fb.simulate(s, parts, 2.5).items()}
    return out


@pytest.fixture(scope="module")
def free_runs():
    """Dynamics, 10 N m on link 1 from rest (Sec. 12.3.4).

    :return: {model: dict of arrays t, z, zdot}
    :rtype: dict
    """
    out = {}
    for model in (1, 2):
        s, parts = fb.build(model, torque=fb.DRIVE_TORQUE,
                            state=fb.assembled_state(model))
        s.set_integrator("rk45")
        s.init()
        assert s.get_dof() == 1
        out[model] = {k: np.array(v) for k, v
                      in fb.simulate(s, parts, 2.0).items()}
    return out


@pytest.mark.parametrize("model", MODELS_FIG)
def test_driving_torque_matches_fig_12_3_1a(inverse_runs, model):
    """Fig. 12.3.1(a) (p. 460): driving torque on link 1."""
    r = inverse_runs[model]
    bad = _bad_points(r["t"], r["torque"],
                      DATA["inverse_dynamics"]["torque_figure"], model)
    assert not bad, (f"model {model}: points outside the read uncertainty: "
                     f"{bad}")


@pytest.mark.parametrize("model", MODELS_FIG)
def test_reaction_x_at_a_matches_fig_12_3_1b(inverse_runs, model):
    """Fig. 12.3.1(b) (p. 461): x reaction of revolute A on the ground."""
    r = inverse_runs[model]
    bad = _bad_points(r["t"], r["fx_a"],
                      DATA["inverse_dynamics"]["reaction_figure"], model)
    assert not bad, (f"model {model}: points outside the read uncertainty: "
                     f"{bad}")


@pytest.mark.parametrize("model", MODELS_FIG)
def test_rocker_height_matches_fig_12_3_2a(free_runs, model):
    """Fig. 12.3.2(a) (p. 461): rocker centroid height."""
    r = free_runs[model]
    bad = _bad_points(r["t"], r["z"], DATA["dynamics"]["z_figure"], model)
    assert not bad, (f"model {model}: points outside the read uncertainty: "
                     f"{bad}")


@pytest.mark.parametrize("model", MODELS_FIG)
def test_rocker_height_extremum_times_match_fig_12_3_2a(free_runs, model):
    """Fig. 12.3.2(a): times of the lowest and highest rocker positions.

    Tolerance u_t = 0.02 s (1 % of the time axis).
    """
    fig = DATA["dynamics"]["z_figure"]
    ut, _ = _spans(fig["axes"])
    r = free_runs[model]
    bad = []
    for p in fig["extremum_times"]:
        if p["curve"] != model:
            continue
        w = (r["t"] > p["t"] - 0.3) & (r["t"] < p["t"] + 0.3)
        k = (np.argmin(r["z"][w]) if p["kind"] == "min"
             else np.argmax(r["z"][w]))
        if abs(r["t"][w][k] - p["t"]) > ut:
            bad.append((p["kind"], p["t"], round(float(r["t"][w][k]), 3)))
    assert not bad, (f"model {model}: extremum times outside the read "
                     f"uncertainty: {bad}")


@pytest.mark.parametrize("model", MODELS_FIG)
def test_rocker_velocity_matches_fig_12_3_2b(free_runs, model):
    """Fig. 12.3.2(b) (p. 462): rocker centroid z velocity."""
    r = free_runs[model]
    bad = _bad_points(r["t"], r["zdot"], DATA["dynamics"]["zdot_figure"],
                      model)
    assert not bad, (f"model {model}: points outside the read uncertainty: "
                     f"{bad}")


def _rotation(e):
    """Haug A(p), body-to-global (Eq. 6.1.13).

    :param e: Euler parameters [e0, e1, e2, e3]
    :type e: list
    :return: 3x3 matrix
    :rtype: numpy.ndarray
    """
    e0, e1, e2, e3 = e
    return 2.0 * np.array([
        [e0 * e0 + e1 * e1 - 0.5, e1 * e2 - e0 * e3, e1 * e3 + e0 * e2],
        [e1 * e2 + e0 * e3, e0 * e0 + e2 * e2 - 0.5, e2 * e3 - e0 * e1],
        [e1 * e3 - e0 * e2, e2 * e3 + e0 * e1, e0 * e0 + e3 * e3 - 0.5]])


@pytest.mark.parametrize("model", [1, 2])
def test_energy_balance_of_free_run(model):
    """IV check for Sec. 12.3.4: T + V - W stays constant.

    T uses the Table 12.3.1 inertias (zero Iz'z' included), V = sum m g z,
    and W = n (theta - theta_0) is the work of the 10 N m torque. The rk45
    error control (rtol 1e-6 class) keeps the drift far below the 33 J of
    work done in 2 s; 1e-3 J is the stated bound.
    """
    s, _ = fb.build(model, torque=fb.DRIVE_TORQUE,
                    state=fb.assembled_state(model))
    s.set_integrator("rk45")
    s.init()
    names = ["link1", "link2", "link3"] if model == 1 else ["link1", "link3"]

    def energy(theta_unwrapped):
        pos = np.asarray(s.get_body_positions())
        vel = np.asarray(s.get_body_velocities())
        omega = np.asarray(s.get_body_angular_velocities())
        quat = np.asarray(s.get_body_quaternions())
        total = 0.0
        for i, n in enumerate(names, start=1):
            mass, inertia = fb.INERTIA[n]
            wb = _rotation(quat[i]).T @ np.array(omega[i])
            total += 0.5 * mass * float(np.dot(vel[i], vel[i]))
            total += (0.5 * float(np.dot(inertia, wb * wb))
                      + mass * fb.GRAVITY * pos[i][2])
        return total - fb.DRIVE_TORQUE * theta_unwrapped

    theta = _theta(np.asarray(s.get_body_quaternions())[1])
    e_start = energy(theta)
    worst = 0.0
    while s.get_sim_time() < 2.0 - 1e-9:
        s.solve()
        new = _theta(np.asarray(s.get_body_quaternions())[1])
        theta += math.atan2(math.sin(new - theta), math.cos(new - theta))
        worst = max(worst, abs(energy(theta) - e_start))
    assert worst < 1e-3

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
