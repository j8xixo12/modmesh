# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Haug Section 12.5: vehicle, book comparison.

Book data: ``tests/data/multibody/haug_12_5.json``. Builder:
``solvcon/multibody/models/haug/vehicle.py``.

What the solver can check:

* DOF of models 1 and 2 (p. 471).
* The joint data of Tables 12.5.2-12.5.6 evaluated at the equilibrium pose
  of Table 12.5.9: every constraint must vanish to the table precision.
* The solver constraint rows equal an independent evaluation of Haug
  Eqs. 9.4.29 and 9.4.34 (XM), so a residual in the previous check comes
  from the book data and not from the joint formulation.
* The rack input of Fig. 12.5.6.

What it cannot check: the equilibrium analysis (Table 12.5.9, Figs.
12.5.4-12.5.5) and every dynamic result (Figs. 12.5.7-12.5.13). They depend
on the tire model of pp. 473-474 (vertical tire spring on the spindle
height, slip-angle lateral force, friction limit). The solver has no such
force element, and the book does not print the tire parameters.
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

from solvcon.multibody.models.haug import vehicle  # noqa: E402

DATA = json.loads((DATA_DIR / "haug_12_5.json").read_text())

# Table 12.5.9 prints 4 decimals: each value has a rounding error <= 5e-5.
# A body rotation from rounded e1, e2, e3 is <= 2 * sqrt(3) * 5e-5 = 1.7e-4
# rad. A joint row between two bodies with points up to 1.5 m (chassis) and 0.4
# m (other bodies) from the body origin therefore errs by at most
# 2 * sqrt(3) * 5e-5 + 1.7e-4 * (1.5 + 0.4) = 5e-4 (m for position rows,
# rad for axis rows, m^2 for the 0.5 (d.d - L^2) distance rows with L < 1 m).
POSE_TOL = 5e-4


@pytest.mark.parametrize("model,dof", [(1, 10), (2, 12)])
@pytest.mark.parametrize("front", ["book", "fitted"])
def test_dof(model, dof, front):
    """p. 471: model 2 has DOF = 56 - 44 = 12; model 1 has two bodies and
    two revolute joints less, so DOF = 10."""
    s, _, _ = vehicle.build(model=model, pose="equilibrium", gravity=0.0,
                            front=front)
    s.init()
    assert s.get_dof() == dof


#: Joints whose printed data do not fit the printed pose (book data, not
#: solver).
BOOK_INCONSISTENT = {
    "A-B": "Table 12.5.2 p.471 (E-F mirrored) vs Table 12.5.9 p.476: |AB| = "
           "0.3227 vs printed distance 0.203, and h_A . AB = 0.048 m (B not "
           "in the plane normal to the pivot axis)",
    "B-C": "Table 12.5.3 p.472 (F-G mirrored) vs Table 12.5.9 p.476: wheel "
           "strut point is 0.100 m (x) and 0.057 m (y) from the vertical "
           "chassis strut axis",
    "K": "Table 12.5.9 p.476: body 8 e3 = -0.0002 but chassis e3 = -0.0005; a "
         "revolute about x' cannot change e3, axis rows give 6.0e-4 rad > "
         "5e-4 rounding bound",
}
BOOK_INCONSISTENT["E-F"] = BOOK_INCONSISTENT["A-B"]
BOOK_INCONSISTENT["F-G"] = BOOK_INCONSISTENT["B-C"]


def _joint_names():
    out = []
    for name in [j[0] for j in vehicle.JOINTS] + ["driver D"]:
        if name in BOOK_INCONSISTENT:
            mark = pytest.mark.xfail(strict=True,
                                     reason=BOOK_INCONSISTENT[name])
            out.append(pytest.param(name, marks=mark))
        else:
            out.append(name)
    return out


@pytest.mark.parametrize("joint", _joint_names())
def test_book_equilibrium_satisfies_joint(joint):
    """Tables 12.5.2-12.5.6 at the Table 12.5.9 pose: residual <= rounding."""
    s, _, layout = vehicle.build(model=2, pose="equilibrium", gravity=0.0,
                                 front="book")
    res = vehicle.joint_residuals(s, layout)
    assert res[joint] <= POSE_TOL, "%s residual %.5f" % (joint, res[joint])


def _vec(a, b):
    return [a[k] - b[k] for k in range(3)]


def _dot(a, b):
    return sum(a[k] * b[k] for k in range(3))


@pytest.mark.parametrize("name", ["A-B", "E-F", "B-C", "F-G"])
def test_front_joint_rows_match_haug_equations(name):
    """XM: solver rows equal Eqs. 9.4.29 (revolute-spherical) and 9.4.34
    (strut).

    Both sides use the printed book data at the Table 12.5.9 pose. Agreement
    to round-off shows that the residual of these joints in
    test_book_equilibrium_satisfies_joint is a property of the book data.
    """
    entry = next(j for j in vehicle.JOINTS if j[0] == name)
    _, kind, bi, bj, pi, qi, pj, _, dist = entry
    s, _, layout = vehicle.build(model=2, pose="equilibrium", gravity=0.0,
                                 front="book")
    c = list(result_arrays(s.evaluate_constraints(0.0))["C"])
    row = 7
    for jname, n in layout:
        if jname == name:
            break
        row += n
    rows = c[row:row + 2]
    a_i = vehicle.rotation(vehicle._quat(vehicle.EQUILIBRIUM[bi][3:]))
    h = vehicle._mul(a_i, _vec(qi, pi))
    norm = math.sqrt(_dot(h, h))
    h = [x / norm for x in h]
    d = _vec(vehicle.global_point(bi, pi), vehicle.global_point(bj, pj))
    if kind == "rs":
        expected = [0.5 * (_dot(d, d) - dist * dist), _dot(h, d)]
        assert rows[0] == pytest.approx(expected[0], abs=1e-12)
        assert abs(rows[1]) == pytest.approx(abs(expected[1]), abs=1e-12)
    else:
        # Strut rows are d . f_i and d . g_i for a perpendicular pair f_i, g_i;
        # their norm is the distance of P_j from the axis line.
        perp = [d[k] - _dot(h, d) * h[k] for k in range(3)]
        assert math.hypot(*rows) == pytest.approx(math.sqrt(_dot(perp, perp)),
                                                  abs=1e-12)


def test_fitted_front_assembles_at_book_equilibrium():
    """IV: with the fitted front joints, assembly from Table 12.5.9 stays at
    Table 12.5.9 to the table precision (positions, POSE_TOL)."""
    s, _, _ = vehicle.build(model=2, pose="equilibrium", gravity=0.0,
                            front="fitted")
    s.init()
    pos = np.asarray(s.get_body_positions())
    for num in range(1, 9):
        book = vehicle.EQUILIBRIUM[num][:3]
        err = max(abs(pos[num][k] - book[k]) for k in range(3))
        assert err <= POSE_TOL, "body %d moved %.5f m" % (num, err)


FIG_12_5_6 = [
    ("lane1", 0.5, 0.0), ("lane1", 1.75, 0.00142), ("lane1", 2.5, 0.0),
    ("lane1", 3.25, -0.00140), ("lane1", 5.0, 0.0),
    ("lane2", 1.75, 0.00258), ("lane2", 3.3, -0.00253), ("lane2", 5.0, 0.0),
    ("circle", 1.75, 0.00258), ("circle", 3.0, 0.00258),
    ("circle", 5.5, 0.00258),
]


@pytest.mark.parametrize("kind,t,value", FIG_12_5_6)
def test_rack_input_matches_fig_12_5_6(kind, t, value):
    """Fig. 12.5.6 readings; read uncertainty from the JSON (5e-5 m)."""
    tol = DATA["figures"]["fig_12_5_6_rack_driver"]["read_uncertainty_m"]
    assert abs(vehicle.rack_position(kind, t) - value) <= tol
    c, _, _ = vehicle.steering_function(kind).eval(t)
    assert abs(c - value) <= tol


# Executable evidence of the book inconsistencies, from the JSON data alone.

def _rot(e):
    """Haug A(p) (Eq. 9.3.6) from printed e1, e2, e3; e0 from normalization."""
    e1, e2, e3 = e
    e0 = math.sqrt(1.0 - e1 * e1 - e2 * e2 - e3 * e3)
    return [
        [2 * (e0 * e0 + e1 * e1) - 1, 2 * (e1 * e2 - e0 * e3),
         2 * (e1 * e3 + e0 * e2)],
        [2 * (e1 * e2 + e0 * e3), 2 * (e0 * e0 + e2 * e2) - 1,
         2 * (e2 * e3 - e0 * e1)],
        [2 * (e1 * e3 - e0 * e2), 2 * (e2 * e3 + e0 * e1),
         2 * (e0 * e0 + e3 * e3) - 1],
    ]


def _json_point(body, local):
    q = DATA["equilibrium"][str(body)]
    a = _rot(q[3:])
    return [q[k] + sum(a[k][m] * local[m] for m in range(3)) for k in range(3)]


@pytest.mark.parametrize("name", ["A-B", "E-F"])
def test_evidence_revolute_spherical_distance(name):
    """Table 12.5.2 p.471 at Table 12.5.9 p.476: |AB| misses the printed 0.203
    by > 0.1 m."""
    j = DATA["joints"]["revolute_spherical"][name]
    d = _vec(_json_point(j["body_i"], j["P_i"]),
             _json_point(j["body_j"], j["P_j"]))
    assert abs(math.sqrt(_dot(d, d)) - j["distance"]) > 0.1


@pytest.mark.parametrize("name", ["B-C", "F-G"])
def test_evidence_strut_axis_offset(name):
    """Table 12.5.3 p.472 at Table 12.5.9 p.476: strut point is > 0.09 m off
    the axis."""
    j = DATA["joints"]["strut"][name]
    a = _rot(DATA["equilibrium"][str(j["body_i"])][3:])
    h_l = _vec(j["Q_i"], j["P_i"])
    h = [sum(a[k][m] * h_l[m] for m in range(3)) for k in range(3)]
    n = math.sqrt(_dot(h, h))
    h = [x / n for x in h]
    d = _vec(_json_point(j["body_i"], j["P_i"]),
             _json_point(j["body_j"], j["P_j"]))
    perp = [d[k] - _dot(h, d) * h[k] for k in range(3)]
    assert math.sqrt(_dot(perp, perp)) > 0.09


def test_evidence_stabilizer_bar_yaw():
    """Table 12.5.9 p.476: bar 8 and chassis e3 differ by 3e-4, three times the
    1e-4 rounding bound (2 x 5e-5); revolute K (axis x') cannot change e3."""
    eq = DATA["equilibrium"]
    assert abs(eq["8"][5] - eq["1"][5]) > 2.9e-4

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
