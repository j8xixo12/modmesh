# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Haug Section 12.6: governor mechanism, book comparison.

Book data and figure readings: ``tests/data/multibody/haug_12_6.json``.
Builder: ``solvcon/multibody/models/haug/governor.py``.

Checks:

* DOF of the model (p. 483).
* Steady-state spindle speed at a 45 degree arm slope (Section 12.6.2).
* The same speed from an analytic force balance (AN).
* Time histories of the collar height and the spindle speed under the load
  torque of Fig. 12.6.2 and the engine feedback of Eq. 12.6.1
  (Figs. 12.6.3 to 12.6.6).

The engine feedback torque depends on the collar position. The solver has no
such force element, so the builder holds the torque constant over each
coupling step and rebuilds the system (see the module docstring of the
builder). The coupling error at dt = 0.02 s is below 4e-6 in both outputs
(collar in 0.01 m units, omega in rad/s; checked against dt = 0.0025 s).
"""

import json
import math
from pathlib import Path

import pytest

from solvcon import multibody

DATA_DIR = Path(__file__).parent / "data" / "multibody"

if not multibody.HAS_MULTIBODY:
    pytest.skip("solvcon is built without the multibody engine",
                allow_module_level=True)

from solvcon.multibody.models.haug import governor  # noqa: E402

DATA = json.loads((DATA_DIR / "haug_12_6.json").read_text())
DYN = DATA["dynamic_analysis"]

COUPLING_DT = 0.02
T_END = 10.0


def test_dof_matches_book():
    """p. 483: DOF = 35 - 33 = 2."""
    s, _ = governor.build()
    s.init()
    assert s.get_dof() == DATA["model_count"]["dof"]


def test_steady_state_speed_matches_book():
    """Section 12.6.2: omega = 11.0174 rad/s keeps the collar stationary."""
    omega = governor.steady_state_omega()
    # The book prints 6 significant digits. One unit in the last digit
    # (1e-4 rad/s) covers rounding or truncation of the printed value.
    # g is not printed; g = 9.8 is inferred (g = 9.80665 gives 11.0212).
    assert abs(omega - DATA["steady_state"]["omega_rad_s"]) <= 1e-4


def _analytic_omega(gravity):
    """Force balance of the arm angle phi in the rotating frame.

    Virtual work on phi with ball mass m_b at arm a_b, coupler from the arm
    point at a to the collar (length L), collar weight m_c g, and isotropic
    ball inertia (no gyroscopic moment):
    omega^2 = (2 m_b g a_b + m_c g a (1 + a sin(phi) / q)) / (2 m_b a_b^2
    sin(phi)), q = h - a sin(phi), h = hinge-to-collar distance.
    """
    a_b, a = governor.ARM, governor.COUPLER_ARM
    h = governor.SPINDLE_Y - governor.COLLAR_Y
    sin_phi = (a * a + h * h - governor.COUPLER_LENGTH ** 2) / (2.0 * a * h)
    q = h - a * sin_phi
    m_b, m_c = governor.MASS["ball1"], governor.MASS["collar"]
    num = (2.0 * m_b * gravity * a_b
           + m_c * gravity * a * (1.0 + a * sin_phi / q))
    return math.sqrt(num / (2.0 * m_b * a_b * a_b * sin_phi))


def test_steady_state_speed_matches_analytic():
    """AN: the solver root equals the closed-form force balance."""
    omega = governor.steady_state_omega(tol=1e-11)
    # Bisection stops at a 1e-11 bracket; the initial acceleration solve is
    # exact to round-off. 1e-8 rad/s leaves margin for round-off only.
    assert abs(omega - _analytic_omega(governor.GRAVITY)) <= 1e-8


CASES = {
    "fig_12_6_3": ("collar_y", 12500.0,
                   {"k1000": 1000.0, "k2000": 2000.0, "k3000": 3000.0}),
    "fig_12_6_4": ("omega", 12500.0,
                   {"k1000": 1000.0, "k2000": 2000.0, "k3000": 3000.0}),
    "fig_12_6_5": ("collar_y", None,
                   {"C7500": 7500.0, "C12500": 12500.0, "C17500": 17500.0}),
    "fig_12_6_6": ("omega", None,
                   {"C7500": 7500.0, "C12500": 12500.0, "C17500": 17500.0}),
}

#: 1 percent of the plotted axis span (read uncertainty of a printed curve).
READ_TOL = {"fig_12_6_3": 0.004, "fig_12_6_4": 0.02, "fig_12_6_5": 0.005,
            "fig_12_6_6": 0.01}


def _ripple(fig, curve, t):
    """Ripple half amplitude of the book curve at time t (JSON readings)."""
    rip = DYN[fig].get("ripple_half_amplitude", {})
    for key, amp in rip.items():
        name, cond = key.split("_t_ge_")
        if name == curve and t >= float(cond):
            return amp
    return 0.0


#: Figs. 12.6.3-12.6.4 p.487: the k = 2000 and k = 3000 curves do not fit their
#: labels.
LABEL_REASON = (
    "Fig 12.6.3-12.6.4 p.487 labels: book k=2000 curve matches solver k=3000 "
    "(t=5: omega 10.154 vs 10.144, collar 4.844 vs 4.843); book k=3000 curve "
    "matches solver k=5000 (t=8: omega 9.536 vs 9.514); both lie below the "
    "closed-form final omega of their labeled k (10.247, 9.900) while still "
    "decreasing"
)


def _points():
    out = []
    for fig, (_, _, curves) in CASES.items():
        for curve in curves:
            for t in DYN[fig][curve]:
                # fig_12_6_3 k2000 t=8 lies within tolerance: the curves
                # converge to the common collar value 4.800 (torque balance),
                # so it is not marked.
                mislabeled = (fig in ("fig_12_6_3", "fig_12_6_4")
                              and curve in ("k2000", "k3000")
                              and (fig, curve, t)
                              != ("fig_12_6_3", "k2000", "8"))
                if mislabeled:
                    mark = pytest.mark.xfail(strict=True, reason=LABEL_REASON)
                    out.append(pytest.param(fig, curve, t, marks=mark))
                else:
                    out.append((fig, curve, t))
    return out


@pytest.fixture(scope="module")
def runs():
    """Simulate each (k, C) case once; key (k, C)."""
    cache = {}

    def get(k, gain):
        if (k, gain) not in cache:
            cache[(k, gain)] = governor.simulate(k=k, gain=gain, t_end=T_END,
                                                 dt=COUPLING_DT)
        return cache[(k, gain)]

    return get


@pytest.mark.parametrize("fig,curve,t", _points())
def test_time_history_matches_figure(runs, fig, curve, t):
    """Figs. 12.6.3-12.6.6: collar height and spindle speed after the load
    step.

    The reading is corrected by the anchor offset: the figure value of the
    initial plateau minus its exact value (collar 5.000, omega 11.0174).
    Tolerance: 1 percent of the axis span plus the book ripple half
    amplitude where the book curve ripples (the ripple phase of an
    independent integration need not match point by point).
    """
    key, gain, curves = CASES[fig]
    if gain is None:
        k, gain = 1000.0, curves[curve]
    else:
        k = curves[curve]
    run = runs(k, gain)
    i = int(round(float(t) / COUPLING_DT))
    assert abs(run["t"][i] - float(t)) < 1e-9
    scale = 100.0 if key == "collar_y" else 1.0
    value = scale * run[key][i]
    anchor = DYN[fig]["anchor"]
    book = DYN[fig][curve][t] - (anchor["reading"] - anchor["exact"])
    tol = READ_TOL[fig] + _ripple(fig, curve, float(t))
    assert abs(value - book) <= tol, (
        "%s %s t=%s: solver %.4f, book %.4f, tol %.4f" % (fig, curve, t, value,
                                                          book, tol)
    )


# Executable evidence of the figure inconsistency, from the JSON data alone.

def _final_omega(k, gain):
    """Closed-form final steady state after the load step (Eq. 12.6.1 and
    statics).

    Torque balance: gain * dl = 25 N m. The spring pulls the collar up with
    k * dl, so the coupler load is (m_c g - k dl); the arm force balance of
    _analytic_omega then gives omega at the collar drop dl.
    """
    j = DATA["joints"]
    a_b = -j["REV2"]["P_j"][0]
    a = -j["DIST1"]["P_i"][0]
    length = j["DIST1"]["distance"]
    m_b = DATA["bodies"]["3_ball1"][0]
    m_c = DATA["bodies"]["5_collar"][0]
    g = DATA["gravity"]["value"]
    st = DATA["steady_state"]
    dl = 25.0 / gain
    h = st["2_spindle"][1] - st["5_collar"][1] + dl
    sin_phi = (a * a + h * h - length ** 2) / (2.0 * a * h)
    q = h - a * sin_phi
    num = (2.0 * m_b * g * a_b
           + (m_c * g - k * dl) * a * (1.0 + a * sin_phi / q))
    return math.sqrt(num / (2.0 * m_b * a_b * a_b * sin_phi))


def _corrected(fig, curve, t):
    anchor = DYN[fig]["anchor"]
    return DYN[fig][curve][t] - (anchor["reading"] - anchor["exact"])


def test_evidence_k1000_final_state_matches_closed_form():
    """Figs. 12.6.3-12.6.4 p.487, k = 1000: late readings equal the closed-form
    final state (collar 5.000 - 100 * 25 / 12500 = 4.800, omega 10.5818) within
    the read tolerance plus ripple of the figure."""
    assert abs(_corrected("fig_12_6_3", "k1000", "9")
               - (5.0 - 100.0 * 25.0 / 12500.0)) <= (
        READ_TOL["fig_12_6_3"] + 0.003)
    assert abs(_corrected("fig_12_6_4", "k1000", "9")
               - _final_omega(1000.0, 12500.0)) <= (
        READ_TOL["fig_12_6_4"])


def test_evidence_k2000_k3000_labels():
    """Figs. 12.6.3-12.6.4 p.487: the book k = 2000 and k = 3000 omega curves
    decrease monotonically and are already below the closed-form final omega
    of their labeled k, so they cannot settle there. They are above the final
    omega for k = 3000 and k = 5000, which fits a label error."""
    for curve, labeled, fitted, t in (("k2000", 2000.0, 3000.0, "9"),
                                      ("k3000", 3000.0, 5000.0, "8")):
        values = list(DYN["fig_12_6_4"][curve].values())
        assert all(b < a for a, b in zip(values, values[1:]))
        omega = _corrected("fig_12_6_4", curve, t)
        assert omega < _final_omega(labeled, 12500.0) - 0.25
        assert omega > _final_omega(fitted, 12500.0)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
