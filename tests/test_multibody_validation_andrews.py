# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Validation of the dynamics against a published benchmark: Andrews' squeezer
mechanism (Schiehlen 1990), a stiff planar seven-link mechanism with a
spring and a driving torque.

* Against the Bari IVP test set (Hairer-Wanner formulation, 16-digit
  reference at t = 0.03 s): the solver models the mechanism as rigid bodies
  with revolute joints
  (solvcon/multibody/models/validation/andrews_squeezer.py), which is a
  different formulation; the seven angles must agree.
* Energy balance (the IFToMM benchmark criterion, below 1e-5 J): kinetic +
  spring energy - initial energy - work of the torque stays at zero.
* Against the IFToMM results files (beta0 = -0.062 rad): the crank tip F
  is as close to each published solution as the two are to each other.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from solvcon import multibody

DATA_DIR = Path(__file__).parent / "data" / "multibody"
REF = json.loads((DATA_DIR / "andrews" / "reference.json").read_text())


@pytest.fixture(scope="module")
def andrews():
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    from solvcon.multibody.models.validation import andrews_squeezer
    return andrews_squeezer


def _angles(mod, s, prev):
    q = np.array(s.get_body_quaternions())
    out = {}
    for num, name in enumerate(mod.ORDER, start=1):
        th = 2.0 * math.atan2(q[num][3], q[num][0])
        if prev is not None:
            th += 2.0 * math.pi * round((prev[name] - th) / (2.0 * math.pi))
        out[name] = th
    return out


@pytest.mark.slow
def test_bari_reference_angles(andrews):
    bari = REF["bari"]
    dt = 1e-4
    s, h = andrews.build(beta0=bari["beta0"], dt=dt, rtol=1e-12, atol=1e-14)
    ang = _angles(andrews, s, None)
    for _ in range(int(round(bari["t_end"] / dt))):
        s.solve()
        ang = _angles(andrews, s, ang)
    assert s.get_sim_time() == pytest.approx(bari["t_end"], abs=1e-12)
    # Hairer-Wanner angles from the absolute link angles.
    beta, delta = ang["OF"], ang["AG"]
    eps = ang["AH"] + math.pi / 2
    ours = [beta, ang["EF"] - beta, ang["EBD"],
            ang["GE"] - delta + math.pi / 2, delta, ang["HE"] - eps, eps]
    # The crank turns 15.8 rad; measured agreement is 1e-9 rad (relative
    # 5e-11), so 1e-7 rad leaves room for the rk45 tolerance.
    np.testing.assert_allclose(ours, bari["y_end"], rtol=0, atol=1e-7)


@pytest.mark.slow
def test_energy_balance_and_iftomm_results(andrews):
    rows = andrews.simulate(t_end=0.05, dt=1e-4)
    e0 = rows[0]["T"] + rows[0]["V"]
    drift = max(abs(r["T"] + r["V"] - e0 - r["W"]) for r in rows)
    assert drift < 1e-8  # measured 7e-10 J; the benchmark asks below 1e-5 J
    ref = REF["iftomm"]["rows"]
    by_t = {round(r["t"], 6): r for r in rows}
    for t_text, (x1, y1, t1, v1, w1) in ref["res1"].items():
        t = float(t_text)
        r = by_t[round(t, 6)]
        x2, y2 = ref["res2"][t_text][:2]
        if t == 0.0:
            # Same assembled start: the published values to their 10 digits.
            assert r["xF"] == pytest.approx(x1, abs=1e-12)
            assert r["yF"] == pytest.approx(y1, abs=1e-12)
            assert r["V"] == pytest.approx(v1, rel=1e-10)
            continue
        # The two published solutions differ from each other by up to
        # 4.3e-6 m (yF at 0.02 s). This model, which matches the Bari
        # reference to 1e-9 rad, must be as close to each of them as they
        # are to each other (measured: 3.5e-6 m at most).
        spread = max(max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                     for a, b
                     in zip(ref["res1"].values(), ref["res2"].values()))
        for x_ref, y_ref in ((x1, y1), (x2, y2)):
            assert abs(r["xF"] - x_ref) <= spread, t
            assert abs(r["yF"] - y_ref) <= spread, t
        assert r["T"] == pytest.approx(t1, rel=1e-3)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
