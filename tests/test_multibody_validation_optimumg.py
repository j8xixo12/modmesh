# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Validation against a real car: the OptimumG generic Formula Student case
study (OptimumKinematics results), front and rear double wishbones.

The hardpoints (tests/data/multibody/optimumg_fsae/front.json, rear.json) and
the published results (published.json) come from the case study; see the
``source`` fields.  The checks:

* the static steering axis values (KPI, caster, scrub, trail) and the roll
  center height equal the published values to their printed precision;
* camber and toe against heave (front and rear) and against steering wheel
  angle (front) equal the published plots within the plot-reading
  uncertainty.

Conventions of the case study are converted here: toe-out positive (the
solver reports toe-in), heave positive is wheel rebound, rack travel is
85.788 mm per steering wheel revolution, positive turns left.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.suspension import kc
from solvcon.multibody.suspension.model import CornerModel

DATA = Path(__file__).parent / "data" / "multibody" / "optimumg_fsae"
PUB = json.loads((DATA / "published.json").read_text())
DEG = 180.0 / math.pi


@pytest.fixture
def mbs():
    """Import the compiled solver module or skip.

    :return: compiled solver module
    :rtype: module
    """
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    return multibody


def _corner(side):
    return CornerModel.load(DATA / f"{side}.json")


@pytest.mark.parametrize("side", ["front", "rear"])
def test_static_alignment_equals_published(mbs, side):
    model = _corner(side)
    res = kc.bump_sweep(mbs, model, [0.0])
    want = PUB["static"][side]
    assert res["kingpin_method"] == "ball_joints"
    # Published to 0.1 deg and 0.1 mm: the printed value is within 0.05.
    got = {"kpi": res["kpi"][0] * DEG, "caster": res["caster"][0] * DEG,
           "scrub": res["scrub"][0] * 1e3, "trail": res["trail"][0] * 1e3,
           "roll_center": res["roll_center"][0] * 1e3,
           "camber": res["camber"][0] * DEG, "toe": -res["toe"][0] * DEG}
    for key, value in want.items():
        assert got[key] == pytest.approx(value, abs=0.05 + 1e-9), (side, key,
                                                                   got[key])


@pytest.mark.parametrize("side", ["front", "rear"])
def test_heave_curves_equal_published(mbs, side):
    model = _corner(side)
    heave = np.array(PUB["heave"]["heave"], float)
    res = kc.bump_sweep(mbs, model, list(-heave / 1e3))
    assert res["completed"]
    unc = PUB["heave"]["reading_uncertainty"]
    camber = res["camber"] * DEG
    toe_out = -res["toe"] * DEG
    np.testing.assert_allclose(camber, PUB["heave"][f"{side}_camber"], rtol=0,
                               atol=unc["camber"])
    np.testing.assert_allclose(toe_out, PUB["heave"][f"{side}_toe"], rtol=0,
                               atol=unc[f"toe_{side}"])


def test_steering_curves_equal_published(mbs):
    model = _corner("front")
    pub = PUB["steer"]
    swa = np.array(pub["steering_wheel_angle"], float)
    res = kc.steer_sweep(mbs, model, list(85.788 * swa / 360.0 / 1e3))
    assert len(res["rack"]) == len(swa)
    camber = res["camber"] * DEG
    toe_out = -res["toe"] * DEG
    for k in range(len(swa)):
        assert camber[k] == pytest.approx(pub["camber"][k],
                                          abs=pub["camber_uncertainty"][k])
        assert toe_out[k] == pytest.approx(pub["toe"][k],
                                           abs=pub["toe_uncertainty"][k])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
