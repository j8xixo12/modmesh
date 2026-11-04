# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for the suspension K&C application (roadmap P9.1-P9.5).

Rules under test:

* XM: every K&C metric of the engine sweep equals the independent geometric
  reference solver (suspension/reference.py) at every travel point.
* AN: parallel, equal, horizontal wishbones with a parallel tie rod give
  zero camber change, zero bump steer, and a roll center on the ground.
* At design (spring unloaded) the wheel rate from driver efforts equals
  k * MR^2 (virtual work).
* Adjusters change static alignment with the documented signs, and the
  engine and the reference solver agree after an adjustment.
* A sweep beyond the reach of the linkage stops and reports the value.
"""

import copy
import math

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.suspension import kc
from solvcon.multibody.suspension.model import CornerModel
from solvcon.multibody.suspension.reference import ReferenceCorner

TRAVELS = list(np.linspace(-0.04, 0.04, 9))


@pytest.fixture
def mbs():
    """Import the compiled solver module or skip.

    :return: compiled solver module
    :rtype: module
    """
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    return multibody


@pytest.fixture
def model():
    return CornerModel.load("race_front_corner")


def test_engine_matches_reference_solver(mbs, model):
    res = kc.bump_sweep(mbs, model, TRAVELS)
    assert res["completed"]
    ref = ReferenceCorner(model)
    z0 = model.hp["wheel_center"][2]
    for k, t in enumerate(TRAVELS):
        r = ref.metrics(z0 + t)
        # Exact quantities (both from converged Newton): 1e-10 rad / m.
        assert res["camber"][k] == pytest.approx(r["camber"], abs=1e-10)
        assert res["toe"][k] == pytest.approx(r["toe"], abs=1e-10)
        # Quantities from sensitivities: the reference uses central differences
        # with h = 1e-6, error ~1e-9.
        for key in ("caster", "kpi", "roll_center", "scrub", "trail"):
            assert res[key][k] == pytest.approx(r[key], abs=2e-8), (key, t)


def test_static_alignment_is_plausible(mbs, model):
    res = kc.bump_sweep(mbs, model, [0.0])
    deg = math.degrees
    assert abs(deg(res["camber"][0])) < 1e-9  # design: wheel plane vertical
    assert abs(deg(res["toe"][0])) < 1e-9
    assert 1.0 < deg(res["caster"][0]) < 3.0
    assert 5.0 < deg(res["kpi"][0]) < 9.0
    assert 0.5 < res["motion_ratio"][0] < 1.5


def parallel_wishbones(model):
    m = copy.deepcopy(model)
    lo_in, lo_out = np.array([0.0, 0.25, 0.15]), np.array([0.0, 0.74, 0.15])
    dz = np.array([0.0, 0.0, 0.33])
    for name, sign in (("lower_front", 1), ("lower_rear", -1)):
        m.hp[name + "_inner"] = lo_in + np.array([sign * 0.2, 0, 0])
        m.hp[name + "_outer"] = lo_out
    for name, sign in (("upper_front", 1), ("upper_rear", -1)):
        m.hp[name + "_inner"] = lo_in + dz + np.array([sign * 0.2, 0, 0])
        m.hp[name + "_outer"] = lo_out + dz
    m.hp["tie_rod_inner"] = np.array([0.09, 0.25, 0.25])
    m.hp["tie_rod_outer"] = np.array([0.09, 0.74, 0.25])
    return m


def test_parallel_equal_wishbones_analytic(mbs, model):
    m = parallel_wishbones(model)
    res = kc.bump_sweep(mbs, m, TRAVELS)
    assert res["completed"]
    np.testing.assert_allclose(res["camber"], 0.0, atol=1e-10)
    np.testing.assert_allclose(res["toe"], 0.0, atol=1e-10)
    mid = len(TRAVELS) // 2
    assert res["roll_center"][mid] == pytest.approx(0.0, abs=1e-10)


def test_wheel_rate_equals_k_mr_squared_at_design(mbs, model):
    res = kc.bump_sweep(mbs, model, [-0.001, 0.0, 0.001])
    mr = res["motion_ratio"][1]
    # Central difference of the effort over +/- 1 mm: error ~ (1 mm)^2 * k
    # MR''.
    assert res["wheel_rate"][1] == pytest.approx(model.spring_k * mr ** 2,
                                                 rel=1e-4)


def test_adjuster_signs_and_reference_agreement(mbs, model):
    base = kc.bump_sweep(mbs, model, [0.0])
    # Physical signs: a longer front-mounted tie rod pushes the front of the
    # wheel out (toe-out); a shim that spaces the upper ball joint outboard on
    # the upright tilts the wheel top inboard (more negative camber), because
    # the upper links hold the joint in place.
    cases = [("toe_link", "toe", -1), ("camber_shim", "camber", -1),
             ("caster_shim", "caster", -1)]
    for adjuster, metric, sign in cases:
        m = copy.deepcopy(model)
        m.set_adjusters(**{adjuster: 0.002})
        res = kc.bump_sweep(mbs, m, [0.0])
        change = res[metric][0] - base[metric][0]
        assert sign * change > 0, (adjuster, change)
        ref = ReferenceCorner(m).metrics(m.hp["wheel_center"][2])
        assert res[metric][0] == pytest.approx(ref[metric], abs=2e-8)


def test_sweep_beyond_reach_stops_with_message(mbs, model):
    res = kc.bump_sweep(mbs, model, [0.0, 0.05, 2.0, 0.0])
    assert not res["completed"]
    assert len(res["travel"]) == 2
    assert "2" in res["message"]


def test_camber_sign_convention():
    # Unit check of the ISO sign: rotate the left wheel about +x so that the
    # top moves inboard (-y): camber must be negative.
    from solvcon.multibody.suspension.metrics import wheel_geometry
    th = math.radians(1.0)
    a = np.array([[1, 0, 0], [0, math.cos(th), -math.sin(th)],
                  [0, math.sin(th), math.cos(th)]])
    top = a @ np.array([0.0, 0.0, 1.0])
    assert top[1] < 0  # top moved inboard for the left wheel
    wg = wheel_geometry(np.zeros(3), a, np.array([0.0, 1.0, 0.0]), 0.3, +1)
    assert wg["camber"] == pytest.approx(-th, abs=1e-12)


def test_mirrored_corner_has_same_alignment(mbs, model):
    left = kc.bump_sweep(mbs, model, TRAVELS)
    right = kc.bump_sweep(mbs, model.mirrored(), TRAVELS)
    for key in ("camber", "toe", "caster", "kpi", "scrub", "trail",
                "motion_ratio", "roll_center"):
        np.testing.assert_allclose(right[key], left[key], atol=1e-10,
                                   err_msg=key)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
