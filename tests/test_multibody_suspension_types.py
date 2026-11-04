# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for the suspension types of the model library (roadmap P15).

Rules under test:

* Topology: every library model holds the upright with 5 constraints; a
  model with another count, an arm without a shared ball joint, or an
  adjuster the model lacks is rejected.  Rigid wishbone arms give the same
  kinematics as their two links.
* Solver: over the full stop range of every library model, the engine bump
  sweep equals the independent reference solver (pose from the constraint
  equations), and the motion ratio stays positive (no over-center rocker or
  locked linkage).
* Real geometry against published results:
  - MacPherson strut, CEVT car (Yang 2019, Chalmers) against ADAMS/Car:
    camber, toe, KPI, caster, roll center, steer angle;
  - semi-trailing arm, Skoda 130 RS measured (Vondrak 2013, Brno) against
    ADAMS (wheel marker angles) and the closed form of a rotation about a
    fixed axis;
  - five-link (Grzyb & Kowalski 2001): camber and steer angle;
  - four-link with a virtual kingpin, Audi A4 B5 (Knapczyk & Maniowski
    2003): camber, toe, wheel center path within the fit the paper reports
    for its mechanism, and the static values of its Table 1;
  - FSAE pullrod on the upper wishbone (OptimumG case study): the coil-over
    motion ratio.
* Dynamics: an 80 mm stone at 30 km/h keeps every library corner in its
  travel range with a positive motion ratio, and the car settles.
"""

import copy
import json
import math
import os

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.suspension import kc
from solvcon.multibody.suspension.corner import kc_rig
from solvcon.multibody.suspension.metrics import rot
from solvcon.multibody.suspension.model import DATA_DIR, CornerModel
from solvcon.multibody.suspension.reference import ReferenceCorner
from solvcon.multibody.tools.arrays import result_arrays

DEG = 180.0 / math.pi
TESTS = os.path.join(os.path.dirname(__file__), "data", "multibody")
LIBRARY = sorted(p[:-5] for p in os.listdir(DATA_DIR) if p.endswith(".json"))


@pytest.fixture
def mbs():
    """The engine, or skip when solvcon was built without it."""
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon built without the multibody engine")
    return multibody


def _data(name):
    with open(os.path.join(TESTS, name)) as fobj:
        return json.load(fobj)


def _model(name):
    return CornerModel.load(os.path.join(TESTS, name))


def _interp(curve, t_mm, ours):
    pt = np.array(curve["travel"], dtype="float64")
    pv = np.array([np.nan if v is None else v for v in curve["value"]],
                  dtype="float64")
    return np.interp(pt, t_mm, ours), pv


# Topology.


def test_library_has_the_suspension_types():
    types = {CornerModel.load(n).type for n in LIBRARY}
    assert {"macpherson", "double_wishbone_pullrod", "five_link",
            "semi_trailing_arm", "four_link_virtual_kingpin"} <= types


def test_a_wrong_constraint_count_is_rejected():
    raw = copy.deepcopy(CornerModel.load("five_link_rear").raw)
    del raw["links"]["link_5"]
    with pytest.raises(ValueError, match="give 4 constraints"):
        CornerModel(raw)
    raw = copy.deepcopy(CornerModel.load("macpherson_front").raw)
    raw["arm"] = {"inner": [0, 300, 200], "outer": [100, 500, 200]}
    with pytest.raises(ValueError, match="give 10 constraints"):
        CornerModel(raw)


def test_arm_and_adjuster_rules():
    raw = copy.deepcopy(CornerModel.load("five_link_rear").raw)
    raw["arms"] = {"bad": {"links": ["link_1", "link_2"]}}
    with pytest.raises(ValueError, match="share"):
        CornerModel(raw)
    model = CornerModel.load("five_link_rear")
    with pytest.raises(ValueError, match="no caster_shim"):
        model.set_adjusters(caster_shim=0.001)


def test_rigid_arms_equal_their_links(mbs):
    base = CornerModel.load("race_front_corner")
    raw = copy.deepcopy(base.raw)
    raw["arms"] = {"upper": {"links": ["upper_front", "upper_rear"]},
                   "lower": {"links": ["lower_front", "lower_rear"]}}
    armed = CornerModel(raw)
    travels = list(np.linspace(-0.05, 0.04, 7))
    for adjust in ({}, {"camber_shim": 0.002, "caster_shim": 0.003,
                        "toe_link": 0.001}):
        base.set_adjusters(**adjust)
        armed.set_adjusters(**adjust)
        a = kc.bump_sweep(mbs, base, travels)
        b = kc.bump_sweep(mbs, armed, travels)
        for key in ("camber", "toe", "caster", "kpi", "roll_center",
                    "motion_ratio"):
            np.testing.assert_allclose(b[key], a[key], atol=1e-9, rtol=0)


# The solver over the full travel range.


@pytest.mark.parametrize("name", LIBRARY)
def test_library_model_against_the_reference(mbs, name):
    m = CornerModel.load(name)
    lo = m.stops["droop_travel"] - 0.01
    hi = m.stops["bump_travel"] + 0.01
    res = kc.bump_sweep(mbs, m, list(np.linspace(lo, hi, 15)))
    assert res["completed"], res["message"]
    # No over-center rocker or locked linkage in the travel range.
    assert res["motion_ratio"].min() > 0.3
    assert res["rcond"].min() > 1e-3
    ref = ReferenceCorner(m)
    z0 = m.hp["wheel_center"][2]
    for k in np.argsort(np.abs(res["travel"])):
        r = ref.metrics(z0 + res["travel"][k])
        for key in ("camber", "toe", "caster", "kpi", "scrub", "trail",
                    "spring_length", "motion_ratio"):
            if np.isnan(r[key]):
                assert np.isnan(res[key][k]), key
            else:
                assert abs(res[key][k] - r[key]) < 1e-8, (key, k)
        assert abs(res["roll_center"][k] - r["roll_center"]) < 1e-8


@pytest.mark.parametrize("name", LIBRARY)
def test_compliance_converges_to_the_kinematics(mbs, name):
    # Rubber bushings: a few hundredths to tenths of a degree per kN.  The
    # compliance is linear in 1/k, so bushings 100 times stiffer give a
    # change 100 times smaller: the rigid kinematics in the limit.
    from solvcon.multibody.suspension import compliance
    m = CornerModel.load(name)
    for case in ("lateral", "braking"):
        soft = compliance.compliance_sweep(mbs, m, case, 1000.0, n=3)
        stiff = compliance.compliance_sweep(
            mbs, m, case, 1000.0, n=3, k_trans=100.0 * m.bushing_k,
            k_rot=100.0 * m.bushing_k_rot)
        assert soft["completed"] and stiff["completed"]
        for key in ("toe", "camber"):
            assert abs(soft[key][-1]) * DEG < 0.5, (case, key)
            assert soft[key][-1] / stiff[key][-1] == pytest.approx(
                100.0, rel=0.05), (case, key)


# Published results.


def test_macpherson_against_adams(mbs):
    pub = _data("chalmers_macpherson/published.json")["curves"]
    m = _model("chalmers_macpherson/model.json")
    travels = np.arange(-110, 81, 5) * 1e-3
    res = kc.bump_sweep(mbs, m, list(travels))
    assert res["completed"]
    t = travels * 1e3
    for key, ours, scale, tol in (("camber_adams", "camber", DEG, 0.02),
                                  ("kingpin_inclination_adams", "kpi", DEG,
                                   0.03),
                                  ("caster_adams", "caster", DEG, 0.05)):
        ov, pv = _interp(pub[key], t, res[ours] * scale)
        assert np.abs(ov - pv).max() < tol, key
    # Toe: the published static toe is 0.04 deg, a toe setting the listed
    # points do not include; the curve shape agrees.
    ov, pv = _interp(pub["toe_adams"], t, res["toe"] * DEG)
    assert pv[np.argmin(np.abs(np.array(pub["toe_adams"]["travel"])))] \
        == pytest.approx(0.04, abs=0.001)
    assert np.abs((ov + 0.0404) - pv).max() < 0.02
    # Roll center: ADAMS gives the height above the moved wheel pad, the
    # engine above the body-fixed ground; the two differ by the travel.
    # The curve is read to +-3 mm; the difference grows to 5.7 mm at -75 mm.
    ov, pv = _interp(pub["roll_center_height_adams"], t,
                     res["roll_center"] * 1e3 - t)
    assert np.abs(ov - pv).max() < 6.0
    at0 = pub["roll_center_height_adams"]["travel"].index(0.0)
    assert abs(ov[at0] - pv[at0]) < 2.0
    # Steer angle against rack travel (the published right wheel curve, the
    # mirror of this corner's convention).
    racks = np.array(pub["steer_angle_right_wheel_vs_rack"]["rack"]) * 1e-3
    st = kc.steer_sweep(mbs, m, list(racks))
    np.testing.assert_allclose(
        st["toe"] * DEG,
        np.array(pub["steer_angle_right_wheel_vs_rack"]["value"]) - 0.04,
        atol=0.4)


def _arm_poses(mbs, m, travels):
    s, c = kc_rig(mbs, m, travel=float(travels[0]))
    res = result_arrays(s.sweep(c.travel, [c.z0 + t for t in travels]))
    up = c.nums["upright"]
    return res["positions"][:, up], [rot(q) for q in
                                     res["quaternions"][:, up]]


def test_semi_trailing_arm_against_adams(mbs):
    pub = _data("vut_semi_trailing/published.json")
    m = _model("vut_semi_trailing/model.json")
    travels = np.arange(0, 121, 5) * 1e-3
    _, rots = _arm_poses(mbs, m, travels)
    # The published 'toe' and 'camber' are the 3-2-1 yaw and roll + 90 deg
    # of a wheel marker: z on the spin axis, x horizontal at the measured
    # pose.
    spin = m.spin_axis / np.linalg.norm(m.spin_axis)
    x = -np.cross([0.0, 0.0, 1.0], spin)
    x /= np.linalg.norm(x)
    r0 = np.column_stack([x, np.cross(spin, x), spin])
    yaw, roll = [], []
    for rm in rots:
        r = rm @ r0
        yaw.append(math.atan2(r[1, 0], r[0, 0]) * DEG)
        roll.append(math.atan2(r[2, 1], r[2, 2]) * DEG + 90.0)
    t = travels * 1e3
    ov, pv = _interp(pub["curves"]["toe_adams"], t, yaw)
    assert np.abs(ov - pv).max() < 0.03
    ov, pv = _interp(pub["curves"]["camber_adams"], t, roll)
    assert np.abs(ov - pv).max() < 0.06


def test_semi_trailing_arm_closed_form(mbs):
    # A rigid rotation about the pivot axis k: with r from the axis to the
    # wheel center, dz/dtheta = (k x r)_z, and the spin axis s turns at
    # ds/dtheta = k x s.  The camber (top out positive, -asin of s_z for a
    # left wheel) changes at -(k x s)_z / sqrt(1 - s_z^2) per unit angle.
    # (k x s)_z = k_x s_y - k_y s_x: the toe-in of 0.47 deg (s_x) takes 5
    # percent off the first-order value k_x.
    m = _model("vut_semi_trailing/model.json")
    p_in = m.hp["arm_inner"]
    k = m.hp["arm_outer"] - p_in
    k /= np.linalg.norm(k)
    r = m.hp["wheel_center"] - p_in
    s = m.spin_axis / np.linalg.norm(m.spin_axis)
    exact = -np.cross(k, s)[2] / math.sqrt(1.0 - s[2] ** 2) \
        / np.cross(k, r)[2]
    res = kc.bump_sweep(mbs, m, [-1e-4, 0.0, 1e-4])
    gain = (res["camber"][2] - res["camber"][0]) / 2e-4
    assert gain == pytest.approx(exact, rel=2e-3)
    # The source's first-order value 0.024 deg/mm, from k_x alone.
    assert -k[0] / np.cross(k, r)[2] * DEG * 1e-3 == pytest.approx(
        -0.024, abs=0.001)


def test_five_link_against_published(mbs):
    pub = _data("grzyb_five_link/published.json")
    b6 = np.array(pub["points"]["B6"]) * 1e-3
    b7 = np.array(pub["points"]["B7"]) * 1e-3
    for tag, fname in (("changed", "model.json"),
                       ("original", "model_original.json")):
        m = _model(f"grzyb_five_link/{fname}")
        travels = np.arange(-100, 101, 5) * 1e-3
        pos, rots = _arm_poses(mbs, m, travels)
        steer, camber = [], []
        for p7, rm in zip(pos, rots):
            p6 = p7 + rm @ (b6 - b7)
            # The angle definitions of the paper, eqs (3.5) and (3.6).
            steer.append(math.degrees(math.atan((p6[0] - p7[0])
                                                / (p7[1] - p6[1]))))
            camber.append(math.degrees(math.atan(
                (p7[2] - p6[2]) / math.hypot(p7[0] - p6[0], p7[1] - p6[1]))))
        t = travels * 1e3
        ov, pv = _interp(pub["curves"][f"steer_angle_{tag}"], t, steer)
        assert np.abs(ov - pv).max() < 0.03, tag
        ov, pv = _interp(pub["curves"][f"camber_{tag}"], t, camber)
        assert np.abs(ov - pv).max() < 0.12, tag


def test_audi_four_link_against_published(mbs):
    pub = _data("knapczyk_audi_a4/published.json")
    m = _model("knapczyk_audi_a4/model.json")
    travels = np.arange(-80, 81, 5) * 1e-3
    pos, rots = _arm_poses(mbs, m, travels)
    spin = [rm @ m.spin_axis for rm in rots]
    t = travels * 1e3
    # The target curves of the paper; its synthesized mechanism (these
    # points) meets them with fit_II = 0.104 deg camber, 0.229 deg toe,
    # 0.45 mm wheel-center x.
    gamma = [math.degrees(math.atan2(s[2], math.hypot(s[0], s[1])))
             for s in spin]
    delta = [math.degrees(math.atan2(-s[0], s[1])) for s in spin]
    for key, ours, rms in (("camber_gamma", gamma, 0.1),
                           ("toe_delta", delta, 0.3),
                           ("wheel_center_x", pos[:, 0] * 1e3, 1.0),
                           ("wheel_center_y", pos[:, 1] * 1e3, 0.5)):
        ov, pv = _interp(pub["curves"][key], t, ours)
        assert np.sqrt(np.nanmean((ov - pv) ** 2)) < rms, key
    # Table 1 (synthesis result, s_w = -20 mm) on the library model, whose
    # wheel center is on the tire center plane.
    lib = CornerModel.load("virtual_kingpin_front")
    res = kc.bump_sweep(mbs, lib, [-0.02])
    assert res["kpi"][0] * DEG == pytest.approx(4.5, abs=0.3)
    assert res["caster"][0] * DEG == pytest.approx(3.4, abs=0.1)
    assert res["scrub"][0] * 1e3 == pytest.approx(-11.0, abs=1.5)
    assert 2e3 * res["contact_patch"][0][1] == pytest.approx(1495.0, abs=6.0)


def test_fsae_pullrod_motion_ratio(mbs):
    # OptimumG slide 113: the front coil-over ratio, wheel over coil-over
    # displacement from design, with the pullrod on the upper wishbone.
    pub = _data("optimumg_fsae/published.json")["motion_ratio"]
    m = CornerModel.load("fsae_front_pullrod")
    travels = np.linspace(-0.05, 0.05, 101)
    res = kc.bump_sweep(mbs, m, list(travels))
    length0 = np.interp(0.0, res["travel"], res["spring_length"])
    for heave, value in zip(pub["heave"], pub["front"]):
        travel = -heave * 1e-3
        if 0 == heave:
            ours = 1.0 / np.interp(0.0, res["travel"], res["motion_ratio"])
        else:
            ours = abs(travel) / abs(np.interp(travel, res["travel"],
                                               res["spring_length"])
                                     - length0)
        # A constant offset of about 0.02 (1.8 percent) remains.
        assert ours == pytest.approx(value, abs=0.045), heave


# Dynamics.


@pytest.mark.parametrize("name", LIBRARY)
def test_stone_keeps_every_corner_in_range(mbs, name):
    from solvcon.multibody.suspension import road, vehicle
    m = CornerModel.load(name)
    mass = m.vehicle.get("corner_mass", 200.0)
    deflection = vehicle.static_tire_deflection(m, mass)[0]
    cam = road.TandemCam.for_tire(m.wheel_radius, deflection)
    rin = road.road_input(road.RoadProfile.bump(0.08, 0.15), 30.0 / 3.6, cam,
                          duration=2.0)
    s, hd = vehicle.quarter_car(mbs, m, mass, road=rin, dt=2e-3)
    s.set_integrator("rk45")
    s.set_tolerances(1e-7, 1e-10)
    up = hd["corner"].nums["upright"]
    z0 = np.asarray(s.get_body_positions())
    travel = []
    for _ in range(int(round(rin.duration / 2e-3)) - 1):
        s.solve()
        pos = np.asarray(s.get_body_positions())
        travel.append((pos[up][2] - z0[up][2]) - (pos[1][2] - z0[1][2]))
    travel = np.array(travel)
    assert travel.max() < m.stops["bump_travel"] + 0.015
    assert travel.min() > m.stops["droop_travel"] - 0.005
    ext = kc.bump_sweep(mbs, m, [float(travel.min()), float(travel.max())])
    assert ext["motion_ratio"].min() > 0.3
    assert abs(travel[-1]) < 0.005

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
