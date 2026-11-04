# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for the suspension engineering workflow (roadmap P11.4, P11.6-P11.9,
P11.11, P11.12).

Rules under test:

* P11.4: the front- and side-view instant centers, as points of the
  upright, have zero in-plane velocity; the roll center point height equals
  the ``roll_center`` metric; parallel equal wishbones give no FVIC (NaN);
  the kingpin axis is the ball joint line (double wishbone) or the
  steering screw axis (virtual kingpin).
* P11.6: after the alignment solve, the bump sweep at design gives the
  targets within 1e-8 rad and the reference solver agrees; an unreachable
  target reports failure and leaves the adjusters unchanged.
* P11.7: the gradient at zero load equals the linear compliance from
  independent central differences; stiff bushings scale the gradient by the
  stiffness ratio; Maxwell-Betti reciprocity holds between the lateral and
  braking cases and between the lateral and aligning cases.
* P11.8, P11.12: a hardpoint edit gives the same sweep as a model built
  with the edited point; an edit outside the limit box raises and keeps the
  model.
* P11.9: each point of a hardpoint sweep equals a direct sweep of the
  edited model; the bump steer of the tie rod inner z changes sign near the
  shipped value; the sweep stops at the limit box.
* P11.11: heave-only, rack-only, and roll-only profiles equal the bump,
  steer, and roll sweeps; a combined row equals a direct engine solve and
  the independent reference solver.
"""

import copy
import json
import math

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.tools.arrays import result_arrays
from solvcon.multibody.suspension import (alignment, axle, compliance, corner,
                                          kc, sensitivity)
from solvcon.multibody.suspension import metrics as mt
from solvcon.multibody.suspension.model import DATA_DIR, CornerModel
from solvcon.multibody.suspension.reference import ReferenceCorner

from test_multibody_suspension import parallel_wishbones
from test_multibody_suspension_compliance import compliance_matrix

TRAVELS = list(np.linspace(-0.04, 0.04, 9))
DEG = math.pi / 180.0


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


def raw_data():
    with open(DATA_DIR / "race_front_corner.json") as f:
        return json.load(f)


# ---------------------------------------------------------------- P11.4


def rig_at(mbs, model, travel):
    """Engine rig at one travel, solved exactly as ``bump_sweep([travel])``
    does.

    :return: (system, corner handles, bump sweep result at the pose)
    :rtype: tuple
    """
    s, c = corner.kc_rig(mbs, model, travel=travel)
    res = result_arrays(s.sweep(c.travel, [c.z0 + travel]))
    s.sweep(c.travel, [res["values"][0]])
    return s, c, res


def test_instant_centers_have_zero_in_plane_velocity(mbs, model):
    res = kc.bump_sweep(mbs, model, TRAVELS)
    assert res["completed"]
    for key in ("fvic", "svic", "roll_center_point", "kingpin_point",
                "kingpin_dir"):
        assert res[key].shape == (len(TRAVELS), 3), key
    for t in TRAVELS:
        # One-point sweeps: the test rig below gives the same sensitivities.
        # (Across separate solves the poses agree only to the engine
        # tolerance, which moves the rates by ~1e-12 to 1e-10.)
        one = kc.bump_sweep(mbs, model, [t])
        _, c, rr = rig_at(mbs, model, t)
        up = c.nums["upright"]
        r, dr, dw = (rr["positions"][0, up], rr["dr_ds"][0, up],
                     rr["domega_ds"][0, up])
        cp = one["contact_patch"][0]
        fv, sv = one["fvic"][0], one["svic"][0]
        assert np.all(np.isfinite(fv)) and np.all(np.isfinite(sv))
        # plane x = contact patch x
        assert fv[0] == pytest.approx(cp[0], abs=1e-15)
        # plane y = contact patch y
        assert sv[1] == pytest.approx(cp[1], abs=1e-15)
        v_fv = dr + np.cross(dw, fv - r)
        v_sv = dr + np.cross(dw, sv - r)
        # The rates are O(1) m/m and the ICs are a few m away: rounding of the
        # products is ~1e-15, so 1e-12 m/m is a strict zero.
        assert abs(v_fv[1]) < 1e-12 and abs(v_fv[2]) < 1e-12, (t, v_fv)
        assert abs(v_sv[0]) < 1e-12 and abs(v_sv[2]) < 1e-12, (t, v_sv)


def test_instant_centers_agree_with_reference_solver(mbs, model):
    # Independent check: the IC velocity as a material point of the upright
    # from central differences of reference-solver poses (no engine
    # sensitivities).
    res = kc.bump_sweep(mbs, model, TRAVELS)
    ref = ReferenceCorner(model)
    z0 = model.hp["wheel_center"][2]
    h = 1e-6
    for k, t in enumerate(TRAVELS):
        r, a = ref.solve(z0 + t)
        r1, a1 = ref.solve(z0 + t + h)
        r0, a0 = ref.solve(z0 + t - h)
        for key, plane in (("fvic", (1, 2)), ("svic", (0, 2))):
            loc = a.T @ (res[key][k] - r)
            v = (r1 + a1 @ loc - r0 - a0 @ loc) / (2 * h)
            # Reference poses solve to ~1e-15 m; over 2h = 2e-6 m that is
            # ~1e-9 m/m; the truncation error h^2 * x''' is ~1e-12.
            assert abs(v[plane[0]]) < 2e-8 and abs(v[plane[1]]) < 2e-8, (key,
                                                                         t, v)


def test_roll_center_point_and_kingpin(mbs, model):
    res = kc.bump_sweep(mbs, model, TRAVELS)
    for k, t in enumerate(TRAVELS):
        rc = res["roll_center_point"][k]
        cp = res["contact_patch"][k]
        assert rc[1] == 0.0
        assert rc[0] == pytest.approx(cp[0], abs=1e-15)
        # Same line, two formulas (through the FVIC, and perpendicular to the
        # contact patch velocity): rounding only.
        assert rc[2] == pytest.approx(res["roll_center"][k], abs=1e-12)
        # The roll center lies on the line from the contact patch to the FVIC.
        d1 = res["fvic"][k] - cp
        d2 = rc - cp
        assert abs(d1[1] * d2[2] - d1[2] * d2[1]) < 1e-12 * np.linalg.norm(d1)
        n = res["kingpin_dir"][k]
        p = res["kingpin_point"][k]
        assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-14)
        assert n[2] > 0
        sa = mt.steering_axis_metrics(n, p, cp, model.side)
        assert sa["caster"] == pytest.approx(res["caster"][k], abs=1e-14)
        assert sa["kpi"] == pytest.approx(res["kpi"][k], abs=1e-14)
        assert p[2] == pytest.approx(cp[2], abs=1e-12)  # the ground point
        assert res["kingpin_method"] == "ball_joints"


def test_kingpin_is_the_ball_joint_line(mbs, model):
    # A double wishbone steers about the line through its ball joints.
    import copy
    res = kc.bump_sweep(mbs, model, TRAVELS)
    lower, upper = model.kingpin_joints()
    assert (lower, upper) == ("lower_front_outer", "upper_front_outer")
    for k, t in enumerate(TRAVELS):
        s, c, _ = rig_at(mbs, model, t)
        up = c.nums["upright"]
        r = result_arrays(s.sweep(c.travel, [c.z0 + t]))
        pos = r["positions"][0, up]
        a = mt.rot(r["quaternions"][0, up])
        pts = model.upright_points()
        n, p = res["kingpin_dir"][k], res["kingpin_point"][k]
        for name in (lower, upper):
            q = pos + a @ (pts[name] - pts["wheel_center"])
            # Both ball joints lie on the axis: rounding of the pose (1e-10 m
            # Newton tolerance) only.
            off = q - p
            assert np.linalg.norm(off - (off @ n) * n) < 1e-9, (name, t)
    # The instant-axis definition stays available for virtual kingpins.
    inst = copy.deepcopy(model)
    inst.steer_axis = "instant"
    assert inst.kingpin_joints() is None
    one = kc.bump_sweep(mbs, inst, [0.0])
    assert one["kingpin_method"] == "instant_axis"


def test_instant_kingpin_is_the_screw_axis(mbs, model):
    import copy
    model = copy.deepcopy(model)
    model.steer_axis = "instant"
    for t in TRAVELS:
        # On the screw axis, the velocity of an upright point for a rack
        # motion is parallel to the axis. Same solve as a one-point sweep (see
        # test_instant_centers_have_zero_in_plane_velocity).
        one = kc.bump_sweep(mbs, model, [t])
        n, p = one["kingpin_dir"][0], one["kingpin_point"][0]
        s, c, _ = rig_at(mbs, model, t)
        st = result_arrays(s.sweep(c.steer, [0.0]))
        up = c.nums["upright"]
        v = (st["dr_ds"][0, up]
             + np.cross(st["domega_ds"][0, up], p - st["positions"][0, up]))
        # |v| ~ 0.015 m/m; rounding of the axis point (~1e-16 m) times the
        # angular rate (~10 rad/m) is ~1e-15.
        assert np.linalg.norm(v - (v @ n) * n) < 1e-12, t


def test_parallel_wishbones_give_fvic_at_infinity(mbs, model):
    m = parallel_wishbones(model)
    res = kc.bump_sweep(mbs, m, TRAVELS)
    assert res["completed"]
    assert np.all(np.isnan(res["fvic"]))
    # The roll center still exists: the line is perpendicular to the contact
    # patch path, which is vertical here, so the roll center is on the ground.
    mid = len(TRAVELS) // 2
    assert res["roll_center_point"][mid][2] == pytest.approx(0.0, abs=1e-10)
    assert np.all(np.isfinite(res["roll_center_point"]))


def test_ic_helpers_at_infinity():
    cp = np.array([0.0, 0.8, 0.0])
    v = np.array([0.0, 0.0, 1.0])
    assert np.all(np.isnan(mt.front_view_ic(cp, v, np.zeros(3))))
    assert np.all(np.isnan(mt.side_view_ic(cp, v, np.zeros(3))))
    # A pure rotation about x through the point (0, 0, 0): the IC is there.
    w = np.array([0.5, 0.0, 0.0])
    v = np.cross(w, cp - np.zeros(3))
    np.testing.assert_allclose(mt.front_view_ic(cp, v, w), [0.0, 0.0, 0.0],
                               atol=1e-15)


# ---------------------------------------------------------------- P11.6


def static_alignment(mbs, model):
    r = kc.bump_sweep(mbs, model, [0.0])
    return {k: r[k][0] for k in ("camber", "toe", "caster")}


def test_alignment_reaches_all_targets(mbs, model):
    targets = {"camber": -1.5 * DEG, "toe": 0.1 * DEG, "caster": 1.69 * DEG}
    before = copy.deepcopy(model.adjust)
    out = alignment.solve_alignment(mbs, model, targets)
    assert out["converged"], out["message"]
    assert model.adjust == before  # the input model is not modified
    m = copy.deepcopy(model)
    m.set_adjusters(**out["adjusters"])
    got = static_alignment(mbs, m)
    ref = ReferenceCorner(m).metrics(m.hp["wheel_center"][2])
    for key, value in targets.items():
        assert got[key] == pytest.approx(value, abs=1e-8), key
        assert out["achieved"][key] == pytest.approx(got[key], abs=1e-12), key
        # Reference caster comes from central differences (h = 1e-6, error
        # ~1e-9), as in tests/test_suspension.py.
        assert ref[key] == pytest.approx(value, abs=2e-8), key
    assert out["iterations"] <= 20


def test_alignment_subset_keeps_other_adjusters(mbs, model):
    model.set_adjusters(camber_shim=0.0, caster_shim=0.001, toe_link=0.0005)
    out = alignment.solve_alignment(mbs, model, {"camber": -2.0 * DEG})
    assert out["converged"], out["message"]
    assert out["adjusters"]["caster_shim"] == 0.001
    assert out["adjusters"]["toe_link"] == 0.0005
    m = copy.deepcopy(model)
    m.set_adjusters(**out["adjusters"])
    assert static_alignment(mbs, m)["camber"] == pytest.approx(-2.0 * DEG,
                                                               abs=1e-8)


def test_alignment_already_at_target_takes_no_step(mbs, model):
    model.set_adjusters(toe_link=0.001)
    toe = static_alignment(mbs, model)["toe"]
    out = alignment.solve_alignment(mbs, model, {"toe": toe})
    assert out["converged"]
    assert out["iterations"] == 0
    assert out["adjusters"]["toe_link"] == 0.001


def test_alignment_unreachable_target_reports_failure(mbs, model):
    model.set_adjusters(toe_link=0.001)
    before = copy.deepcopy(model.adjust)
    out = alignment.solve_alignment(mbs, model, {"camber": -30.0 * DEG})
    assert not out["converged"]
    assert "camber_shim" in out["message"]
    assert out["adjusters"] == before  # failure leaves the adjusters unchanged
    assert model.adjust == before


def test_alignment_rejects_unknown_target(mbs, model):
    with pytest.raises(ValueError, match="ride_height"):
        alignment.solve_alignment(mbs, model, {"ride_height": 0.1})


# ---------------------------------------------------------------- P11.7


def test_compliance_sweep_validates_input(mbs, model):
    with pytest.raises(ValueError, match="odd"):
        compliance.compliance_sweep(mbs, model, "lateral", 1000.0, n=10)
    with pytest.raises(ValueError, match="cornering"):
        compliance.compliance_sweep(mbs, model, "cornering", 1000.0)


@pytest.mark.parametrize("case",
                         ["lateral", "braking", "traction", "aligning"])
def test_compliance_sweep_shape(mbs, model, case):
    out = compliance.compliance_sweep(mbs, model, case, 1000.0, n=5)
    assert out["completed"], out["message"]
    np.testing.assert_allclose(out["load"],
                               [-1000.0, -500.0, 0.0, 500.0, 1000.0])
    assert out["patch"].shape == (5, 3)
    assert out["toe"][2] == 0.0 and out["camber"][2] == 0.0
    assert np.all(out["patch"][2] == 0.0)
    for key in ("toe", "camber", "patch_x", "patch_y", "patch_z"):
        assert np.isfinite(out["gradient"][key])


def test_compliance_gradient_equals_linear_compliance(mbs, model):
    model.set_adjusters(toe_link=0.001)
    f = 200.0
    c, toe = compliance_matrix(mbs, model, f)
    for case, j in (("braking", 0), ("lateral", 1)):
        # step 200 N
        out = compliance.compliance_sweep(mbs, model, case, 2 * f, n=5)
        g = out["gradient"]
        got = np.array([g["patch_x"], g["patch_y"], g["patch_z"]])
        # Same equilibria (unique) reached by a ramp instead of a fresh start:
        # equal to the equilibrium tolerance, ~1e-6 of the largest compliance.
        scale = np.max(np.abs(c))
        np.testing.assert_allclose(got, c[:, j], atol=1e-6 * scale,
                                   err_msg=case)
        assert g["toe"] == pytest.approx(toe[j], rel=1e-6,
                                         abs=1e-6 * abs(toe[1]))


def test_compliance_maxwell_betti(mbs, model):
    model.set_adjusters(toe_link=0.001)
    lat = compliance.compliance_sweep(mbs, model, "lateral", 400.0,
                                      n=5)["gradient"]
    brk = compliance.compliance_sweep(mbs, model, "braking", 400.0,
                                      n=5)["gradient"]
    mz = compliance.compliance_sweep(mbs, model, "aligning", 40.0,
                                     n=5)["gradient"]
    # Fy at the patch and Fx at the patch: d u_x / d Fy = d u_y / d Fx.
    # tests/test_suspension_compliance.py measures the asymmetry at ~7e-9 of
    # max |C| (Newton tolerance); 1e-6 bounds it.
    scale = max(abs(lat["patch_y"]), abs(brk["patch_x"]))
    assert lat["patch_x"] == pytest.approx(brk["patch_y"], abs=1e-6 * scale)
    # Mz and Fy: the work conjugate of Mz is the upright rotation about z, so
    # d rot_z / d Fy = d u_y / d Mz, with the same 1e-6 bound. (The toe
    # change is not the conjugate: with static toe and camber it differs from
    # -rot_z by ~4e-5 relative, through the camber rotation.)
    assert abs(lat["rot_z"]) > 1e-9
    assert mz["patch_y"] == pytest.approx(lat["rot_z"], rel=1e-6)
    # The toe change is -side * rot_z to first order in the static camber
    # and toe (both below 1 deg here): 1e-4 bounds the ~4e-5 difference.
    assert -model.side * lat["toe"] == pytest.approx(lat["rot_z"], rel=1e-4)


def test_compliance_stiff_bushings_scale_gradient(mbs, model):
    base = compliance.compliance_sweep(mbs, model, "lateral", 400.0, n=5)
    f = 100.0
    # Every elastic element scales by f: the bushings and the spring, which
    # acts in parallel through the pushrod (the travel driver holds only the
    # wheel-center height). With no preload at zero load, the linear
    # compliance is then exactly proportional to 1 / f. What remains is the
    # cubic term of the +/-200 N central difference in the base gradient
    # (measured ~1e-8 relative; it shrinks with the stiffness) and the
    # equilibrium tolerance: 1e-6 bounds both.
    m = copy.deepcopy(model)
    m.spring_k *= f
    stiff = compliance.compliance_sweep(mbs, m, "lateral", 400.0, n=5,
                                        k_trans=f * model.bushing_k,
                                        k_rot=f * model.bushing_k_rot)
    for key in ("toe", "patch_y"):
        assert f * stiff["gradient"][key] == pytest.approx(
            base["gradient"][key], rel=1e-6), key
    very = compliance.compliance_sweep(mbs, model, "lateral", 400.0, n=5,
                                       k_trans=1e12, k_rot=1e6)
    assert (abs(very["gradient"]["patch_y"])
            < 1e-4 * abs(base["gradient"]["patch_y"]))


def test_compliance_stops_at_failed_equilibrium(mbs, model):
    # A 500 MN lateral load would deflect the 20 kN/mm bushings by ~25 m: far
    # beyond the reach of the equilibrium Newton from the unloaded pose.
    out = compliance.compliance_sweep(mbs, model, "lateral", 5e8, n=3)
    assert not out["completed"]
    assert "load -5e+08" in out["message"]
    assert len(out["load"]) < 3


# ---------------------------------------------------------------- P11.8,
# P11.12


def sweep_equal(a, b, atol):
    for key in ("camber", "toe", "caster", "kpi", "scrub", "trail",
                "motion_ratio", "roll_center"):
        np.testing.assert_allclose(a[key], b[key], atol=atol, err_msg=key)


@pytest.mark.parametrize("name,link,end", [("tie_rod_inner", "tie_rod",
                                            "rack"),
                                           ("upper_front_inner", "upper_front",
                                            "chassis"),
                                           ("lower_front_outer", "lower_front",
                                            "upright")])
def test_set_hardpoint_equals_built_model(mbs, model, name, link, end):
    data = raw_data()
    data["links"][link][end][2] += 3.0
    built = CornerModel(data)
    model.set_hardpoint(name, model.hp[name] + np.array([0.0, 0.0, 0.003]))
    # The two models differ only by the rounding of mm -> m (1 ulp).
    np.testing.assert_allclose(model.hp[name], built.hp[name], atol=1e-16)
    sweep_equal(kc.bump_sweep(mbs, model, TRAVELS),
                kc.bump_sweep(mbs, built, TRAVELS), 1e-11)


def test_set_hardpoint_validates_name_and_shape(model):
    with pytest.raises(ValueError, match="no_such_point"):
        model.set_hardpoint("no_such_point", [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        model.set_hardpoint("tie_rod_inner", [0.0, 0.0])
    names = model.hardpoint_names()
    assert set(names) == set(model.hp)
    assert "tie_rod_inner" in names and "wheel_center" in names


def test_hardpoint_and_adjusters_compose(mbs, model):
    p = model.hp["upper_front_outer"] + np.array([0.002, -0.001, 0.0])
    a = copy.deepcopy(model)
    a.set_hardpoint("upper_front_outer", p)
    a.set_adjusters(camber_shim=0.002, toe_link=0.001)
    b = copy.deepcopy(model)
    b.set_adjusters(camber_shim=0.002, toe_link=0.001)
    b.set_hardpoint("upper_front_outer", p)
    # Rule: hp holds the unadjusted design point; adjusters apply on top.
    np.testing.assert_array_equal(a.hp["upper_front_outer"], p)
    shift = np.array([0.0, model.side * 0.002, 0.0])
    np.testing.assert_allclose(a.upright_points()["upper_front_outer"],
                               p + shift, atol=1e-16)
    sweep_equal(kc.bump_sweep(mbs, a, TRAVELS), kc.bump_sweep(mbs, b, TRAVELS),
                0.0)


def test_limits_from_json_and_rejection(tmp_path, model):
    data = raw_data()
    data["limits"] = {"tie_rod_inner": {"min": [80.0, 270.0, 210.0],
                                        "max": [100.0, 290.0, 230.0]}}
    path = tmp_path / "corner.json"
    path.write_text(json.dumps(data))
    m = CornerModel.load(path)
    np.testing.assert_allclose(m.limits["tie_rod_inner"]["min"],
                               [0.08, 0.27, 0.21])
    np.testing.assert_allclose(m.limits["tie_rod_inner"]["max"],
                               [0.10, 0.29, 0.23])
    before = m.hp["tie_rod_inner"].copy()
    with pytest.raises(ValueError, match=r"tie_rod_inner.*z.*0\.23"):
        m.set_hardpoint("tie_rod_inner", before + np.array([0.0, 0.0, 0.02]))
    np.testing.assert_array_equal(m.hp["tie_rod_inner"], before)
    m.set_hardpoint("tie_rod_inner", before + np.array([0.0, 0.0, 0.005]))
    assert m.hp["tie_rod_inner"][2] == pytest.approx(0.224)
    # Mirror: the y bounds swap and change sign.
    r = m.mirrored()
    np.testing.assert_allclose(r.limits["tie_rod_inner"]["min"],
                               [0.08, -0.29, 0.21])
    np.testing.assert_allclose(r.limits["tie_rod_inner"]["max"],
                               [0.10, -0.27, 0.23])
    # inside the mirrored box
    r.set_hardpoint("tie_rod_inner", r.hp["tie_rod_inner"])


def test_set_limit(model):
    p = model.hp["rocker_pivot"].copy()
    with pytest.raises(ValueError, match="min"):
        model.set_limit("rocker_pivot", p + 0.01, p - 0.01)
    with pytest.raises(ValueError, match="rocker_pivot"):
        # excludes the point
        model.set_limit("rocker_pivot", p + 0.001, p + 0.01)
    model.set_limit("rocker_pivot", p - 0.01, p + 0.01)
    with pytest.raises(ValueError, match="rocker_pivot.*x"):
        model.set_hardpoint("rocker_pivot", p + np.array([-0.02, 0.0, 0.0]))
    np.testing.assert_array_equal(model.hp["rocker_pivot"], p)
    model.set_limit("rocker_pivot", None, None)
    assert "rocker_pivot" not in model.limits
    model.set_hardpoint("rocker_pivot", p + np.array([-0.02, 0.0, 0.0]))


# ---------------------------------------------------------------- P11.9


SUMMARY = ("camber_gain", "bump_steer", "roll_center", "caster", "kpi",
           "scrub", "trail", "motion_ratio")


def test_hardpoint_sweep_equals_direct_sweeps(mbs, model):
    offsets = [-0.003, 0.0, 0.002]
    h = 0.005
    out = sensitivity.hardpoint_sweep(mbs, model, "upper_rear_inner", 1,
                                      offsets, h=h)
    assert out["completed"], out["message"]
    np.testing.assert_array_equal(out["offset"], offsets)
    for k, d in enumerate(offsets):
        m = copy.deepcopy(model)
        p = m.hp["upper_rear_inner"].copy()
        p[1] += d
        m.set_hardpoint("upper_rear_inner", p)
        r = kc.bump_sweep(mbs, m, [-h, 0.0, h])
        expect = {"camber_gain": (r["camber"][2] - r["camber"][0]) / (2 * h),
                  "bump_steer": (r["toe"][2] - r["toe"][0]) / (2 * h)}
        for key in SUMMARY[2:]:
            expect[key] = r[key][1]
        for key in SUMMARY:
            # Same computation on the same model: bit-identical.
            assert out[key][k] == expect[key], (key, d)
    np.testing.assert_array_equal(
        model.hp["upper_rear_inner"],
        CornerModel.load("race_front_corner").hp["upper_rear_inner"])


def test_tie_rod_height_bump_steer_changes_sign_near_design(mbs, model):
    offsets = list(np.linspace(-0.005, 0.005, 11))
    out = sensitivity.hardpoint_sweep(mbs, model, "tie_rod_inner", 2, offsets)
    assert out["completed"], out["message"]
    bs = out["bump_steer"]
    # monotonic: higher inner point, more toe-out in bump
    assert np.all(np.diff(bs) < 0)
    k = int(np.flatnonzero(np.sign(bs[:-1]) != np.sign(bs[1:]))[0])
    zero = (offsets[k]
            - bs[k] * (offsets[k + 1] - offsets[k]) / (bs[k + 1] - bs[k]))
    # The shipped point minimizes bump steer over +/-40 mm (Nelder-Mead to
    # 2e-4 deg residual); the local zero at design is within 0.1 mm of it.
    assert abs(zero) < 1e-4, zero
    assert abs(bs[5]) < 0.02 * abs(bs[0])


def test_hardpoint_sweep_stops_at_limit_box(mbs, model):
    p = model.hp["tie_rod_inner"]
    model.set_limit("tie_rod_inner", p - 0.002, p + 0.002)
    offsets = [0.0, 0.001, 0.002, 0.003, 0.004]
    out = sensitivity.hardpoint_sweep(mbs, model, "tie_rod_inner", 2, offsets)
    assert not out["completed"]
    assert len(out["offset"]) == 3
    assert "tie_rod_inner" in out["message"] and "0.003" in out["message"]
    for key in SUMMARY:
        assert len(out[key]) == 3


def test_hardpoint_sweep_validates_axis(mbs, model):
    with pytest.raises(ValueError, match="axis"):
        sensitivity.hardpoint_sweep(mbs, model, "tie_rod_inner", 3, [0.0])


# ---------------------------------------------------------------- P11.11


def test_profile_heave_only_equals_bump_sweep(mbs, model):
    rows = [(t, 0.0, 0.0) for t in TRAVELS]
    out = axle.profile_sweep(mbs, model, rows)
    assert out["completed"], out["message"]
    left = kc.bump_sweep(mbs, model, TRAVELS)
    right = kc.bump_sweep(mbs, model.mirrored(), TRAVELS)
    # Both solve the same constraints to the engine tolerance 1e-10 m; over
    # the ~0.1 m levers that is ~1e-9 rad.
    np.testing.assert_allclose(out["camber_left"], left["camber"], atol=2e-9)
    np.testing.assert_allclose(out["toe_left"], left["toe"], atol=2e-9)
    np.testing.assert_allclose(out["camber_right"], right["camber"], atol=2e-9)
    np.testing.assert_allclose(out["toe_right"], right["toe"], atol=2e-9)
    np.testing.assert_allclose(out["travel_left"], TRAVELS, atol=1e-15)


def test_profile_rack_only_equals_steer_sweep(mbs, model):
    racks = list(np.linspace(-0.02, 0.02, 9))
    out = axle.profile_sweep(mbs, model, [(0.0, 0.0, r) for r in racks],
                             with_arb=False)
    assert out["completed"], out["message"]
    left = kc.steer_sweep(mbs, model, racks)
    # A common rack moves both inner tie rod points by the same global
    # displacement; the mirrored right corner has its rack axis along -y.
    right = kc.steer_sweep(mbs, model.mirrored(), [-r for r in racks])
    np.testing.assert_allclose(out["toe_left"], left["toe"], atol=2e-9)
    np.testing.assert_allclose(out["camber_left"], left["camber"], atol=2e-9)
    np.testing.assert_allclose(out["toe_right"], right["toe"], atol=2e-9)
    np.testing.assert_allclose(out["camber_right"], right["camber"], atol=2e-9)
    # Both wheels steer the same way: toe-in on one side is toe-out on the
    # other.
    assert out["toe_left"][-1] * out["toe_right"][-1] < 0


def test_profile_roll_only_equals_roll_sweep(mbs, model):
    svals = list(np.linspace(-0.03, 0.03, 7))
    half_track = abs(model.hp["wheel_center"][1])
    rolls = [math.atan2(s, half_track) for s in svals]
    out = axle.profile_sweep(mbs, model, [(0.0, phi, 0.0) for phi in rolls])
    assert out["completed"], out["message"]
    rs = axle.roll_sweep(mbs, model, svals)
    np.testing.assert_allclose(out["roll"], rs["roll"], atol=1e-15)
    raw = rs["raw"]
    for side, num in (("left", rs["axle"]["left"].nums["upright"]),
                      ("right", rs["axle"]["right"].nums["upright"])):
        m = rs["axle"]["models"][side]
        for k in range(len(svals)):
            wg = mt.wheel_geometry(raw["positions"][k, num],
                                   mt.rot(raw["quaternions"][k, num]),
                                   m.spin_axis, m.wheel_radius, m.side)
            assert out["camber_" + side][k] == pytest.approx(wg["camber"],
                                                             abs=2e-9)
            assert out["toe_" + side][k] == pytest.approx(wg["toe"], abs=2e-9)
    np.testing.assert_allclose(out["travel_left"], svals, atol=1e-15)
    np.testing.assert_allclose(out["travel_right"], [-s for s in svals],
                               atol=1e-15)


def test_profile_combined_row_equals_direct_solve(mbs, model):
    rows = [(0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.015, 1.0 * DEG, 0.004),
            (0.02, 1.5 * DEG, 0.008)]
    out = axle.profile_sweep(mbs, model, rows)
    assert out["completed"], out["message"]
    heave, roll, rack = rows[-1]
    s, ax = axle.kc_axle(mbs, model, True)
    z0 = ax["z0"]
    sroll = abs(model.hp["wheel_center"][1]) * math.tan(roll)
    drivers = [ax["left"].travel, ax["right"].travel, ax["left"].steer,
               ax["right"].steer]
    direct = result_arrays(s.sweep(
        drivers, [[z0 + heave + sroll, z0 + heave - sroll, rack, -rack]]))
    assert direct["completed"]
    for side in ("left", "right"):
        num = ax[side].nums["upright"]
        np.testing.assert_allclose(out["positions"][-1, num],
                                   direct["positions"][0, num], atol=1e-9)
        m = ax["models"][side]
        wg = mt.wheel_geometry(direct["positions"][0, num],
                               mt.rot(direct["quaternions"][0, num]),
                               m.spin_axis, m.wheel_radius, m.side)
        assert out["camber_" + side][-1] == pytest.approx(wg["camber"],
                                                          abs=2e-9)
        assert out["toe_" + side][-1] == pytest.approx(wg["toe"], abs=2e-9)
        # Independent reference solver of each corner.
        sign = 1.0 if side == "left" else -1.0
        r = ReferenceCorner(m).metrics(z0 + heave + sign * sroll, sign * rack)
        assert out["camber_" + side][-1] == pytest.approx(r["camber"],
                                                          abs=2e-9)
        assert out["toe_" + side][-1] == pytest.approx(r["toe"], abs=2e-9)


def test_profile_reports_failure(mbs, model):
    out = axle.profile_sweep(mbs, model, [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)])
    assert not out["completed"]
    assert len(out["heave"]) == 1
    assert out["message"]


def test_read_profile_csv(tmp_path):
    p = tmp_path / "profile.csv"
    p.write_text("heave_mm,roll_deg,rack_mm\n0,0,0\n10,1.5,-4\n\n")
    rows = axle.read_profile_csv(p)
    assert len(rows) == 2
    np.testing.assert_allclose(rows[1], [0.01, 1.5 * DEG, -0.004], rtol=1e-15)
    q = tmp_path / "bad.csv"
    q.write_text("heave_mm,pitch_deg\n0,0\n")
    with pytest.raises(ValueError, match="pitch_deg"):
        axle.read_profile_csv(q)
    r = tmp_path / "text.csv"
    r.write_text("heave_mm,roll_deg,rack_mm\n0,a,0\n")
    with pytest.raises(ValueError, match="line 2"):
        axle.read_profile_csv(r)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
