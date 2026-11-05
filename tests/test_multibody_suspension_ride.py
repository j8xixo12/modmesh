# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Ride comfort and road holding (``suspension.ride``), validated against
published data.

* The ISO 2631-1 weightings Wk and Wd equal the tabulated 1/3-octave values
  (``tests/data/multibody/iso2631_weightings``).
* The ISO 8608 classes: the n0 = 0.1 cycles/m values and the Omega0 = 1
  rad/m values of the standard follow from each other; a generated profile
  has the variance of its PSD and the PSD itself.
* The 2-DOF light vehicle of Ribeiro & Cunha (arXiv 2605.05235, 2026) on
  classes B, C, and D at 40 m/s: the tire load ratio of the paper, in the
  linear theory and in an engine model whose tire leaves the road.
* A library corner: the engine quarter car on a random road equals the
  linear model on the same input record, and the record statistics are
  near the random vibration theory.
"""

import json
import math
import os

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.suspension import ride
from solvcon.multibody.suspension.model import CornerModel

DATA = os.path.join(os.path.dirname(__file__), "data", "multibody")
Z = [0.0, 0.0, 0.0]
EZ = [0.0, 0.0, 1.0]


def _load(name):
    with open(os.path.join(DATA, name, "published.json")) as fobj:
        return json.load(fobj)


@pytest.mark.parametrize("kind", ["k", "d"])
def test_weighting_equals_the_iso_2631_table(kind):
    pub = _load("iso2631_weightings")
    f = np.array(pub["frequency_hz"])
    db = 20.0 * np.log10(np.abs(ride.weighting(f, kind)))
    # The table rounds to 0.01 dB.
    np.testing.assert_allclose(db, pub[f"W{kind}_db"], atol=0.006)


def test_weighted_rms_of_a_sine():
    # A 1 m/s^2 sine at 1 Hz reads 0.482 x its RMS through Wk (ISO 2631-1
    # Table 3 lists 482 / 1000 at 1 Hz).
    t = np.arange(0.0, 64.0, 1e-3)
    a = np.sin(2.0 * math.pi * 1.0 * t)
    a_w, a_rms = ride.weighted_rms(t, a, 'k', settle=4.0)
    assert a_rms == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-3)
    assert a_w / a_rms == pytest.approx(10.0 ** (-6.33 / 20.0), rel=2e-3)


def test_comfort_reactions():
    assert ride.comfort_reaction(0.2) == "not uncomfortable"
    assert ride.comfort_reaction(0.4) == "a little uncomfortable"
    # The ranges of ISO 2631-1 Annex C overlap.
    assert ride.comfort_reaction(0.55) == \
        "a little uncomfortable to fairly uncomfortable"
    assert ride.comfort_reaction(3.0) == "extremely uncomfortable"


def test_iso_8608_classes_in_both_frequency_forms():
    pub = _load("ribeiro_cunha_2026")["iso_8608_gd_n0_1e-6_m3"]
    for name, value in pub.items():
        assert ride.road_psd(ride.N0, name) == pytest.approx(value * 1e-6)
    # ISO 8608 also states G_d(Omega0) at 1 rad/m: A 1, B 4, C 16, D 64,
    # E 256 (1e-6 m^3), rounded; some papers quote these as G_d(n0).
    angular = {"A": 1.0, "B": 4.0, "C": 16.0, "D": 64.0, "E": 256.0}
    for name, value in angular.items():
        assert ride.road_psd_angular(1.0, name) == pytest.approx(
            value * 1e-6, rel=0.01)
    with pytest.raises(ValueError):
        ride.road_psd(0.1, "J")


def test_random_profile_has_its_psd():
    band = (0.05, 2.0)
    prof = ride.random_profile("C", 2000.0, seed=11, band=band, count=4000,
                               dx=0.02)
    n = np.linspace(band[0], band[1], 100001)
    variance = np.trapezoid(ride.road_psd(n, "C"), n)
    assert np.var(prof.z) == pytest.approx(variance, rel=0.05)
    half = ride.random_profile("C", 2000.0, seed=11, band=band, count=4000,
                               dx=0.02, convention="half")
    assert np.var(half.z) == pytest.approx(0.5 * np.var(prof.z), rel=1e-9)
    # The one-sided PSD of the profile (periodogram averaged over octave
    # bands) follows G_d(n).
    z = prof.z - np.mean(prof.z)
    spec = np.abs(np.fft.rfft(z)) ** 2 * 2.0 * 0.02 / z.size
    freq = np.fft.rfftfreq(z.size, 0.02)
    for lo in (0.1, 0.2, 0.4, 0.8):
        sel = (freq >= lo) & (freq < 2.0 * lo)
        measured = np.mean(spec[sel] * freq[sel] ** 2)
        expect = ride.road_psd(1.0, "C")  # G_d(n) n^2 is constant
        assert measured == pytest.approx(expect, rel=0.15)


def _benchmark_car():
    pub = _load("ribeiro_cunha_2026")
    veh = pub["vehicle"]
    ks = veh["ms"] * (2.0 * math.pi * veh["fn"]) ** 2
    cs = veh["zeta"] * 2.0 * math.sqrt(ks * veh["ms"])
    return pub, veh, ks, cs


@pytest.mark.parametrize("road_class", ["B", "C"])
def test_benchmark_tire_load_ratio_linear(road_class):
    pub, veh, ks, cs = _benchmark_car()
    car = ride.TwoDofQuarterCar(veh["ms"], veh["mu"], ks, cs, veh["kt"])
    out = car.random_response(road_class, veh["speed"], convention="half")
    assert out["tire_load_ratio"] == pytest.approx(
        pub["tire_load_ratio"][road_class], rel=0.05)


@pytest.mark.parametrize("road_class", ["B", "C"])
def test_benchmark_minimum_tire_load_ratio(road_class):
    # Over symmetric damping ratios: the paper's optimum (over bump and
    # rebound damping) is 0.140 on B and about 0.26 on C.
    pub, veh, ks, _ = _benchmark_car()
    best = min(
        ride.TwoDofQuarterCar(
            veh["ms"], veh["mu"], ks, z * 2.0 * math.sqrt(ks * veh["ms"]),
            veh["kt"]).random_response(road_class, veh["speed"],
                                       convention="half")["tire_load_ratio"]
        for z in np.linspace(0.1, 1.0, 91))
    assert best == pytest.approx(
        pub["minimum_tire_load_ratio"][road_class], rel=0.06)


def _two_dof_engine(mbs, ms, mu, ks, cs, kt, road_input, radius=0.3,
                    gravity=9.81):
    """The 2-DOF quarter car in the engine: two bodies on vertical sliders,
    the suspension spring-damper, and the one-sided tire on the road."""
    deflection = (ms + mu) * gravity / kt
    s = mbs.DynamicsSystem(1e-3)
    ground = mbs.create_ground(0)
    wheel = mbs.create_mobilized_body_quat(
        1, [0, 0, radius], Z, Z, [1, 0, 0, 0], Z, Z, mu, [1, 1, 1],
        [0, 0, -mu * gravity], Z)
    body = mbs.create_mobilized_body_quat(
        2, [0, 0, radius + 0.4], Z, Z, [1, 0, 0, 0], Z, Z, ms, [1, 1, 1],
        [0, 0, -ms * gravity], Z)
    for item in (ground, wheel, body):
        s.Add(item)
    s.Add(mbs.create_joint(2, Z, Z, Z, Z, EZ, EZ, ground, wheel))
    s.Add(mbs.create_joint(2, Z, Z, Z, Z, EZ, EZ, ground, body))
    s.Add(mbs.create_spring_damper(wheel, body, Z, Z, ks, cs,
                                   0.4 + ms * gravity / ks))
    height, slope = road_input.drive_functions(mbs)
    tire = mbs.create_tire(ground, wheel, [0, 1, 0], radius + deflection,
                           kt, 0.0, road_height=height, road_slope=slope)
    s.Add(tire)
    s.set_integrator("rk45")
    s.set_tolerances(1e-7, 1e-10)
    s.init()
    return s, tire


@pytest.mark.parametrize("road_class", ["B", "C", "D"])
def test_benchmark_tire_load_ratio_engine(road_class):
    # The engine tire carries no tension: on class D it leaves the road and
    # the tire load ratio drops below the linear value, as in the paper
    # (0.53 against 0.57).
    pub, veh, ks, cs = _benchmark_car()
    length, speed = veh["road_length"], veh["speed"]
    prof = ride.random_profile(road_class, length + 10.0, seed=7,
                               convention="half")
    duration = length / speed
    s, tire = _two_dof_engine(multibody, veh["ms"], veh["mu"], ks, cs,
                              veh["kt"],
                              ride.point_road_input(prof, speed, duration))
    force = []
    for _ in range(int(duration / 1e-3) - 2):
        s.solve()
        force.append(multibody.get_tire_state(tire)["normal_force"])
    force = np.array(force[2000:])
    ratio = np.std(force) / ((veh["ms"] + veh["mu"]) * 9.81)
    assert ratio == pytest.approx(pub["tire_load_ratio"][road_class],
                                  rel=0.07)
    linear = ride.TwoDofQuarterCar(veh["ms"], veh["mu"], ks, cs, veh["kt"])
    lin = linear.random_response(road_class, speed, convention="half")
    if "D" == road_class:
        assert np.mean(force <= 0.0) > 0.01
        assert ratio < lin["tire_load_ratio"]
    elif "C" == road_class:
        # "a limiting condition where the contact force approaches zero":
        # a 3.4 sigma excursion, a few samples in 1e4.
        assert np.mean(force <= 0.0) < 2e-3
        assert ratio == pytest.approx(lin["tire_load_ratio"], rel=0.03)
    else:
        assert np.all(force > 0.0)
        assert ratio == pytest.approx(lin["tire_load_ratio"], rel=0.03)


def test_corner_engine_equals_linear_model_on_the_same_record():
    model = CornerModel.load("macpherson_front")
    mass = model.raw["vehicle"]["corner_mass"]
    speed = 60.0 / 3.6
    prof = ride.random_profile("C", speed * 12.0 + 5.0, seed=3)
    run = ride.simulate(multibody, model, mass,
                        ride.point_road_input(prof, speed, 12.0),
                        settle=2.0)
    assert run["airborne"] == 0.0
    assert len(run["poses"]) == len(run["pose_times"])
    assert {"chassis", "upright", "road"} <= set(run["poses"][0])
    lin = ride.LinearCorner(multibody, model, mass)
    # The linear model driven by the recorded road (frequency domain).
    t, y = run["time"], run["road"]
    spec = np.fft.rfft(y - np.mean(y))
    freq = np.maximum(np.fft.rfftfreq(y.size, t[1] - t[0]), 1e-6)
    resp = lin.response(freq)
    keep = t >= 2.0
    same = {key: np.fft.irfft(spec * resp[key], y.size) for key in resp}
    assert run["body_acc_rms"] == pytest.approx(
        np.std(same["body_acc"][keep]), rel=0.03)
    assert run["body_acc_w"] == pytest.approx(
        ride.weighted_rms(t, same["body_acc"], "k", 2.0)[0], rel=0.03)
    assert run["tire_load_ratio"] == pytest.approx(
        np.std(same["tire_force"][keep]) / lin.static_load, rel=0.05)
    assert run["travel_std"] == pytest.approx(
        np.std(same["travel"][keep]), rel=0.03)
    # One 10 s record against the ensemble of the random vibration theory.
    theory = lin.random_response("C", speed)
    assert run["body_acc_w"] == pytest.approx(theory["body_acc_w"],
                                              rel=0.15)
    assert run["tire_load_ratio"] == pytest.approx(
        theory["tire_load_ratio"], rel=0.15)


@pytest.mark.parametrize("stem", [
    "race_front_corner", "macpherson_front", "fsae_front_pullrod",
    "five_link_rear", "semi_trailing_rear", "virtual_kingpin_front"])
def test_linear_corner_modes_equal_the_engine_modes(stem):
    # The damped modes of the linear corner (eigenvalues of the first
    # order form) are the engine modal analysis of the quarter car at its
    # static equilibrium (ride and wheel hop: frequency and damping ratio).
    from solvcon.multibody.suspension import vehicle
    model = CornerModel.load(stem)
    mass = model.raw.get("vehicle", {}).get("corner_mass") or 200.0
    lin = ride.LinearCorner(multibody, model, mass)
    n = lin.M.shape[0]
    first = np.block([[np.zeros((n, n)), np.eye(n)],
                      [-np.linalg.solve(lin.M, lin.K),
                       -np.linalg.solve(lin.M, lin.C)]])
    lam = np.linalg.eigvals(first)
    lam = lam[lam.imag > 0.0]
    lam = lam[np.argsort(np.abs(lam))]
    s, _ = vehicle.quarter_car(multibody, model, mass)
    s.find_equilibrium(tol=1e-9)
    modal = multibody.modal_analysis(s)
    freq = np.asarray(modal.frequencies)
    zeta = np.asarray(modal.damping_ratios)
    order = np.argsort(freq)
    np.testing.assert_allclose(np.abs(lam) / (2.0 * math.pi), freq[order],
                               rtol=1e-3)
    np.testing.assert_allclose(-lam.real / np.abs(lam), zeta[order],
                               rtol=1e-3)
    modes = lin.modes()
    np.testing.assert_allclose([m[0] for m in modes], freq[order],
                               rtol=1e-3)
    assert all(abs(abs(m[2]).max() - 1.0) < 1e-12 for m in modes)
