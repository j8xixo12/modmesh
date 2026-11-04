# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for the road input of the dynamics.

Rules under test:

* Engine: after find_equilibrium a driver that moves at t = 0 has its
  velocity, so RK4 follows it without a Baumgarte transient; the tire pushes
  along the road normal only in compression; a dropped wheel bounces with
  the analytic contact time.
* Enveloping (tandem cam, Schmeitz 2004): a flat road gives a zero
  effective road; a road much longer than the tire is followed; a narrow
  cleat is enveloped to the analytic height; a step rises monotonically
  between its reach limits; a symmetric bump gives a symmetric height and an
  antisymmetric slope.
* Quarter car: it starts at rest at the design pose; on a slow rig it
  follows the plate from t = 0; at low speed the wheel follows the effective
  road of a bump; the tire force tilts with the effective slope (pushed back
  climbing, forward descending); a pothole at speed lifts the tire off the
  road and the force is zero while it is off.
* Travel stops: a length stop has no force between its limits, pushes the
  points apart below the bump limit and pulls them together above the droop
  limit with k d + k2 d^2 + c dd/dt, and never pulls on the bump side.  On
  an 80 mm stone the quarter car stays in its travel range (before the
  stops, the pushrod rocker went over center near +90 mm and the linkage
  locked near +148 mm), and in the air the suspension extends.
"""

import math
import os
import tempfile

import numpy as np
import pytest

from solvcon import multibody
from solvcon.multibody.suspension import road, vehicle
from solvcon.multibody.suspension.model import CornerModel

Z = [0.0, 0.0, 0.0]
EZ = [0.0, 0.0, 1.0]


@pytest.fixture
def mbs():
    """The engine, or skip when solvcon was built without it.

    :return: the engine module
    :rtype: module
    """
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon built without the multibody engine")
    return multibody


@pytest.fixture
def model():
    return CornerModel.load("race_front_corner")


# The engine.


def _driven_pair(mbs, amplitude, omega, dt=1e-3):
    s = mbs.DynamicsSystem(dt)
    g = mbs.create_ground(0)
    s.Add(g)
    plate = mbs.create_mobilized_body_quat(1, Z, Z, Z, [1, 0, 0, 0], Z, Z,
                                           1.0, [0.1, 0.1, 0.1], Z, Z)
    mass = mbs.create_mobilized_body_quat(2, [0.0, 0.0, 0.5], Z, Z,
                                          [1, 0, 0, 0], Z, Z, 10.0,
                                          [0.1, 0.1, 0.1],
                                          [0.0, 0.0, -98.1], Z)
    s.Add(plate)
    s.Add(mass)
    s.Add(mbs.create_joint(2, Z, Z, Z, Z, EZ, EZ, g, plate))
    s.Add(mbs.create_joint(2, [0.0, 0.0, 0.5], Z, Z, Z, EZ, EZ, g, mass))
    s.Add(mbs.create_absolute_driver(
        body=plate, ground=g, point=Z, direction=EZ,
        func=mbs.DriveFunction.sine(amplitude=amplitude, omega=omega)))
    s.Add(mbs.create_spring_damper(plate, mass, Z, Z, 1.0e4, 50.0, 0.5))
    s.init()
    return s


def test_equilibrium_start_is_velocity_consistent(mbs):
    amplitude, omega = 0.005, 2.0 * math.pi * 3.0
    s = _driven_pair(mbs, amplitude, omega)
    assert s.find_equilibrium(tol=1e-10)["converged"]
    ev = s.evaluate_constraints(s.get_sim_time())
    vel = np.asarray(s.get_body_velocities())
    qdot = []
    for i in range(3):
        qdot += list(vel[i]) + [0.0] * 4
    # Only translations move here; the p_dot entries stay zero.
    residual = np.asarray(ev["Cq"]) @ np.array(qdot) + np.asarray(ev["Ct"])
    assert np.abs(residual).max() < 1e-12
    assert vel[1][2] == pytest.approx(amplitude * omega, rel=1e-12)
    assert vel[2] == pytest.approx([0.0, 0.0, 0.0], abs=1e-14)
    # RK4 follows the driver (before the fix: a 3.47 mm Baumgarte
    # transient, the initial velocity error A omega decaying at 10 1/s).
    err = 0.0
    for _ in range(500):
        s.solve()
        z = np.asarray(s.get_body_positions())[1][2]
        err = max(err, abs(z - amplitude * math.sin(omega * s.get_sim_time())))
    assert err < 1e-6


def _wheel(mbs, z0, mass=20.0, dt=1e-4):
    s = mbs.DynamicsSystem(dt)
    g = mbs.create_ground(0)
    s.Add(g)
    wheel = mbs.create_mobilized_body_quat(
        1, [0.0, 0.0, z0], Z, Z, [1, 0, 0, 0], Z, Z, mass, [0.5, 0.8, 0.5],
        [0.0, 0.0, -9.81 * mass], Z)
    s.Add(wheel)
    s.Add(mbs.create_joint(2, [0.0, 0.0, z0], Z, Z, Z, EZ, EZ, g, wheel))
    return s, g, wheel


def test_tire_force_only_in_compression(mbs):
    radius, k = 0.3, 2.0e5
    s, g, wheel = _wheel(mbs, 0.31)
    tire = mbs.create_tire(g, wheel, [0.0, 1.0, 0.0], radius, k)
    s.Add(tire)
    s.init()
    state = mbs.get_tire_state(tire)
    assert not state["in_contact"]
    assert state["normal_force"] == 0.0
    assert state["deflection"] == pytest.approx(-0.01, abs=1e-12)
    s2, g2, wheel2 = _wheel(mbs, 0.299)
    tire2 = mbs.create_tire(g2, wheel2, [0.0, 1.0, 0.0], radius, k)
    s2.Add(tire2)
    s2.init()
    assert s2.find_equilibrium(tol=1e-10)["converged"]
    st = mbs.get_tire_state(tire2)
    assert st["normal_force"] == pytest.approx(20.0 * 9.81, rel=1e-9)
    assert st["deflection"] == pytest.approx(20.0 * 9.81 / k, rel=1e-9)


def test_tire_on_a_moving_sloped_road(mbs):
    radius, k, b, h = 0.3, 2.0e5, math.radians(10.0), 0.004
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    s.Add(g)
    wheel = mbs.create_mobilized_body_quat(1, [0.0, 0.0, 0.3], Z, Z,
                                           [1, 0, 0, 0], Z, Z, 20.0,
                                           [0.5, 0.8, 0.5], Z, Z)
    s.Add(wheel)
    s.Add(mbs.create_joint(2, [0.0, 0.0, 0.3], Z, Z, Z, EZ, EZ, g, wheel))
    s.Add(mbs.create_absolute_driver(
        body=wheel, ground=g, point=Z, direction=EZ,
        func=mbs.DriveFunction.parameter(0.3)))
    tire = mbs.create_tire(g, wheel, [0.0, 1.0, 0.0], radius, k,
                           road_height=mbs.DriveFunction.parameter(h),
                           road_slope=mbs.DriveFunction.parameter(b))
    s.Add(tire)
    s.init()
    loads = mbs.get_force_element_loads(tire)
    state = mbs.get_tire_state(tire)
    n = np.array([-math.sin(b), 0.0, math.cos(b)])
    contact = np.array([0.0, 0.0, 0.3]) - radius * n
    expected = float(np.dot(np.array([0.0, 0.0, h]) - contact, n))
    assert state["deflection"] == pytest.approx(expected, rel=1e-12)
    force = np.asarray(loads["force_j"])
    np.testing.assert_allclose(force, k * expected * n, rtol=1e-12)
    assert force[0] < 0.0  # a rising road pushes the wheel back
    np.testing.assert_allclose(np.asarray(loads["force_i"]), -force,
                               rtol=1e-15)


def test_drop_bounce_contact_time(mbs):
    # A 20 kg wheel falls 5 mm onto the road on the tire spring alone. With
    # the compression x: x'' = -omega^2 x + g from x = 0, x' = |v_impact|;
    # contact ends when x returns to 0, after (2 / omega)(pi - atan(b / a))
    # with a = g / omega^2, b = |v_impact| / omega.
    mass, radius, k = 20.0, 0.3, 2.0e5
    s, g, wheel = _wheel(mbs, radius + 0.005, mass=mass, dt=2e-5)
    s.set_integrator("rk45")
    s.set_tolerances(1e-10, 1e-12)
    tire = mbs.create_tire(g, wheel, [0.0, 1.0, 0.0], radius, k)
    s.Add(tire)
    s.init()
    v_impact = math.sqrt(2.0 * 9.81 * 0.005)
    touch = leave = v_leave = None
    for _ in range(6000):
        s.solve()
        state = mbs.get_tire_state(tire)
        if state["in_contact"] and touch is None:
            touch = s.get_sim_time()
        if touch is not None and not state["in_contact"]:
            leave = s.get_sim_time()
            v_leave = np.asarray(s.get_body_velocities())[1][2]
            break
    assert touch is not None and leave is not None
    omega = math.sqrt(k / mass)
    duration = (2.0 / omega) * (math.pi - math.atan(
        (v_impact / omega) / (9.81 / omega ** 2)))
    # The 2e-5 s output step brackets each event, so 4e-5 s bounds it.
    assert leave - touch == pytest.approx(duration, abs=4e-5)
    # Undamped: it leaves with the impact speed, reversed.
    assert v_leave == pytest.approx(v_impact, rel=2e-3)


# Enveloping.


def _cam():
    return road.TandemCam.for_tire(0.33, 0.008)


def test_flat_road_has_a_zero_effective_road():
    cam = _cam()
    w, beta = cam.effective(road.RoadProfile.flat(), np.linspace(-1, 1, 21))
    np.testing.assert_allclose(w, 0.0, atol=1e-15)
    np.testing.assert_allclose(beta, 0.0, atol=1e-15)


def test_long_road_waves_are_followed():
    # On a wave much longer than the tire, each cam center sits b_e above
    # the road under it, to an error e (the c_e < 2 cam touches with its
    # pointed bottom, so e is tiny).  The effective height is then the mean
    # of the road at the two cams within e, and the slope the chord slope
    # over l_s within 2 e / l_s.  The profile is sampled densely so its
    # linear interpolation (A (1 - cos(pi / n)) per sample) stays below
    # 1e-7 m.
    cam = _cam()
    amp, wavelength = 0.01, 20.0
    profile = road.RoadProfile.sine(amp, wavelength, cycles=2,
                                    n_per_cycle=1024)
    x = np.linspace(5.0, 35.0, 61)
    w, beta = cam.effective(profile, x)
    front, rear = x + 0.5 * cam.l_s, x - 0.5 * cam.l_s
    e = max(np.abs(cam.cam_height(profile, p) - cam.b_e
                   - profile.height(p)).max() for p in (front, rear))
    mean = 0.5 * (profile.height(front) + profile.height(rear))
    assert np.abs(w - mean).max() <= e + 1e-15
    chord = np.arctan((profile.height(front) - profile.height(rear))
                      / cam.l_s)
    assert np.abs(beta - chord).max() <= 2.0 * e / cam.l_s + 1e-15
    # The chord mean attenuates the wave by 1 - cos(k l_s / 2), 1.7e-4 here.
    exact = amp * np.sin(2 * math.pi * x / wavelength)
    k = 2 * math.pi / wavelength
    bound = amp * (1.0 - math.cos(0.5 * k * cam.l_s)) + e + 1e-7
    assert np.abs(w - exact).max() <= bound


def test_narrow_cleat_is_enveloped():
    # A 10 mm cleat of 10 mm height centered under the wheel: each cam
    # touches the cleat corner nearest to it, at |x| = l_s / 2 - width / 2,
    # so w = h + z_e(l_s / 2 - width / 2) - b_e (when that beats the flat
    # road under the cam, b_e).
    cam = _cam()
    h, width = 0.01, 0.01
    profile = road.RoadProfile("cleat", [-0.5, -width / 2 - 1e-9,
                                         -width / 2, width / 2,
                                         width / 2 + 1e-9, 0.5],
                               [0.0, 0.0, h, h, 0.0, 0.0])
    w, beta = cam.effective(profile, [0.0])
    corner = 0.5 * cam.l_s - 0.5 * width
    expected = max(cam.b_e, h + float(cam.cam_offset(corner))) - cam.b_e
    # The scan step (1 mm) misses the exact corner by at most half a step;
    # z_e' there is below 0.1, so 1e-4 m bounds it.
    assert w[0] == pytest.approx(expected, abs=1e-4)
    assert w[0] < h  # enveloped: lower than the cleat
    assert beta[0] == pytest.approx(0.0, abs=1e-12)


def test_step_rises_monotonically_within_its_reach():
    cam = _cam()
    h = 0.02
    profile = road.RoadProfile.step(h)
    x = np.linspace(-1.0, 1.0, 401)
    w, beta = cam.effective(profile, x)
    assert np.all(np.diff(w) >= -1e-15)
    far = np.abs(x) > cam.reach + 1e-3
    np.testing.assert_allclose(w[far & (x < 0)], 0.0, atol=1e-15)
    np.testing.assert_allclose(w[far & (x > 0)], h, atol=1e-15)
    assert np.all(beta >= -1e-15)
    assert beta.max() > 0.0


def test_symmetric_bump_gives_symmetric_effective_road():
    cam = _cam()
    profile = road.RoadProfile.bump(0.02, 0.2)
    x = 0.1 + np.linspace(-0.6, 0.6, 121)
    w, beta = cam.effective(profile, x)
    np.testing.assert_allclose(w, w[::-1], atol=1e-12)
    np.testing.assert_allclose(beta, -beta[::-1], atol=1e-12)


def test_profile_csv():
    fd, path = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    try:
        with open(path, 'w') as fobj:
            fobj.write("# measured\nx_mm,z_mm\n0,0\n100,5\n200,0\n")
        profile = road.RoadProfile.from_csv(path)
        np.testing.assert_allclose(profile.height([0.05, 0.1]),
                                   [0.0025, 0.005])
        with open(path, 'w') as fobj:
            fobj.write("t,z\n0,0\n")
        with pytest.raises(ValueError, match="x_mm,z_mm"):
            road.RoadProfile.from_csv(path)
    finally:
        os.remove(path)


# The quarter car.


def _run(mbs, model, road_input, ms=200.0, dt=1e-3):
    s, hd = vehicle.quarter_car(mbs, model, ms, road=road_input, dt=dt)
    assert s.find_equilibrium(tol=1e-9)["converged"]
    s.set_integrator("rk45")
    s.set_tolerances(1e-7, 1e-10)
    up = hd["corner"].nums["upright"]
    rows = []
    n = int(round(road_input.duration / dt)) - 1
    for _ in range(n):
        s.solve()
        pos = np.asarray(s.get_body_positions())
        state = mbs.get_tire_state(hd["tire"])
        loads = mbs.get_force_element_loads(hd["tire"])
        t = s.get_sim_time()
        h, b, _ = road_input.at(t)
        rows.append((t, pos[1][2], pos[up][2] - model.hp["wheel_center"][2],
                     h, b, state["normal_force"], state["in_contact"],
                     np.asarray(loads["force_j"])[0]))
    return np.array(rows, dtype='float64'), hd


def test_quarter_car_starts_at_rest_on_a_flat_road(mbs, model):
    flat = road.road_input(road.RoadProfile.flat(), 10.0, _cam(),
                           duration=0.3)
    rows, hd = _run(mbs, model, flat)
    # Equilibrium at the design pose: nothing moves.
    assert np.abs(rows[:, 1]).max() < 1e-8
    assert np.abs(rows[:, 2]).max() < 1e-8
    np.testing.assert_allclose(rows[:, 5], hd["weight"], rtol=1e-8)


def _linear_reference(mbs, hd, ms, rin, times):
    """Body and wheel center heights of the exact linear quarter car
    (suspension.vehicle.linear_quarter_car, built from K&C sensitivities)
    driven by the effective road of ``rin``, at ``times``."""
    from solvcon.multibody.suspension import kc
    m = hd["model"]
    kw = kc.bump_sweep(mbs, m, [-1e-4, 0.0, 1e-4])["wheel_rate"][1]
    lin = vehicle.linear_quarter_car(mbs, m, ms, kw, m.tire_k,
                                     tire_load=hd["weight"])
    a, a_x, weight = lin["a"], lin["a_x"], hd["weight"]
    cw = m.spring_c * hd["motion_ratio"] ** 2
    cc = np.array([[m.tire_c, m.tire_c * a],
                   [m.tire_c * a, cw + m.tire_c * a * a]])
    hfn, bfn = rin.drive_functions(mbs)
    minv = np.linalg.inv(lin["M"])
    # The deflection normal to the tilted plane is R - (z_w - w) cos(beta):
    # with z_w - w close to the loaded radius R_l, a tilt adds
    # R_l (1 - cos(beta)) of compression, second order in beta but large
    # against a small w.  The forward force component, to first order
    # -W beta, loads the travel through the contact forward rate a_x
    # (anti-dive geometry).
    r_load = m.wheel_radius

    def deriv(t, y):
        w, wd, _ = hfn.eval(t)
        beta, betad, _ = bfn.eval(t)
        w_eff = w + r_load * (1.0 - math.cos(beta))
        wd_eff = wd + r_load * math.sin(beta) * betad
        f = np.array([1.0, a]) * (m.tire_k * w_eff + m.tire_c * wd_eff)
        f = f + np.array([0.0, a_x]) * (-weight * beta)
        acc = minv @ (f - cc @ y[2:] - lin["K"] @ y[:2])
        return np.concatenate([y[2:], acc])

    # RK4 at 1e-4 s: the wheel hop period is about 0.04 s.
    y, t, h = np.zeros(4), 0.0, 1e-4
    ref = {}
    for _ in range(int(round(times[-1] / h))):
        k1 = deriv(t, y)
        k2 = deriv(t + h / 2, y + h / 2 * k1)
        k3 = deriv(t + h / 2, y + h / 2 * k2)
        k4 = deriv(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        ref[round(t, 6)] = (y[0], y[0] + y[1])
    body = np.array([ref[round(t, 6)][0] for t in times])
    wheel = np.array([ref[round(t, 6)][1] for t in times])
    return body, wheel


def test_slow_rig_is_followed_from_the_start(mbs, model):
    # 0.1 Hz is far below the ride frequency (2.8 Hz): the body and the
    # wheel follow the plate, which starts at rest (faded in), lagging only
    # by the damper (c_w h' / k_w, about 0.08 mm here).  The engine equals
    # the linear quarter car within 1 % of the 5 mm amplitude.
    ms = 200.0
    rig = road.rig_input(0.005, 0.1, 2.0)
    rows, hd = _run(mbs, model, rig, ms=ms, dt=2e-3)
    body, wheel = _linear_reference(mbs, hd, ms, rig, rows[:, 0])
    assert np.abs(rows[:, 1] - body).max() < 0.01 * 0.005
    assert np.abs(rows[:, 2] - wheel).max() < 0.01 * 0.005
    assert np.abs(rows[:, 1] - rows[:, 3]).max() < 0.2e-3


def test_small_bump_matches_the_linear_quarter_car(mbs, model):
    # A 3 mm bump at 10 km/h keeps the tire on the road, so the contact
    # stays linear.  The engine equals the linear quarter car driven by the
    # same effective road; geometric nonlinearity over the travel and the
    # damper rate MR^2 frozen at design leave 1.8 % (measured), bounded by
    # 3 % of the peak.
    ms = 200.0
    cam = road.TandemCam.for_tire(model.wheel_radius, 0.008)
    bump = road.RoadProfile.bump(0.003, 0.2)
    rin = road.road_input(bump, 10.0 / 3.6, cam, duration=1.2)
    rows, hd = _run(mbs, model, rin, ms=ms, dt=1e-3)
    assert np.all(rows[:, 6] > 0.5)
    body, wheel = _linear_reference(mbs, hd, ms, rin, rows[:, 0])
    assert np.abs(rows[:, 1] - body).max() < 0.03 * np.abs(body).max()
    assert np.abs(rows[:, 2] - wheel).max() < 0.03 * np.abs(wheel).max()


def test_bump_tilts_the_tire_force(mbs, model):
    cam = road.TandemCam.for_tire(model.wheel_radius, 0.008)
    bump = road.RoadProfile.bump(0.03, 0.15)
    fast = road.road_input(bump, 30.0 / 3.6, cam)
    rows, _ = _run(mbs, model, fast, dt=5e-4)
    on = rows[:, 6] > 0.5
    _, bfn = fast.drive_functions(mbs)
    beta = np.array([bfn.eval(t)[0] for t in rows[:, 0]])
    fn, fx = rows[:, 5], rows[:, 7]
    # The force is along the effective road normal: fx = -Fn sin(beta),
    # backward while climbing, forward while descending.
    np.testing.assert_allclose(fx[on], -fn[on] * np.sin(beta[on]),
                               rtol=1e-12, atol=1e-9)
    climbing = on & (beta > 1e-3)
    descending = on & (beta < -1e-3)
    assert climbing.any() and descending.any()
    assert np.all(fx[climbing] < 0.0)
    assert np.all(fx[descending] > 0.0)


def test_pothole_lifts_the_tire_off_the_road(mbs, model):
    cam = road.TandemCam.for_tire(model.wheel_radius, 0.008)
    hole = road.RoadProfile.pothole(0.06, 0.3, edge=0.01)
    fast = road.road_input(hole, 40.0 / 3.6, cam)
    rows, hd = _run(mbs, model, fast, dt=5e-4)
    on = rows[:, 6] > 0.5
    assert not on.all()  # the tire leaves the road
    assert np.all(rows[:, 5] >= 0.0)  # never pulls
    np.testing.assert_allclose(rows[~on, 5], 0.0, atol=0.0)
    # Landing on the far edge loads the tire well above the static load.
    assert rows[:, 5].max() > 2.0 * hd["weight"]


# The travel stops.


def _stop_pair(mbs, length, rate=0.0, **stop):
    """Two free bodies on the x axis at ``length`` apart, body j moving
    away from body i at ``rate``, with a length stop between them."""
    s = mbs.DynamicsSystem(1e-3)
    g = mbs.create_ground(0)
    s.Add(g)
    bi = mbs.create_mobilized_body_quat(1, Z, Z, Z, [1, 0, 0, 0], Z, Z, 1.0,
                                        [0.1, 0.1, 0.1], Z, Z)
    bj = mbs.create_mobilized_body_quat(2, [length, 0.0, 0.0], [rate, 0, 0],
                                        Z, [1, 0, 0, 0], Z, Z, 1.0,
                                        [0.1, 0.1, 0.1], Z, Z)
    s.Add(bi)
    s.Add(bj)
    fe = mbs.create_length_stop(bi, bj, Z, Z, **stop)
    s.Add(fe)
    s.init()
    return fe


@pytest.mark.parametrize("length, rate, engaged, pen, rate_pen", [
    (0.20, 0.0, 0, 0.0, 0.0),  # between the limits
    (0.09, 0.0, -1, 0.01, 0.0),  # bump stop, static
    (0.09, -0.5, -1, 0.01, 0.5),  # bump stop, closing
    (0.32, 0.0, 1, 0.02, 0.0),  # droop stop, static
    (0.32, 0.5, 1, 0.02, 0.5),  # droop stop, opening
])
def test_length_stop_law(mbs, length, rate, engaged, pen, rate_pen):
    k, k2, c = 2.0e5, 2.0e8, 2.0e3
    fe = _stop_pair(mbs, length, rate, min_length=0.1, max_length=0.3, k=k,
                    k2=k2, c=c)
    state = mbs.get_length_stop_state(fe)
    assert state["engaged"] == engaged
    assert state["length"] == pytest.approx(length, rel=1e-14)
    assert state["penetration"] == pytest.approx(pen, abs=1e-14)
    force = k * pen + k2 * pen * pen + c * rate_pen
    assert state["force"] == pytest.approx(force, rel=1e-12, abs=1e-12)
    loads = mbs.get_force_element_loads(fe)
    fj = np.asarray(loads["force_j"])
    # Bump side pushes body j away (+x); droop side pulls it back (-x).
    expected = -engaged * force
    assert fj == pytest.approx([expected, 0.0, 0.0], abs=1e-9)
    assert np.asarray(loads["force_i"]) == pytest.approx(-fj, abs=1e-9)


def test_length_stop_never_pulls(mbs):
    # Leaving the bump stop fast: the damping would exceed the spring term.
    fe = _stop_pair(mbs, 0.099, 2.0, min_length=0.1, max_length=0.3,
                    k=2.0e5, c=2.0e3)
    state = mbs.get_length_stop_state(fe)
    assert state["engaged"] == -1
    assert state["force"] == 0.0
    # One limit only: None disables the other one.
    fe = _stop_pair(mbs, 5.0, 0.0, min_length=0.1, k=2.0e5)
    assert mbs.get_length_stop_state(fe)["engaged"] == 0


def test_length_stop_rejects_bad_limits(mbs):
    with pytest.raises(ValueError, match="min_length"):
        _stop_pair(mbs, 0.2, min_length=0.3, max_length=0.1, k=1.0)
    with pytest.raises(ValueError, match="negative"):
        _stop_pair(mbs, 0.2, min_length=0.1, max_length=0.3, k=-1.0)


def test_stone_stays_in_the_travel_range(mbs, model):
    # An 80 mm stone at 30 km/h throws the car: the bump stop takes the
    # hit before the rocker goes over center, and in the air the wheel
    # hangs down from the body.
    from solvcon.multibody.suspension import kc
    stone = road.RoadProfile.bump(0.08, 0.15)
    cam = road.TandemCam.for_tire(model.wheel_radius,
                                  vehicle.static_tire_deflection(
                                      model, 200.0)[0])
    rin = road.road_input(stone, 30.0 / 3.6, cam, duration=0.5)
    s, hd = vehicle.quarter_car(mbs, model, 200.0, road=rin, dt=1e-3)
    s.set_integrator("rk45")
    s.set_tolerances(1e-7, 1e-10)
    up = hd["corner"].nums["upright"]
    z0 = np.asarray(s.get_body_positions())
    travel, airborne, states = [], [], set()
    for _ in range(int(round(rin.duration / 1e-3)) - 1):
        s.solve()
        pos = np.asarray(s.get_body_positions())
        travel.append((pos[up][2] - z0[up][2]) - (pos[1][2] - z0[1][2]))
        airborne.append(not mbs.get_tire_state(hd["tire"])["in_contact"])
        states.add(mbs.get_length_stop_state(hd["stop"])["engaged"])
    travel = np.array(travel)
    airborne = np.array(airborne)
    assert -1 in states  # the bump stop took the hit
    # The rocker motion ratio is still positive at the peak travel.
    peak = float(travel.max())
    assert model.stops["bump_travel"] < peak < 0.07
    mr = kc.bump_sweep(mbs, model, [peak])["motion_ratio"][0]
    assert mr > 0.2
    # In the air the suspension extends below the design travel.
    assert airborne.any()
    assert travel[airborne].min() < -0.015
    assert travel.min() > model.stops["droop_travel"] - 0.005

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
