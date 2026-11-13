# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Replay a measured acceleration history on the full vehicle: the body
motion (roll, pitch, heave, wheel travel, tire loads) that the suspension
gives for the accelerations of a log (for example a RaceChrono export,
:mod:`.racechrono`).

The vehicle rides on the rig of :func:`suspension.car.steady_state` and
:func:`suspension.longitudinal.steady_state`, now in time: in the frame of
the car every body carries the D'Alembert load -m (a_x, a_y) and its
weight, and the tires carry the reaction at their contact patches:

* the side force of each tire is M a_y F_zi / sum F_z (each tire uses the
  same share of its load, as at the fixed point of the steady rig);
* the longitudinal force is M a_x, split between the axles by the brake
  bias (braking, at the contact patch: outboard brakes) or the drive split
  (traction, at the wheel center, the shaft moment on the body).

The applied loads balance in x and y (the tire forces are M a).  The
road holds each contact patch sideways by the slip damping c_y = C_alpha /
V of the tire: the single contact point tire with the relaxation length
sigma is a spring C_alpha / sigma in series with that damper (Pacejka
2012, Ch. 7), and below V / sigma (about 7 Hz at 20 m/s) the damper
governs.  A damper carries nothing in a steady state, so the replay keeps
the loads of the steady rig.  Without it the wheels slide sideways freely
and the body rolls about the mass center of the car instead of its roll
axis (a 30 percent higher roll frequency than the roll model of the
closed form).  A very soft support of the chassis (:data:`SUPPORT_HZ`,
critically damped) stops the drift in x and yaw.  Roll, pitch, and heave
are free, so the suspension, the bars, the dampers, the stops, and the
tires set them.  No tire slip curve enters (only the slip damping of the
patches): the replay checks the suspension and body model against the
measured body rates, independent of the tire curve.

The accelerations are taken in the road plane.  An accelerometer tilts with
the body, so a raw reading holds g sin(roll), about 0.035 g per 2 deg of
roll; whether a logger removes it is the logger's (RaceChrono documents no
such step for its calculated channels).
"""

import copy
import math

import numpy as np

from . import car
from .metrics import rot

__all__ = [
    'steer_replay',
    'acceleration_replay',
    'compare',
    'patch_support',
    'RIGID_TIRE',
]

#: The frequency (Hz) of the chassis on its support in x, y, and yaw.
SUPPORT_HZ = 0.2
#: The factor on the radial stiffness and damping of the tires of a replay
#: with rigid tires.  The wheel center then keeps its height over the road
#: (a tire of 200 N/mm gives way 1 mm instead of 20 mm at 4 kN), so the
#: body moves on the springs, the bars, and the linkage alone.  On a 967 kg
#: car the roll gradient is within 3 percent of the one at half the factor;
#: at 50 the start equilibrium does not converge.
RIGID_TIRE = 20.0


def _smooth(time, values, window):
    """Centered moving average over ``window`` seconds (0: none)."""
    if window <= 0.0 or len(time) < 3:
        return np.asarray(values, dtype='float64')
    step = float(np.median(np.diff(time)))
    n = max(1, int(round(window / step)))
    if n < 2:
        return np.asarray(values, dtype='float64')
    kernel = np.ones(n) / n
    pad = n // 2
    padded = np.pad(values, (pad, n - 1 - pad), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def patch_support(vm, speed):
    """The lateral stiffness (0) and damping c_y = C_alpha / V of the
    contact patches, front and rear (see the module docstring)."""
    speed = max(abs(float(speed)), 1.0)
    return tuple((0.0, c / speed) for c in vm.cornering_stiffness)


def acceleration_replay(mbs, vm, time, ax, ay, gravity=9.81, dt=2e-3,
                        brake_bias=0.6, drive_split=0.0, pose_dt=0.02,
                        smooth=0.0, record_dt=None, speed=20.0, rack=None,
                        rigid_tire=False):
    """The body motion of the vehicle under a measured acceleration
    history.

    :param time: sample times (s, increasing)
    :param ax: longitudinal acceleration (m/s^2, + forward) at the times
    :param ay: lateral acceleration (m/s^2, + to the left: a left turn)
    :param brake_bias: front share of the braking force
    :param drive_split: front share of the drive force
    :param smooth: centered moving average of the inputs (s), 0 for none
    :param record_dt: the record interval (s), default ``dt``
    :param speed: the speed (m/s) of the slip damping C_alpha / V of the
        contact patches (the mean speed of the log)
    :param rack: None, or the rack travel (m) at the times, with the sign
        of the linkage (:func:`car.kinematic_steer` gives the wheel angle
        of a travel): the front wheels steer as measured, so the camber, the
        caster lift, and the scrub of the steered linkage enter the travel
        and the loads.  The applied forces stay in the road axes of the
        car: the steer angle does not turn them.
    :param rigid_tire: leave the tire deflection out (:data:`RIGID_TIRE`):
        the wheels stay on the road at their radius, a lifted wheel still
        carries no load
    :return: dict with ``time``, ``ax``, ``ay`` (the inputs used), ``roll``
        (rad, + right side down: the outside of a left turn), ``pitch``
        (rad, + nose down), ``heave`` (m, the CG rise from the start),
        ``roll_rate`` and ``pitch_rate`` (rad/s about the chassis x and
        y), ``travel`` (m, (n, 4) wheel travel to the body from the start,
        + bump, :data:`car.CORNERS` order), ``loads`` (N, (n, 4)),
        ``support`` (largest support force, N), ``rack`` (m, zeros without
        one), ``poses`` and ``pose_times``, ``models``
    """
    time = np.asarray(time, dtype='float64')
    ax = _smooth(time, np.asarray(ax, dtype='float64'), smooth)
    ay = _smooth(time, np.asarray(ay, dtype='float64'), smooth)
    if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("replay: time must increase")
    models = car.corner_models(mbs, vm, gravity)
    if rigid_tire:
        models = {name: copy.copy(model) for name, model in models.items()}
        for model in models.values():
            model.tire_k *= RIGID_TIRE
            model.tire_c *= RIGID_TIRE
    mass_total = vm.sprung_mass + sum(
        2.0 * car._corner_mass(models[f"{a}_left"]) for a in ("front",
                                                              "rear"))
    inertia_z = float(vm.sprung_inertia[2])
    omega = 2.0 * math.pi * SUPPORT_HZ
    stiffness = (mass_total * omega ** 2, inertia_z * omega ** 2)
    damping = (2.0 * mass_total * omega, 2.0 * inertia_z * omega)
    steer = None
    if rack is not None:
        rack = np.asarray(rack, dtype='float64')
        if rack.shape != time.shape:
            raise ValueError("replay: one rack travel for each time")
        # The table holds its last value past the end (a step may look
        # ahead).
        rel = time - time[0]
        stamps = [float(v) for v in rel] + [float(rel[-1]) + 1.0]
        table = [float(v) for v in rack] + [float(rack[-1])]
        steer = (mbs.DriveFunction.table(stamps, table),
                 mbs.DriveFunction.table(stamps, [-v for v in table]))
    s, h = car.build(mbs, vm, 0.0, gravity,
                     side_forces={name: 0.0 for name in car.CORNERS},
                     steer_func=steer,
                     dt=dt, models=models, support=stiffness,
                     support_damping=damping,
                     patch_support=patch_support(vm, speed))
    uprights = {h["corners"][name].nums["upright"]: name
                for name in car.CORNERS}
    total = sum(mass for _, mass in h["bodies"])

    def loads():
        return np.array([float(mbs.get_tire_state(h["tires"][name])
                               ["normal_force"]) for name in car.CORNERS])

    def apply(a_x, a_y, fz):
        share = (brake_bias if a_x < 0.0 else drive_split)
        braking = a_x < 0.0
        weight = max(float(fz.sum()), 1e-9)
        field = np.array([-a_x, -a_y, -gravity])
        pos = np.asarray(s.get_body_positions())
        quat = np.asarray(s.get_body_quaternions())
        chassis_torque = np.zeros(3)
        applied = {}
        for num, mass in h["bodies"]:
            force = mass * field
            torque = np.zeros(3)
            if num in uprights:
                name = uprights[num]
                k = car.CORNERS.index(name)
                front = name.startswith("front")
                fy = total * a_y * fz[k] / weight
                fx = 0.5 * total * a_x * (share if front else 1.0 - share)
                spin = rot(quat[num]) @ models[name].spin_axis
                down = np.array([0.0, 0.0, -1.0]) + spin[2] * spin
                down /= np.linalg.norm(down)
                lever = pos[num][2] / -down[2]
                patch = np.array([0.0, fy, 0.0])
                drive = np.array([fx, 0.0, 0.0])
                if braking:
                    patch = patch + drive
                else:
                    # The shaft takes the moment of the drive force about
                    # the wheel center to the body.
                    force = force + drive
                    chassis_torque += np.cross(lever * down, drive)
                force = force + patch
                torque = np.cross(lever * down, patch)
            applied[num] = (force, torque)
        for num, (force, torque) in applied.items():
            if 1 == num:
                torque = torque + chassis_torque
            s.set_applied_load(num, list(force), list(torque))

    # The start: the static fixed point at the first sample.
    fz = np.array([(vm.corner_sprung_mass(name.startswith("front"))
                    + car._corner_mass(models[name])) * gravity
                   for name in car.CORNERS])
    for _ in range(30):
        apply(ax[0], ay[0], fz)
        res = s.find_equilibrium(tol=1e-10)
        if not res["converged"]:
            raise RuntimeError(f"replay: no equilibrium at the start: "
                               f"{res['message']}")
        new = loads()
        done = np.max(np.abs(new - fz)) < 1e-6 * np.max(new)
        fz = new
        if done:
            break
    s.set_integrator("rk45")
    s.set_tolerances(1e-7, 1e-10)
    quat0 = np.asarray(s.get_body_quaternions())
    pos0 = np.asarray(s.get_body_positions())
    r0 = rot(quat0[1])

    def corner_offsets(pos, quat):
        r_ch = rot(quat[1])
        return np.array([float((r_ch.T @ (pos[h["corners"][name].nums[
            "upright"]] - pos[1]))[2]) for name in car.CORNERS])

    travel0 = corner_offsets(pos0, quat0)
    record_dt = dt if record_dt is None else float(record_dt)
    every = max(1, int(round(record_dt / dt)))
    pose_every = max(1, int(round(pose_dt / dt)))
    rec = {"time": [], "ax": [], "ay": [], "rack": [], "roll": [],
           "pitch": [], "heave": [], "roll_rate": [], "pitch_rate": [],
           "travel": [], "loads": [], "support": []}
    poses, pose_times = [], []
    steps = int(math.floor((time[-1] - time[0]) / dt + 1e-9))
    t0 = time[0]
    for step in range(steps + 1):
        t = t0 + step * dt
        a_x = float(np.interp(t, time, ax))
        a_y = float(np.interp(t, time, ay))
        if step > 0:
            apply(a_x, a_y, fz)
            s.solve()
            fz = loads()
        if 0 == step % every:
            pos = np.asarray(s.get_body_positions())
            quat = np.asarray(s.get_body_quaternions())
            omega = np.asarray(s.get_body_angular_velocities())
            r_ch = rot(quat[1])
            w = r_ch.T @ omega[1]
            rec["time"].append(t - t0)
            rec["ax"].append(a_x)
            rec["ay"].append(a_y)
            rec["rack"].append(0.0 if rack is None else float(
                np.interp(t, time, rack)))
            rec["roll"].append(car._roll_angle(quat[1]))
            rec["pitch"].append(float(math.atan2(r_ch[0, 2], r_ch[2, 2])))
            rec["heave"].append(float(pos[1][2] - pos0[1][2]))
            rec["roll_rate"].append(float(w[0]))
            rec["pitch_rate"].append(float(w[1]))
            rec["travel"].append(corner_offsets(pos, quat) - travel0)
            rec["loads"].append(fz)
            sup = mbs.get_force_element_loads(h["chassis_support"])
            rec["support"].append(float(np.linalg.norm(
                np.asarray(sup["force_j"])[:2])))
        if 0 == step % pose_every:
            poses.append(car.vehicle_pose(s, h, follow=True))
            pose_times.append(t - t0)
    out = {key: np.array(val) for key, val in rec.items()}
    # The start may be a turn: the angles are from the design pose (the
    # chassis starts level only at a = 0).
    out["roll0"] = car._roll_angle(quat0[1])
    out["pitch0"] = float(math.atan2(r0[0, 2], r0[2, 2]))
    out["poses"] = poses
    out["pose_times"] = np.array(pose_times)
    out["models"] = models
    out["mass"] = total
    return out


#: The speed error gain (1/s) of :func:`steer_replay`.
SPEED_GAIN = 2.0
#: The share of the grip left by the side force that the longitudinal
#: demand may use (an anti-lock and traction control: past the peak a wheel
#: locks or spins and loses its side force).
GRIP_SHARE = 0.9


def _demand_limit(tire, states):
    """The largest |F_x / F_z| of an axle: the friction ellipse of each
    wheel at its side force, the smaller of the two."""
    limit = math.inf
    for state in states:
        fz = float(state["normal_force"])
        if fz <= 0.0:
            continue
        mu_x, mu_y = tire.friction(fz)
        used = min(1.0, abs(float(state["lateral_force"])) / (mu_y * fz))
        limit = min(limit, GRIP_SHARE * mu_x * math.sqrt(1.0 - used * used))
    return 0.0 if math.isinf(limit) else limit


def steer_replay(mbs, vm, time, rack, speed, gravity=9.81, dt=1e-3,
                 brake_bias=0.6, drive_split=0.0, pose_dt=0.02,
                 record_dt=None, speed_gain=SPEED_GAIN, stop_sideslip=None):
    """The motion of the vehicle under a measured steering and speed
    history: the car drives on its tires (the Magic Formula slip forces of
    :attr:`car.VehicleModel.tires`), open loop in the steering.

    The rack follows ``rack``.  The speed follows ``speed``: the tires take
    the longitudinal force M (dV/dt + ``speed_gain`` (V - v)), v the speed
    of the model, as a demand F_x / F_z that is the same for the two
    wheels of an axle (the force shares their loads: no wheel of an
    unloaded inside corner locks or spins), the braking force by
    ``brake_bias`` and the drive force by ``drive_split`` between the
    axles.  The demand stays inside :data:`GRIP_SHARE` of the friction
    ellipse of each wheel at its side force (:func:`_demand_limit`); the
    speed then lags its target.  The tire force acts at the contact patch:
    right for a brake on the upright; for the drive, the moment of the
    force about the wheel center goes from the upright to the chassis (the
    shaft of an inboard differential, as in :func:`acceleration_replay`).

    Nothing steers the car back to the measured path: past the grip of the
    model tires the run departs from the log (see ``sideslip``).

    :param time: sample times (s, increasing)
    :param rack: rack travel (m, + steers to the left) at the times
    :param speed: forward speed (m/s) at the times
    :param brake_bias: front share of the braking force; None shares it
        by the axle loads (every tire brakes at the same F_x / F_z)
    :param drive_split: front share of the drive force; None as above
    :param stop_sideslip: None, or the sideslip (rad) past which the run
        stops: the car has left the path of the log
    :return: dict with ``time``, ``rack`` and ``target_speed`` (the
        inputs), ``speed`` (m/s), ``yaw_rate`` (rad/s, + to the left),
        ``ay`` and ``ax`` (m/s^2, the chassis acceleration normal to and
        along its heading in the road plane), ``sideslip`` (rad),
        ``roll`` (rad, + right side down), ``pitch`` (rad, + nose down),
        ``heave`` (m), ``roll_rate`` and ``pitch_rate`` (rad/s about the
        chassis x and y), ``travel`` (m, (n, 4), + bump), ``loads`` (N,
        (n, 4)), ``demand`` ((n, 2), F_x / F_z front and rear), ``poses``
        (following the car) and ``pose_times``, ``models``, and
        ``departed`` (the time the run stopped at ``stop_sideslip``, else
        None)
    """
    time = np.asarray(time, dtype='float64')
    rack = np.asarray(rack, dtype='float64')
    speed = np.asarray(speed, dtype='float64')
    if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("replay: time must increase")
    if np.min(speed) <= 0.0:
        raise ValueError("replay: the speed must stay above zero")
    t0 = time[0]
    rel = time - t0
    # The table holds its last value past the end (a step may look ahead).
    stamps = [float(v) for v in rel] + [float(rel[-1]) + 1.0]
    table = [float(v) for v in rack] + [float(rack[-1])]
    steer = (mbs.DriveFunction.table(stamps, table),
             mbs.DriveFunction.table(stamps, [-v for v in table]))
    models = car.corner_models(mbs, vm, gravity)
    s, h = car.build(mbs, vm, 0.0, gravity, steer_func=steer, dt=dt,
                     speed=float(speed[0]), with_slip=True, models=models)
    s.set_integrator("rk45")
    s.set_tolerances(1e-7, 1e-10)
    total = sum(mass for _, mass in h["bodies"])
    tires = [h["tires"][name] for name in car.CORNERS]
    masses = dict(h["bodies"])
    uprights = [(num, masses[num]) for num in (
        h["corners"][name].nums["upright"] for name in car.CORNERS)]
    rate = np.gradient(speed, rel)
    pos0 = np.asarray(s.get_body_positions())
    quat0 = np.asarray(s.get_body_quaternions())

    def corner_offsets(pos, quat):
        r_ch = rot(quat[1])
        return np.array([float((r_ch.T @ (pos[h["corners"][name].nums[
            "upright"]] - pos[1]))[2]) for name in car.CORNERS])

    travel0 = corner_offsets(pos0, quat0)
    record_dt = dt if record_dt is None else float(record_dt)
    every = max(1, int(round(record_dt / dt)))
    pose_every = max(1, int(round(pose_dt / dt)))
    rec = {key: [] for key in (
        "time", "rack", "target_speed", "speed", "yaw_rate", "sideslip",
        "roll", "pitch", "heave", "roll_rate", "pitch_rate", "travel",
        "loads", "demand", "velocity", "heading")}
    poses, pose_times = [], []
    steps = int(math.floor(rel[-1] / dt + 1e-9))
    states = [mbs.get_tire_state(tire) for tire in tires]
    fz = np.array([float(state["normal_force"]) for state in states])
    departed = None
    for step in range(steps + 1):
        t = step * dt
        quat = np.asarray(s.get_body_quaternions())
        vel = np.asarray(s.get_body_velocities())
        r_ch = rot(quat[1])
        heading = r_ch[:2, 0] / np.linalg.norm(r_ch[:2, 0])
        v_xy = np.asarray(vel[1][:2], dtype="float64")
        forward = float(v_xy @ heading)
        if stop_sideslip is not None and abs(math.atan2(
                float(v_xy[1] * heading[0] - v_xy[0] * heading[1]),
                forward)) > stop_sideslip:
            departed = t
            break
        target = float(np.interp(t, rel, speed))
        a_x = float(np.interp(t, rel, rate)) + speed_gain * (target
                                                             - forward)
        share = brake_bias if a_x < 0.0 else drive_split
        if share is None:
            share = float(fz[0] + fz[1]) / max(float(fz.sum()), 1.0)
        demand = []
        pos = np.asarray(s.get_body_positions())
        along = np.array([heading[0], heading[1], 0.0])
        shaft = np.zeros(3)
        for k, part in ((0, share), (2, 1.0 - share)):
            load = max(float(fz[k] + fz[k + 1]), 1.0)
            limit = _demand_limit(vm.tire(0 == k), states[k:k + 2])
            ratio = float(np.clip(total * a_x * part / load, -limit,
                                  limit))
            demand.append(ratio)
            for j in (k, k + 1):
                mbs.set_tire_longitudinal(tires[j], ratio)
                num, mass = uprights[j]
                moment = np.zeros(3)
                if ratio > 0.0:
                    # The drive force at the patch, about the wheel center.
                    moment = np.cross([0.0, 0.0, -pos[num][2]],
                                      ratio * fz[j] * along)
                    shaft += moment
                s.set_applied_load(num, [0.0, 0.0, -mass * gravity],
                                   list(-moment))
        s.set_applied_load(1, [0.0, 0.0, -vm.sprung_mass * gravity],
                           list(shaft))
        if 0 == step % every:
            omega = np.asarray(s.get_body_angular_velocities())
            w = r_ch.T @ omega[1]
            lateral = np.array([-heading[1], heading[0]])
            rec["time"].append(t)
            rec["rack"].append(float(np.interp(t, rel, rack)))
            rec["target_speed"].append(target)
            rec["speed"].append(forward)
            rec["yaw_rate"].append(float(omega[1][2]))
            rec["sideslip"].append(math.atan2(float(v_xy @ lateral),
                                              forward))
            # The angles of the chassis to the road, whatever its heading.
            rec["roll"].append(float(math.atan2(r_ch[2, 1], r_ch[2, 2])))
            rec["pitch"].append(float(-math.asin(r_ch[2, 0])))
            rec["heave"].append(float(pos[1][2] - pos0[1][2]))
            rec["roll_rate"].append(float(w[0]))
            rec["pitch_rate"].append(float(w[1]))
            rec["travel"].append(corner_offsets(pos, quat) - travel0)
            rec["loads"].append(fz)
            rec["demand"].append(demand)
            rec["velocity"].append(v_xy)
            rec["heading"].append(heading)
        if 0 == step % pose_every:
            poses.append(car.vehicle_pose(s, h, follow=True))
            pose_times.append(t)
        if step < steps:
            s.solve()
            if vm.relaxation_length > 0.0:
                for tire in tires:
                    mbs.commit_tire(tire)
            states = [mbs.get_tire_state(tire) for tire in tires]
            fz = np.array([float(state["normal_force"])
                           for state in states])
    out = {key: np.array(val) for key, val in rec.items()}
    # The chassis acceleration (central differences of the velocity) along
    # and normal to the heading.
    accel = np.gradient(out.pop("velocity"), out["time"], axis=0)
    heading = out.pop("heading")
    out["ax"] = np.einsum("ij,ij->i", accel, heading)
    out["ay"] = (accel[:, 1] * heading[:, 0] - accel[:, 0] * heading[:, 1])
    out["poses"] = poses
    out["pose_times"] = np.array(pose_times)
    out["models"] = models
    out["mass"] = total
    out["departed"] = departed
    return out


def compare(sim_time, sim, log_time, measured, max_lag=0.5):
    """Compare a simulated rate with a measured one on the log times.

    :return: dict with ``correlation`` (at zero lag), ``gain`` (the least
        squares factor of the simulation onto the measurement),
        ``rms_measured``, ``rms_error`` (no fitting), and ``lag`` (s, the
        delay of the simulation with the best correlation within
        ``max_lag``: + when the simulation leads the measurement)
    """
    log_time = np.asarray(log_time, dtype='float64')
    measured = np.asarray(measured, dtype='float64')
    inside = (log_time >= sim_time[0]) & (log_time <= sim_time[-1])
    t, m = log_time[inside], measured[inside]
    y = np.interp(t, sim_time, sim)
    mc, yc = m - m.mean(), y - y.mean()
    den = math.sqrt(float(mc @ mc) * float(yc @ yc))
    corr = float(mc @ yc) / den if den > 0.0 else 0.0
    gain = float(yc @ mc) / float(yc @ yc) if float(yc @ yc) > 0 else 0.0
    step = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    best, lag = -2.0, 0.0
    if step > 0.0 and float(mc @ mc) > 0.0:
        for k in range(-int(max_lag / step), int(max_lag / step) + 1):
            shifted = np.interp(t, sim_time + k * step, sim)
            sc = shifted - shifted.mean()
            d = math.sqrt(float(mc @ mc) * float(sc @ sc))
            c = float(mc @ sc) / d if d > 0.0 else 0.0
            if c > best:
                best, lag = c, k * step
    return {"correlation": corr, "gain": gain,
            "rms_measured": float(np.sqrt(np.mean(m * m))),
            "rms_error": float(np.sqrt(np.mean((y - m) ** 2))),
            "lag": lag}

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
