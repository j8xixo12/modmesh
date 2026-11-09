# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Experimental two-axle model with rotating wheels and relaxed tire slip.

State: x, y, yaw, u, v, yaw rate, boost, front/rear wheel angular
speed, front/rear relaxed longitudinal slip and front/rear slip angle.
Each axle represents two identical wheels.  Only forward motion is
supported: the model holds ``u >= 0`` and wheel speeds ``>= 0``, and lateral
load transfer and differential behavior are omitted.

The driveline is an ideal torque-splitting centre differential with the
front share ``s``; the carrier speed is ``s w_f + (1 - s) w_r`` and the
engine speed is the overall gear ratio times the carrier speed.  The
engine and gearbox inertia reflected through the ratio couples the two
axle equations.  Brakes act as Coulomb friction on the wheels: a brake
torque opposes the wheel rotation, and a wheel at rest stays locked while
the brake capacity holds it.  Longitudinal load transfer is quasi-static
from the delivered acceleration, as in :mod:`.bicycle`.
"""

import numpy as np
from .bicycle import (
    LOAD_ITERATIONS,
    LOAD_TOLERANCE,
    TIRE,
    Vehicle,
    axle_loads,
    boost_target,
    engine_torque,
    gear_ratio,
)
from .tire import GRAVITY, _force, relaxation_rate

#: Speed floor (m/s) of the slip definitions and the slip relaxation.
MIN_SLIP_SPEED = 1.0


def initial_state(speed, vehicle=Vehicle()):
    if not np.isfinite(speed) or speed < 2:
        raise ValueError(
            'This model requires an initial speed of at least 2 m/s'
        )
    return np.array(
        [
            0,
            0,
            0,
            speed,
            0,
            0,
            0,
            speed / vehicle.radius,
            speed / vehicle.radius,
            0,
            0,
            0,
            0,
        ],
        dtype='float64',
    )


def engine_rpm(p, front_omega, rear_omega, gear):
    """Return the engine speed in rpm from the driven wheel speeds.

    :param p: The vehicle
    :type p: Vehicle
    :param front_omega: The front wheel angular speed in rad/s
    :type front_omega: float
    :param rear_omega: The rear wheel angular speed in rad/s
    :type rear_omega: float
    :param gear: The forward gear, from 1
    :type gear: int
    :return: The engine speed, without the idle floor
    :rtype: float
    """
    share = p.front_drive_share
    carrier = share * front_omega + (1 - share) * rear_omega
    return gear_ratio(p, gear) * carrier * 60 / (2 * np.pi)


def inertia_matrix(p, gear, wheel_inertia=None):
    """Return the 2x2 front/rear axle rotational inertia in kg m^2.

    The engine and gearbox inertia ``I_e`` turns at ``n (s w_f + (1 - s)
    w_r)``, so its kinetic energy adds ``I_e n^2 [s, 1 - s]^T [s, 1 - s]``
    to the two-wheel inertia of each axle.
    """
    wheel = p.wheel_inertia if wheel_inertia is None else wheel_inertia
    share = np.array(
        [p.front_drive_share, 1 - p.front_drive_share], dtype='float64'
    )
    driveline = (p.engine_inertia + p.gearbox_inertia) * gear_ratio(p, gear)
    driveline *= gear_ratio(p, gear)
    return 2 * wheel * np.eye(2) + driveline * np.outer(share, share)


def wheel_accelerations(inertia, torque, capacity, omega):
    """Solve the axle accelerations with Coulomb brakes.

    :param inertia: The 2x2 axle inertia matrix
    :type inertia: numpy.ndarray
    :param torque: The axle torques without the brakes, in N m
    :type torque: numpy.ndarray
    :param capacity: The nonnegative brake torque capacity per axle
    :type capacity: numpy.ndarray
    :param omega: The axle angular speeds in rad/s
    :type omega: numpy.ndarray
    :return: The angular accelerations and the applied brake torques
    :rtype: tuple
    """
    # A turning wheel gets the full brake torque against its rotation.
    brake = np.where(omega > 0, -capacity, 0.0)
    held = (omega <= 0) & (capacity > 0)
    for _ in range(3):
        free = ~held
        accel = np.zeros(2, dtype='float64')
        if free.any():
            accel[free] = np.linalg.solve(
                inertia[np.ix_(free, free)], (torque + brake)[free]
            )
        # The brake torque that keeps a held wheel at rest.
        needed = inertia @ accel - torque
        slipping = held & (np.abs(needed) > capacity)
        if not slipping.any():
            brake[held] = needed[held]
            return accel, brake
        # The brake saturates and resists the breakaway direction.
        brake[slipping] = np.sign(needed[slipping]) * capacity[slipping]
        held &= ~slipping
    return accel, brake


def forces(p, tire, state, controls, curve, wheel_inertia=None):
    """Return the tire, load and driveline quantities of a state."""
    throttle, brake, steer, gear = controls
    if not (
        0 <= throttle <= 1
        and 0 <= brake <= 1
        and np.isfinite(steer)
        and 1 <= gear <= len(p.gears)
    ):
        raise ValueError('Invalid driver input')
    if wheel_inertia is not None and (
        not np.isfinite(wheel_inertia) or wheel_inertia <= 0
    ):
        raise ValueError('wheel inertia must be positive')
    _, _, yaw, u, v, r, boost, wf, wr, kf, kr, af, ar = state
    a = p.wheelbase * (1 - p.front_fraction)
    b = p.wheelbase * p.front_fraction
    c, s = np.cos(steer), np.sin(steer)
    front_u = u * c + (v + a * r) * s
    front_v = -u * s + (v + a * r) * c
    rear_u, rear_v = u, v - b * r
    front_speed = max(abs(front_u), MIN_SLIP_SPEED)
    rear_speed = max(abs(rear_u), MIN_SLIP_SPEED)
    targets = (
        (p.radius * wf - front_u) / front_speed,
        (p.radius * wr - rear_u) / rear_speed,
        np.arctan2(front_v, max(front_u, MIN_SLIP_SPEED)),
        np.arctan2(rear_v, max(rear_u, MIN_SLIP_SPEED)),
    )
    drag = (
        0.5 * p.air_density * p.cda * u * u
        + p.rolling_coefficient * p.mass * GRAVITY
    )
    ax = 0.0
    for _ in range(LOAD_ITERATIONS):
        front_load, rear_load = axle_loads(p, ax)
        ffx, ffy = (2 * x for x in _force(tire, front_load / 2, kf, af))
        frx, fry = (2 * x for x in _force(tire, rear_load / 2, kr, ar))
        fx, fy = ffx * c - ffy * s, ffx * s + ffy * c
        delivered = (fx + frx - drag) / p.mass
        converged = abs(delivered - ax) <= LOAD_TOLERANCE
        ax = delivered
        if converged:
            break
    rpm = engine_rpm(p, wf, wr, gear)
    ratio = gear_ratio(p, gear)
    driveline_torque = (
        engine_torque(p, rpm, throttle, boost, curve) * ratio * p.efficiency
    )
    share = p.front_drive_share
    torque = np.array(
        [
            driveline_torque * share - ffx * p.radius,
            driveline_torque * (1 - share) - frx * p.radius,
        ],
        dtype='float64',
    )
    total = brake * 4 * p.max_brake_torque
    capacity = np.array(
        [total * p.front_brake_share, total * (1 - p.front_brake_share)],
        dtype='float64',
    )
    accel, brakes = wheel_accelerations(
        inertia_matrix(p, gear, wheel_inertia),
        torque,
        capacity,
        np.array([wf, wr], dtype='float64'),
    )
    return dict(
        front_x=ffx,
        front_y=ffy,
        rear_x=frx,
        rear_y=fry,
        body_x=fx,
        body_y=fy,
        front_load=front_load,
        rear_load=rear_load,
        drag=drag,
        ax=ax,
        rpm=rpm,
        wheel_accel=accel,
        brake_torque=brakes,
        slip_targets=targets,
        relaxation=(
            relaxation_rate(tire, front_speed),
            relaxation_rate(tire, rear_speed),
        ),
    )


def derivative(p, tire, state, controls, curve, wheel_inertia=None):
    f = forces(p, tire, state, controls, curve, wheel_inertia)
    _, _, yaw, u, v, r, boost, wf, wr, kf, kr, af, ar = state
    a = p.wheelbase * (1 - p.front_fraction)
    b = p.wheelbase * p.front_fraction
    du = (f['body_x'] + f['rear_x'] - f['drag']) / p.mass + r * v
    if u <= 0 and du < 0:
        du = 0.0
    forward = max(u, 0.0)
    rate_f, rate_r = f['relaxation']
    kf_target, kr_target, af_target, ar_target = f['slip_targets']
    return np.array(
        [
            forward * np.cos(yaw) - v * np.sin(yaw),
            forward * np.sin(yaw) + v * np.cos(yaw),
            r,
            du,
            (f['body_y'] + f['rear_y']) / p.mass - r * u,
            (a * f['body_y'] - b * f['rear_y']) / p.yaw_inertia,
            (boost_target(p, controls[0], f['rpm']) - boost) / p.turbo_tau,
            f['wheel_accel'][0],
            f['wheel_accel'][1],
            rate_f * (kf_target - kf),
            rate_r * (kr_target - kr),
            rate_f * (af_target - af),
            rate_r * (ar_target - ar),
        ],
        dtype='float64',
    )


def simulate(
    initial,
    controls,
    duration,
    dt=0.0005,
    vehicle=Vehicle(),
    tire=TIRE,
    curve=None,
):
    if not np.isfinite([duration, dt]).all() or duration <= 0 or dt <= 0:
        raise ValueError('duration and step must be finite and positive')
    n = round(duration / dt)
    if n < 1 or not np.isclose(n * dt, duration, rtol=1e-10, atol=1e-12):
        raise ValueError('duration must be a multiple of dt')
    time = np.arange(n + 1, dtype='float64') * dt
    result = np.zeros((n + 1, 13), dtype='float64')
    result[0] = initial
    if curve is None:
        raise ValueError('Supply a torque curve loaded from car data')
    for i in range(n):
        t, y = time[i], result[i]

        def rhs(ti, yi):
            return derivative(vehicle, tire, yi, controls(ti), curve)

        k1 = rhs(t, y)
        k2 = rhs(t + dt / 2, y + dt * k1 / 2)
        k3 = rhs(t + dt / 2, y + dt * k2 / 2)
        k4 = rhs(t + dt, y + dt * k3)
        step = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # A brake cannot reverse a wheel and the model is forward only:
        # a step that crosses zero ends at rest (locked).
        step[3] = max(step[3], 0.0)
        step[7:9] = np.maximum(step[7:9], 0.0)
        step[6] = np.clip(step[6], 0, vehicle.max_boost)
        result[i + 1] = step
        if not np.isfinite(result[i + 1]).all():
            raise ValueError('nonfinite state; reduce integration step')
    return time, result


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
