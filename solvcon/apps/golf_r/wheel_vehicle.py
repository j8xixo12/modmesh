# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Experimental two-axle model with rotating wheels and relaxed tire slip.

State: x, y, yaw, u, v, yaw rate, boost, front/rear wheel angular
speed, front/rear relaxed longitudinal slip and front/rear slip angle.
Each axle represents two identical wheels. Only forward, moving conditions
are supported; low-speed standstill and differential behavior are omitted.
"""

import numpy as np
from .bicycle import Vehicle
from .tire import Tire, force


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


def derivative(p, tire, state, controls, curve, wheel_inertia=1.8):
    throttle, brake, steer, gear = controls
    if not (
        0 <= throttle <= 1
        and 0 <= brake <= 1
        and np.isfinite(steer)
        and 1 <= gear <= len(p.gears)
    ):
        raise ValueError('Invalid driver input')
    if not np.isfinite(wheel_inertia) or wheel_inertia <= 0:
        raise ValueError('wheel inertia must be positive')
    _, _, yaw, u, v, r, boost, wf, wr, kf, kr, af, ar = state
    if u < 1:
        raise ValueError('wheel model does not support speeds below 1 m/s')
    a = p.wheelbase * (1 - p.front_fraction)
    b = p.wheelbase * p.front_fraction
    c, s = np.cos(steer), np.sin(steer)
    front_u = u * c + (v + a * r) * s
    front_v = -u * s + (v + a * r) * c
    rear_u, rear_v = u, v - b * r
    kf_target = (p.radius * wf - front_u) / max(abs(front_u), 1.0)
    kr_target = (p.radius * wr - rear_u) / max(abs(rear_u), 1.0)
    af_target = np.arctan2(front_v, max(front_u, 1.0))
    ar_target = np.arctan2(rear_v, max(rear_u, 1.0))
    # Static axle loads: transient pitch/load transfer is not yet coupled.
    front_load = p.mass * 9.81 * p.front_fraction / 2
    rear_load = p.mass * 9.81 * (1 - p.front_fraction) / 2
    ffx, ffy = 2 * force(tire, front_load, kf, af)
    frx, fry = 2 * force(tire, rear_load, kr, ar)
    fx, fy = ffx * c - ffy * s, ffx * s + ffy * c
    drag = (
        0.5 * p.air_density * p.cda * u * u
        + p.rolling_coefficient * p.mass * 9.81
    )
    ratio = p.gears[gear - 1] * p.final_drive
    rpm = max(850.0, u / p.radius * ratio * 60 / (2 * np.pi))
    torque = throttle * np.interp(rpm, curve[:, 0], curve[:, 1]) * (1 + boost)
    if rpm >= 8000:
        torque = 0.0
    driveline_torque = torque * ratio * p.efficiency
    brakes = brake * 4 * p.max_brake_torque
    boost_target = p.max_boost * throttle * min((rpm / 2200) ** 2.5, 1.0)
    return np.array(
        [
            u * np.cos(yaw) - v * np.sin(yaw),
            u * np.sin(yaw) + v * np.cos(yaw),
            r,
            (fx + frx - drag) / p.mass + r * v,
            (fy + fry) / p.mass - r * u,
            (a * fy - b * fry) / p.yaw_inertia,
            (boost_target - boost) / p.turbo_tau,
            (
                driveline_torque * p.front_drive_share
                - brakes * p.front_brake_share
                - ffx * p.radius
            )
            / (2 * wheel_inertia),
            (
                driveline_torque * (1 - p.front_drive_share)
                - brakes * (1 - p.front_brake_share)
                - frx * p.radius
            )
            / (2 * wheel_inertia),
            abs(front_u) / tire.relaxation_length * (kf_target - kf),
            abs(rear_u) / tire.relaxation_length * (kr_target - kr),
            abs(front_u) / tire.relaxation_length * (af_target - af),
            abs(rear_u) / tire.relaxation_length * (ar_target - ar),
        ],
        dtype='float64',
    )


def simulate(
    initial,
    controls,
    duration,
    dt=0.0005,
    vehicle=Vehicle(),
    tire=Tire(),
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
        result[i + 1] = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if not np.isfinite(result[i + 1]).all():
            raise ValueError('nonfinite state; reduce integration step')
    return time, result


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
