# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Provisional planar bicycle dynamics with traction and braking limits.

The tire curves, drive split, turbo time constant, efficiency, inertia and
drag are assumptions. This is not an Assetto Corsa tire or AWD model.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Vehicle:
    mass: float = 1654.0
    wheelbase: float = 2.61
    front_fraction: float = 0.60
    cg_height: float = 0.55
    yaw_inertia: float = 2200.0
    front_stiffness: float = 2 * 1200 * 180 / np.pi
    rear_stiffness: float = 2 * 1200 * 180 / np.pi
    mu: float = 1.0
    radius: float = 0.32
    final_drive: float = 4.47
    efficiency: float = 0.85
    front_drive_share: float = 0.6
    front_brake_share: float = 0.76
    max_brake_torque: float = 2400.0
    cda: float = 0.34 * 2.04
    rolling_coefficient: float = 0.012
    air_density: float = 1.225
    turbo_tau: float = 0.4
    max_boost: float = 1.9
    gears: tuple = (3.19, 2.75, 1.90, 1.04, 0.86, 0.79, 0.66)


def torque_curve(data_dir):
    points = []
    for line in (Path(data_dir) / 'power.lut').read_text().splitlines():
        if '|' in line:
            points.append([float(x) for x in line.split('|')])
    return np.asarray(points, dtype='float64')


def forces(p, state, throttle, brake, steer, gear, curve):
    """Return axle forces and loads for the planar state."""
    if not (
        0 <= throttle <= 1
        and 0 <= brake <= 1
        and np.isfinite(steer)
        and 1 <= gear <= len(p.gears)
    ):
        raise ValueError('Invalid driver input or gear')
    _, _, _, u, v, yaw_rate, boost = state
    speed = max(float(u), 0.0)
    ratio = p.gears[gear - 1] * p.final_drive
    rpm = max(850.0, speed / p.radius * ratio * 60 / (2 * np.pi))
    engine_torque = np.interp(rpm, curve[:, 0], curve[:, 1]) * (1 + boost)
    if rpm >= 8000:
        engine_torque = 0.0
    drive = throttle * engine_torque * ratio * p.efficiency / p.radius
    braking = brake * 4 * p.max_brake_torque / p.radius
    drag = 0.5 * p.air_density * p.cda * speed**2
    rolling = p.rolling_coefficient * p.mass * 9.81 if speed > 0 else 0.0
    requested = drive - braking
    estimated_ax = (requested - drag - rolling) / p.mass
    transfer = p.mass * estimated_ax * p.cg_height / p.wheelbase
    front_load = np.clip(
        p.mass * 9.81 * p.front_fraction - transfer, 0, p.mass * 9.81
    )
    rear_load = p.mass * 9.81 - front_load
    front_x = drive * p.front_drive_share - braking * p.front_brake_share
    rear_x = drive * (1 - p.front_drive_share) - braking * (
        1 - p.front_brake_share
    )
    front_x = np.clip(front_x, -p.mu * front_load, p.mu * front_load)
    rear_x = np.clip(rear_x, -p.mu * rear_load, p.mu * rear_load)
    a = p.wheelbase * (1 - p.front_fraction)
    b = p.wheelbase * p.front_fraction
    front_slip = steer - np.arctan2(v + a * yaw_rate, max(speed, 1.0))
    rear_slip = -np.arctan2(v - b * yaw_rate, max(speed, 1.0))
    # Friction circles combine the longitudinal and lateral demand per axle.
    front_cap = np.sqrt(max(0.0, (p.mu * front_load) ** 2 - front_x**2))
    rear_cap = np.sqrt(max(0.0, (p.mu * rear_load) ** 2 - rear_x**2))
    front_y = np.clip(p.front_stiffness * front_slip, -front_cap, front_cap)
    rear_y = np.clip(p.rear_stiffness * rear_slip, -rear_cap, rear_cap)
    return dict(
        front_x=front_x,
        rear_x=rear_x,
        front_y=front_y,
        rear_y=rear_y,
        front_load=front_load,
        rear_load=rear_load,
        drag=drag + rolling,
        rpm=rpm,
    )


def derivative(p, state, throttle, brake, steer, gear, curve):
    x, y, heading, u, v, yaw_rate, boost = state
    f = forces(p, state, throttle, brake, steer, gear, curve)
    front_x = f['front_x'] * np.cos(steer) - f['front_y'] * np.sin(steer)
    front_y = f['front_y'] * np.cos(steer) + f['front_x'] * np.sin(steer)
    du = (front_x + f['rear_x'] - f['drag']) / p.mass + yaw_rate * v
    if u <= 0 and du < 0:
        du = 0.0
    dv = (front_y + f['rear_y']) / p.mass - yaw_rate * u
    a, b = p.wheelbase * (1 - p.front_fraction), p.wheelbase * p.front_fraction
    dr = (a * front_y - b * f['rear_y']) / p.yaw_inertia
    boost_target = p.max_boost * throttle * min((f['rpm'] / 2200) ** 2.5, 1.0)
    forward_speed = max(u, 0.0)
    return np.array(
        [
            forward_speed * np.cos(heading) - v * np.sin(heading),
            forward_speed * np.sin(heading) + v * np.cos(heading),
            yaw_rate,
            du,
            dv,
            dr,
            (boost_target - boost) / p.turbo_tau,
        ],
        dtype='float64',
    )


def simulate(p, initial, controls, duration, dt, curve=None):
    if duration <= 0 or dt <= 0 or not np.isfinite([duration, dt]).all():
        raise ValueError('duration and dt must be finite and positive')
    n = int(round(duration / dt))
    if n < 1 or not np.isclose(n * dt, duration, atol=1e-12, rtol=1e-10):
        raise ValueError('duration must be an integer multiple of dt')
    if curve is None:
        raise ValueError('Supply a torque curve loaded from car data')
    t = np.arange(n + 1, dtype='float64') * dt
    states = np.zeros((n + 1, 7), dtype='float64')
    states[0] = initial
    for i in range(n):
        now, state = t[i], states[i]

        def rhs(time, value):
            return derivative(p, value, *controls(time), curve)

        k1 = rhs(now, state)
        k2 = rhs(now + dt / 2, state + dt * k1 / 2)
        k3 = rhs(now + dt / 2, state + dt * k2 / 2)
        k4 = rhs(now + dt, state + dt * k3)
        states[i + 1] = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        states[i + 1, 3] = max(states[i + 1, 3], 0)
        states[i + 1, 6] = np.clip(states[i + 1, 6], 0, p.max_boost)
    return t, states


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
