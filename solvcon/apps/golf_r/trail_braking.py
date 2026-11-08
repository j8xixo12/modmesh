# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Scripted trail-braking scenario and independently defined road geometry."""

import math
from dataclasses import dataclass
import numpy as np
from .bicycle import Vehicle, derivative, forces, torque_curve


@dataclass(frozen=True)
class Scenario:
    entry_x: float = 35.0
    radius: float = 45.0
    corner_angle: float = 1.35
    road_width: float = 10.0
    initial_speed: float = 20.0
    duration: float = 9.0
    dt: float = 0.01


def road(scenario=Scenario(), spacing=0.5):
    """Straight, circular left bend, and straight exit, in metres."""
    initial = np.arange(-35, scenario.entry_x, spacing, dtype='float64')
    angles = np.arange(
        0, scenario.corner_angle, spacing / scenario.radius, dtype='float64'
    )
    end = scenario.corner_angle
    exit_distance = np.arange(0, 55 + spacing / 2, spacing, dtype='float64')
    approach = np.column_stack((initial, np.zeros_like(initial)))
    bend = np.column_stack(
        (
            scenario.entry_x + scenario.radius * np.sin(angles),
            scenario.radius * (1 - np.cos(angles)),
        )
    )
    end_point = np.array(
        [
            scenario.entry_x + scenario.radius * math.sin(end),
            scenario.radius * (1 - math.cos(end)),
        ],
        dtype='float64',
    )
    exit_path = end_point + exit_distance[:, None] * [
        math.cos(end),
        math.sin(end),
    ]
    return np.vstack((approach, bend, exit_path))


def driver(t, state, path, vehicle):
    """Open-loop pedals with geometric look-ahead steering."""
    brake = float(
        np.interp(
            t, [0, 0.8, 1.15, 1.8, 3.4, 3.6, 9], [0, 0, 0.3, 0.3, 0, 0, 0]
        )
    )
    throttle = float(
        np.interp(t, [0, 3.3, 4.0, 4.7, 9], [0, 0, 0, 0.23, 0.23])
    )
    nearest = int(np.argmin(np.sum((path - state[:2]) ** 2, axis=1)))
    lookahead = max(8.0, state[3] * 0.65)
    target = min(nearest + round(lookahead / 0.5), len(path) - 1)
    dx, dy = path[target] - state[:2]
    angle = math.atan2(dy, dx) - state[2]
    steering = math.atan2(
        2 * vehicle.wheelbase * math.sin(angle), max(math.hypot(dx, dy), 1)
    )
    return throttle, brake, float(np.clip(steering, -0.22, 0.22)), 3


def run(data_dir, scenario=Scenario(), dt=None):
    p, curve, path = Vehicle(), torque_curve(data_dir), road(scenario)
    dt = scenario.dt if dt is None else dt
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    n = round(scenario.duration / dt)
    if n < 1 or not math.isclose(n * dt, scenario.duration, abs_tol=1e-10):
        raise ValueError('duration must be a multiple of dt')
    time = np.arange(n + 1, dtype='float64') * dt
    states = np.zeros((n + 1, 7), dtype='float64')
    states[0, 3] = scenario.initial_speed
    controls = np.zeros((n + 1, 4), dtype='float64')
    loads = np.zeros((n + 1, 4), dtype='float64')
    for i, t in enumerate(time):
        state = states[i]
        control = driver(t, state, path, p)
        controls[i] = control
        f = forces(p, state, *control, curve)
        loads[i] = [
            f['front_x'],
            f['front_y'],
            f['front_load'],
            f['rear_load'],
        ]
        if i == n:
            break

        def rhs(ti, yi):
            return derivative(p, yi, *driver(ti, yi, path, p), curve)

        k1 = rhs(t, state)
        k2 = rhs(t + dt / 2, state + dt * k1 / 2)
        k3 = rhs(t + dt / 2, state + dt * k2 / 2)
        k4 = rhs(t + dt, state + dt * k3)
        states[i + 1] = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    wheel_angles = np.zeros(n + 1, dtype='float64')
    wheel_angles[1:] = np.cumsum(
        (states[:-1, 3] + states[1:, 3]) * dt / (2 * p.radius)
    )
    return time, states, controls, loads, wheel_angles, path


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
