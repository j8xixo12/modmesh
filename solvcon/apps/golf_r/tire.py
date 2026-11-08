# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Experimental smooth combined-slip tire, independent of AC's tire model."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Tire:
    longitudinal_stiffness: float = 80000.0
    cornering_stiffness: float = 68755.0
    friction: float = 1.0
    reference_load: float = 4000.0
    load_sensitivity: float = 0.1
    relaxation_length: float = 0.072

    def __post_init__(self):
        values = (
            self.longitudinal_stiffness,
            self.cornering_stiffness,
            self.friction,
            self.reference_load,
            self.relaxation_length,
        )
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError(
                'Tire stiffness, friction, load and length must be positive'
            )
        if not 0 <= self.load_sensitivity < 1:
            raise ValueError('load sensitivity must be in [0, 1)')


def force(tire, load, slip_ratio, slip_angle):
    """Fx positive for driving; Fy opposes positive slip angle.

    Uses a smooth saturating law, then a friction-circle projection. Values
    are synthetic assumptions and not fitted to a measured tire curve.
    """
    if not np.isfinite([load, slip_ratio, slip_angle]).all():
        raise ValueError('Tire inputs must be finite')
    if load <= 0:
        return np.zeros(2, dtype='float64')
    mu = max(
        0.05,
        tire.friction
        * (1 - tire.load_sensitivity * (load / tire.reference_load - 1)),
    )
    cap = mu * load
    fx = cap * math.tanh(tire.longitudinal_stiffness * slip_ratio / cap)
    fy = -cap * math.tanh(tire.cornering_stiffness * slip_angle / cap)
    magnitude = math.hypot(fx, fy)
    if magnitude > cap:
        fx, fy = fx * cap / magnitude, fy * cap / magnitude
    return np.array([fx, fy], dtype='float64')


def relax(tire, previous, target, speed, dt):
    """Exact first-order slip relaxation for constant target and speed."""
    if not np.isfinite([previous, target, speed, dt]).all() or dt < 0:
        raise ValueError('Relaxation inputs must be finite and dt nonnegative')
    decay = math.exp(-abs(speed) * dt / tire.relaxation_length)
    return target + (previous - target) * decay


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
