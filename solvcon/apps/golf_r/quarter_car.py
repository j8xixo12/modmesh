# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Linear quarter-car experiment using Golf R wheel rates in SI units.

Masses and damping are explicit assumptions, not measurements of the car.
The model is linearized about static equilibrium, so coordinates are
displacements and gravity is absent from the dynamic equations.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Parameters:
    sprung_mass: float
    unsprung_mass: float
    wheel_rate: float
    damping: float
    tire_rate: float

    def __post_init__(self):
        values = (
            self.sprung_mass,
            self.unsprung_mass,
            self.wheel_rate,
            self.tire_rate,
        )
        if not all(np.isfinite(x) and x > 0 for x in values):
            raise ValueError(
                'Masses and stiffnesses must be finite and positive'
            )
        if not np.isfinite(self.damping) or self.damping < 0:
            raise ValueError('Damping must be finite and nonnegative')

    def matrices(self):
        m = np.diag([self.sprung_mass, self.unsprung_mass])
        k, kt, c = self.wheel_rate, self.tire_rate, self.damping
        stiffness = np.array([[k, -k], [-k, k + kt]], dtype='float64')
        damping = np.array([[c, -c], [-c, c]], dtype='float64')
        return m, damping, stiffness

    def natural_frequencies(self):
        m, _, k = self.matrices()
        values = np.linalg.eigvalsh(
            k / np.sqrt(np.outer(np.diag(m), np.diag(m)))
        )
        return np.sqrt(values) / (2 * np.pi)


def derivative(p, state, road):
    zs, zu, vs, vu = state
    fs = p.wheel_rate * (zs - zu) + p.damping * (vs - vu)
    ft = p.tire_rate * (zu - road)
    return np.array(
        [vs, vu, -fs / p.sprung_mass, (fs - ft) / p.unsprung_mass],
        dtype='float64',
    )


def simulate(p, road, duration, dt, initial=None):
    if duration <= 0 or dt <= 0 or not np.isfinite([duration, dt]).all():
        raise ValueError('duration and dt must be finite and positive')
    n = int(round(duration / dt))
    if n < 1 or not np.isclose(n * dt, duration, atol=1e-12, rtol=1e-10):
        raise ValueError('duration must be an integer multiple of dt')
    time = np.arange(n + 1, dtype='float64') * dt
    state = np.zeros((n + 1, 4), dtype='float64')
    if initial is not None:
        state[0] = np.asarray(initial, dtype='float64')
    for i in range(n):
        t, y = time[i], state[i]
        k1 = derivative(p, y, road(t))
        k2 = derivative(p, y + dt * k1 / 2, road(t + dt / 2))
        k3 = derivative(p, y + dt * k2 / 2, road(t + dt / 2))
        k4 = derivative(p, y + dt * k3, road(t + dt))
        state[i + 1] = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return time, state


def energy(p, state):
    zs, zu, vs, vu = np.asarray(state).T
    return (
        0.5 * p.sprung_mass * vs**2
        + 0.5 * p.unsprung_mass * vu**2
        + 0.5 * p.wheel_rate * (zs - zu) ** 2
        + 0.5 * p.tire_rate * zu**2
    )


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
