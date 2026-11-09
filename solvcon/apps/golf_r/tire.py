# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Experimental smooth combined-slip tire, independent of AC's tire model.

This is the single tire model of the ``golf_r`` bicycle and wheel models.
Combined slip uses the theoretical slip quantities of Pacejka, Tire and
Vehicle Dynamics, 3rd ed. (2012), ch. 3 (brush model) and ch. 4:

.. math::

    \\sigma_x = \\frac{\\kappa}{1 + \\kappa}, \\quad
    \\sigma_y = \\frac{\\tan\\alpha}{1 + \\kappa}, \\quad
    \\sigma = \\sqrt{\\sigma_x^2 + \\sigma_y^2}

    F_x = \\frac{\\sigma_x}{\\sigma} F_{x0}(\\sigma), \\quad
    F_y = -\\frac{\\sigma_y}{\\sigma} F_{y0}(\\sigma)

The pure-slip curves are :math:`F_0(\\sigma) = \\mu F_z \\tanh(C
\\sigma / \\mu F_z)`.  The force points along the slip velocity when the
contact patch slides, so a locked wheel (:math:`\\kappa = -1`) gives
:math:`|F_y| / |F| = \\sin\\alpha`.  Stiffness values are synthetic
assumptions and are not fitted to a measured tire curve.
"""

from dataclasses import dataclass
import math
import numpy as np

#: Standard gravity in m/s^2, shared by the golf_r models.
GRAVITY = 9.81
#: The largest driving slip ratio that the force inversion searches.
MAX_DRIVE_SLIP = 1.0


@dataclass(frozen=True)
class Tire:
    """Per-tire parameters in SI units.

    ``relaxation_length`` is 0.45 m, inside the typical passenger-car
    range of 0.3 m to 0.6 m at nominal load (Pacejka 2012, ch. 7,
    single contact point transient tire model).  The AC ``tyres.ini``
    value RELAXATION_LENGTH=0.072 is a parameter of the AC solver and is
    not used here.  The models use one length for the longitudinal and
    the lateral slip.
    """

    longitudinal_stiffness: float = 80000.0
    cornering_stiffness: float = 68755.0
    friction: float = 1.0
    reference_load: float = 4000.0
    load_sensitivity: float = 0.1
    relaxation_length: float = 0.45

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


def friction_limit(tire, load):
    """Return the largest force magnitude of the tire at a vertical load.

    :param tire: The tire parameters
    :type tire: Tire
    :param load: The vertical load in N
    :type load: float
    :return: The friction limit in N, zero for a nonpositive load
    :rtype: float
    """
    if load <= 0:
        return 0.0
    mu = max(
        0.05,
        tire.friction
        * (1 - tire.load_sensitivity * (load / tire.reference_load - 1)),
    )
    return mu * load


def _force(tire, load, slip_ratio, slip_angle):
    """Return (Fx, Fy) without input checks; see :func:`force`."""
    cap = friction_limit(tire, load)
    tan_alpha = math.tan(slip_angle)
    slip = math.hypot(slip_ratio, tan_alpha)
    if cap == 0 or slip == 0:
        return 0.0, 0.0
    rolling = 1 + slip_ratio
    # A locked or reversed wheel slides on the whole contact patch.
    sigma = slip / rolling if rolling > 0 else math.inf
    fx0 = cap * math.tanh(tire.longitudinal_stiffness * sigma / cap)
    fy0 = cap * math.tanh(tire.cornering_stiffness * sigma / cap)
    # sigma_x / sigma = kappa / slip; the factor 1 + kappa cancels.
    return slip_ratio / slip * fx0, -tan_alpha / slip * fy0


def force(tire, load, slip_ratio, slip_angle):
    """Return the combined-slip tire force.

    Fx is positive for driving; Fy opposes a positive slip angle.

    :param tire: The tire parameters
    :type tire: Tire
    :param load: The vertical load in N
    :type load: float
    :param slip_ratio: The longitudinal slip ratio kappa; -1 is locked
    :type slip_ratio: float
    :param slip_angle: The slip angle in rad, inside (-pi/2, pi/2)
    :type slip_angle: float
    :return: The force (Fx, Fy) in N
    :rtype: numpy.ndarray
    :raises ValueError: When an input is not finite or the slip angle is
        out of range
    """
    if not np.isfinite([load, slip_ratio, slip_angle]).all():
        raise ValueError('Tire inputs must be finite')
    if abs(slip_angle) >= math.pi / 2:
        raise ValueError('slip angle must be inside (-pi/2, pi/2)')
    return np.array(
        _force(tire, load, slip_ratio, slip_angle), dtype='float64'
    )


def force_for_demand(tire, load, demand, slip_angle, tolerance=1e-7):
    """Return the tire force that delivers a longitudinal force demand.

    The function solves the slip ratio for which :func:`force` gives
    ``demand`` at ``slip_angle`` by bisection on [-1, MAX_DRIVE_SLIP].  If
    the demand is beyond the tire limit, the wheel locks (braking) or
    spins at MAX_DRIVE_SLIP (driving), and the function returns the
    force at that slip ratio.

    :param tire: The tire parameters
    :type tire: Tire
    :param load: The vertical load in N
    :type load: float
    :param demand: The requested longitudinal force in N
    :type demand: float
    :param slip_angle: The slip angle in rad
    :type slip_angle: float
    :param tolerance: Force tolerance relative to the friction limit
    :type tolerance: float
    :return: The delivered force (Fx, Fy) in N and the slip ratio
    :rtype: tuple
    """
    if demand == 0:
        return (*_force(tire, load, 0.0, slip_angle), 0.0)
    low, high = -1.0, MAX_DRIVE_SLIP
    fx_low, fy_low = _force(tire, load, low, slip_angle)
    if demand <= fx_low:
        return fx_low, fy_low, low
    fx_high, fy_high = _force(tire, load, high, slip_angle)
    if demand >= fx_high:
        return fx_high, fy_high, high
    limit = tolerance * max(friction_limit(tire, load), 1.0)
    for _ in range(100):
        middle = 0.5 * (low + high)
        fx, fy = _force(tire, load, middle, slip_angle)
        if abs(fx - demand) <= limit:
            break
        if fx < demand:
            low = middle
        else:
            high = middle
    return fx, fy, middle


def relaxation_rate(tire, speed):
    """Return the first-order slip relaxation rate in 1/s.

    The slip state ``s`` follows ``ds/dt = rate * (target - s)`` with
    ``rate = |speed| / relaxation_length``.

    :param tire: The tire parameters
    :type tire: Tire
    :param speed: The contact patch travel speed in m/s
    :type speed: float
    :return: The relaxation rate
    :rtype: float
    """
    return abs(speed) / tire.relaxation_length


def relax(tire, previous, target, speed, dt):
    """Exact first-order slip relaxation for constant target and speed."""
    if not np.isfinite([previous, target, speed, dt]).all() or dt < 0:
        raise ValueError('Relaxation inputs must be finite and dt nonnegative')
    decay = math.exp(-relaxation_rate(tire, speed) * dt)
    return target + (previous - target) * decay


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
