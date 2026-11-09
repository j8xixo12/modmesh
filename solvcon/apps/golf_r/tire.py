# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""The tire of the ``golf_r`` models: the reduced Magic Formula of the
multibody engine (:class:`solvcon.multibody.MagicFormulaTire`).

The bicycle model, the wheel model, and the multibody vehicle use the same
slip model.  Each direction has the pure-slip curve

.. math::

    F_0(\\sigma) = D \\sin(C \\arctan(B \\sigma - E (B \\sigma -
    \\arctan B \\sigma))), \\quad
    D = \\mu F_z (F_z / F_{z0})^{n - 1}

and the combined slip uses the theoretical slip quantities of Pacejka,
Tire and Vehicle Dynamics, 3rd ed. (2012), Sec. 4.3.2:

.. math::

    \\sigma_x = \\frac{\\kappa}{1 + \\kappa}, \\quad
    \\sigma_y = \\frac{\\tan\\alpha}{1 + \\kappa}, \\quad
    F_x = \\frac{\\sigma_x}{\\sigma} F_{x0}(\\sigma), \\quad
    F_y = -\\frac{\\sigma_y}{\\sigma} F_{y0}(\\sigma)

A locked wheel (:math:`\\kappa = -1`) slides; with equal curves in both
directions the force points along the sliding velocity, :math:`|F_y| / |F|
= \\sin\\alpha`.  :func:`ac_tire` maps the AC ``tyres.ini`` data to the
model; the default :class:`Tire` holds assumed values of a passenger tire.
"""

from dataclasses import dataclass, field
import math
import numpy as np

from solvcon import multibody

#: Standard gravity in m/s^2, shared by the golf_r models.
GRAVITY = 9.81
#: The largest driving slip ratio that the force inversion uses.
MAX_DRIVE_SLIP = 1.0


@dataclass(frozen=True)
class Tire:
    """Per-tire parameters in SI units.

    The stiffnesses hold at ``reference_load``: the cornering stiffness
    dF_y/dalpha (N/rad) and the longitudinal slip stiffness dF_x/dkappa
    (N).  The peak friction is ``friction`` at the reference load and
    scales as (F_z / F_z0)^(n - 1) with the load exponent n; the slip
    stiffness then scales as F_z^n.  ``shape`` is the Magic Formula C of
    both directions (1.3: the typical lateral value of Pacejka 2012, Sec.
    4.3.2) and ``curvature`` its E.  None for a longitudinal value means
    the lateral one.

    ``relaxation_length`` is 0.45 m, inside the typical passenger-car
    range of 0.3 m to 0.6 m at nominal load (Pacejka 2012, ch. 7); the
    tire of the AC data uses its ``RELAXATION_LENGTH`` (see
    :func:`ac_tire`).  The models use one length for the longitudinal and
    the lateral slip.
    """

    longitudinal_stiffness: float = 80000.0
    cornering_stiffness: float = 68755.0
    friction: float = 1.0
    reference_load: float = 4000.0
    load_exponent: float = 0.9
    shape: float = 1.3
    curvature: float = 0.0
    relaxation_length: float = 0.45
    longitudinal_friction: float = None
    longitudinal_load_exponent: float = None
    model: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        values = (
            self.longitudinal_stiffness,
            self.cornering_stiffness,
            self.friction,
            self.reference_load,
            self.relaxation_length,
            self.load_exponent,
        )
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError(
                'Tire stiffness, friction, load, length, and load exponent '
                'must be positive'
            )
        object.__setattr__(
            self, 'model', multibody.MagicFormulaTire(**self.parameters())
        )

    def parameters(self):
        """The :class:`solvcon.multibody.MagicFormulaTire` arguments.

        :rtype: dict
        """
        mu_x = (
            self.friction
            if self.longitudinal_friction is None
            else self.longitudinal_friction
        )
        n_x = (
            self.load_exponent
            if self.longitudinal_load_exponent is None
            else self.longitudinal_load_exponent
        )
        fz0 = self.reference_load

        def curve(mu, n, stiffness):
            return {
                'mu': mu,
                'load_exponent': n,
                'stiffness': stiffness / (self.shape * mu * fz0),
                'shape': self.shape,
                'curvature': self.curvature,
            }

        return {
            'fz0': fz0,
            'longitudinal': curve(mu_x, n_x, self.longitudinal_stiffness),
            'lateral': curve(
                self.friction, self.load_exponent, self.cornering_stiffness
            ),
        }


def ac_tire(section):
    """The :class:`Tire` of an AC ``tyres.ini`` section (tyre model v10).

    * The peak friction is ``DY_REF`` (lateral) and ``DX_REF``
      (longitudinal) at ``FZ0`` with the load exponents ``LS_EXPY`` and
      ``LS_EXPX``: mu(F_z) = DY_REF (F_z / FZ0)^(LS_EXPY - 1).
    * The shape C makes the sliding level D sin(C pi / 2) equal to
      ``FALLOFF_LEVEL`` times the peak (E = 0).
    * The lateral peak is at ``FRICTION_LIMIT_ANGLE``: B tan(alpha_peak) =
      tan(pi / (2 C)).
    * ``CX_MULT`` scales B of the longitudinal curve (assumed meaning).
    * ``RELAXATION_LENGTH`` is the relaxation length (m).

    :param section: the key-value pairs of a ``[FRONT]`` or ``[REAR]``
        section
    :type section: dict
    :return: the tire and the assumptions of the mapping
    :rtype: tuple
    """
    falloff = float(section['FALLOFF_LEVEL'])
    if not 0.0 < falloff < 1.0:
        raise ValueError('tyres.ini: FALLOFF_LEVEL must be in (0, 1)')
    shape = 2.0 - 2.0 / math.pi * math.asin(falloff)
    peak = math.radians(float(section['FRICTION_LIMIT_ANGLE']))
    b_y = math.tan(0.5 * math.pi / shape) / math.tan(peak)
    b_x = float(section.get('CX_MULT', '1')) * b_y
    fz0 = float(section['FZ0'])
    mu_y = float(section['DY_REF'])
    mu_x = float(section['DX_REF'])
    tire = Tire(
        longitudinal_stiffness=b_x * shape * mu_x * fz0,
        cornering_stiffness=b_y * shape * mu_y * fz0,
        friction=mu_y,
        reference_load=fz0,
        load_exponent=float(section['LS_EXPY']),
        shape=shape,
        curvature=0.0,
        relaxation_length=float(section['RELAXATION_LENGTH']),
        longitudinal_friction=mu_x,
        longitudinal_load_exponent=float(section['LS_EXPX']),
    )
    assumptions = [
        'tyre: Magic Formula with the AC peak friction, load exponents, '
        'peak slip angle, and fall-off level; E = 0, so the curve shape '
        'between the peak and the sliding level is assumed',
        'tyre: CX_MULT scales the longitudinal stiffness factor B '
        '(its AC meaning is not documented)',
        'tyre: no camber thrust, pressure, temperature, or aligning moment '
        '(DCAMBER_*, PRESSURE_*, the thermal model are not used)',
    ]
    return tire, assumptions


def friction_limit(tire, load):
    """Return the lateral peak force of the tire at a vertical load.

    :param tire: The tire parameters
    :type tire: Tire
    :param load: The vertical load in N
    :type load: float
    :return: The peak force in N, zero for a nonpositive load
    :rtype: float
    """
    if load <= 0:
        return 0.0
    return tire.model.friction(load)[1] * load


def _force(tire, load, slip_ratio, slip_angle):
    """Return (Fx, Fy) without input checks; see :func:`force`."""
    return tire.model.force(load, slip_ratio, slip_angle)


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


def force_for_demand(tire, load, demand, slip_angle):
    """Return the tire force that delivers a longitudinal force demand.

    The slip ratio solves :func:`force` = ``demand`` at ``slip_angle`` on
    the stable branch of the curve (between zero and the peak).  If the
    demand is beyond the peak, the wheel locks (braking, kappa = -1) or
    spins at MAX_DRIVE_SLIP (driving), and the function returns the force
    at that slip ratio.

    :param tire: The tire parameters
    :type tire: Tire
    :param load: The vertical load in N
    :type load: float
    :param demand: The requested longitudinal force in N
    :type demand: float
    :param slip_angle: The slip angle in rad
    :type slip_angle: float
    :return: The delivered force (Fx, Fy) in N and the slip ratio
    :rtype: tuple
    """
    return tire.model.force_for_demand(load, demand, slip_angle)


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
