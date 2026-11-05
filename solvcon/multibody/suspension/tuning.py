# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tuning: the setup of a corner, the parts a race engineer changes at the
track, and the numbers that say what a change does.

A setup is a dict in the units of a setup sheet:

* ``spring_rate`` (N/mm): the coil spring.
* ``ride_height`` (mm, + raises the corner): the body length of a
  full-length adjustable coilover.  The spring then holds the corner at
  the static wheel travel -ride_height, so the geometry at ride height
  (camber, roll center, motion ratio) moves along the K&C curves; the
  damper moves with it, so the stroke to the droop stop stays.
* ``bump_stop_gap`` (mm): the travel from static to the bump stop.
* ``bump_clicks``, ``rebound_clicks``: the adjusters of a click damper (0
  is soft); or ``damper_bump`` and ``damper_rebound`` (N.s/mm at the
  damper, the low-speed slopes) for a damper without clicks.
* ``arb_rate`` (N m/deg): the anti-roll bar torsion rate, when the corner
  has a bar.
* ``tire_rate`` (N/mm): the tire vertical rate (pressure).

:func:`readouts` gives the engineering numbers at the static travel: the
motion ratio, the wheel rate, the ride rate (the wheel rate in series with
the tire), the ride frequency of Milliken & Milliken (RCVD, 1995, Ch. 16),
f = sqrt(K_ride / m_s) / (2 pi), the damped ride and wheel hop modes of
the linear corner (equal to the engine modal analysis), the bump and
rebound damping ratios c MR^2 / (2 sqrt(K_ride m_s)), the static
alignment and roll center, and the travel left to the stops.
"""

import copy
import json
import math

import numpy as np

from . import axle
from . import kc
from . import ride
from .model import MM, curve_slope, damper_table

__all__ = [  # noqa: F822
    'FIELDS',
    'apply',
    'baseline',
    'load',
    'readouts',
    'save',
]

#: The setup fields: key, label, unit.
FIELDS = (
    ('spring_rate', "spring rate", "N/mm"),
    ('ride_height', "ride height", "mm"),
    ('bump_stop_gap', "bump stop gap", "mm"),
    ('bump_clicks', "damper bump", "clicks"),
    ('rebound_clicks', "damper rebound", "clicks"),
    ('damper_bump', "damper bump", "N.s/mm"),
    ('damper_rebound', "damper rebound", "N.s/mm"),
    ('arb_rate', "anti-roll bar", "N m/deg"),
    ('tire_rate', "tire rate", "N/mm"),
)


def _adjustable(model):
    spec = model.damper_spec
    return spec is not None and "adjustable" == spec.get("kind")


def baseline(model):
    """The setup a corner model is written with.

    :return: setup dict; only the fields the model has
    """
    out = {'spring_rate': model.spring_k * 1e-3,
           'ride_height': -model.static_travel / MM,
           'tire_rate': model.tire_k * 1e-3}
    if model.stops is not None:
        out['bump_stop_gap'] = (model.stops["bump_travel"]
                                - model.static_travel) / MM
    if _adjustable(model):
        out['bump_clicks'] = int(model.damper_spec.get("bump_clicks", 0))
        out['rebound_clicks'] = int(model.damper_spec.get(
            "rebound_clicks", 0))
    elif model.damper_curve is not None:
        out['damper_bump'] = curve_slope(model.damper_curve, -1e-9) * 1e-3
        out['damper_rebound'] = curve_slope(model.damper_curve,
                                            1e-9) * 1e-3
    else:
        out['damper_bump'] = out['damper_rebound'] = model.spring_c * 1e-3
    if model.arb is not None:
        out['arb_rate'] = model.arb["k"] * math.pi / 180.0
    return out


def click_range(model):
    """The number of clicks of the damper, or None without clicks."""
    return int(model.damper_spec["clicks"]) if _adjustable(model) else None


def apply(model, setup):
    """A copy of ``model`` with ``setup`` applied.

    :param setup: setup dict; missing fields keep the model values
    :raises ValueError: for a field the model does not have or a value out
        of range
    """
    base = baseline(model)
    unknown = set(setup) - set(base)
    if unknown:
        raise ValueError(f"model '{model.name}' has no setup field "
                         f"{', '.join(sorted(unknown))}")
    m = copy.deepcopy(model)
    v = dict(base, **setup)
    if v['spring_rate'] <= 0.0 or v['tire_rate'] <= 0.0:
        raise ValueError("the spring and tire rates must be positive")
    m.spring_k = float(v['spring_rate']) * 1e3
    m.tire_k = float(v['tire_rate']) * 1e3
    m.static_travel = -float(v['ride_height']) * MM
    if m.stops is not None:
        # A full-length adjustable coilover: the ride height moves the
        # damper with the corner, so the stroke to the droop stop stays;
        # the bump stop gap is its own setting.
        gap = float(v['bump_stop_gap']) * MM
        if gap <= 0.0:
            raise ValueError("the bump stop gap must be positive")
        shift = m.static_travel - model.static_travel
        m.stops = dict(m.stops, bump_travel=m.static_travel + gap,
                       droop_travel=m.stops["droop_travel"] + shift)
    if _adjustable(m):
        spec = dict(m.damper_spec, bump_clicks=v['bump_clicks'],
                    rebound_clicks=v['rebound_clicks'])
        m.damper_spec = spec
        m.damper_curve = damper_table(spec)
    else:
        bump, rebound = float(v['damper_bump']), float(v['damper_rebound'])
        if bump < 0.0 or rebound < 0.0:
            raise ValueError("the damper slopes must not be negative")
        if m.damper_curve is None and bump == rebound:
            m.spring_c = bump * 1e3
        else:
            # Scale each side of the curve to its new low-speed slope.
            old = m.damper_curve or damper_table(
                {"kind": "asymmetric", "bump": m.spring_c * 1e-3,
                 "rebound": m.spring_c * 1e-3})
            sb = curve_slope(old, -1e-9) * 1e-3
            sr = curve_slope(old, 1e-9) * 1e-3
            vel, force = old
            force = [f * (bump / sb if x < 0.0 else rebound / sr)
                     if (sb if x < 0.0 else sr) > 0.0 else 0.0
                     for x, f in zip(vel, force)]
            m.damper_curve = (list(vel), force)
    if m.damper_curve is not None:
        m.spring_c = 0.5 * (curve_slope(m.damper_curve, -1e-9)
                            + curve_slope(m.damper_curve, 1e-9))
    if 'arb_rate' in v:
        if v['arb_rate'] < 0.0:
            raise ValueError("the bar rate must not be negative")
        m.arb = dict(m.arb, k=float(v['arb_rate']) * 180.0 / math.pi)
    return m


def readouts(mbs, model, sprung_mass, gravity=9.81, axle_roll=True):
    """The engineering numbers of a corner at its static travel.

    :param model: the corner model with its setup applied
    :param sprung_mass: the sprung mass the corner carries (kg)
    :param axle_roll: also the roll stiffness of the axle of two such
        corners, the springs alone and with the bar (N m/deg)
    :return: dict of floats in setup sheet units (mm, N/mm, Hz, deg)
    """
    lin = ride.LinearCorner(mbs, model, sprung_mass, gravity)
    s0 = float(model.static_travel)
    k_w = lin.wheel_rate
    k_t = model.tire_k
    k_ride = k_w * k_t / (k_w + k_t)
    n = lin.M.shape[0]
    first = np.block([[np.zeros((n, n)), np.eye(n)],
                      [-np.linalg.solve(lin.M, lin.K),
                       -np.linalg.solve(lin.M, lin.C)]])
    lam = np.linalg.eigvals(first)
    lam = lam[lam.imag > 0.0]
    lam = lam[np.argsort(np.abs(lam))]
    freq = np.abs(lam) / (2.0 * math.pi)
    zeta = -lam.real / np.abs(lam)
    mr = lin.motion_ratio
    curve = model.damper_curve
    if curve is None:
        c_b = c_r = model.spring_c
    else:
        c_b, c_r = curve_slope(curve, -1e-9), curve_slope(curve, 1e-9)
    crit = 2.0 * math.sqrt(k_ride * sprung_mass)
    sweep = kc.bump_sweep(mbs, model, [s0])
    deg = 180.0 / math.pi
    out = {
        'static_travel': s0 / MM,
        'motion_ratio': mr,
        'wheel_rate': k_w * 1e-3,
        'ride_rate': k_ride * 1e-3,
        'ride_frequency': math.sqrt(k_ride / sprung_mass) / (2 * math.pi),
        'ride_mode': float(freq[0]), 'ride_mode_damping': float(zeta[0]),
        'hop_mode': float(freq[1]), 'hop_mode_damping': float(zeta[1]),
        'damping_ratio_bump': c_b * mr * mr / crit,
        'damping_ratio_rebound': c_r * mr * mr / crit,
        'camber': float(sweep['camber'][0]) * deg,
        'toe': float(sweep['toe'][0]) * deg,
        'roll_center': float(sweep['roll_center'][0]) / MM,
        'static_load': lin.static_load,
    }
    if not math.isnan(float(sweep['caster'][0])):
        out['caster'] = float(sweep['caster'][0]) * deg
        out['kpi'] = float(sweep['kpi'][0]) * deg
    if model.stops is not None:
        out['bump_travel_left'] = (model.stops["bump_travel"] - s0) / MM
        out['droop_travel_left'] = (s0 - model.stops["droop_travel"]) / MM
    if axle_roll:
        travels = [-2e-3, 0.0, 2e-3]
        springs = axle.roll_sweep(mbs, model, travels, False, offset=s0)
        out['roll_stiffness_springs'] = \
            float(springs['roll_stiffness'][1]) * math.pi / 180.0
        if model.arb is not None:
            both = axle.roll_sweep(mbs, model, travels, True, offset=s0)
            out['roll_stiffness'] = \
                float(both['roll_stiffness'][1]) * math.pi / 180.0
        else:
            out['roll_stiffness'] = out['roll_stiffness_springs']
    return out


#: The readouts in the order of a setup sheet: key, label, unit, format.
READOUTS = (
    ('ride_frequency', "ride frequency (K_ride, m_s)", "Hz", ".2f"),
    ('ride_mode', "ride mode (damped, engine)", "Hz", ".2f"),
    ('ride_mode_damping', "ride mode damping", "", ".2f"),
    ('damping_ratio_bump', "damping ratio, bump", "", ".2f"),
    ('damping_ratio_rebound', "damping ratio, rebound", "", ".2f"),
    ('hop_mode', "wheel hop mode", "Hz", ".1f"),
    ('hop_mode_damping', "wheel hop damping", "", ".2f"),
    ('motion_ratio', "motion ratio (spring / wheel)", "", ".3f"),
    ('wheel_rate', "wheel rate", "N/mm", ".1f"),
    ('ride_rate', "ride rate (with the tire)", "N/mm", ".1f"),
    ('static_travel', "static travel from design (+ bump)", "mm", ".1f"),
    ('camber', "static camber", "deg", "+.3f"),
    ('toe', "static toe (+ in)", "deg", "+.3f"),
    ('caster', "static caster", "deg", ".2f"),
    ('kpi', "static KPI", "deg", ".2f"),
    ('roll_center', "roll center height", "mm", ".1f"),
    ('roll_stiffness_springs', "axle roll stiffness, springs", "N m/deg",
     ".0f"),
    ('roll_stiffness', "axle roll stiffness, springs and bar", "N m/deg",
     ".0f"),
    ('bump_travel_left', "travel to the bump stop", "mm", ".1f"),
    ('droop_travel_left', "travel to the droop stop", "mm", ".1f"),
)


def save(path, setups, note=""):
    """Write ``{name: setup}`` (for example ``{"corner": ...}`` or
    ``{"front": ..., "rear": ...}``) as a setup sheet file."""
    with open(path, 'w') as fobj:
        json.dump({"setup_sheet": 1, "note": note, "setups": setups}, fobj,
                  indent=2)
        fobj.write("\n")


def load(path):
    """Read a setup sheet file written by :func:`save`.

    :return: ``{name: setup}``
    :raises ValueError: when the file is not a setup sheet
    """
    with open(path) as fobj:
        data = json.load(fobj)
    if not isinstance(data, dict) or "setups" not in data:
        raise ValueError(f"{path}: not a setup sheet")
    return data["setups"]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
