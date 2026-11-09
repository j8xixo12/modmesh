# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The key performance indicators (KPIs) of the overview: their definitions,
units, the engineering question each answers, and the target bands.

A target band is given only where a published source gives one; every
other KPI shows its value with no band until the user loads targets (see
:func:`load_targets`).  The sources:

* Ride frequency: M. Giaraffa, "Tech Tip: Springs & Dampers, Part One",
  OptimumG: 0.5 - 1.5 Hz for passenger cars, 1.5 - 2.0 Hz for sedan
  racecars and moderate downforce formula cars, 3.0 - 5.0+ Hz for high
  downforce racecars.
* Damping ratio: OptimumG, "Tech Tip: Springs & Dampers, Episode Four" (as
  cited by a secondary copy): about 0.25 for passenger car ride comfort,
  0.65 - 0.70 as a racecar baseline.
"""

import json

__all__ = [
    'FIELD_HELP',
    'GROUPS',
    'KPIS',
    'PROFILES',
    'band',
    'load_targets',
    'status',
]

#: The target profiles: key, label.
PROFILES = (
    ('road', "Road car (comfort)"),
    ('track', "Sedan racecar / track"),
    ('downforce', "High-downforce racecar"),
)

_RIDE = "M. Giaraffa, Tech Tip: Springs & Dampers, Part One (OptimumG)"
_DAMP = ("OptimumG, Tech Tip: Springs & Dampers, Episode Four (secondary "
         "citation)")

#: The engineering questions in the order of the overview: key, title,
#: what the group answers.
GROUPS = (
    ('ride', "Ride (corner)",
     "Is the corner soft or stiff, and is the body motion controlled?"),
    ('kc', "Kinematics (corner)",
     "How does the wheel move in bump: camber, toe, roll center?"),
    ('handling', "Roll and handling (vehicle)",
     "How much does the car roll, which end slides first, how fast does "
     "it turn in?"),
    ('braking', "Braking and traction (vehicle)",
     "Which axle locks or spins first, and how much does the nose dive?"),
)

#: ``{key: dict(group, label, unit, fmt, definition, targets, source)}``;
#: ``targets`` is ``{profile: (lo, hi)}`` or empty.
KPIS = {
    'ride_frequency': dict(
        group='ride', label="ride frequency", unit="Hz", fmt=".2f",
        definition="f = sqrt(K_ride / m_s) / (2 pi): the undamped body "
                   "bounce frequency of the corner, K_ride the wheel rate "
                   "in series with the tire, m_s the corner sprung mass "
                   "(Milliken, RCVD, Ch. 16).  Higher is stiffer: less "
                   "travel and quicker response, less comfort and "
                   "mechanical grip.",
        targets={'road': (0.5, 1.5), 'track': (1.5, 2.0),
                 'downforce': (3.0, 5.0)},
        source=_RIDE),
    'damping_ratio_bump': dict(
        group='ride', label="damping ratio, bump", unit="", fmt=".2f",
        definition="zeta = c MR^2 / (2 sqrt(K_ride m_s)) with the "
                   "low-speed bump slope c: 1 is critical damping.",
        targets={'road': (0.2, 0.3), 'track': (0.65, 0.70),
                 'downforce': (0.65, 0.70)},
        source=_DAMP),
    'damping_ratio_rebound': dict(
        group='ride', label="damping ratio, rebound", unit="", fmt=".2f",
        definition="zeta with the low-speed rebound slope.",
        targets={'road': (0.2, 0.3), 'track': (0.65, 0.70),
                 'downforce': (0.65, 0.70)},
        source=_DAMP),
    'wheel_rate': dict(
        group='ride', label="wheel rate", unit="N/mm", fmt=".1f",
        definition="the spring rate at the wheel center: spring rate "
                   "times the motion ratio squared.",
        targets={}, source=""),
    'motion_ratio': dict(
        group='ride', label="motion ratio", unit="", fmt=".3f",
        definition="spring travel per unit wheel travel.", targets={},
        source=""),
    'hop_mode': dict(
        group='ride', label="wheel hop", unit="Hz", fmt=".1f",
        definition="the damped unsprung (wheel hop) mode of the corner.",
        targets={}, source=""),
    'roll_stiffness': dict(
        group='ride', label="axle roll stiffness", unit="N m/deg",
        fmt=".0f",
        definition="roll moment per degree of axle roll, springs and bar.",
        targets={}, source=""),
    'camber_gain': dict(
        group='kc', label="camber change in bump", unit="deg/10 mm",
        fmt="+.3f",
        definition="camber change per 10 mm of bump at static: negative "
                   "gains negative camber in bump (keeps the outer tire "
                   "upright in roll).",
        targets={}, source=""),
    'bump_steer': dict(
        group='kc', label="bump steer", unit="deg/10 mm", fmt="+.3f",
        definition="toe change per 10 mm of bump at static (+ toe-in).  "
                   "Bump steer turns the car on bumps and in roll.",
        targets={}, source=""),
    'roll_center': dict(
        group='kc', label="roll center height", unit="mm", fmt=".0f",
        definition="the kinematic roll center height at static.",
        targets={}, source=""),
    'kpi': dict(
        group='kc', label="kingpin inclination", unit="deg", fmt=".2f",
        definition="the steering axis angle in front view.", targets={},
        source=""),
    'caster': dict(
        group='kc', label="caster", unit="deg", fmt=".2f",
        definition="the steering axis angle in side view.", targets={},
        source=""),
    'bump_travel_left': dict(
        group='kc', label="travel to the bump stop", unit="mm", fmt=".1f",
        definition="wheel travel from static to the bump stop.",
        targets={}, source=""),
    'roll_gradient': dict(
        group='handling', label="roll gradient", unit="deg/g", fmt=".2f",
        definition="body roll per g of steady lateral acceleration.",
        targets={}, source=""),
    'lltd': dict(
        group='handling', label="front lateral load transfer", unit="%",
        fmt=".1f",
        definition="the front share of the total lateral load transfer "
                   "(LLTD).  More front share: more understeer at the "
                   "limit.",
        targets={}, source=""),
    'understeer_gradient': dict(
        group='handling', label="understeer gradient", unit="deg/g",
        fmt="+.2f",
        definition="K = d(delta - L/R)/d(a_y): + understeer, - oversteer "
                   "(SAE J670).",
        targets={}, source=""),
    'lateral_limit': dict(
        group='handling', label="steady lateral limit", unit="g",
        fmt=".3f",
        definition="the largest steady lateral acceleration of the "
                   "handling diagram.",
        targets={}, source=""),
    'yaw_gain': dict(
        group='handling', label="steady yaw gain", unit="1/s", fmt=".2f",
        definition="steady yaw rate per front wheel angle of the step "
                   "steer (deg/s per deg).",
        targets={}, source=""),
    'step_ay': dict(
        group='handling', label="step steer lateral acceleration",
        unit="g", fmt=".3f",
        definition="the steady lateral acceleration of the step steer.",
        targets={}, source=""),
    'brake_limit': dict(
        group='braking', label="braking limit", unit="g", fmt=".3f",
        definition="the deceleration where the first axle locks.",
        targets={}, source=""),
    'anti_dive': dict(
        group='braking', label="anti-dive", unit="%", fmt=".0f",
        definition="the share of the braking load transfer the front "
                   "geometry carries instead of the springs.",
        targets={}, source=""),
    'traction_limit': dict(
        group='braking', label="traction limit", unit="g", fmt=".3f",
        definition="the acceleration where the first driven axle spins.",
        targets={}, source=""),
}


#: The setup fields in words: what the part is and what a change does.
FIELD_HELP = {
    'spring_rate': "coil spring rate at the spring (N/mm).  The wheel rate "
                   "is this times the motion ratio squared.  Stiffer: "
                   "higher ride frequency, less roll and pitch, less "
                   "mechanical grip on bumps.",
    'ride_height': "ride height change (mm, + raises the corner) of a "
                   "full-length adjustable coilover; the geometry moves "
                   "along its K&C curves.",
    'bump_stop_gap': "wheel travel from static to the bump stop (mm).",
    'damper_bump': "damper low-speed bump slope at the shaft (N.s/mm): "
                   "the force per shaft speed below the knee of the "
                   "force-velocity curve; the whole bump side scales "
                   "with it.  Controls body motion in compression.",
    'damper_rebound': "damper low-speed rebound slope at the shaft "
                      "(N.s/mm); the whole rebound side scales with it.  "
                      "Controls the body rising and the wheel drop.",
    'bump_clicks': "bump click of a damper with a measured click map; a "
                   "click is the dyno curve of that damper at that click.",
    'rebound_clicks': "rebound click of a damper with a measured click "
                      "map.",
    'arb_rate': "anti-roll bar torsion rate (N m/deg of bar twist).  "
                "Stiffer: less roll, more load transfer on that axle (the "
                "front: more understeer).",
    'tire_rate': "tire vertical rate (N/mm), set by the pressure.",
}


def band(key, profile, custom=None):
    """The target band ``(lo, hi)`` of a KPI, or None.

    :param custom: ``{key: (lo, hi)}`` the user loaded; it wins over the
        published band
    """
    if custom and key in custom:
        return tuple(custom[key])
    return KPIS[key]['targets'].get(profile)


def status(value, target):
    """``'in'``, ``'out'``, or ``'none'`` (no target)."""
    if target is None or value is None:
        return 'none'
    lo, hi = target
    return 'in' if lo <= value <= hi else 'out'


def load_targets(path):
    """Read user targets: a JSON object ``{kpi: [lo, hi]}``.

    :raises ValueError: for an unknown KPI or a bad band
    """
    with open(path) as fobj:
        data = json.load(fobj)
    out = {}
    for key, value in data.items():
        if key not in KPIS:
            raise ValueError(f"targets: unknown KPI {key!r}")
        if len(value) != 2 or not float(value[0]) <= float(value[1]):
            raise ValueError(f"targets: {key} needs [lo, hi] with lo <= hi")
        out[key] = (float(value[0]), float(value[1]))
    return out

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
