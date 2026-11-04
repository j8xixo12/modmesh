# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Suspension K&C metrics from a pose and its sensitivities (ISO 8855).

Sign conventions (left corner, side = +1; a right corner uses side = -1):

* camber: negative when the top of the wheel leans toward the vehicle;
* toe: positive for toe-in (front of the wheel toward the centerline);
* caster: positive when the top of the steering axis is rearward;
* kingpin inclination (KPI): positive when the top of the axis is inboard;
* scrub radius: positive when the contact patch is outboard of the steering
  axis ground point;
* mechanical trail: positive when the steering axis ground point is ahead of
  the contact patch;
* motion ratio: spring compression per unit wheel bump;
* roll center height: from the front-view line through the contact patch,
  perpendicular to the contact patch velocity, at the vehicle centerline.

The steering axis is the instantaneous screw axis of the upright for a rack
motion, so the same code covers a double wishbone (axis through the ball
joints) and a multi-link with a virtual kingpin.
"""

import math

import numpy as np


def rot(q):
    """Haug A(p): body-to-global rotation matrix.

    :param q: Euler parameters [e0, e1, e2, e3]
    :type q: list
    :return: 3x3 matrix
    :rtype: numpy.ndarray
    """
    e0, e1, e2, e3 = q
    e = np.array([e1, e2, e3])
    et = np.array([[0, -e3, e2], [e3, 0, -e1], [-e2, e1, 0]])
    return (2 * e0 * e0 - 1) * np.eye(3) + 2 * (np.outer(e, e) + e0 * et)


def wheel_geometry(r_wc, a_up, spin_local, radius, side):
    """Camber, toe, and contact patch of a wheel.

    :param r_wc: wheel center (m)
    :type r_wc: numpy.ndarray
    :param a_up: upright rotation matrix
    :type a_up: numpy.ndarray
    :param spin_local: outward spin axis in the upright frame
    :type spin_local: numpy.ndarray
    :param radius: loaded tire radius (m)
    :type radius: float
    :param side: +1 left, -1 right
    :type side: int
    :return: dict with camber, toe (rad), spin_axis, contact_patch
    :rtype: dict
    """
    s = a_up @ spin_local
    s = s / np.linalg.norm(s)
    ez = np.array([0.0, 0.0, 1.0])
    # Top of the wheel inboard tilts the outward spin axis up (s_z > 0): that
    # is negative camber.
    camber = -math.asin(float(np.clip(s @ ez, -1.0, 1.0)))
    h = side * np.cross(s, ez)
    toe = -side * math.atan2(h[1], h[0])
    down = -ez - (-ez @ s) * s
    cp = r_wc + radius * down / np.linalg.norm(down)
    return {"camber": camber, "toe": toe, "spin_axis": s, "contact_patch": cp}


def screw_axis(r_ref, dr_ds, dw_ds):
    """Instantaneous screw axis of a body from its sensitivities.

    :param r_ref: reference point of the body (m)
    :type r_ref: numpy.ndarray
    :param dr_ds: velocity of the reference point per unit input
    :type dr_ds: numpy.ndarray
    :param dw_ds: angular velocity (global) per unit input
    :type dw_ds: numpy.ndarray
    :return: (unit axis direction, point on the axis) or (None, None) for pure
        translation
    :rtype: tuple
    """
    w2 = float(dw_ds @ dw_ds)
    if w2 < 1e-24:
        return None, None
    n = dw_ds / math.sqrt(w2)
    p = r_ref + np.cross(dw_ds, dr_ds) / w2
    return n, p


def ball_joint_axis(p_lower, p_upper):
    """Steering axis of a double wishbone: the line through the ball joints.

    With the wishbones held, the upright can only turn about this line, so
    it is the steering (kingpin) axis that OptimumK, Lotus, and the Adams
    geometric method report.

    :param p_lower: lower ball joint (global, m)
    :type p_lower: numpy.ndarray
    :param p_upper: upper ball joint (global, m)
    :type p_upper: numpy.ndarray
    :return: (unit direction, point on the axis)
    :rtype: tuple
    """
    d = np.asarray(p_upper, float) - np.asarray(p_lower, float)
    return d / np.linalg.norm(d), np.asarray(p_lower, float)


def steering_axis_metrics(n, p, cp, side):
    """Caster, KPI, scrub radius, and mechanical trail from the steering axis.

    :param n: steering axis direction
    :type n: numpy.ndarray
    :param p: point on the steering axis
    :type p: numpy.ndarray
    :param cp: contact patch center
    :type cp: numpy.ndarray
    :param side: +1 left, -1 right
    :type side: int
    :return: dict with caster, kpi (rad), scrub, trail (m), ground_point
    :rtype: dict
    """
    if n[2] < 0:
        n = -n
    caster = math.atan2(-n[0], n[2])
    kpi = math.atan2(-side * n[1], n[2])
    t = (cp[2] - p[2]) / n[2]
    gp = p + t * n
    return {"caster": caster, "kpi": kpi, "scrub": side * (cp[1] - gp[1]),
            "trail": gp[0] - cp[0], "ground_point": gp}


def point_rate(r_body, a_body, local, dr_ds, dw_ds):
    """Position and rate of a body-fixed point.

    :return: (position, d position / ds)
    :rtype: tuple
    """
    arm = a_body @ np.asarray(local)
    return r_body + arm, dr_ds + np.cross(dw_ds, arm)


def roll_center_height(cp, v_cp):
    """Front-view roll center height at the centerline y = 0.

    :param cp: contact patch center
    :type cp: numpy.ndarray
    :param v_cp: contact patch velocity per unit bump (upright-fixed point)
    :type v_cp: numpy.ndarray
    :return: roll center height (m)
    :rtype: float
    """
    u = np.array([v_cp[1], v_cp[2]])
    d = np.array([-u[1], u[0]])  # perpendicular to the front-view velocity
    if abs(d[0]) < 1e-15:
        return math.inf
    t = -cp[1] / d[0]
    return float(cp[2] + t * d[1])


def anti_dive(v_cp, wheelbase, cg_height, front_brake_fraction):
    """Side-view anti-dive angle and fraction for outboard front brakes.

    The side-view swing arm line from the contact patch to the side-view
    instant center is perpendicular to the contact patch path, so its angle
    from the ground is atan2(v_x, v_z) of the contact patch velocity per unit
    bump. Positive when the contact patch moves forward in bump.

    :param v_cp: contact patch velocity per unit bump
    :type v_cp: numpy.ndarray
    :param wheelbase: wheelbase (m)
    :type wheelbase: float
    :param cg_height: center of gravity height (m)
    :type cg_height: float
    :param front_brake_fraction: front share of the braking force
    :type front_brake_fraction: float
    :return: (angle in rad, anti-dive fraction, 1.0 = 100 %)
    :rtype: tuple
    """
    angle = math.atan2(v_cp[0], v_cp[2])
    return (angle,
            math.tan(angle) * wheelbase / cg_height * front_brake_fraction)


# An instant center farther than this from the contact patch is treated as
# at infinity (in-plane angular rate ~ 0, e.g. parallel equal wishbones).
# 1000 km is far beyond any real swing arm, and far below the distance that
# rounding noise of a zero angular rate (~1e-15 rad/m) would give.
IC_MAX_DISTANCE = 1e6


def _plane_ic(cp, v_cp, w, axis):
    """Instant center of the upright motion projected on a plane.

    The plane holds ``cp`` and is normal to the global ``axis``. The
    in-plane velocity of an upright point p is v_cp + w x (p - cp) without
    the ``axis`` component; it is zero at p = cp + (w_n e_n x v_cp) / w_n^2.

    :return: instant center (m), or NaN row if it is at infinity
    :rtype: numpy.ndarray
    """
    cp = np.asarray(cp, float)
    v = np.array(v_cp, float)
    v[axis] = 0.0
    wn = float(w[axis])
    if abs(wn) * IC_MAX_DISTANCE <= float(np.linalg.norm(v)):
        return np.full(3, np.nan)
    e = np.zeros(3)
    e[axis] = 1.0
    return cp + np.cross(e, v) / wn


def front_view_ic(cp, v_cp, w):
    """Front-view instant center (FVIC) in the plane x = cp_x.

    :param cp: contact patch center (m)
    :type cp: numpy.ndarray
    :param v_cp: contact patch velocity per unit bump (upright-fixed point)
    :type v_cp: numpy.ndarray
    :param w: upright angular rate per unit bump (rad/m, global)
    :type w: numpy.ndarray
    :return: FVIC (m), or a NaN row when it is at infinity
    :rtype: numpy.ndarray
    """
    return _plane_ic(cp, v_cp, w, 0)


def side_view_ic(cp, v_cp, w):
    """Side-view instant center (SVIC) in the plane y = cp_y.

    :param cp: contact patch center (m)
    :type cp: numpy.ndarray
    :param v_cp: contact patch velocity per unit bump (upright-fixed point)
    :type v_cp: numpy.ndarray
    :param w: upright angular rate per unit bump (rad/m, global)
    :type w: numpy.ndarray
    :return: SVIC (m), or a NaN row when it is at infinity
    :rtype: numpy.ndarray
    """
    return _plane_ic(cp, v_cp, w, 1)


def roll_center_point(cp, v_cp, fvic):
    """Roll center: the contact patch to FVIC line at the centerline y = 0.

    If the FVIC is at infinity, the line is perpendicular to the front-view
    contact patch velocity (the same line as in ``roll_center_height``).

    :param cp: contact patch center (m)
    :type cp: numpy.ndarray
    :param v_cp: contact patch velocity per unit bump
    :type v_cp: numpy.ndarray
    :param fvic: front-view instant center (m) or NaN row
    :type fvic: numpy.ndarray
    :return: roll center point (cp_x, 0, height) in m; NaN row if the line
        is parallel to y = 0
    :rtype: numpy.ndarray
    """
    cp = np.asarray(cp, float)
    if np.all(np.isfinite(fvic)):
        d = np.asarray(fvic, float) - cp
    else:
        d = np.array([0.0, -v_cp[2], v_cp[1]])
    if abs(d[1]) < 1e-15 * max(1.0, abs(d[2])):
        return np.full(3, np.nan)
    t = -cp[1] / d[1]
    return np.array([cp[0], 0.0, cp[2] + t * d[2]])


def rotation_vector(a):
    """Rotation vector (axis times angle) of a rotation matrix.

    :param a: 3x3 rotation matrix
    :type a: numpy.ndarray
    :return: rotation vector (rad)
    :rtype: numpy.ndarray
    """
    c = float(np.clip(0.5 * (np.trace(a) - 1.0), -1.0, 1.0))
    th = math.acos(c)
    w = np.array([a[2, 1] - a[1, 2], a[0, 2] - a[2, 0], a[1, 0] - a[0, 1]])
    if th < 1e-9:  # sin(th) / th = 1 to rounding
        return 0.5 * w
    return th / (2.0 * math.sin(th)) * w

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
