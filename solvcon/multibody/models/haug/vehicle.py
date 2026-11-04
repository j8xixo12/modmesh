# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Haug Section 12.5: dynamics of a vehicle (passenger car of Fig. 1.1.10).

Reference: E. J. Haug, "Computer-Aided Kinematics and Dynamics of Mechanical
Systems", Vol. I, Section 12.5, pp. 468-481, Figs. 12.5.1 to 12.5.13,
Tables 12.5.1 to 12.5.9. The transcribed data are in
``tests/data/multibody/haug_12_5.json``.

Bodies (book numbers equal solver numbers; solver body 0 is a ground body
that no joint uses)::

    1 chassis
    2 front-right wheel assembly (McPherson strut)
    3 front-left wheel assembly (McPherson strut)
    4 rear-right wheel assembly (trailing arm)
    5 rear-left wheel assembly (trailing arm)
    6 steering rack
    7 front roll-stabilizing bar (model 2 only)
    8 rear roll-stabilizing bar (model 2 only)

Global frame: z up, the car drives along +y, x points to the right side
(wheels 2 and 4 have x > 0).

Joints (Tables 12.5.2 to 12.5.6): revolute-spherical A-B and E-F (lower
control arms), strut B-C and F-G, translational D (rack), revolute H and I
(trailing arms), revolute J and K (stabilizer bars), distance 2-6 and 3-6
(tie rods), relative translational driver D (steering, Fig. 12.5.6).
Suspension: eight TSDA elements (Table 12.5.7).

Missing in the solver: the tire model of p. 473-474. The book defines a
vertical tire force F_ver = k_t * d (d from the spindle height above the
road), a lateral force F_lat = C_alpha * alpha with alpha = psi - psi_v
(Eqs. 12.5.1-12.5.3), and the friction limit |F_lat| <= mu * F_ver. The
solver has no point-on-plane contact force, no slip-angle force, and no
friction saturation. The book also does not print k_t, C_alpha, mu, the
tire radius, or g. This builder therefore has no tire forces, and no result
of Sections 12.5.2 and 12.5.4 can be reproduced with it.
"""

import math

from .... import multibody as mbs
from ...tools.arrays import as_list

Z = [0.0, 0.0, 0.0]

#: Gravitational acceleration [m/s^2]. Not printed in Section 12.5; the
#: value of Section 12.6 (inferred from its steady state) is used.
GRAVITY = 9.8

#: 55 mph in m/s (Section 12.5.4, p. 478).
MPH = 0.44704

#: Table 12.5.1 (p. 470): mass [kg] and Ixx, Iyy, Izz [kg m^2].
INERTIA = {
    1: (1247.5, [1240.1, 209.6, 139.9]),
    2: (37.6, [1.0, 0.5, 1.0]),
    3: (37.6, [1.0, 0.5, 1.0]),
    4: (43.0, [0.67, 1.5, 1.2]),
    5: (43.0, [0.67, 1.5, 1.2]),
    6: (0.1, [0.01, 0.01, 0.01]),
    7: (1.0, [0.1, 0.01, 0.01]),
    8: (1.0, [0.1, 0.01, 0.01]),
}

#: Table 12.5.8 (p. 475): position estimates; all Euler parameters e1-e3 = 0.
#: The book prints x = 0.58 for wheel 5; the symmetric value -0.58 is used
#: because wheel 5 is the rear-left wheel (joint I at x = -0.2026).
ESTIMATE = {
    1: [0.0, 0.0, 0.55],
    2: [0.6, 1.17, 0.34],
    3: [-0.6, 1.17, 0.34],
    4: [0.58, -1.29, 0.26],
    5: [-0.58, -1.29, 0.26],
    6: [0.0, 1.26, 0.16],
    7: [0.0, 1.05, 0.21],
    8: [0.0, -1.40, 0.38],
}

#: Table 12.5.9 (p. 476): equilibrium configuration of model 1, as
#: (x, y, z, e1, e2, e3). e0 follows from the normalization constraint.
EQUILIBRIUM = {
    1: [0.0, 0.0, 0.5385, -0.0065, 0.0, -0.0005],
    2: [0.6014, 1.1761, 0.3449, -0.0022, -0.0065, -0.001],
    3: [-0.5992, 1.1772, 0.3449, -0.0023, 0.0065, -0.0002],
    4: [0.5788, -1.2918, 0.2617, -0.0075, -0.0003, -0.0005],
    5: [-0.5812, -1.2907, 0.2617, -0.0075, 0.0003, -0.0005],
    6: [0.0012, 1.2625, 0.1471, -0.0065, 0.0, -0.0005],
    7: [0.001, 1.0566, 0.1983, 0.0675, 0.0, -0.0005],
    8: [-0.0013, -1.4097, 0.3756, -0.0508, 0.0, -0.0002],
}

#: Table 12.5.7 (p. 473): TSDA (body i, body j, P_i, P_j, k [N/m], c [N s/m],
#: L0 [m]).
TSDA = [
    (1, 2, [0.5011, 1.1162, 0.1998], Z, 38600.0, 150.0, 0.4992),
    (1, 3, [-0.5011, 1.1162, 0.1998], Z, 38600.0, 150.0, 0.4992),
    (1, 4, [0.58, -1.2875, 0.2], Z, 38600.0, 200.0, 0.5641),
    (1, 5, [-0.58, -1.2875, 0.2], Z, 38600.0, 200.0, 0.5641),
    (7, 2, [0.509, 0.1405, 0.02], [-0.0897, 0.0168, -0.2119], 83400.0, 0.0,
     0.1011),
    (7, 3, [-0.509, 0.1405, 0.02], [0.0897, 0.0168, -0.2119], 83400.0, 0.0,
     0.1011),
    (8, 4, [0.392, 0.168, -0.0474], [-0.1832, 0.048, -0.0423], 9330.0, 0.0,
     0.0943),
    (8, 5, [-0.392, 0.168, -0.0474], [0.1832, 0.048, -0.0423], 9330.0, 0.0,
     0.0943),
]


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


#: Joint data, Tables 12.5.2 to 12.5.6 (pp. 471-473). Each entry:
#: (name, kind, body i, body j, P_i, Q_i, P_j, Q_j, distance). The joint axis
#: of body i is Q_i - P_i. Kinds: rs = revolute-spherical, strut, trans,
#: rev, dist.
JOINTS = [
    ("A-B", "rs", 1, 2, [0.33, 1.1735, -0.3923], [0.322, 0.8675, -0.3893],
     [0.0507, 0.0321, -0.2044], [0.0507, 0.0321, 1.0], 0.203),
    ("E-F", "rs", 1, 3, [-0.33, 1.1735, -0.3923], [-0.322, 0.8675, -0.3893],
     [-0.0507, 0.0321, -0.2044], [-0.0507, 0.0321, 1.0], 0.203),
    ("B-C", "strut", 1, 2, [0.5011, 1.1162, 0.1998], [0.5011, 1.1162, 1.0],
     Z, [1.0, 0.0, 0.0], None),
    ("F-G", "strut", 1, 3, [-0.5011, 1.1162, 0.1998], [-0.5011, 1.1162, 1.0],
     Z, [1.0, 0.0, 0.0], None),
    ("D", "trans", 1, 6, [0.0, 1.2675, -0.3748], [1.0, 1.2675, -0.3748],
     Z, [1.0, 0.0, 0.0], None),
    ("H", "rev", 1, 4, [0.2026, -1.0095, -0.285], [-0.7633, -1.2683, -0.285],
     [-0.3774, 0.278, 0.0093], [-1.343, 0.0192, 0.0093], None),
    ("I", "rev", 1, 5, [-0.2026, -1.0095, -0.285], [0.7633, -1.2683, -0.285],
     [0.3774, 0.278, 0.0093], [1.343, 0.0192, 0.0093], None),
    ("J", "rev", 1, 7, [0.0, 1.061, -0.3263], [1.0, 1.061, -0.3263],
     Z, [1.0, 0.0, 0.0], None),
    ("K", "rev", 1, 8, [0.0, -1.4075, -0.1813], [1.0, -1.4075, -0.1813],
     Z, [1.0, 0.0, 0.0], None),
    ("2-6", "dist", 2, 6, [0.07, 0.155, -0.186], None, [0.305, 0.0, 0.0], None,
     0.3742),
    ("3-6", "dist", 3, 6, [-0.07, 0.155, -0.186], None, [-0.305, 0.0, 0.0],
     None, 0.3742),
]

#: Number of constraint rows of each joint kind (book p. 471).
ROWS = {"rs": 2, "strut": 2, "trans": 5, "rev": 5, "dist": 1, "driver": 1}

#: Fig. 12.5.6 (p. 477): rack amplitudes [m] read from the figure.
RACK_AMPLITUDE = {"lane1": 0.00142, "lane2": 0.00258, "circle": 0.00258}


def rack_position(kind, t):
    """Return the rack translation of Fig. 12.5.6 relative to the chassis.

    The figure shows a full sine period between t = 1 s and t = 4 s for the
    lane changes and the first quarter period followed by a constant value
    for the circle. The sine shape is an interpretation of the figure.

    :param kind: "lane1", "lane2", or "circle"
    :type kind: str
    :param t: time [s]
    :type t: float
    :return: rack translation along chassis x [m]
    :rtype: float
    """
    amp = RACK_AMPLITUDE[kind]
    if t <= 1.0:
        return 0.0
    if kind == "circle":
        if t >= 1.75:
            return amp
    elif t >= 4.0:
        return 0.0
    return amp * math.sin(2.0 * math.pi * (t - 1.0) / 3.0)


def steering_function(kind, t_end=6.0, dt=0.01):
    """Return a spline drive function of the rack translation.

    :param kind: "lane1", "lane2", "circle", or None for a fixed rack
    :type kind: str or None
    :param t_end: last time of the table [s]
    :type t_end: float
    :param dt: spacing of the spline knots [s]
    :type dt: float
    :return: drive function C(t)
    :rtype: solvcon.multibody.DriveFunction
    """
    if kind is None:
        return mbs.DriveFunction.polynomial([0.0])
    n = int(round(t_end / dt))
    times = [i * dt for i in range(n + 1)]
    return mbs.DriveFunction.spline(times,
                                    [rack_position(kind, t) for t in times])


def _quat(e):
    e1, e2, e3 = e
    return [math.sqrt(max(0.0, 1.0 - e1 * e1 - e2 * e2 - e3 * e3)), e1, e2, e3]


def rotation(p):
    """Return the body-to-global matrix A(p) of Haug Eq. 9.3.6 as nested lists.

    :param p: Euler parameters [e0, e1, e2, e3]
    :type p: list
    :return: 3 x 3 matrix
    :rtype: list
    """
    e0, e1, e2, e3 = p
    return [
        [2 * (e0 * e0 + e1 * e1) - 1, 2 * (e1 * e2 - e0 * e3),
         2 * (e1 * e3 + e0 * e2)],
        [2 * (e1 * e2 + e0 * e3), 2 * (e0 * e0 + e2 * e2) - 1,
         2 * (e2 * e3 - e0 * e1)],
        [2 * (e1 * e3 - e0 * e2), 2 * (e2 * e3 + e0 * e1),
         2 * (e0 * e0 + e3 * e3) - 1],
    ]


def _mul(a, x):
    return [sum(a[i][k] * x[k] for k in range(3)) for i in range(3)]


def _mul_t(a, x):
    return [sum(a[k][i] * x[k] for k in range(3)) for i in range(3)]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def global_point(num, local):
    """Return the global position of a body point at the Table 12.5.9 pose.

    :param num: book body number
    :type num: int
    :param local: body-fixed coordinates of the point
    :type local: list
    :return: global coordinates
    :rtype: list
    """
    r = EQUILIBRIUM[num][:3]
    a = _mul(rotation(_quat(EQUILIBRIUM[num][3:])), local)
    return [r[0] + a[0], r[1] + a[1], r[2] + a[2]]


def fitted_front_joint(kind, bi, bj, pi, qi, pj):
    """Return chassis-side data of a front joint that fit the Table 12.5.9
    pose.

    Tables 12.5.2 and 12.5.3 do not agree with Tables 12.5.8 and 12.5.9: at
    the book equilibrium the strut points miss the chassis strut axis by
    0.10 m, and the lower ball joint is 0.32 m (not 0.203 m) from point A
    and 0.048 m out of the plane normal to the pivot axis. This function
    keeps every wheel-side point and the pivot axis direction, and changes
    only the chassis side:

    * strut: the axis runs from point C (chassis) to the strut point of the
      wheel assembly at the book equilibrium;
    * revolute-spherical: point A moves along the pivot axis into the plane
      of point B, and the distance is |AB| at the book equilibrium.

    :param kind: "rs" or "strut"
    :type kind: str
    :param bi: chassis body number
    :type bi: int
    :param bj: wheel assembly body number
    :type bj: int
    :param pi: chassis point of Tables 12.5.2-12.5.3
    :type pi: list
    :param qi: chassis axis point of Tables 12.5.2-12.5.3
    :type qi: list
    :param pj: wheel assembly point of Tables 12.5.2-12.5.3
    :type pj: list
    :return: (chassis point, chassis axis, distance or None)
    :rtype: tuple
    """
    a_i = rotation(_quat(EQUILIBRIUM[bi][3:]))
    pi_g = global_point(bi, pi)
    pj_g = global_point(bj, pj)
    if kind == "strut":
        return pi, _mul_t(a_i, _sub(pi_g, pj_g)), None
    h = _sub(qi, pi)
    n = math.sqrt(_dot(h, h))
    h = [x / n for x in h]
    shift = _dot(h, _mul_t(a_i, _sub(pj_g, pi_g)))
    pi_new = [pi[k] + shift * h[k] for k in range(3)]
    d = _sub(pj_g, global_point(bi, pi_new))
    return pi_new, h, math.sqrt(_dot(d, d))


def build(model=2, pose="estimate", steering=None, speed_mph=0.0,
          gravity=GRAVITY, dt=1e-3, front="book", t_end=6.0):
    """Build vehicle model 1 or 2 of Section 12.5.1 without tire forces.

    :param model: 1 (bodies 1-6) or 2 (bodies 1-8 with stabilizer bars)
    :type model: int
    :param pose: "estimate" (Table 12.5.8) or "equilibrium" (Table 12.5.9)
    :type pose: str
    :param steering: rack input of Fig. 12.5.6: "lane1", "lane2", "circle",
        or None for a fixed rack
    :type steering: str or None
    :param speed_mph: initial forward speed of all bodies along +y [mph]
    :type speed_mph: float
    :param gravity: gravitational acceleration along -z [m/s^2]
    :type gravity: float
    :param dt: time step [s]
    :type dt: float
    :param front: "book" uses Tables 12.5.2-12.5.3 as printed; "fitted"
        uses :func:`fitted_front_joint` for joints A-B, E-F, B-C, F-G
    :type front: str
    :param t_end: last time of the steering table [s]
    :type t_end: float
    :return: system, dict of bodies by book number, list of (name, rows)
    :rtype: tuple
    """
    nb = 6 if model == 1 else 8
    s = mbs.DynamicsSystem(dt)
    ground = mbs.create_ground(0)
    s.Add(ground)
    bodies = {}
    v0 = [0.0, speed_mph * MPH, 0.0]
    for num in range(1, nb + 1):
        if pose == "estimate":
            r, p = ESTIMATE[num], [1.0, 0.0, 0.0, 0.0]
        else:
            r, p = EQUILIBRIUM[num][:3], _quat(EQUILIBRIUM[num][3:])
        mass, inertia = INERTIA[num]
        body = mbs.create_mobilized_body_quat(
            num, list(r), list(v0), Z, p, Z, Z, mass, inertia,
            [0.0, 0.0, -mass * gravity], Z,
        )
        s.Add(body)
        bodies[num] = body

    layout = []
    for name, kind, bi, bj, pi, qi, pj, qj, dist in JOINTS:
        if bi > nb or bj > nb:
            continue
        b_i, b_j = bodies[bi], bodies[bj]
        if kind in ("rs", "strut"):
            if front == "fitted":
                pi, axis, dist = fitted_front_joint(kind, bi, bj, pi, qi, pj)
            else:
                axis = _sub(qi, pi)
        if kind == "rs":
            joint = mbs.create_revolute_spherical_joint(b_i, b_j, pi, pj, axis,
                                                        dist)
        elif kind == "strut":
            joint = mbs.create_strut_joint(b_i, b_j, pi, pj, axis)
        elif kind == "trans":
            ax = _sub(qi, pi)
            joint = mbs.create_joint(2, pi, pj, Z, Z, ax, _sub(qj, pj), b_i,
                                     b_j)
        elif kind == "rev":
            joint = mbs.create_joint(1, pi, pj, Z, Z, _sub(qi, pi),
                                     _sub(qj, pj), b_i, b_j)
        else:
            joint = mbs.create_joint_distance(6, pi, pj, Z, Z, Z, Z, b_i, b_j,
                                              dist)
        s.Add(joint)
        layout.append((name, ROWS[kind]))

    # Steering driver (p. 471, Section 12.5.3): rack translation along chassis
    # x'.
    s.Add(mbs.create_translational_driver(bodies[1], bodies[6],
                                          [0.0, 1.2675, -0.3748], Z,
                                          [1.0, 0.0, 0.0],
                                          steering_function(steering, t_end)))
    layout.append(("driver D", ROWS["driver"]))

    for bi, bj, pi, pj, k, c, l0 in TSDA:
        if bi <= nb and bj <= nb:
            s.Add(mbs.create_spring_damper(bodies[bi], bodies[bj], pi, pj, k,
                                           c, l0))
    return s, bodies, layout


def joint_residuals(s, layout, t=0.0):
    """Return the largest constraint residual of each joint at time t.

    The constraint vector starts with seven ground rows; the joint rows
    follow in insertion order.

    :param s: dynamics system
    :type s: solvcon.multibody.DynamicsSystem
    :param layout: list of (name, rows) from :func:`build`
    :type layout: list
    :param t: time [s]
    :type t: float
    :return: residual by joint name
    :rtype: dict
    """
    c = as_list(s.evaluate_constraints(t)["C"])
    out = {}
    row = 7
    for name, n in layout:
        out[name] = max(abs(x) for x in c[row:row + n])
        row += n
    return out


if __name__ == "__main__":
    for model in (1, 2):
        sysm, _, lay = build(model=model, pose="equilibrium")
        print("model", model)
        for jname, res in joint_residuals(sysm, lay).items():
            print("  %-9s %.5f" % (jname, res))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
