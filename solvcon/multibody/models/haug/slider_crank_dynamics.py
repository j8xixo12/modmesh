# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Haug Section 12.2: dynamic analysis of the spatial slider-crank.

Reference: E. J. Haug, "Computer-Aided Kinematics and Dynamics of
Mechanical Systems, Vol. I", Section 12.2, pp. 455-458. The kinematic model
is the model of Section 10.2 (pp. 396-399, Tables 10.2.1 to 10.2.7). The
inertia data are in Table 12.2.1. The results are Fig. 12.2.1 (driving
torque, inverse dynamics) and Figs. 12.2.2 to 12.2.4 (slider position,
velocity, and acceleration, dynamics). The transcribed book data are in
``tests/data/multibody/haug_12_2.json``.

Body numbers. Haug numbers the ground 4. The solver needs the ground as
body 0, so the ground is body 0 here. Crank 1, connecting rod 2, and
slider 3 keep their numbers.

Joints (Tables 10.2.1 to 10.2.5, body-fixed coordinates):

* revolute A, crank - ground: crank P = (0, 0, 0), ground
  P = (0, 0.1, 0.12), axis PQ = x on both bodies;
* spherical B, crank - rod: crank P = (0, 0.08, 0), rod P = (-l/2, 0, 0);
* revolute-cylindrical C, rod - slider: rod P = (l/2, 0, 0), rod axis y',
  slider axis x';
* translational D, slider - ground along x;
* distance constraint, rod P = (l/2, 0, 0) to slider P = (1, 0, 0),
  length 1. It fixes point C of the rod at the slider origin.

The book gives the joint data for l = 0.3 m only. For l = 0.27 m and
l = 0.24 m this module moves the rod end points to -l/2 and +l/2 and keeps
the inertia data of Table 12.2.1. The book does not state how it changed
the model.

Forces. The book does not print the force data. Gravity along -z is
applied. Without gravity the driving torque of Fig. 12.2.1 does not match
the book: at t = 0.8 s the book shows -0.29 N m, the model gives -0.28 N m
with gravity and -0.10 N m without. The book does not print g. ``GRAVITY =
9.8`` follows the value that reproduces the governor results of Section 12.6
with the same code (DADS). The difference to 9.81 is 0.1 %, below the figure
read uncertainty.

Sense of rotation. Fig. 10.2.3 and Fig. 12.2.1 (inverse dynamics) are
reproduced with the crank turning in the negative sense about the global
x axis. Figs. 12.2.2 and 12.2.3 (dynamics) start with a positive slider
velocity, that is, with the crank turning in the positive sense. The
functions below take the signed crank rate.

Initial crank speed of the dynamic runs. The book states 6, 2 pi, and
8 rad/s (p. 458). The plotted initial slider velocities (Fig. 12.2.3) need
crank rates about 1.5 times larger; for 2 pi and 8 rad/s no crank angle
gives the plotted value. :func:`crank_rate_from_slider_velocity` converts a
plotted initial slider velocity into the crank rate.

Run this file to print the driving torque and a short dynamic history::

    python solvcon/multibody/models/haug/slider_crank_dynamics.py
"""

import math

from .... import multibody as mbs
from ...tools.arrays import as_list, reaction_arrays

Z = [0.0, 0.0, 0.0]
AX = [1.0, 0.0, 0.0]
AY = [0.0, 1.0, 0.0]

#: Gravitational acceleration along -z [m/s^2] (assumed, see module text).
GRAVITY = 9.8

#: Table 12.2.1 (p. 456): mass [kg] and principal moments
#: [Ix'x', Iy'y', Iz'z'] [kg m^2]. All products of inertia are zero.
INERTIA = {
    "crank": (0.12, [0.0001, 0.00001, 0.0001]),
    "rod": (0.5, [0.004, 0.0004, 0.004]),
    "slider": (2.0, [0.0001, 0.0001, 0.0001]),
}

#: Revolute A on the ground, Table 10.2.1 [m].
GROUND_A = [0.0, 0.1, 0.12]
#: Spherical B on the crank, Table 10.2.2 [m].
CRANK_B = [0.0, 0.08, 0.0]

#: Assembled crank Euler parameters, Table 10.2.7 (p. 399). e2 and e3 are
#: below 5e-5, the assembly error of the book; the crank turns about x.
CRANK_E0 = 0.72090
CRANK_E1 = 0.69306

#: Nominal connecting rod length [m] (Table 10.2.2, Sec. 12.2.3).
ROD_LENGTH = 0.3


def initial_crank_angle():
    """Return the assembled crank angle about x from Table 10.2.7.

    :return: crank rotation angle about the global x axis [rad]
    :rtype: float
    """
    return 2.0 * math.atan2(CRANK_E1, CRANK_E0)


def _rod_pose(theta, rod_length):
    """Return the rod centroid, rod Euler parameters, and slider x.

    The rod frame has x' along B -> C (Table 10.2.2: B at x' = -l/2) and
    y' perpendicular to the slider axis (revolute-cylindrical joint,
    Table 10.2.3).

    :param theta: crank angle about the global x axis [rad]
    :type theta: float
    :param rod_length: connecting rod length l [m]
    :type rod_length: float
    :return: (rod centroid, rod Euler parameters, slider x)
    :rtype: tuple
    """
    by = GROUND_A[1] + CRANK_B[1] * math.cos(theta)
    bz = GROUND_A[2] + CRANK_B[1] * math.sin(theta)
    xc = math.sqrt(rod_length ** 2 - by * by - bz * bz)
    ex = [xc / rod_length, -by / rod_length, -bz / rod_length]
    # y' = x_global x ex, normalized; z' = ex x y'
    ny = math.hypot(ex[2], ex[1])
    ey = [0.0, -ex[2] / ny, ex[1] / ny]
    ez = [ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2],
          ex[0] * ey[1] - ex[1] * ey[0]]
    quat = _quat_from_columns(ex, ey, ez)
    centroid = [0.5 * xc, 0.5 * by, 0.5 * bz]
    return centroid, quat, xc


def _quat_from_columns(ex, ey, ez):
    """Return Euler parameters of A = [ex ey ez] (body-to-global).

    :param ex: first column of A
    :type ex: list
    :param ey: second column of A
    :type ey: list
    :param ez: third column of A
    :type ez: list
    :return: Euler parameters [e0, e1, e2, e3]
    :rtype: list
    """
    a = [[ex[i], ey[i], ez[i]] for i in range(3)]
    tr = a[0][0] + a[1][1] + a[2][2]
    e0 = 0.5 * math.sqrt(max(0.0, tr + 1.0))
    return [e0, (a[2][1] - a[1][2]) / (4.0 * e0),
            (a[0][2] - a[2][0]) / (4.0 * e0), (a[1][0] - a[0][1]) / (4.0 * e0)]


def build(rod_length=ROD_LENGTH, crank_rate=-2.0 * math.pi, driven=True,
          gravity=GRAVITY, dt=0.001):
    """Build the slider-crank of Sections 10.2 and 12.2.

    The bodies start at the exact closed configuration for the crank angle
    of Table 10.2.7, so ``init()`` does not move them.

    If ``driven`` is true, then a rotational driver (Haug Eq. 9.5.4)
    imposes theta(t) = theta_0 + crank_rate * t on the crank relative to
    the ground about x. The system has 0 DOF and ``solve()`` runs
    kinematic analysis with Lagrange multipliers (inverse dynamics,
    Sec. 12.2.2). The driver effort is the torque on the crank about +x.

    If ``driven`` is false, then the crank starts with angular velocity
    crank_rate about x and the system moves freely (Sec. 12.2.3).

    :param rod_length: connecting rod length l [m]
    :type rod_length: float
    :param crank_rate: signed crank rate about the global x axis [rad/s]
    :type crank_rate: float
    :param driven: add the constant-rate crank driver
    :type driven: bool
    :param gravity: gravitational acceleration along -z [m/s^2]
    :type gravity: float
    :param dt: output step [s]
    :type dt: float
    :return: (system, parts); parts maps names to bodies and the driver
    :rtype: tuple
    """
    theta0 = initial_crank_angle()
    s = mbs.DynamicsSystem(dt)
    ground = mbs.create_ground(0)
    s.Add(ground)

    m, j = INERTIA["crank"]
    crank_q = [math.cos(0.5 * theta0), math.sin(0.5 * theta0), 0.0, 0.0]
    omega = [0.0 if driven else crank_rate, 0.0, 0.0]
    crank = mbs.create_mobilized_body_quat(1, GROUND_A, Z, Z, crank_q, omega,
                                           Z, m, j, [0.0, 0.0, -m * gravity],
                                           Z)
    s.Add(crank)

    centroid, rod_q, xc = _rod_pose(theta0, rod_length)
    m, j = INERTIA["rod"]
    rod = mbs.create_mobilized_body_quat(2, centroid, Z, Z, rod_q, Z, Z, m, j,
                                         [0.0, 0.0, -m * gravity], Z)
    s.Add(rod)
    m, j = INERTIA["slider"]
    slider = mbs.create_mobilized_body_quat(3, [xc, 0.0, 0.0], Z, Z,
                                            [1.0, 0.0, 0.0, 0.0], Z, Z, m, j,
                                            [0.0, 0.0, -m * gravity], Z)
    s.Add(slider)

    half = 0.5 * rod_length
    # Revolute A (Table 10.2.1), spherical B (Table 10.2.2)
    s.Add(mbs.create_joint(1, Z, GROUND_A, Z, Z, AX, AX, crank, ground))
    s.Add(mbs.create_joint(0, CRANK_B, [-half, 0.0, 0.0], Z, Z, AX, AX, crank,
                           rod))
    # Revolute-cylindrical C (Table 10.2.3): rod axis y', slider axis x'
    s.Add(mbs.create_joint(7, [half, 0.0, 0.0], [0.2, 0.0, 0.0], Z, Z, AY, AX,
                           rod, slider))
    # Translational D (Table 10.2.4) and distance constraint (Table 10.2.5)
    s.Add(mbs.create_joint(2, Z, [0.2, 0.0, 0.0], Z, Z, AX, AX, slider,
                           ground))
    s.Add(mbs.create_joint_distance(6, [half, 0.0, 0.0], [1.0, 0.0, 0.0], Z, Z,
                                    AX, AX, rod, slider, 1.0))
    parts = {"ground": ground, "crank": crank, "rod": rod, "slider": slider,
             "driver": None}
    if driven:
        driver = mbs.create_rotational_driver(
            ground, crank, AX, AX,
            mbs.DriveFunction.polynomial([theta0, crank_rate]))
        s.Add(driver)
        parts["driver"] = driver
    return s, parts


def crank_rate_from_slider_velocity(xdot0, rod_length=ROD_LENGTH):
    """Return the crank rate that gives the initial slider velocity xdot0.

    The velocity analysis is linear in the crank rate, so one solve with a
    unit crank rate gives dx/dtheta at the start pose.

    :param xdot0: initial slider velocity along x [m/s]
    :type xdot0: float
    :param rod_length: connecting rod length l [m]
    :type rod_length: float
    :return: signed crank rate about the global x axis [rad/s]
    :rtype: float
    """
    s, _ = build(rod_length=rod_length, crank_rate=1.0, driven=False)
    s.init()
    return xdot0 / float(as_list(s.get_body_velocities())[3][0])


def simulate(system, t_end, driven):
    """Run the analysis and record the book output quantities.

    :param system: initialized solver system
    :type system: solvcon.multibody.DynamicsSystem
    :param t_end: end time [s]
    :type t_end: float
    :param driven: true if the system has the crank driver (the last joint)
    :type driven: bool
    :return: dict of lists: t, x, xdot, xddot of the slider, and torque
        (driver effort about +x, inverse dynamics only)
    :rtype: dict
    """
    out = {"t": [], "x": [], "xdot": [], "xddot": [], "torque": []}

    def record():
        out["t"].append(system.get_sim_time())
        out["x"].append(float(as_list(system.get_body_positions())[3][0]))
        out["xdot"].append(
            float(as_list(system.get_body_velocities())[3][0]))
        out["xddot"].append(
            float(as_list(system.get_body_accelerations())[3][0]))
        if driven:
            reactions = reaction_arrays(system.get_joint_reactions())
            out["torque"].append(float(reactions[-1]["effort"]))

    record()
    while system.get_sim_time() < t_end - 1e-9:
        system.solve()
        record()
    return out


def main():
    """Print the driving torque (l = 0.3 m) and the free motion."""
    s, _ = build()
    s.init()
    res = simulate(s, 1.0, True)
    for k in range(0, len(res["t"]), 100):
        print(f"t = {res['t'][k]:.2f} s  torque = {res['torque'][k]:+.4f} N m")
    s, _ = build(crank_rate=2.0 * math.pi, driven=False)
    s.set_integrator("rk45")
    s.init()
    res = simulate(s, 1.0, False)
    for k in range(0, len(res["t"]), 100):
        print(f"t = {res['t'][k]:.2f} s  x = {res['x'][k]:.4f} m  "
              f"xdot = {res['xdot'][k]:+.4f} m/s")


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
