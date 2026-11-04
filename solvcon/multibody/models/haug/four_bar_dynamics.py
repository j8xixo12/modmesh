# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Haug Section 12.3: dynamic analysis of a spatial four-bar mechanism.

Reference: E. J. Haug, "Computer-Aided Kinematics and Dynamics of
Mechanical Systems, Vol. I", Section 12.3, pp. 458-462. The kinematic
models are the models of Section 10.3 (pp. 402-408, Tables 10.3.1 to
10.3.6). The inertia data are in Table 12.3.1. The results are Tables
12.3.2 and 12.3.3 (equilibrium), Fig. 12.3.1 (inverse dynamics), and
Fig. 12.3.2 (dynamics). The transcribed book data are in
``tests/data/multibody/haug_12_3.json``.

Body numbers. Haug numbers the ground 4. The solver needs the ground as
body 0, so the ground is body 0 here.

* Model 1 (Fig. 10.3.1): link 1 (crank AB), link 2 (coupler BC), link 3
  (rocker CD). Revolute A (ground - link 1), universal B (link 1 - link 2),
  spherical C (link 2 - link 3), revolute D (link 3 - ground).
* Model 2 (Fig. 10.3.2): link 1 and link 2 (the rocker CD). Link BC is a
  spherical-spherical composite joint, that is a distance constraint
  between B on link 1 and C on the rocker. The rocker has the inertia of
  link 3 of Table 12.3.1 ("ignoring link 2", p. 459).

Joint data (body-fixed coordinates):

* revolute A, Table 10.3.1: P = (0, 0, 0) on both bodies, axis x;
* universal B, Table 10.3.2: link 1 P = (0, 0, 2), axis (0.750, -0.662, 0);
  link 2 P = (0, 6.1, 0), axis (0, 0, 1);
* spherical C, Table 10.3.3: link 2 P = (0, -6.1, 0), link 3
  P = (0, -3.7, 0);
* revolute D, Table 10.3.1: link 3 P = (0, 3.7, 0), axis x'; ground
  P = (-4, -8.5, 0), axis PQ = (0, -1, 0);
* model 2 distance B-C: link 1 P = (0, 0, 2), rocker P = (0, -3.7, 0)
  (Table 10.3.4). The length is 12.2 m, the length 6.1 + 6.1 of model 1.
  The book prints 12.19 m (p. 406); the equilibrium of Table 12.3.3 gives
  |BC| = 12.200 m.

Forces. Gravity acts along -z (Sec. 12.3.2). The book does not print g.
``GRAVITY = 9.8`` is the value that reproduces Section 12.6 with the same
code. The equilibrium configuration does not depend on g.

Driver (Sec. 10.3.3, 12.3.3): theta = pi t about the global x axis, from
the configuration of Table 10.3.6. Dynamics (Sec. 12.3.4): a constant
torque n = 10 N m on link 1, counterclockwise about +x, from rest.

Run this file to print the equilibrium and the dynamic response::

    python solvcon/multibody/models/haug/four_bar_dynamics.py
"""

import math

from .... import multibody as mbs
from ...tools.arrays import as_list, reaction_arrays

Z = [0.0, 0.0, 0.0]
AX = [1.0, 0.0, 0.0]

#: Gravitational acceleration along -z [m/s^2] (assumed, see module text).
GRAVITY = 9.8

#: Table 12.3.1 (p. 459): mass [kg] and [Ix'x', Iy'y', Iz'z'] [kg m^2].
#: All products of inertia are zero.
INERTIA = {
    "link1": (2.0, [4.0, 2.0, 0.0]),
    "link2": (1.0, [12.4, 0.01, 0.0]),
    "link3": (1.0, [4.54, 0.01, 0.0]),
}

#: Length of link BC [m], 6.1 + 6.1 (Tables 10.3.2 and 10.3.3).
COUPLER_LENGTH = 12.2

#: Driver rate theta = pi t (Sec. 10.3.3) [rad/s].
CRANK_SPEED = math.pi

#: Applied torque on link 1 for dynamics (Sec. 12.3.4) [N m].
DRIVE_TORQUE = 10.0

#: Initial estimates, Table 10.3.5 (p. 406): position and (e1, e2, e3).
#: e0 >= 0 follows from the normalization constraint (Eq. 9.6.3).
ESTIMATES = {
    "link1": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    "link2": ([-3.75, -4.25, 4.25], [-0.29, -0.27, -0.26]),
    "link3": ([-5.75, -8.5, 3.25], [-0.36, 0.36, -0.61]),
}


def _quat(vec):
    """Return unit Euler parameters from the vector part.

    :param vec: Euler parameters (e1, e2, e3)
    :type vec: list
    :return: Euler parameters [e0, e1, e2, e3]
    :rtype: list
    """
    e1, e2, e3 = vec
    return [math.sqrt(max(0.0, 1.0 - e1 * e1 - e2 * e2 - e3 * e3)), e1, e2, e3]


def build(model=1, crank=None, torque=0.0, gravity=GRAVITY, state=None,
          dt=0.001):
    """Build model 1 or model 2 of the spatial four-bar.

    :param model: 1 (four bodies, universal and spherical joints) or 2
        (three bodies, spherical-spherical link)
    :type model: int
    :param crank: crank driver. None: no driver (free, 1 DOF). A float:
        theta = crank * t (Eq. 9.5.4) from theta = 0. The string
        ``"parameter"``: theta set by ``DriveFunction.parameter`` for a
        quasi-static sweep.
    :type crank: float or str or None
    :param torque: constant torque on link 1 about +x [N m]
    :type torque: float
    :param gravity: gravitational acceleration along -z [m/s^2]
    :type gravity: float
    :param state: (positions, quaternions) per body, solver order, to
        start from instead of the Table 10.3.5 estimates. The bodies start
        at rest.
    :type state: tuple or None
    :param dt: output step [s]
    :type dt: float
    :return: (system, parts); parts maps names to bodies and joints
    :rtype: tuple
    """
    s = mbs.DynamicsSystem(dt)
    ground = mbs.create_ground(0)
    s.Add(ground)
    names = ["link1", "link2", "link3"] if model == 1 else ["link1", "link3"]
    bodies = {}
    for num, name in enumerate(names, start=1):
        mass, inertia = INERTIA[name]
        if state is None:
            pos, vec = ESTIMATES[name]
            quat = _quat(vec)
        else:
            pos = [float(v) for v in state[0][num]]
            quat = [float(v) for v in state[1][num]]
        bodies[name] = mbs.create_mobilized_body_quat(
            num, pos, Z, Z, quat, Z, Z, mass, inertia,
            [0.0, 0.0, -mass * gravity], Z)
        s.Add(bodies[name])
    link1 = bodies["link1"]
    rocker = bodies["link3"]

    joint_a = mbs.create_joint(1, Z, Z, Z, Z, AX, AX, ground, link1)
    s.Add(joint_a)
    if model == 1:
        s.Add(mbs.create_joint(4, [0.0, 0.0, 2.0], [0.0, 6.1, 0.0], Z, Z,
                               [0.750, -0.662, 0.0], [0.0, 0.0, 1.0], link1,
                               bodies["link2"]))
        s.Add(mbs.create_joint(0, [0.0, -6.1, 0.0], [0.0, -3.7, 0.0], Z, Z, AX,
                               AX, bodies["link2"], rocker))
    else:
        s.Add(mbs.create_joint_distance(6, [0.0, 0.0, 2.0], [0.0, -3.7, 0.0],
                                        Z, Z, AX, AX, link1, rocker,
                                        COUPLER_LENGTH))
    s.Add(mbs.create_joint(1, [0.0, 3.7, 0.0], [-4.0, -8.5, 0.0], Z, Z,
                           AX, [0.0, -1.0, 0.0], rocker, ground))

    parts = {"ground": ground, "link1": link1, "rocker": rocker,
             "joint_a": joint_a, "driver": None, "model": model}
    if model == 1:
        parts["link2"] = bodies["link2"]
    if crank is not None:
        if crank == "parameter":
            func = mbs.DriveFunction.parameter(0.0)
        else:
            func = mbs.DriveFunction.polynomial([0.0, float(crank)])
        parts["driver"] = mbs.create_rotational_driver(ground, link1, AX, AX,
                                                       func)
        s.Add(parts["driver"])
    if torque != 0.0:
        s.Add(mbs.create_rsda(ground, link1, AX, AX, 0.0, 0.0,
                              mbs.DriveFunction.polynomial([float(torque)])))
    return s, parts


def assembled_state(model=1, theta=0.0, steps=None):
    """Return the closed configuration at crank angle theta.

    The model assembles from the Table 10.3.5 estimates at theta = 0 with
    a parameter driver. A quasi-static sweep then turns link 1 to theta in
    small steps, so the rocker stays on the assembly branch of Table 10.3.6.

    :param model: 1 or 2
    :type model: int
    :param theta: crank angle about +x [rad]
    :type theta: float
    :param steps: number of sweep steps (default: one per 2 degrees)
    :type steps: int or None
    :return: (positions, quaternions) per body in solver order
    :rtype: tuple
    """
    s, parts = build(model, crank="parameter")
    s.init()
    if theta != 0.0:
        n = steps or max(2, int(abs(theta) / math.radians(2.0)) + 1)
        values = [theta * (k + 1) / n for k in range(n)]
        res = s.sweep(parts["driver"], values)
        if not res["completed"]:
            raise RuntimeError("four-bar sweep failed: " + res["message"])
    return ([list(p) for p in as_list(s.get_body_positions())],
            [list(q) for q in as_list(s.get_body_quaternions())])


def reaction_a(system):
    """Return the reaction force of revolute A on the ground (global).

    The book plots the "local reaction x force" at A (Fig. 12.3.1(b)).
    Joint A lists the ground first (Table 10.3.1), and the ground frame is
    the global frame.

    :param system: solver system with joint A as the first joint
    :type system: solvcon.multibody.DynamicsSystem
    :return: force [Fx, Fy, Fz] [N]
    :rtype: list
    """
    reactions = reaction_arrays(system.get_joint_reactions())
    return [float(v) for v in reactions[0]["force_i"]]


def simulate(system, parts, t_end):
    """Run the analysis and record the book output quantities.

    :param system: initialized solver system
    :type system: solvcon.multibody.DynamicsSystem
    :param parts: parts dictionary from :func:`build`
    :type parts: dict
    :param t_end: end time [s]
    :type t_end: float
    :return: dict of lists: t, z and zdot of the rocker centroid, torque
        (driver effort about +x, driven systems only), fx_a (reaction x
        at A on the ground)
    :rtype: dict
    """
    nb = 3 if parts["model"] == 1 else 2
    driven = parts["driver"] is not None
    out = {"t": [], "z": [], "zdot": [], "torque": [], "fx_a": []}

    def record():
        out["t"].append(system.get_sim_time())
        out["z"].append(float(as_list(system.get_body_positions())[nb][2]))
        out["zdot"].append(
            float(as_list(system.get_body_velocities())[nb][2]))
        reactions = reaction_arrays(system.get_joint_reactions())
        out["fx_a"].append(float(reactions[0]["force_i"][0]))
        if driven:
            out["torque"].append(float(reactions[-1]["effort"]))

    record()
    while system.get_sim_time() < t_end - 1e-9:
        system.solve()
        record()
    return out


def main():
    """Print the equilibrium of both models and the dynamic response."""
    for model, theta in ((1, math.radians(-123.6)), (2, math.radians(59.9))):
        s, _ = build(model, state=assembled_state(model, theta))
        s.init()
        res = s.find_equilibrium()
        print(f"model {model}: {res}")
        for p, q in zip(as_list(s.get_body_positions()),
                        as_list(s.get_body_quaternions())):
            print(f"  r = ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})  "
                  f"e = ({q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}, {q[3]:+.4f})")
    for model in (1, 2):
        s, parts = build(model, torque=DRIVE_TORQUE,
                         state=assembled_state(model))
        s.set_integrator("rk45")
        s.init()
        res = simulate(s, parts, 2.0)
        for k in range(0, len(res["t"]), 250):
            print(f"model {model} t = {res['t'][k]:.2f} s  z = "
                  f"{res['z'][k]:.4f} m  "
                  f"zdot = {res['zdot'][k]:+.4f} m/s")


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
