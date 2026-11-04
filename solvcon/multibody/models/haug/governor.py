# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Haug Section 12.6: dynamic analysis of a governor mechanism.

Reference: E. J. Haug, "Computer-Aided Kinematics and Dynamics of Mechanical
Systems", Vol. I, Section 12.6, pp. 481-488, Fig. 12.6.1, Tables 12.6.1 to
12.6.6, Figs. 12.6.2 to 12.6.6. The transcribed data are in
``tests/data/multibody/haug_12_6.json``.

Model (book body numbers in parentheses, solver numbers first)::

    0 (1) ground
    1 (2) spindle, rotates about the global y axis (REV1)
    2 (3) ball 1, hinged to the spindle by REV2 (axis z)
    3 (4) ball 2, hinged to the spindle by REV3 (axis z)
    4 (5) collar, slides on the spindle along y (TRANS1)
    couplers DIST1 (ball 1 - collar) and DIST2 (ball 2 - collar)
    TSDA spindle - collar: k = 1000, 2000 or 3000 N/m, c = 30 kg/s, L0 = 0.15 m

DOF = 35 - 33 = 2 (book p. 483).

Gravity acts along -y. The book does not print the value of g. The book
steady-state speed (11.0174 rad/s, p. 485) is reproduced with g = 9.8 m/s^2
and not with g = 9.80665 m/s^2 (11.0231 rad/s), so g = 9.8 is the default.

Engine torque (Eq. 12.6.1): ``T_s = C * dl``, where ``dl`` is the spring
deformation, that is, the downward movement of the collar from the steady
state. The solver has no force element whose torque depends on the state.
:func:`simulate` therefore holds ``T_e + T_s`` constant over each time step,
applies it as a body torque on the spindle, and rebuilds the system from the
current state at every step. A predictor-corrector on ``dl`` (trapezoidal
rule) makes the coupling second-order accurate in the step size.
"""

import math

from .... import multibody as mbs
from ...tools.arrays import as_list

Z = [0.0, 0.0, 0.0]

#: Gravitational acceleration [m/s^2], inferred from the book steady state.
GRAVITY = 9.8

#: Table 12.6.1 (p. 482): mass [kg] and principal moments [kg m^2].
MASS = {"spindle": 200.0, "ball1": 1.0, "ball2": 1.0, "collar": 1.0}
INERTIA = {
    "spindle": [25.0, 50.0, 25.0],
    "ball1": [0.1, 0.1, 0.1],
    "ball2": [0.1, 0.1, 0.1],
    "collar": [0.15, 0.125, 0.15],
}

#: Fig. 12.6.1 and Table 12.6.3: arm length from hinge to ball center [m].
ARM = 0.16
#: Table 12.6.2: coupler attachment on the arm, measured from the hinge [m].
COUPLER_ARM = 0.08
#: Table 12.6.2: coupler length [m].
COUPLER_LENGTH = 0.10922
#: Table 12.6.6: global y of the spindle origin and of the collar [m].
SPINDLE_Y = 0.2
COLLAR_Y = 0.05
#: Table 12.6.5: free length [m] and damping [kg/s] of the TSDA.
FREE_LENGTH = 0.15
DAMPING = 30.0
#: Section 12.6.2 (p. 485): steady-state spindle speed [rad/s].
OMEGA_BOOK = 11.0174


def external_torque(t):
    """Return the load torque T_e(t) of Fig. 12.6.2 about the global y axis.

    T_e is zero up to t = 1 s, falls linearly to -25 N m at t = 2 s, and
    stays at -25 N m after that.

    :param t: time [s]
    :type t: float
    :return: torque [N m]
    :rtype: float
    """
    if t <= 1.0:
        return 0.0
    if t >= 2.0:
        return -25.0
    return -25.0 * (t - 1.0)


def steady_state(omega=OMEGA_BOOK, collar_y=COLLAR_Y):
    """Return a body state of the rotating governor.

    The collar height fixes the arm angle through the coupler length. Every
    body turns with the spindle at the rate ``omega`` about the global y axis.
    At the book values the state is Table 12.6.6. That table prints
    e3 = +0.3827 for ball 1, but REV2 (Table 12.6.3) puts the hinge at
    x' = -0.16 of ball 1, which needs e3 = -0.3827. This function uses the
    sign that satisfies REV2.

    :param omega: spindle angular velocity [rad/s]
    :type omega: float
    :param collar_y: global y of the collar [m]
    :type collar_y: float
    :return: per body (solver numbers 1 to 4) a dict with keys r, v, p, w
    :rtype: list
    """
    # Hinge-to-collar distance h and arm angle phi below the horizontal plane:
    # L^2 = (a cos phi)^2 + (h - a sin phi)^2 with a = COUPLER_ARM.
    h = SPINDLE_Y - collar_y
    a = COUPLER_ARM
    sin_phi = (a * a + h * h - COUPLER_LENGTH ** 2) / (2.0 * a * h)
    phi = math.asin(sin_phi)
    c, s = math.cos(phi), math.sin(phi)
    w = [0.0, omega, 0.0]
    half = 0.5 * phi
    states = [
        {"r": [0.0, SPINDLE_Y, 0.0], "v": Z, "p": [1.0, 0.0, 0.0, 0.0],
         "w": w},
        {
            "r": [ARM * c, SPINDLE_Y - ARM * s, 0.0],
            "v": [0.0, 0.0, -omega * ARM * c],
            "p": [math.cos(half), 0.0, 0.0, -math.sin(half)],
            "w": w,
        },
        {
            "r": [-ARM * c, SPINDLE_Y - ARM * s, 0.0],
            "v": [0.0, 0.0, omega * ARM * c],
            "p": [math.cos(half), 0.0, 0.0, math.sin(half)],
            "w": w,
        },
        {"r": [0.0, collar_y, 0.0], "v": Z, "p": [1.0, 0.0, 0.0, 0.0], "w": w},
    ]
    return states


def build(k=1000.0, state=None, spindle_torque=0.0, gravity=GRAVITY, dt=1e-3):
    """Build the governor of Fig. 12.6.1 with the data of Tables 12.6.1-12.6.5.

    :param k: TSDA spring constant [N/m] (Table 12.6.5: 1000, 2000, 3000)
    :type k: float
    :param state: body states from :func:`steady_state` or :func:`read_state`;
        None uses the book steady state
    :type state: list or None
    :param spindle_torque: constant torque on the spindle about y [N m]
    :type spindle_torque: float
    :param gravity: gravitational acceleration along -y [m/s^2]
    :type gravity: float
    :param dt: time step of the system [s]
    :type dt: float
    :return: system and dict of bodies keyed by name
    :rtype: tuple
    """
    if state is None:
        state = steady_state()
    s = mbs.DynamicsSystem(dt)
    ground = mbs.create_ground(0)
    s.Add(ground)
    names = ["spindle", "ball1", "ball2", "collar"]
    bodies = {"ground": ground}
    for num, (name, st) in enumerate(zip(names, state), start=1):
        torque = [0.0, spindle_torque, 0.0] if name == "spindle" else Z
        body = mbs.create_mobilized_body_quat(
            num, list(st["r"]), list(st["v"]), Z, list(st["p"]), list(st["w"]),
            Z, MASS[name], INERTIA[name], [0.0, -MASS[name] * gravity, 0.0],
            torque,
        )
        s.Add(body)
        bodies[name] = body
    sp, b1, b2, co = (bodies[n] for n in names)
    ey = [0.0, 1.0, 0.0]
    ez = [0.0, 0.0, 1.0]
    # REV1 (Table 12.6.3): ground P = (0, 0, 0), spindle P = (0, -0.2, 0), axis
    # y.
    s.Add(mbs.create_joint(1, Z, [0.0, -SPINDLE_Y, 0.0], Z, Z, ey, ey, ground,
                           sp))
    # REV2 and REV3 (Table 12.6.3): hinges at the spindle origin, axis z.
    s.Add(mbs.create_joint(1, Z, [-ARM, 0.0, 0.0], Z, Z, ez, ez, sp, b1))
    s.Add(mbs.create_joint(1, Z, [ARM, 0.0, 0.0], Z, Z, ez, ez, sp, b2))
    # TRANS1 (Table 12.6.4): collar slides along the spindle y axis.
    s.Add(mbs.create_joint(2, Z, Z, Z, Z, ey, ey, sp, co))
    # DIST1 and DIST2 (Table 12.6.2): couplers from the arm midpoints to the
    # collar.
    s.Add(mbs.create_joint_distance(6, [-COUPLER_ARM, 0.0, 0.0], Z, Z, Z, ez,
                                    ez, b1, co, COUPLER_LENGTH))
    s.Add(mbs.create_joint_distance(6, [COUPLER_ARM, 0.0, 0.0], Z, Z, Z, ez,
                                    ez, b2, co, COUPLER_LENGTH))
    # TSDA (Table 12.6.5): spindle origin to collar origin.
    s.Add(mbs.create_spring_damper(sp, co, Z, Z, k, DAMPING, FREE_LENGTH))
    return s, bodies


def read_state(s):
    """Read the state of bodies 1 to 4 from a system.

    :param s: dynamics system
    :type s: solvcon.multibody.DynamicsSystem
    :return: per body a dict with keys r, v, p, w (global angular velocity)
    :rtype: list
    """
    r = as_list(s.get_body_positions())
    v = as_list(s.get_body_velocities())
    p = as_list(s.get_body_quaternions())
    w = as_list(s.get_body_angular_velocities())
    return [{"r": r[i], "v": v[i], "p": p[i], "w": w[i]} for i in range(1, 5)]


def collar_acceleration(omega, gravity=GRAVITY):
    """Return the collar acceleration along y at the 45 degree configuration.

    :param omega: spindle angular velocity [rad/s]
    :type omega: float
    :param gravity: gravitational acceleration [m/s^2]
    :type gravity: float
    :return: collar y acceleration [m/s^2]
    :rtype: float
    """
    s, _ = build(state=steady_state(omega), gravity=gravity)
    s.set_integrator("rk45")
    s.init()
    return as_list(s.get_body_accelerations())[4][1]


def steady_state_omega(gravity=GRAVITY, lo=10.5, hi=11.5, tol=1e-10):
    """Find the spindle speed at which the collar is stationary (Section
    12.6.2).

    The function brackets the root of :func:`collar_acceleration` and applies
    bisection.

    :param gravity: gravitational acceleration [m/s^2]
    :type gravity: float
    :param lo: lower bound of the bracket [rad/s]
    :type lo: float
    :param hi: upper bound of the bracket [rad/s]
    :type hi: float
    :param tol: bracket width at which the search stops [rad/s]
    :type tol: float
    :return: steady-state angular velocity [rad/s]
    :rtype: float
    """
    f_lo = collar_acceleration(lo, gravity)
    f_hi = collar_acceleration(hi, gravity)
    if f_lo * f_hi > 0.0:
        raise ValueError("steady-state speed is not inside the bracket")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        f_mid = collar_acceleration(mid, gravity)
        if f_mid * f_lo > 0.0:
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def simulate(k=1000.0, gain=12500.0, t_end=10.0, dt=0.005, gravity=GRAVITY,
             integrator="rk45"):
    """Simulate Section 12.6.4: load torque of Fig. 12.6.2 and engine feedback.

    The spindle torque over [t, t + dt] is ``T_e(t + dt/2) + C * dl``, where
    ``dl`` is the mean of the collar drop at the start and at the end of the
    step. The end value comes from a predictor step (Heun scheme).

    :param k: TSDA spring constant [N/m]
    :type k: float
    :param gain: feedback constant C of Eq. 12.6.1 [N m / m]
    :type gain: float
    :param t_end: end time [s]
    :type t_end: float
    :param dt: coupling step [s]
    :type dt: float
    :param gravity: gravitational acceleration [m/s^2]
    :type gravity: float
    :param integrator: solver integrator name
    :type integrator: str
    :return: dict with lists t, collar_y, omega
    :rtype: dict
    """
    state = steady_state(OMEGA_BOOK)
    out = {"t": [0.0], "collar_y": [state[3]["r"][1]],
           "omega": [state[0]["w"][1]]}

    def step(st, torque):
        s, _ = build(k=k, state=st, spindle_torque=torque, gravity=gravity,
                     dt=dt)
        s.set_integrator(integrator)
        s.init()
        s.solve()
        return read_state(s)

    n = int(round(t_end / dt))
    for i in range(n):
        t = i * dt
        te = external_torque(t + 0.5 * dt)
        drop0 = COLLAR_Y - state[3]["r"][1]
        pred = step(state, te + gain * drop0)
        drop1 = COLLAR_Y - pred[3]["r"][1]
        state = step(state, te + gain * 0.5 * (drop0 + drop1))
        out["t"].append(t + dt)
        out["collar_y"].append(state[3]["r"][1])
        out["omega"].append(state[0]["w"][1])
    return out


if __name__ == "__main__":
    print("steady-state omega (g = %.5f): %.6f rad/s, book %.4f"
          % (GRAVITY, steady_state_omega(), OMEGA_BOOK))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
