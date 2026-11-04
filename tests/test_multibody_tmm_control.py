# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests of solvcon.multibody.tmm.control: controlled multibody systems, mixed,
FE and finite segment TMM (Ch. 9).

Book: Rui, Wang, Zhang, Transfer Matrix Method for Multibody Systems (Wiley
2018). Reference data: tests/data/multibody/tmm_ch9_examples.json (book page
numbers inside). Every independent model in this file (global M, C, K matrices,
state-space matrices, absolute-coordinate DAE) is assembled here, not taken
from solvcon.multibody.tmm.control.
"""

import functools
import json
import math
from pathlib import Path

import numpy as np
import pytest


from solvcon.multibody.tmm import control as ctl
from solvcon.multibody.tmm import library as lib

DATA = Path(__file__).parent / "data" / "multibody"
with open(DATA / "tmm_ch9_examples.json") as _fh:
    _CH9 = json.load(_fh)
EX = _CH9["examples"]
CH = _CH9["chain_6_7"]
OMEGA = CH["Omega"]
F4 = CH["F4c"] - 1j * CH["F4s"]  # amplitude of F^c cos + F^s sin (Eq. 9.15)
BOOK_TOL = 0.5e-4  # half a unit of the 4th printed decimal


# ----------------------------------------------------------------------
# Models of Examples 9.6, 9.7, 9.9 (Fig. 9.27, 9.30)
# ----------------------------------------------------------------------
def _elements(force=F4):
    (m2, m4, m6), (k1, k3, k5), (c1, c3, c5) = CH["m"], CH["K"], CH["C"]
    return [lib.Spring(k1, c1), lib.LumpedMass(m2), lib.Spring(k3, c3),
            lib.LumpedMass(m4, force=force), lib.Spring(k5, c5),
            lib.LumpedMass(m6)]


def _tmm(example, delay=None, force=F4):
    """ControlledChain of Example 9.6 (actuator mass 2) or 9.7 (actuator spring
    3)."""
    actuator = 2 if example in ("9.7", "9.14") else 1
    d = EX[example].get("delay", 0.0) if delay is None else delay
    return ctl.ControlledChain(_elements(force), sensor=5, actuator=actuator,
                               gains=EX[example]["gains"], delay=d)


def _global(example, s=None, delay=0.0, control=True):
    """Independent global model M, C, K (Example 9.6 Eq. a, Example 9.7 Eq. a).

    The control force F_c = -(Ka xdd_6 + Kv xd_6 + Kd x_6)(t - tau) acts on
    mass 2 (Example 9.6) or pushes mass 4 and pulls mass 2 (Example 9.7).
    With s given, the delay factor exp(-s tau) multiplies the control columns.
    """
    (m2, m4, m6), (k1, k3, k5), (c1, c3, c5) = CH["m"], CH["K"], CH["C"]
    ka, kv, kd = EX[example]["gains"]
    m = np.diag([m2, m4, m6]).astype(complex)
    c = np.array([[c1 + c3, -c3, 0], [-c3, c3 + c5, -c5], [0, -c5, c5]],
                 dtype=complex)
    k = np.array([[k1 + k3, -k3, 0], [-k3, k3 + k5, -k5], [0, -k5, k5]],
                 dtype=complex)
    b = _control_vector(example)
    e = 1.0 if s is None else np.exp(-s * delay)
    if not control:
        e = 0.0
    m[:, 2] += ka * b * e
    c[:, 2] += kv * b * e
    k[:, 2] += kd * b * e
    return m, c, k


def _control_vector(example):
    """Generalized force of a unit control force (mass 2, or spring 3
    actuator)."""
    if example in ("9.7", "9.14"):
        return np.array([-1.0, 1.0, 0.0])
    return np.array([1.0, 0.0, 0.0])


def _global_steady(example, delay=0.0):
    s = 1j * OMEGA
    m, c, k = _global(example, s, delay)
    return np.linalg.solve(s * s * m + s * c + k, np.array([0.0, F4, 0.0]))


def _state_space(example):
    m, c, k = (x.real for x in _global(example))
    mi = np.linalg.inv(m)
    return np.block([[np.zeros((3, 3)), np.eye(3)], [-mi @ k, -mi @ c]])


# ----------------------------------------------------------------------
# BK: Examples 9.6, 9.7, 9.9 steady state (Eq. 9.12-9.30, 9.40-9.47)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("example", ["9.6", "9.7"])
def test_book_steady_state_real_time(example):
    """BK Examples 9.6 (p. 352 Eq. c) and 9.7 (p. 355 Eq. h): x = X^r cos - X^i
    sin."""
    z = _tmm(example).steady_state(OMEGA)
    x = z[[1, 3, 5], 0]
    np.testing.assert_allclose(x.real, EX[example]["cos"], atol=BOOK_TOL)
    np.testing.assert_allclose(-x.imag, EX[example]["sin"], atol=BOOK_TOL)


@pytest.mark.parametrize("example", ["9.6", "9.7"])
def test_steady_state_equals_global_and_book_matrix(example):
    """XM: TMM steady state = independent global frequency solve; Eq. (9.26)
    form = augmented form.

    Tolerance 1e-12 relative: both are exact linear solves of a 3-DOF system.
    """
    ch = _tmm(example)
    z = ch.steady_state(OMEGA)
    np.testing.assert_allclose(z[[1, 3, 5], 0], _global_steady(example),
                               rtol=1e-12)
    # atol 1e-13: the free-end force is a difference of terms of size 10 N
    np.testing.assert_allclose(ch.steady_state_book(OMEGA), z, rtol=1e-12,
                               atol=1e-13)


def test_example_9_9_delay_sign_inconsistency():
    """BK Example 9.9 (p. 361): the printed values are the response of tau =
    -0.001 s.

    Book inconsistency: Eq. (9.40)-(9.43) with tau = +0.001 s (a delay)
    differ from the printed values by up to 2.5 units of the last digit;
    the advanced control (tau = -0.001 s) reproduces every printed value.
    """
    ref = EX["9.9"]
    x_delay = _tmm("9.9").steady_state(OMEGA)[[1, 3, 5], 0]
    np.testing.assert_allclose(x_delay, _global_steady("9.9", ref["delay"]),
                               rtol=1e-12)
    book = np.array(ref["cos"]) - 1j * np.array(ref["sin"])
    assert np.max(np.abs(x_delay.real - book.real)) > 2 * BOOK_TOL
    x_adv = _tmm("9.9", delay=-ref["delay"]).steady_state(OMEGA)[[1, 3, 5], 0]
    np.testing.assert_allclose(x_adv.real, ref["cos"], atol=BOOK_TOL)
    np.testing.assert_allclose(-x_adv.imag, ref["sin"], atol=BOOK_TOL)


@pytest.mark.parametrize("example", ["9.6", "9.7"])
def test_steady_state_solves_time_domain_equations(example):
    """XM: RK4 of the independent state-space model from the TMM steady-state
    initial values.

    The trajectory must stay on x(t) = Re(X e^{i Omega t}). RK4 with h = 1e-3
    over 3 s has a global error near h^4 omega_max^5 t ~ 1e-8 relative to |X|;
    tolerance 1e-7 absolute (|X| ~ 1e-2).
    """
    x_amp = _tmm(example).steady_state(OMEGA)[[1, 3, 5], 0]
    a = _state_space(example)
    m, _, _ = (x.real for x in _global(example))
    mi = np.linalg.inv(m)

    def rhs(t, y):
        f = np.array([0.0, CH["F4c"] * math.cos(OMEGA * t)
                      + CH["F4s"] * math.sin(OMEGA * t), 0.0])
        return a @ y + np.concatenate([np.zeros(3), mi @ f])

    y = np.concatenate([x_amp.real, (1j * OMEGA * x_amp).real])
    h, n = 1e-3, 3000
    for i in range(n):
        t = i * h
        k1 = rhs(t, y)
        k2 = rhs(t + h / 2, y + h / 2 * k1)
        k3 = rhs(t + h / 2, y + h / 2 * k2)
        k4 = rhs(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    exact = (x_amp * np.exp(1j * OMEGA * n * h)).real
    np.testing.assert_allclose(y[:3], exact, atol=1e-7)


# ----------------------------------------------------------------------
# Closed-loop eigenvalues (mandatory check)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("example", ["9.6", "9.7", "9.14"])
def test_closed_loop_eigenvalues_vs_state_space(example):
    """XM: TMM closed-loop roots (continuation, Eq. 10.104) = eigenvalues of
    the state matrix.

    The state matrix contains the non-symmetric Ka, Kv, Kd terms. Tolerance
    1e-9 relative: the secant iteration stops at 1e-12 relative step.
    """
    lam = _tmm(example, delay=0.0).eigenvalues(60.0)
    ev = np.linalg.eigvals(_state_space(example))
    ev = np.array(sorted(ev[ev.imag > 0], key=lambda z: z.imag))
    np.testing.assert_allclose(lam, ev, rtol=1e-9)


def test_example_9_14_closed_loop_is_unstable():
    """Book observation: the gains of Example 9.14 give an unstable closed loop
    (Fig. 9.46 is figure-only). One eigenvalue has a positive real part with
    and without the delay."""
    lam0 = _tmm("9.14", delay=0.0).eigenvalues(60.0)
    lam1 = _tmm("9.14").eigenvalues(60.0)
    assert max(lam0.real) > 0.02
    assert max(lam1.real) > 0.02


def _quasi_polynomial_root(example, delay, lam0):
    """Newton iteration on det(lambda^2 M + lambda C + K + e^{-lambda tau} G)
    (global model)."""
    lam = complex(lam0)

    def det(x):
        m, c, k = _global(example, x, delay)
        return np.linalg.det(x * x * m + x * c + k)

    for _ in range(50):
        h = 1e-7 * abs(lam)
        d = det(lam)
        step = d / ((det(lam + h) - det(lam - h)) / (2 * h))
        lam -= step
        if abs(step) < 1e-14 * abs(lam):
            break
    return lam


@pytest.mark.parametrize("example", ["9.9", "9.14"])
def test_delay_roots_vs_quasi_polynomial(example):
    """XM: TMM roots with delay = roots of the global characteristic
    quasi-polynomial.

    The global Newton iteration starts from the tau = 0 state-space
    eigenvalue. Tolerance 1e-8 relative (central-difference Newton derivative).
    """
    lam = _tmm(example).eigenvalues(60.0)
    base = "9.6" if example == "9.9" else "9.7"
    ex = example if example == "9.14" else base
    ev = np.linalg.eigvals(_state_space(ex))
    ev = np.array(sorted(ev[ev.imag > 0], key=lambda z: z.imag))
    roots = np.array([_quasi_polynomial_root(ex, EX[example]["delay"],
                                             z) for z in ev])
    np.testing.assert_allclose(lam, roots, rtol=1e-8)
    assert np.max(np.abs(lam - ev)) > 1e-6  # the delay moves the roots


def test_characteristic_has_no_loop_pole():
    """IV: D(s) is finite where det(I - W U_c) = 0 and Eq. (9.26) has a pole.

    For Example 9.6 the loop factor 1 - c(s) vanishes on the real axis of s;
    D(s) equals -det(I - W U_c) times the plain determinant (module docstring).
    """
    ch = _tmm("9.6")
    for s in (0.3 + 2.0j, 1.0 + 7.0j, -0.5 + 15.0j):
        u = ch.overall_extended(s)
        n = ch.size
        p, k = ch.actuator, ch.sensor
        w = ch._product(p + 1, k, s, 1.0)
        loop = np.linalg.det(np.eye(n + 1) - w @ ch.control_matrix(s))
        phase, logabs = ch.characteristic(s)
        d = phase * np.exp(logabs)
        assert d == pytest.approx(-loop * u[1, 1], rel=1e-10)


def test_controlled_element_matrices():
    """Eq. (9.19)/(15.203) and Example 9.7 Eq. (c) at s = i Omega."""
    s = 1j * OMEGA
    u, uc = ctl.controlled_mass_matrices(10.0, s, force=F4,
                                         gains=(5.0, 1.0, 2.0))
    assert u[1, 0] == pytest.approx(10.0 * OMEGA**2)
    assert u[1, 2] == F4
    assert uc[1, 0] == pytest.approx(5.0 * OMEGA**2 - 1j * OMEGA - 2.0)
    u, uc = ctl.controlled_spring_matrices(150.0, 0.5, s,
                                           gains=(5.0, 5.0, 5.0))
    z = 150.0 + 0.5j * OMEGA
    assert u[0, 1] == pytest.approx(-1.0 / z)
    assert uc[0, 0] == pytest.approx((5.0 * OMEGA**2 - 5.0j * OMEGA - 5.0) / z)
    # delay factor of Eq. (9.43)
    _, ucd = ctl.controlled_mass_matrices(10.0, s, gains=(5.0, 0.0, 0.0),
                                          delay=0.01)
    assert ucd[1,
               0] == pytest.approx(5.0 * OMEGA**2 * np.exp(-1j * OMEGA * 0.01))


# ----------------------------------------------------------------------
# DTTMM of controlled chains (Sec. 9.6, 15.31): Examples 9.10, 9.11, 9.14
# ----------------------------------------------------------------------
def _force(t):
    return np.array([0.0, CH["F4c"] * math.cos(OMEGA * t)
                     + CH["F4s"] * math.sin(OMEGA * t), 0.0])


def _dt_chain(example, dt, delay_steps=0, solver="riccati"):
    kind, act = ("spring", 1) if example in ("9.7", "9.14") else ("mass", 0)
    return ctl.DTControlledChain(CH["m"], CH["K"], CH["C"], _force, 2, act,
                                 EX[example]["gains"], dt, actuator_kind=kind,
                                 delay_steps=delay_steps, solver=solver)


def _global_newmark(example, dt, t_end, delay_steps=0):
    """Independent Newmark (beta = 1/4, gamma = 1/2) of the global model."""
    m, c, k = (x.real for x in _global(example, control=delay_steps == 0))
    b = _control_vector(example)
    x = np.zeros(3)
    v = np.zeros(3)
    a = np.linalg.solve(m, _force(0.0) - c @ v - k @ x)
    hist = [(x, v, a)]
    n = int(round(t_end / dt))
    beta, gamma = 0.25, 0.5
    for i in range(1, n + 1):
        a_c = 1.0 / (beta * dt * dt)
        b_c = -(a_c * x + v / (beta * dt) + (0.5 / beta - 1.0) * a)
        c_c = gamma * dt * a_c
        d_c = v + dt * (1.0 - gamma) * a + gamma * dt * b_c
        f = _force(i * dt)
        if delay_steps:
            j = i - delay_steps
            hx, hv, ha = hist[j] if j >= 0 else (np.zeros(3),) * 3
            ka, kv, kd = EX[example]["gains"]
            f = f + b * (-ka * ha[2] - kv * hv[2] - kd * hx[2])
        x = np.linalg.solve(a_c * m + c_c * c + k, f - m @ b_c - c @ d_c)
        a = a_c * x + b_c
        v = c_c * x + d_c
        hist.append((x, v, a))
    return np.array([h[0] for h in hist])


@pytest.mark.parametrize("example,delay_steps", [("9.6", 0), ("9.7", 0),
                                                 ("9.6", 1), ("9.14", 1)])
def test_dttmm_equals_global_newmark(example, delay_steps):
    """XM: Examples 9.10, 9.11 (real time) and 9.14 (delay 0.001 s) by DTTMM =
    global Newmark.

    Both apply the same Newmark scheme; the Riccati DTTMM keeps the result to
    round-off. Tolerance 1e-11 absolute over 2000 steps (|x| ~ 0.1).
    """
    dt, t_end = 1e-3, 2.0
    _, x, _, _ = _dt_chain(example, dt, delay_steps).simulate(t_end)
    xg = _global_newmark(example, dt, t_end, delay_steps)
    np.testing.assert_allclose(x, xg, atol=1e-11)


def test_dttmm_plain_transfer_loses_precision():
    """Numerical finding: the plain overall transfer equation (Eq. 9.57-9.62)
    is ill-conditioned.

    The element products grow like (m A/K)^3 with A = 1/(beta dt^2):
    dt = 4e-3 gives about 4e12 and agrees with the Riccati DTTMM to 1e-6 after
    500 steps; dt = 1e-3 gives about 2e15 and the error reaches 1e-5 after
    2000 steps.
    """
    for dt, lo, hi in ((4e-3, 0.0, 1e-6), (1e-3, 1e-5, 1.0)):
        _, xr, _, _ = _dt_chain("9.6", dt).simulate(2.0)
        _, xp, _, _ = _dt_chain("9.6", dt, solver="transfer").simulate(2.0)
        err = np.max(np.abs(xr - xp))
        assert lo <= err < hi


def test_dttmm_convergence_order():
    """CV: DTTMM (Example 9.10) error vs RK4 reference with h = 1e-4 is second
    order.

    Reference error ~1e-14; errors at dt = 4e-3, 2e-3, 1e-3 must have ratios
    4 +- 0.8 (order 2 +- 0.3).
    """
    a = _state_space("9.6")
    m, _, _ = (x.real for x in _global("9.6"))
    mi = np.linalg.inv(m)

    def rhs(t, y):
        return a @ y + np.concatenate([np.zeros(3), mi @ _force(t)])

    y = np.zeros(6)
    h = 1e-4
    for i in range(20000):
        t = i * h
        k1 = rhs(t, y)
        k2 = rhs(t + h / 2, y + h / 2 * k1)
        k3 = rhs(t + h / 2, y + h / 2 * k2)
        k4 = rhs(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    errs = []
    for dt in (4e-3, 2e-3, 1e-3):
        _, x, _, _ = _dt_chain("9.6", dt).simulate(2.0)
        errs.append(np.max(np.abs(x[-1] - y[:3])))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(np.abs(orders - 2.0) < 0.3), orders


def test_dttmm_reaches_frequency_domain_steady_state():
    """XM: DTTMM started on the steady state (Example 9.6 TMM amplitudes) stays
    on it.

    Newmark phase error for Omega dt = 8e-3 over 3 s is ~ (Omega dt)^2 Omega t
    / 12 = 1e-4 relative; tolerance 3e-6 absolute (|X| ~ 0.05).
    """
    x_amp = _tmm("9.6").steady_state(OMEGA)[[1, 3, 5], 0]
    dt = 1e-3
    dc = _dt_chain("9.6", dt)
    _, x, _, _ = dc.simulate(3.0, x0=x_amp.real, v0=(1j * OMEGA * x_amp).real)
    exact = (x_amp * np.exp(1j * OMEGA * 3.0)).real
    np.testing.assert_allclose(x[-1], exact, atol=3e-6)


def test_dt_control_matrices():
    """Eq. (9.53), (9.56), (15.214), (15.216): control matrices from the
    Newmark coefficients."""
    a_c, b_c, c_c, d_c = ctl.newmark_coefficients(0.01, 0.2, -1.0, 1e-3)
    x_new = 0.0102
    # xdd = A x + B and xd = C x + D reproduce the Newmark update
    a_new = a_c * x_new + b_c
    v_new = c_c * x_new + d_c
    assert x_new == pytest.approx(0.01 + 1e-3 * 0.2 + 0.25e-6 * (-1.0 + a_new),
                                  rel=1e-12)
    assert v_new == pytest.approx(0.2 + 0.5e-3 * (-1.0 + a_new), rel=1e-12)
    uc = ctl.dt_control_matrix((5.0, 1.0, 2.0), a_c, b_c, c_c, d_c)
    fc = uc[1, 0] * x_new + uc[1, 2]
    # rel 1e-10: A x + B cancels terms of size 4e4 to a result of size 1
    assert fc == pytest.approx(-(5.0 * a_new + 1.0 * v_new + 2.0 * x_new),
                               rel=1e-10)
    ud = ctl.dt_delay_control_matrix((5.0, 1.0, 2.0), 0.1, 0.2, 0.3)
    assert ud[1, 2] == pytest.approx(-(5.0 * 0.3 + 0.2 + 0.2))
    assert np.count_nonzero(ud) == 1


# ----------------------------------------------------------------------
# Planar rigid bodies in acceleration form (Ch. 5, used by Sec. 9.2-9.4)
# ----------------------------------------------------------------------
G = 9.81
PERP = np.array([[0.0, -1.0], [1.0, 0.0]])


def _rot(th):
    return np.array([[math.cos(th), -math.sin(th)],
                     [math.sin(th), math.cos(th)]])


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def test_planar_body_transfer_satisfies_newton_euler():
    """AN: output state of Eq. (5.4)-type matrix satisfies Newton-Euler about
    the mass center.

    Tolerance 1e-12 relative (a few products of O(1) numbers).
    """
    rng = np.random.default_rng(3)
    m, jc = 2.3, 0.41
    c, b = np.array([0.4, 0.1]), np.array([1.1, -0.2])
    th, om = 0.7, -1.3
    f, mom = np.array([0.3, -2.0]), 0.25
    u = ctl.planar_body_transfer(m, jc, c, b, th, om, f, mom)
    z_in = np.concatenate([rng.normal(size=6), [1.0]])
    z_out = u @ z_in
    a_i, al, m_i, q_i = z_in[0:2], z_in[2], z_in[3], z_in[4:6]
    a_o, m_o, q_o = z_out[0:2], z_out[3], z_out[4:6]
    r_ic, r_io = _rot(th) @ c, _rot(th) @ b
    a_c = a_i + al * PERP @ r_ic - om**2 * r_ic
    np.testing.assert_allclose(a_o, a_i + al * PERP @ r_io - om**2 * r_io,
                               rtol=1e-12)
    np.testing.assert_allclose(m * a_c, q_i - q_o + f, rtol=1e-12)
    euler = _cross(-r_ic, q_i) + _cross(r_io - r_ic, -q_o) - m_i + m_o + mom
    assert jc * al == pytest.approx(euler, rel=1e-12)
    assert z_out[2] == al and z_out[6] == 1.0


class _PlanarDAE:
    """Independent absolute-coordinate model: bodies (x_c, y_c, theta),
    particles (x, y), extra linear DOFs with constant M, K; point constraints
    with Lagrange multipliers (index 1)."""

    def __init__(self, n_q):
        self.n_q = n_q
        self.mass = np.zeros((n_q, n_q))
        self.cons = []

    def point(self, q, qd, body, s):
        """Position, Jacobian row block and quadratic term of a body-fixed
        point."""
        if body is None:
            return None
        i, kind = body
        jac = np.zeros((2, self.n_q))
        if kind == "body":
            th, om = q[i + 2], qd[i + 2]
            a = _rot(th)
            jac[:, i:i + 2] = np.eye(2)
            jac[:, i + 2] = PERP @ a @ s
            return q[i:i + 2] + a @ s, jac, -om * om * (a @ s)
        jac[:, i:i + 2] = np.eye(2)
        return q[i:i + 2].copy(), jac, np.zeros(2)

    def accelerations(self, q, qd, forces):
        rows, gam = [], []
        for left, right in self.cons:
            j = np.zeros((2, self.n_q))
            quad = np.zeros(2)
            for sign, item in ((1.0, left), (-1.0, right)):
                kind = item[0]
                if kind == "point":
                    _, jac, qq = self.point(q, qd, item[1], item[2])
                    j += sign * jac
                    quad += sign * qq
                elif kind == "dof_y":  # (x fixed, y = extra dof)
                    j[1, item[1]] += sign
            rows.append(j)
            gam.append(-quad)
        phi = np.vstack(rows)
        n_c = phi.shape[0]
        mat = np.block([[self.mass, phi.T], [phi, np.zeros((n_c, n_c))]])
        sol = np.linalg.solve(mat,
                              np.concatenate([forces, np.concatenate(gam)]))
        return sol[:self.n_q]


def _rod_body_state(rods, thetas, omegas, root, v_root):
    """Mass-center coordinates and velocities of a rod chain (for the DAE)."""
    q, qd = [], []
    r, v = np.asarray(root, dtype=float), np.asarray(v_root, dtype=float)
    for rod, th, om in zip(rods, thetas, omegas):
        a = _rot(th)
        q += list(r + a @ rod.c) + [th]
        qd += list(v + om * PERP @ (a @ rod.c)) + [om]
        r = r + a @ rod.b
        v = v + om * PERP @ (a @ rod.b)
    return np.array(q), np.array(qd)


# ----------------------------------------------------------------------
# Finite segment TMM (Sec. 9.4, Example 9.5)
# ----------------------------------------------------------------------
_E95 = EX["9.5"]["data"]
EI95 = _E95["E"] * _E95["I"]
RA95 = _E95["rho"] * _E95["A"]


def test_finite_segment_accelerations_vs_dae():
    """XM: finite segment TMM accelerations and root moment =
    absolute-coordinate DAE.

    Random state of a 6-segment beam on a base with theta_b, omega_b, alpha_b
    != 0. Tolerance 1e-9 relative (linear solves with stiffness ratio ~1e4).
    """
    n = 6
    base = (0.3, 1.7, 0.8)
    fs = ctl.FiniteSegmentBeam(_E95["l"], EI95, RA95, n, lambda t: base,
                               damping=12.0)
    rng = np.random.default_rng(5)
    th = 0.3 + 0.05 * rng.normal(size=n)
    om = 1.7 + 0.2 * rng.normal(size=n)
    alpha, m_root, q_root = fs.solve(0.0, th, om)
    dae = _PlanarDAE(3 * n)
    forces = np.zeros(3 * n)
    for i, rod in enumerate(fs.rods):
        dae.mass[3 * i:3 * i + 2, 3 * i:3 * i + 2] = rod.mass * np.eye(2)
        dae.mass[3 * i + 2, 3 * i + 2] = rod.j_c
    dae.cons.append((("point", (0, "body"), -fs.rods[0].c), ("ground", None)))
    for i in range(n - 1):
        dae.cons.append((("point", (3 * i, "body"),
                          fs.rods[i].b - fs.rods[i].c),
                         ("point", (3 * i + 3, "body"), -fs.rods[i + 1].c)))
    # torsional springs: root (2 EI/l to the base) and between segments (EI/l)
    k_root, k_in = 2 * EI95 / fs.l_seg, EI95 / fs.l_seg
    mom = k_root * (th[0] - base[0]) + 12.0 * (om[0] - base[1])
    forces[2] -= mom
    for i in range(n - 1):
        mom = k_in * (th[i + 1] - th[i]) + 12.0 * (om[i + 1] - om[i])
        forces[3 * i + 5] -= mom
        forces[3 * i + 2] += mom
    q, qd = _rod_body_state(fs.rods, th, om, np.zeros(2), np.zeros(2))
    acc = dae.accelerations(q, qd, forces)
    np.testing.assert_allclose(alpha, acc[2::3], rtol=1e-9)
    assert m_root == pytest.approx(k_root * (th[0] - base[0])
                                   + 12.0 * (om[0] - base[1]))
    # root reaction: total force on the chain = sum m a_c
    total = sum(rod.mass * acc[3 * i:3 * i + 2] for i, rod
                in enumerate(fs.rods))
    np.testing.assert_allclose(q_root, total, rtol=1e-9)


def test_finite_segment_frequencies_vs_library_and_euler_bernoulli():
    """XM + CV: linearized finite segment TMM = linear library chain; order 2
    to Euler-Bernoulli.

    The Jacobian d(alpha)/d(theta) at rest (central differences, h = 1e-7)
    gives omega^2; the library chain (BeamHinge + TransverseRigidBody,
    Eq. 15.13) is solved by the frequency determinant. Tolerance 1e-7
    relative (difference quotient). Errors vs the cantilever frequencies
    1.8751^2, 4.6941^2 sqrt(EI/(rho A l^4)) must fall by 4 +- 0.8 per halving.
    """
    l_beam = _E95["l"]
    w_eb = np.array([1.875104068711961, 4.694091132974175])**2 * math.sqrt(
        EI95 / (RA95 * l_beam**4))
    errs = []
    for n in (5, 10, 20):
        fs = ctl.FiniteSegmentBeam(l_beam, EI95, RA95, n,
                                   lambda t: (0.0, 0.0, 0.0))
        th, om = np.zeros(n), np.zeros(n)
        jac = np.zeros((n, n))
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1e-7
            jac[:,
                i] = (fs.solve(0.0, th + e, om)[0]
                      - fs.solve(0.0, th - e, om)[0]) / 2e-7
        w_tmm = np.sort(np.sqrt(-np.linalg.eigvals(jac).real))[:2]
        ls = fs.l_seg
        m_s = RA95 * ls
        els = [lib.BeamHinge(kr=2 * EI95 / ls)]
        for i in range(n):
            els.append(lib.TransverseRigidBody(m_s, m_s * ls**2 / 3.0, ls,
                                               ls / 2.0))
            if i < n - 1:
                els.append(lib.BeamHinge(kr=EI95 / ls))
        from solvcon.multibody.tmm.linear import Chain
        w_lib = Chain(els, [0, 1],
                      [2, 3]).natural_frequencies(1.2 * w_eb[1])[:2]
        np.testing.assert_allclose(w_tmm, w_lib, rtol=1e-7)
        errs.append(np.abs(w_lib / w_eb - 1.0))
    errs = np.array(errs)
    orders = np.log2(errs[:-1] / errs[1:])
    assert np.all(np.abs(orders - 2.0) < 0.2), orders


def test_finite_segment_static_tip_deflection_order():
    """AN + CV: quasi-static deflection under a base angular acceleration (Eq.
    9.2-9.4).

    At rest with alpha_b = 1e-3 the beam carries q(x) = -rho A alpha_b x; the
    Euler-Bernoulli tip deflection is -11 rho A alpha_b l^5 / (120 EI). The
    finite segment error falls by 4 +- 0.4 per halving of the segment length.
    """
    al, l_beam = 1e-3, _E95["l"]
    exact = -11.0 * RA95 * al * l_beam**5 / (120.0 * EI95)
    errs = []
    for n in (5, 10, 20):
        fs = ctl.FiniteSegmentBeam(l_beam, EI95, RA95, n,
                                   lambda t: (0.0, 0.0, al))
        th, om = np.zeros(n), np.zeros(n)
        for _ in range(4):  # Newton on alpha_i(theta) = alpha_b (co-rotation)
            r = fs.solve(0.0, th, om)[0] - al
            jac = np.zeros((n, n))
            for i in range(n):
                e = np.zeros(n)
                e[i] = 1e-8
                jac[:,
                    i] = (fs.solve(0.0, th + e, om)[0]
                          - fs.solve(0.0, th - e, om)[0]) / 2e-8
            th = th - np.linalg.solve(jac, r)
        errs.append(abs(fs.tip_deformation(0.0, th)[1] / exact - 1.0))
    ratios = np.array(errs[:-1]) / np.array(errs[1:])
    assert np.all(np.abs(ratios - 4.0) < 0.4), ratios


def test_finite_segment_rk4_order():
    """CV: RK4 of the finite segment TMM during the spin-up of Example 9.5 (Eq.
    9.8), order 4.

    N = 4 segments with an initial deflection (free vibration on top of the
    spin-up) over 1 s; errors at h = 2e-3, 1e-3 against h = 2.5e-4 must give
    order 4 +- 0.2.
    """
    fs = ctl.FiniteSegmentBeam(_E95["l"], EI95, RA95, 4,
                               ctl.spin_up_base(_E95["Omega"], _E95["T"]))
    th0 = np.array([0.05, 0.1, 0.12, 0.13])
    finals = {h: fs.simulate(1.0, h, th0, np.zeros(4))[1][-1] for h
              in (2e-3, 1e-3, 2.5e-4)}
    e1 = np.max(np.abs(finals[2e-3] - finals[2.5e-4]))
    e2 = np.max(np.abs(finals[1e-3] - finals[2.5e-4]))
    assert abs(math.log2(e1 / e2) - 4.0) < 0.2


def test_spin_up_base_motion():
    """AN: Eq. (9.8) base motion: omega = d theta/dt, alpha = d omega/dt,
    continuity at T."""
    base = ctl.spin_up_base(6.0, 15.0)
    for t in (0.5, 7.0, 14.9, 15.0, 20.0):
        h = 1e-5
        th_p, om_p, _ = base(t + h)
        th_m, om_m, _ = base(t - h)
        _, om, al = base(t)
        assert (th_p - th_m) / (2 * h) == pytest.approx(om, rel=1e-8, abs=1e-9)
        if abs(t - 15.0) > 1e-3:
            assert (om_p - om_m) / (2 * h) == pytest.approx(al, rel=1e-7,
                                                            abs=1e-8)
    assert base(15.0)[1] == pytest.approx(6.0)


# ----------------------------------------------------------------------
# FE-TMM for multibody systems (Sec. 9.3, Example 9.4)
# ----------------------------------------------------------------------
def _example_9_4():
    d = EX["9.4"]["data"]
    area = d["side"] ** 2
    ei = d["E"] * d["side"] ** 4 / 12.0
    rods = [ctl.PlanarRod.uniform(d["rod_mass"], d["rod_length"], d["rod_jc"])
            for _ in range(2)]
    sys_ = ctl.BeamRodsSystem((ei, d["rho"] * area, d["L"], d["n_elements"]),
                              rods, gravity=G)
    th0 = -math.radians(d["theta0_deg_below_horizontal"])
    return sys_, th0


def test_hermite_beam_static_tip_load():
    """AN: FE cantilever (Example 9.4 Eq. e-h) reproduces P L^3/(3 EI) and P
    L^2/(2 EI) exactly."""
    _, k, _ = ctl.cantilever_fe(175.0, 0.78, 1.0, 4)
    f = np.zeros(8)
    f[-2] = 2.0
    d = np.linalg.solve(k, f)
    assert d[-2] == pytest.approx(2.0 / (3 * 175.0), rel=1e-12)
    assert d[-1] == pytest.approx(2.0 / (2 * 175.0), rel=1e-12)
    # consistent element mass: a rigid translation carries rho A l
    _, me = ctl.hermite_beam_matrices(175.0, 0.78, 0.25)
    e = np.array([1.0, 0.0, 1.0, 0.0])
    assert e @ me @ e == pytest.approx(0.78 * 0.25, rel=1e-14)


def test_beam_rods_accelerations_vs_dae():
    """XM: FE-TMM accelerations of Example 9.4 = absolute-coordinate DAE with
    the same FE beam.

    Random beam state and rod velocities at the initial angles. Tolerance
    1e-9 relative (stiffness ratio of the FE beam ~1e6).
    """
    sys_, th0 = _example_9_4()
    rng = np.random.default_rng(11)
    nb = sys_.n_b
    delta = 1e-3 * rng.normal(size=nb)
    delta_d = 1e-2 * rng.normal(size=nb)
    th = np.array([th0, th0 + 0.2])
    om = np.array([0.7, -1.1])
    dd, al, _ = sys_.solve(delta, delta_d, th, om)
    nq = nb + 6
    dae = _PlanarDAE(nq)
    dae.mass[:nb, :nb] = sys_.m_b
    forces = np.zeros(nq)
    forces[:nb] = sys_.p_b - sys_.k_b @ delta
    for i, rod in enumerate(sys_.rods):
        j = nb + 3 * i
        dae.mass[j:j + 2, j:j + 2] = rod.mass * np.eye(2)
        dae.mass[j + 2, j + 2] = rod.j_c
        forces[j + 1] = -rod.mass * G
    r3 = sys_.rods[0]
    dae.cons.append((("point", (nb, "body"), -r3.c), ("dof_y", nb - 2)))
    dae.cons.append((("point", (nb, "body"), r3.b - r3.c),
                     ("point", (nb + 3, "body"), -sys_.rods[1].c)))
    q_r, qd_r = _rod_body_state(sys_.rods, th, om, [1.0, delta[-2]],
                                [0.0, delta_d[-2]])
    acc = dae.accelerations(np.concatenate([delta, q_r]),
                            np.concatenate([delta_d, qd_r]), forces)
    np.testing.assert_allclose(dd, acc[:nb], rtol=1e-9,
                               atol=1e-9 * np.max(np.abs(dd)))
    np.testing.assert_allclose(al, acc[nb + 2::3], rtol=1e-9)


def test_beam_rods_energy_and_order():
    """IV + CV: Example 9.4 by the FE-TMM with RK4: energy drift and order 4 +-
    0.2.

    Start of Example 9.4 (beam straight at rest, rods at -60 deg, gravity).
    The highest FE beam frequency is 1.43e4 rad/s. Over 5 ms the RK4 energy
    error at h = 2.5e-5 is below 1e-7 of the energy scale m g l; errors at
    h = 2.5e-5, 1.25e-5 against h = 3.125e-6 give the order.
    """
    sys_, th0 = _example_9_4()
    nq = sys_.n_b + 2
    y0 = np.zeros(2 * nq)
    y0[sys_.n_b:nq] = th0
    finals = {}
    for h in (2.5e-5, 1.25e-5, 3.125e-6):
        y = y0.copy()
        for i in range(int(round(0.005 / h))):
            y = ctl.rk4_step(sys_.rhs, i * h, y, h)
        finals[h] = y
    e_scale = 1.0 * G * 1.0
    assert abs(sys_.energy(finals[2.5e-5]) - sys_.energy(y0)) < 1e-7 * e_scale
    e1 = np.max(np.abs(finals[2.5e-5] - finals[3.125e-6]))
    e2 = np.max(np.abs(finals[1.25e-5] - finals[3.125e-6]))
    assert abs(math.log2(e1 / e2) - 4.0) < 0.2


# ----------------------------------------------------------------------
# Mixed TMM (Sec. 9.2, Examples 9.1 and 9.3)
# ----------------------------------------------------------------------
SPACING_93 = EX["9.3"]["assumed"]["spacing"]
SPACING_91 = EX["9.1"]["assumed"]["spacing"]


def _mixed(example):
    d = EX[example]["data"]
    rod = ctl.PlanarRod.uniform(d["rod_mass"], d["rod_length"], d["rod_jc"])
    if example == "9.3":
        return ctl.mixed_example_9_3(rod, d["particle_mass"], d["spring_k"],
                                     SPACING_93, 5, G)
    return ctl.mixed_example_9_1(rod, d["particle_mass"], d["spring_k"],
                                 SPACING_91, G)


def _absolute(sys_, q, qd):
    """Independent kinematics: rod mass-center states and particle states.

    Returns (rod_q, rod_qd, rod_list, p_pos, p_vel) with rod_q rows (x_c, y_c,
    theta).
    """
    n_p = len(sys_.m)
    pos = np.full((n_p, 2), np.nan)
    vel = np.full((n_p, 2), np.nan)
    for k, i in enumerate(sys_.free_particles):
        pos[i] = q[sys_.n_angles + 2 * k:sys_.n_angles + 2 * k + 2]
        vel[i] = qd[sys_.n_angles + 2 * k:sys_.n_angles + 2 * k + 2]
    rod_q, rod_qd, rod_list = [], [], []
    pending = list(range(len(sys_.chains)))
    while pending:
        for c in list(pending):
            rods, root, tip = sys_.chains[c]
            if root[0] == "ground":
                r, v = np.array(root[1], dtype=float), np.zeros(2)
            elif not np.isnan(pos[root[1]][0]):
                r, v = pos[root[1]].copy(), vel[root[1]].copy()
            else:
                continue
            th = q[sys_.angle_slices[c]]
            om = qd[sys_.angle_slices[c]]
            for rod, t_j, w_j in zip(rods, th, om):
                a = _rot(t_j)
                rod_q.append(np.concatenate([r + a @ rod.c, [t_j]]))
                rod_qd.append(np.concatenate([v + w_j * PERP @ (a @ rod.c),
                                              [w_j]]))
                rod_list.append((c, rod))
                r = r + a @ rod.b
                v = v + w_j * PERP @ (a @ rod.b)
            if tip[0] == "particle":
                pos[tip[1]], vel[tip[1]] = r, v
            pending.remove(c)
    return np.array(rod_q), np.array(rod_qd), rod_list, pos, vel


def _mixed_dae_accelerations(sys_, q, qd):
    """Absolute-coordinate DAE of a MixedSystem; returns accelerations in the
    layout of q."""
    rq, rqd, rods, pos, vel = _absolute(sys_, q, qd)
    n_r, n_p = len(rods), len(sys_.m)
    nq = 3 * n_r + 2 * n_p
    dae = _PlanarDAE(nq)
    forces = np.zeros(nq)
    for k, (_, rod) in enumerate(rods):
        dae.mass[3 * k:3 * k + 2, 3 * k:3 * k + 2] = rod.mass * np.eye(2)
        dae.mass[3 * k + 2, 3 * k + 2] = rod.j_c
        forces[3 * k + 1] = -rod.mass * G
    for i in range(n_p):
        j = 3 * n_r + 2 * i
        dae.mass[j:j + 2, j:j + 2] = sys_.m[i] * np.eye(2)
        forces[j + 1] = -sys_.m[i] * G
    for i, j, k, l0 in sys_.springs:
        d = pos[j] - pos[i]
        t = k * (np.linalg.norm(d) - l0) * d / np.linalg.norm(d)
        forces[3 * n_r + 2 * i:3 * n_r + 2 * i + 2] += t
        forces[3 * n_r + 2 * j:3 * n_r + 2 * j + 2] -= t
    k = 0
    for c, (crods, root, tip) in enumerate(sys_.chains):
        first = k
        if root[0] == "ground":
            dae.cons.append((("point", (3 * k, "body"), -crods[0].c),
                             ("ground", None)))
        else:
            dae.cons.append((("point", (3 * k, "body"), -crods[0].c),
                             ("point", (3 * n_r + 2 * root[1], "particle"),
                              None)))
        for m in range(1, len(crods)):
            dae.cons.append((("point", (3 * (first + m - 1), "body"),
                              crods[m - 1].b - crods[m - 1].c),
                             ("point", (3 * (first + m), "body"),
                              -crods[m].c)))
        last = first + len(crods) - 1
        if tip[0] == "particle":
            dae.cons.append((("point", (3 * last, "body"),
                              crods[-1].b - crods[-1].c),
                             ("point", (3 * n_r + 2 * tip[1], "particle"),
                              None)))
        k += len(crods)
    q_full = np.concatenate([rq.ravel(), pos.ravel()])
    qd_full = np.concatenate([rqd.ravel(), vel.ravel()])
    a_full = dae.accelerations(q_full, qd_full, forces)
    acc = np.zeros(sys_.n_q)
    acc[:sys_.n_angles] = a_full[2:3 * n_r:3]
    for kk, i in enumerate(sys_.free_particles):
        j = 3 * n_r + 2 * i
        acc[sys_.n_angles + 2 * kk:sys_.n_angles + 2 * kk + 2] = a_full[j:j
                                                                        + 2]
    return acc


def _mixed_energy(sys_, y):
    n = sys_.n_q
    rq, rqd, rods, pos, vel = _absolute(sys_, y[:n], y[n:])
    e = 0.0
    for (_, rod), qq, vv in zip(rods, rq, rqd):
        e += (0.5 * rod.mass * vv[:2] @ vv[:2] + 0.5 * rod.j_c * vv[2]**2
              + rod.mass * G * qq[1])
    e += (np.sum(0.5 * sys_.m * np.sum(vel**2, axis=1))
          + np.sum(sys_.m * G * pos[:, 1]))
    for i, j, k, l0 in sys_.springs:
        e += 0.5 * k * (np.linalg.norm(pos[j] - pos[i]) - l0)**2
    return e


@pytest.mark.parametrize("example", ["9.3", "9.1"])
def test_mixed_accelerations_vs_dae(example):
    """XM: mixed TMM (rods by TMM, particles by Newton-Euler) =
    absolute-coordinate DAE.

    Random perturbed state of the Example 9.3 / 9.1 models. Tolerance 1e-9
    relative to the largest acceleration (spring stiffness ratio ~1e5).
    """
    sys_, q0 = _mixed(example)
    rng = np.random.default_rng(2)
    q = q0 + 0.02 * rng.normal(size=q0.size)
    qd = 0.5 * rng.normal(size=q0.size)
    acc = sys_.solve(q, qd)
    ref = _mixed_dae_accelerations(sys_, q, qd)
    np.testing.assert_allclose(acc, ref, rtol=1e-9,
                               atol=1e-9 * np.max(np.abs(ref)))


@pytest.mark.parametrize("example,t_end,hs", [
    ("9.3", 0.1, (2.5e-4, 1.25e-4, 3.125e-5)),
    ("9.1", 0.02, (1e-4, 5e-5, 1.25e-5))])
def test_mixed_energy_and_order(example, t_end, hs):
    """IV + CV: mixed TMM with RK4 from rest under gravity: energy and order 4
    +- 0.2.

    Energy error at the middle step below 1e-6 of m g l (Example 9.1:
    highest mesh frequency 2 sqrt(4 k/m) = 2e3 rad/s). Errors at the two
    larger steps against the smallest step give the order.
    """
    sys_, q0 = _mixed(example)
    y0 = np.concatenate([q0, np.zeros_like(q0)])
    finals = {}
    for h in hs:
        y = y0.copy()
        for i in range(int(round(t_end / h))):
            y = ctl.rk4_step(sys_.rhs, i * h, y, h)
        finals[h] = y
    assert abs(_mixed_energy(sys_, finals[hs[1]])
               - _mixed_energy(sys_, y0)) < 1e-6 * G
    d1 = np.max(np.abs(finals[hs[0]] - finals[hs[2]]))
    d2 = np.max(np.abs(finals[hs[1]] - finals[hs[2]]))
    assert abs(math.log2(d1 / d2) - 4.0) < 0.2


@functools.lru_cache(maxsize=None)
def _spin_up(n_segments):
    """Example 9.5 spin-up with N segments, RK4 h = 2e-3 to t = 16 s
    (cached)."""
    fs = ctl.FiniteSegmentBeam(_E95["l"], EI95, RA95, n_segments,
                               ctl.spin_up_base(_E95["Omega"], _E95["T"]))
    ts, ys = fs.simulate(16.0, 2e-3)
    defo = np.array([fs.tip_deformation(t, y[:n_segments])
                     for t, y in zip(ts, ys)])
    return fs, ts, ys, defo


def _spin_up_extrapolated():
    """Richardson extrapolation (order 2, Eq. 9.2-9.4 convergence) of the tip
    minima, N = 5, 10."""
    d5, d10 = _spin_up(5)[3], _spin_up(10)[3]
    u = d10[:, 0].min() + (d10[:, 0].min() - d5[:, 0].min()) / 3.0
    v = d10[:, 1].min() + (d10[:, 1].min() - d5[:, 1].min()) / 3.0
    return u, v


def _modal_spin_up_tip_min(n_modes=5):
    """Independent model: linear Euler-Bernoulli cantilever in the rotating
    frame with centrifugal stiffening (Galerkin, clamped-free modes), RK4 h =
    1e-3."""
    l_b = _E95["l"]
    roots = [1.875104068711961, 4.694091132974175, 7.854757438237613,
             10.99554073487547, 14.13716839104647][:n_modes]
    xg, wg = np.polynomial.legendre.leggauss(200)
    xg, wg = (xg + 1) * l_b / 2, wg * l_b / 2

    def shapes(x):
        out = []
        for b in roots:
            k = b / l_b
            sg = (math.cosh(b) + math.cos(b)) / (math.sinh(b) + math.sin(b))
            p = (np.cosh(k * x) - np.cos(k * x)
                 - sg * (np.sinh(k * x) - np.sin(k * x)))
            d1 = k * (np.sinh(k * x) + np.sin(k * x)
                      - sg * (np.cosh(k * x) - np.cos(k * x)))
            d2 = k * k * (np.cosh(k * x) + np.cos(k * x)
                          - sg * (np.sinh(k * x) + np.sin(k * x)))
            out.append((p, d1, d2))
        return [np.array([o[i] for o in out]) for i in range(3)]

    p, d1, d2 = shapes(xg)
    m = RA95 * (p * wg) @ p.T
    k = EI95 * (d2 * wg) @ d2.T
    g = RA95 / 2 * (d1 * wg * (l_b**2 - xg**2)) @ d1.T
    f = RA95 * (p * wg) @ xg
    p_tip = shapes(np.array([l_b]))[0][:, 0]
    base = ctl.spin_up_base(_E95["Omega"], _E95["T"])
    n = len(roots)

    def rhs(t, y):
        _, om, al = base(t)
        return np.concatenate([y[n:], np.linalg.solve(
            m, -(k + om**2 * (g - m)) @ y[:n] - al * f)])

    y = np.zeros(2 * n)
    v_min = 0.0
    for i in range(16000):
        y = ctl.rk4_step(rhs, i * 1e-3, y, 1e-3)
        v_min = min(v_min, p_tip @ y[:n])
    return v_min


def test_example_9_5_tip_deflection_vs_beam_theory():
    """XM: finite segment TMM (N -> inf by Richardson from N = 5, 10) = linear
    rotating Euler-Bernoulli modal model with centrifugal stiffening, tip
    transverse minimum.

    Tolerance 0.5 % relative: the modal model is linear in v and drops terms of
    order (v/l)^2 = 0.4 %. Measured: -0.6077 m (TMM) vs -0.6084 m (modal).
    """
    _, v = _spin_up_extrapolated()
    assert v == pytest.approx(_modal_spin_up_tip_min(), rel=5e-3)


@pytest.mark.xfail(strict=True, reason=(
    "Example 9.5 Fig. 9.22 (p. 347): figure-read tip transverse minimum "
    "-0.585 m (+-0.01); finite segment TMM gives -0.6077 m (N -> inf), 3.9 % "
    "lower, and the independent Euler-Bernoulli modal model gives -0.6084 m. "
    "Re-reading p. 344-348 gave no missed data (units SI, zero initial "
    "deformation, tip output, spin-up law Eq. 9.8; the axial springs of Eq. "
    "9.6-9.7 add a centrifugal stretch rho A omega^2 l^3/(3 EA) < 1e-4 m at "
    "the time of the minimum). Suspected cause: the plotted Kane-method curve "
    "uses a different beam model or deformation measure."))
def test_example_9_5_tip_transverse_figure():
    """BK (figure) Example 9.5, Fig. 9.22: tip transverse deformation
    minimum."""
    ref = EX["9.5"]["figure_read"]
    _, v = _spin_up_extrapolated()
    assert (abs(v - ref["tip_transverse_min"])
            < ref["read_precision"]["transverse"])


@pytest.mark.xfail(strict=True, reason=(
    "Example 9.5 Fig. 9.23 (p. 348): figure-read tip longitudinal minimum "
    "-0.0198 m (+-0.0005); finite segment TMM gives -0.0213 m (N -> inf), 7.5 "
    "% lower. The longitudinal value is the foreshortening of the transverse "
    "deflection (about 0.62 v^2/l), so it inherits the 3.9 % transverse "
    "discrepancy twice."))
def test_example_9_5_tip_longitudinal_figure():
    """BK (figure) Example 9.5, Fig. 9.23: tip longitudinal deformation
    minimum."""
    ref = EX["9.5"]["figure_read"]
    u, _ = _spin_up_extrapolated()
    assert (abs(u - ref["tip_longitudinal_min"])
            < ref["read_precision"]["longitudinal"])


def test_example_9_5_drive_moment_figure():
    """BK (figure) Example 9.5, Fig. 9.21 p. 347: peak driving moment of the
    spin-up.

    N = 5 segments, RK4 h = 2e-3 to t = 16 s. The driving moment is
    -m_{0,1} (Eq. 9.11). Figure-read peak 338 N m at t = 7.5 s, read
    precision 5 N m; the N = 5, 10, 20 models give 341.3, 340.4, 340.1 N m.
    """
    ref = EX["9.5"]["figure_read"]
    fs, ts, ys, _ = _spin_up(5)
    moments = np.array([-fs.solve(t, y[:5], y[5:])[1]
                        for t, y in zip(ts[::25], ys[::25])])
    k = int(np.argmax(moments))
    assert abs(moments[k]
               - ref["drive_moment_peak"]) < ref["read_precision"]["moment"]
    assert abs(ts[::25][k] - ref["drive_moment_peak_time"]) < 0.5


# ----------------------------------------------------------------------
# Book data checks without the TMM solver
# ----------------------------------------------------------------------
def test_example_9_9_book_values_are_advanced_control_solver_free():
    """Book inconsistency, Example 9.9 (p. 361), from the transcribed data and
    a 3 x 3 solve.

    The global model of Example 9.6 Eq. (a) with the delayed control force
    exp(-i Omega tau) (Eq. 9.40-9.43, tau = 0.001 s) misses the printed
    amplitudes by more than 2 units of the last digit; with exp(+i Omega tau)
    every printed amplitude is reproduced to half a unit.
    """
    ref = EX["9.9"]
    book_cos, book_sin = np.array(ref["cos"]), np.array(ref["sin"])
    x_delay = _global_steady("9.9", ref["delay"])
    x_adv = _global_steady("9.9", -ref["delay"])
    err_delay = max(np.max(np.abs(x_delay.real - book_cos)),
                    np.max(np.abs(-x_delay.imag - book_sin)))
    err_adv = max(np.max(np.abs(x_adv.real - book_cos)),
                  np.max(np.abs(-x_adv.imag - book_sin)))
    assert err_delay > 2 * BOOK_TOL
    assert err_adv < BOOK_TOL


@pytest.mark.xfail(strict=True, reason=(
    "Example 9.9 (p. 361): TMM with the delay of Eq. (9.40)-(9.43), tau = "
    "+0.001 s, gives x21 = -0.0297 cos, x43 = 0.0086 cos + 0.0169 sin, x65 = "
    "0.0103 cos + 0.0201 sin; printed -0.0295, 0.0084 + 0.0170, 0.0100 + "
    "0.0202 (up to 2.5 units of the last digit). Cause: the printed values "
    "are the response for exp(+i Omega tau) (see the solver-free test)."))
def test_example_9_9_book_values_with_delay():
    """BK Example 9.9: printed steady state vs the delay-controlled TMM."""
    x = _tmm("9.9").steady_state(OMEGA)[[1, 3, 5], 0]
    np.testing.assert_allclose(x.real, EX["9.9"]["cos"], atol=BOOK_TOL)
    np.testing.assert_allclose(-x.imag, EX["9.9"]["sin"], atol=BOOK_TOL)


_MISSING = {
    "9.1": ["L_(i-1,i),j,0"],
    "9.2": ["h", "E", "mu", "rho"],
    "9.3": ["L_(i,i+1),0"],
    "9.8": ["r_0,j (j = 1..4)", "b_5,j (j = 1..4)"],
    "9.12": ["rho, m = rho A3", "b, t_b", "K_p, K_v", "K_ai (i = 1..3)"],
    "9.13": ["m, J", "K, K'", "f"],
}


@pytest.mark.parametrize("example", sorted(_MISSING))
def test_missing_book_data_is_recorded(example):
    """Data record: examples that can not be built completely list the missing
    values (name, symbol, page) in
    tests/data/multibody/tmm_ch9_examples.json."""
    missing = EX[example]["missing"]
    assert all({"name", "symbol", "page"} <= set(item) for item in missing)
    symbols = {item["symbol"] for item in missing}
    assert set(_MISSING[example]) <= symbols

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
