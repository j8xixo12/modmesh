# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests of solvcon.multibody.tmm.continua: nonlinear and two-dimensional TMM
(book Ch. 4, 15.11-15.15).

Book: Rui, Wang, Zhang, Transfer Matrix Method for Multibody Systems (Wiley
2018). Check codes of docs/roadmap.md Sec. 4.2: AN analytic, BK book example,
XM cross-method, CV convergence order, IV invariant. Every comparison states
its tolerance and the reason.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest


from solvcon.multibody.tmm import continua as ct
from solvcon.multibody.tmm import library as lib
from solvcon.multibody.tmm import riccati as ric
from solvcon.multibody.tmm.linear import Chain

DATA = Path(__file__).parent / "data" / "multibody"
with open(DATA / "tmm_ch4_examples.json") as _fh:
    CH4 = json.load(_fh)["examples"]


def _roots(fun, lo, hi, n=400):
    """Sign-change roots of a real scalar function (scan + bisection to
    1e-14)."""
    xs = np.linspace(lo, hi, n)
    vals = [fun(x) for x in xs]
    out = []
    for a, b, fa, fb in zip(xs[:-1], xs[1:], vals[:-1], vals[1:]):
        if fa == 0.0 or fa * fb < 0.0:
            for _ in range(200):
                mid = 0.5 * (a + b)
                fm = fun(mid)
                if fa * fm <= 0.0:
                    b = mid
                else:
                    a, fa = mid, fm
                if b - a < 1e-14 * abs(b):
                    break
            out.append(0.5 * (a + b))
    return np.array(out)


# ----------------------------------------------------------------------
# Bessel functions (needed by Eq. 15.69-15.70, 15.97-15.98)
# ----------------------------------------------------------------------
def test_bessel_reference_values_and_wronskians():
    """AN: tabulated values (16 digits) at x = 1 and the Wronskian identities.

    Tolerance 1e-13 absolute (values of order 1, quadrature converged to
    round-off) and 1e-12 relative for the Wronskians up to x = 17.
    """
    ref = {("J", 0): 0.7651976865579666, ("J", 1): 0.44005058574493355,
           ("Y", 0): 0.08825696421567696, ("Y", 1): -0.7812128213002887,
           ("I", 0): 1.2660658777520084, ("I", 1): 0.565159103992485,
           ("K", 0): 0.42102443824070834, ("K", 1): 0.6019072301972346}
    fun = {"J": ct.bessel_j, "Y": ct.bessel_y, "I": ct.bessel_i,
           "K": ct.bessel_k}
    for (kind, n), val in ref.items():
        assert fun[kind](n, 1.0) == pytest.approx(val, abs=1e-13)
    for x in (0.3, 5.0, 17.0):
        for n in range(4):
            jn, jn1 = ct.bessel_j(n, x), ct.bessel_j(n + 1, x)
            yn, yn1 = ct.bessel_y(n, x), ct.bessel_y(n + 1, x)
            i_n, in1 = ct.bessel_i(n, x), ct.bessel_i(n + 1, x)
            kn, kn1 = ct.bessel_k(n, x), ct.bessel_k(n + 1, x)
            wjy = jn1 * yn - jn * yn1
            wik = i_n * kn1 + in1 * kn
            assert wjy == pytest.approx(2.0 / (math.pi * x), rel=1e-12)
            assert wik == pytest.approx(1.0 / x, rel=1e-12)


# ----------------------------------------------------------------------
# Sec. 4.2 incremental TMM
# ----------------------------------------------------------------------
def _ex41(eps=None, c=None, f3=None, n_harm=3, sub_order=1):
    d = CH4["4.1"]["data"]
    eps = d["eps"] if eps is None else [eps] * 3
    c = d["c"] if c is None else [c] * 3
    forces = list(d["F"]) if f3 is None else [0.0, 0.0, f3]
    springs = [ct.NonlinearSpring(k, cc, e) for k, cc, e
               in zip(d["k"], c, eps)]
    return ct.IncrementalChain(d["m"], springs, forces, n_harm=n_harm,
                               sub_order=sub_order)


def test_example_4_1_eigenfrequencies():
    """BK Example 4.1, p. 139: omega = 0.402, 1.206, 1.616 (half a unit in the
    last digit)."""
    d = CH4["4.1"]["data"]
    els = []
    for k, m in zip(d["k"], d["m"]):
        els += [lib.Spring(k), lib.LumpedMass(m)]
    w = Chain(els, [0], [1]).natural_frequencies(2.0, omega_min=0.01)
    np.testing.assert_allclose(w, CH4["4.1"]["omega"], atol=5e-4)


def test_incremental_linear_limit_equals_linear_tmm():
    """AN: eps = 0 gives the linear steady state of Chain.steady_state (Eq.
    2.119), 1e-10 relative.

    One increment is exact for a linear system (the residual terms R, S of
    Eq. 4.8 vanish after it); harmonics other than 1 stay zero.
    """
    d = CH4["4.1"]["data"]
    omega = 0.9
    inc = _ex41(eps=0.0, c=0.05, f3=0.1 - 0.05j)
    inc.solve(omega)
    els = []
    for i, (k, m) in enumerate(zip(d["k"], d["m"])):
        els += [lib.Spring(k, c=0.05),
                lib.LumpedMass(m, force=(0.1 - 0.05j) if i == 2 else 0.0)]
    z = Chain(els, [0], [1]).steady_state(omega)
    x_lin = np.array([z[2 * i + 1][0] for i in range(3)])
    coef = inc.mass_coefficients()
    np.testing.assert_allclose(coef[:, 1] - 1j * coef[:, 2], x_lin, rtol=1e-10)
    assert np.max(np.abs(coef[:, 3:])) < 1e-14 * np.max(np.abs(x_lin))


def _global_harmonic_balance(m, k, c, eps, f3, omega, n_harm):
    """Independent harmonic balance of m x'' + spring forces = F on the global
    equations."""
    ns = 64
    tau = 2 * np.pi * np.arange(ns) / ns
    nc = 2 * n_harm + 1
    basis = np.ones((ns, nc))
    deriv = np.zeros((ns, nc))
    for j in range(1, n_harm + 1):
        basis[:, 2 * j - 1], basis[:, 2 * j] = np.cos(j * tau), np.sin(j * tau)
        deriv[:, 2 * j - 1], deriv[:, 2 * j] = (-j * np.sin(j * tau),
                                                j * np.cos(j * tau))
    dd = np.zeros((ns, nc))
    for j in range(1, n_harm + 1):
        dd[:, 2 * j - 1] = -j * j * np.cos(j * tau)
        dd[:, 2 * j] = -j * j * np.sin(j * tau)
    n = len(m)

    def residual(flat):
        xs = flat.reshape(n, nc) @ basis.T
        vs = flat.reshape(n, nc) @ deriv.T
        a_s = flat.reshape(n, nc) @ dd.T
        res = []
        for i in range(n):
            def tension(j):
                if j >= n:
                    return 0.0 * tau
                dx = xs[j] - (xs[j - 1] if j > 0 else 0.0)
                dv = vs[j] - (vs[j - 1] if j > 0 else 0.0)
                return k[j] * dx + eps[j] * dx**3 + c[j] * omega * dv
            force = f3 * np.cos(tau) if i == n - 1 else 0.0 * tau
            r_t = (m[i] * omega**2 * a_s[i] + tension(i) - tension(i + 1)
                   - force)
            res.append(basis.T @ r_t / ns)
        return np.concatenate(res)

    x = np.zeros(n * nc)
    for _ in range(40):
        r0 = residual(x)
        jac = np.zeros((x.size, x.size))
        for q in range(x.size):
            dx = np.zeros(x.size)
            dx[q] = 1e-7
            jac[:, q] = (residual(x + dx) - r0) / 1e-7
        step = np.linalg.solve(jac, -r0)
        x += step
        if np.max(np.abs(step)) < 1e-15:
            break
    return x.reshape(n, nc)


def test_incremental_nonlinear_vs_global_harmonic_balance():
    """XM: incremental TMM (Eq. 4.11-4.29) vs harmonic balance of the global
    equations, 1e-10.

    Same truncation N = 5; the two methods solve the same algebraic system
    in different ways, so they agree to the Newton tolerance.
    """
    m, k = [1.0, 1.63, 1.0], [1.0, 1.0, 1.0]
    inc = _ex41(eps=0.3, c=0.05, f3=0.5, n_harm=5)
    its = inc.solve(1.0)
    assert its > 2
    ref = _global_harmonic_balance(m, k, [0.05] * 3, [0.3] * 3, 0.5, 1.0, 5)
    np.testing.assert_allclose(inc.mass_coefficients(), ref,
                               atol=1e-10 * np.max(np.abs(ref)))


def test_incremental_vs_time_integration():
    """BK check of Sec. 4.2 (numerical integration to steady state), own data.

    RK4 with dt = T/400 from rest over 200 periods (slowest decay
    exp(-200 T * 0.025) < 1e-9 for c = 0.2); Fourier coefficients of the last
    period vs the incremental TMM with N = 7. Tolerance 1e-8 of the
    amplitude: measured deviation 2.8e-9 (RK4 error dt^4 plus the truncation
    of N = 7).
    """
    m, k, c, eps, f3, omega = (np.array([1.0, 1.63, 1.0]), 1.0, 0.2, 0.3, 0.5,
                               1.0)
    inc = _ex41(eps=eps, c=c, f3=f3, n_harm=7)
    inc.solve(omega)

    def rhs(t, y):
        x, v = y[:3], y[3:]
        dx = np.diff(np.concatenate([[0.0], x]))
        dv = np.diff(np.concatenate([[0.0], v]))
        ten = k * dx + eps * dx**3 + c * dv
        force = np.array([0.0, 0.0, f3 * math.cos(omega * t)])
        acc = (-ten + np.concatenate([ten[1:], [0.0]]) + force) / m
        return np.concatenate([v, acc])

    period = 2 * math.pi / omega
    nper, nstep = 200, 400
    dt = period / nstep
    y = np.zeros(6)
    t = 0.0
    last = []
    for step in range(nper * nstep):
        k1 = rhs(t, y)
        k2 = rhs(t + dt / 2, y + dt / 2 * k1)
        k3 = rhs(t + dt / 2, y + dt / 2 * k2)
        k4 = rhs(t + dt, y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        if step >= (nper - 1) * nstep:
            last.append(y[:3].copy())
    last = np.array(last)
    tau = omega * dt * np.arange(1, nstep + 1)
    coef_t = np.zeros((3, 2 * 7 + 1))
    coef_t[:, 0] = last.mean(axis=0)
    for j in range(1, 8):
        coef_t[:, 2 * j - 1] = 2 / nstep * last.T @ np.cos(j * tau)
        coef_t[:, 2 * j] = 2 / nstep * last.T @ np.sin(j * tau)
    coef = inc.mass_coefficients()
    np.testing.assert_allclose(coef_t, coef, atol=1e-8 * np.max(np.abs(coef)))
    # the cubic term produces a third harmonic well above the tolerance
    assert np.max(inc.amplitudes()[:, 3]) > 1e-4


def test_subharmonic_time_scale_consistency():
    """IV: tau = Omega t / 3 (Sec. 4.2 item 2) started at zero reproduces the
    harmonic solution.

    Harmonic 3 of the l = 3 solution equals harmonic 1 of the l = 1 solution
    and harmonics 1, 2 stay zero; tolerance 1e-11 (Newton tolerance).
    """
    h1 = _ex41(eps=0.3, c=0.05, f3=0.5, n_harm=3)
    h1.solve(1.0)
    h3 = _ex41(eps=0.3, c=0.05, f3=0.5, n_harm=9, sub_order=3)
    h3.solve(1.0)
    c1, c3 = h1.mass_coefficients(), h3.mass_coefficients()
    np.testing.assert_allclose(c3[:, 5:7], c1[:, 1:3], atol=1e-11)
    assert np.max(np.abs(c3[:, 1:5])) < 1e-11


# ----------------------------------------------------------------------
# Sec. 4.3 FE-TMM for thin plates
# ----------------------------------------------------------------------
def _global_eigs(mesh, bc_x):
    k, m, _ = mesh.global_matrices(bc_x)
    lo = np.linalg.cholesky(m)
    li = np.linalg.inv(lo)
    return np.sqrt(np.linalg.eigvalsh(li @ k @ li.T))


@pytest.mark.parametrize("bc_x,bc_y", [(("S", "S"), ("S", "S")),
                                       (("C", "S"), ("S", "C")),
                                       (("C", "F"), ("S", "S"))])
def test_fe_tmm_frequencies_equal_global_fem(bc_x, bc_y):
    """XM: FE-TMM eigenfrequencies (Eq. 4.77-4.81) equal the global FEM of the
    same mesh, 1e-9.

    Same elements and mesh; the TMM only reorders the algebra. The bound
    covers the conditioning of the transfer product of 4 strips.
    """
    mesh = ct.FEPlateMesh(1.0, 1.0, 4, 4, 1.0, 0.3, 1.0, bc_y=bc_y)
    ref = _global_eigs(mesh, bc_x)
    w = mesh.chain(bc_x).natural_frequencies(1.05 * ref[3], omega_min=0.5,
                                             n_grid=500)
    np.testing.assert_allclose(w, ref[:len(w)], rtol=1e-9)
    assert len(w) >= 4


def test_fe_tmm_convergence_ssss():
    """CV + AN: lambda_11 of the SSSS square plate converges to 2 pi^2
    (Kirchhoff) from below.

    Measured order on the meshes 8 -> 16 is 1.9; the test asks 2 +- 0.2
    (roadmap CV rule). The element is non-conforming, so the error sign is
    not fixed by theory; it is negative on these meshes.
    """
    errs = []
    for n in (4, 8, 16):
        mesh = ct.FEPlateMesh(1.0, 1.0, n, n, 1.0, 0.3, 1.0)
        errs.append(2 * math.pi**2 - _global_eigs(mesh, ("S", "S"))[0])
    assert all(e > 0 for e in errs)
    order = math.log2(errs[1] / errs[2])
    assert 1.8 <= order <= 2.2


def _sections(states, n_sec, nd):
    return np.array([np.real(states[2 * i][:nd])
                     for i in range(n_sec)]).ravel()


def test_fe_tmm_static_vs_global_fem():
    """XM: static response with section loads (Eq. 4.82-4.86, 4.106-4.115) vs
    the global FEM solve.

    Uniform pressure, consistent nodal loads (Eq. 4.60), SSSS, 8 x 8 mesh.
    Riccati TMM (Sec. 4.4.2.2): 1e-12 of the peak (round-off). Plain TMM
    (Eq. 4.85): 1e-9 with 4 strips, but 8 strips lose about 3 digits
    (measured 1.6e-3), the round-off growth that motivates the Riccati form.
    The load enters as Q -> Q - P; with the printed +P of Eq. (4.82) the
    deflection has the opposite sign. AN: the center deflection is within
    3 % of the Navier value 0.00406235 p a^4 / D (measured 1.1 %).
    """
    for n, tol_plain in ((4, 1e-9), (8, None)):
        mesh = ct.FEPlateMesh(1.0, 1.0, n, n, 1.0, 0.3, 1.0)
        nd = mesh.n_dof
        loads = mesh.consistent_pressure(1.0)
        chain = mesh.chain(("S", "S"), loads=loads)
        k, _, free = mesh.global_matrices(("S", "S"))
        d_glob = np.zeros(loads.size)
        d_glob[free] = np.linalg.solve(k, loads.ravel()[free])
        scale = np.max(np.abs(d_glob))
        d_ric = _sections(ric.riccati_steady_state(chain, 0.0)["inputs"],
                          n + 1, nd)
        np.testing.assert_allclose(d_ric, d_glob, atol=1e-12 * scale)
        d_plain = _sections(chain.steady_state(0.0), n + 1, nd)
        err_plain = np.max(np.abs(d_plain - d_glob)) / scale
        if tol_plain is not None:
            assert err_plain < tol_plain
        else:
            assert err_plain > 1e-6
    center = d_glob[4 * nd + mesh.section_dofs.index(3 * 4)]
    assert center == pytest.approx(0.00406235, rel=0.03)
    els = []
    for i, strip in enumerate(mesh.strips() + [None]):
        els.append(ct.SectionLoad(-loads[i]))
        if strip is not None:
            els.append(strip)
    flipped = Chain(els, mesh.section_zero("S"), mesh.section_zero("S"))
    d_flip = _sections(ric.riccati_steady_state(flipped, 0.0)["inputs"], 9, nd)
    np.testing.assert_allclose(d_flip, -d_glob, atol=1e-12 * scale)


def test_table_4_1_ssss_reference_uses_pi_3_14():
    """BK Table 4.1 (p. 151), reference [239], SSSS: 8 of 12 entries are 3.14^2
    (i^2 + j^2).

    The entries (2,3), (3,2) = 128.10, (4,1) = 167.88 and (4,3) = 246.40 fit
    neither 3.14^2 (i^2 + j^2) nor pi^2 (i^2 + j^2) (book data error).
    Tolerance 0.01: one unit in the last printed digit.
    """
    ref = np.array(CH4["4.2"]["ssss_reference"])
    good, bad = [], []
    for i in range(4):
        for j in range(3):
            val = 3.14**2 * ((i + 1) ** 2 + (j + 1) ** 2)
            (good if abs(val - ref[i, j]) <= 0.01
             else bad).append((i + 1, j + 1))
    assert len(good) == 8
    assert bad == [(2, 3), (3, 2), (4, 1), (4, 3)]
    exact = math.pi**2 * 13
    assert abs(exact - 128.10) / exact > 1e-3


def test_table_4_1_csss_reference_vs_levy_exact():
    """BK Table 4.1 group 2 (x = 0 clamped, other edges simply supported) vs
    the exact Levy TMM.

    The Levy strip of Sec. 15.11 is exact. Reference [239] agrees to one
    unit in the last digit for (1,1) and (2,1) only; the other entries are
    off by up to 0.9 % (book data inconsistency).
    """
    ref = np.array(CH4["4.2"]["csss_reference"])
    exact = np.zeros((4, 3))
    for j in range(3):
        strip = ct.LevyPlateStrip(1.0, 1.0, j + 1, 1.0, 1.0, 0.3)
        w = Chain([strip], [0, 1], [0, 2]).natural_frequencies(300.0,
                                                               omega_min=1.0,
                                                               n_grid=800)
        exact[:, j] = w[:4]
    assert exact[0, 0] == pytest.approx(ref[0, 0], abs=0.01)
    assert exact[1, 0] == pytest.approx(ref[1, 0], abs=0.01)


def _csss_exact():
    exact = np.zeros((4, 3))
    for j in range(3):
        strip = ct.LevyPlateStrip(1.0, 1.0, j + 1, 1.0, 1.0, 0.3)
        w = Chain([strip], [0, 1], [0, 2]).natural_frequencies(300.0,
                                                               omega_min=1.0,
                                                               n_grid=800)
        exact[:, j] = w[:4]
    return exact


@pytest.mark.xfail(strict=True, reason=(
    "Table 4.1 group 2 reference [239] vs the exact Levy plate: 10 of 12 "
    "entries differ by more than one unit in the last digit, up to 0.93 % "
    "((1,3): 101.20 vs 100.27). Suspected cause: approximate (Ritz type) "
    "reference solution."))
def test_table_4_1_csss_reference_all_entries():
    """BK Table 4.1 group 2: every reference entry equals the exact Levy value
    within 0.01."""
    np.testing.assert_allclose(_csss_exact(), CH4["4.2"]["csss_reference"],
                               atol=0.01)


@pytest.mark.xfail(strict=True, reason=(
    "Table 4.1 SSSS reference [239] vs exact pi^2 (i^2 + j^2): max difference "
    "0.34 at (4,3) (246.40 vs 246.74). Cause: computed with pi = 3.14 plus 4 "
    "transcription errors (solver-free evidence in "
    "test_table_4_1_ssss_reference_uses_pi_3_14)."))
def test_table_4_1_ssss_reference_equals_navier():
    """BK Table 4.1 group 1: reference entries equal pi^2 (i^2 + j^2) within
    0.01."""
    exact = [[math.pi**2 * (i * i + j * j) for j in (1, 2, 3)] for i
             in (1, 2, 3, 4)]
    np.testing.assert_allclose(exact, CH4["4.2"]["ssss_reference"], atol=0.01)


# ----------------------------------------------------------------------
# Sec. 4.4 FE Riccati TMM
# ----------------------------------------------------------------------
def test_fe_riccati_wilson_theta_equals_global_solver():
    """XM: FE-RTMM (Eq. 4.106-4.115) and the global solve of Eq. (4.100) give
    the same history.

    Plate of Example 4.4 (2.44 m square, h = 6.35 mm, E = 68.95 GPa,
    mu = 0.23, rho = 2524 kg/m3, step pressure 479 Pa, dt = 1.5 ms), all
    edges clamped, 4 x 4 strips, plus cubic node springs k3 w^3 (synthetic
    nonlinearity; the book's geometric nonlinearity is not specified).
    Tolerance 1e-9 of the peak deflection: identical algorithm, the solvers
    differ only in round-off. The Newton loop needs more than one iteration
    per step (nonlinear residual).
    """
    e_mod, h, mu, rho = 6.895e10, 0.00635, 0.23, 2524.0
    d_flex = e_mod * h**3 / (12 * (1 - mu * mu))
    mesh = ct.FEPlateMesh(2.44, 2.44, 4, 4, d_flex, mu, rho * h,
                          bc_y=("C", "C"))
    loads = mesh.consistent_pressure(479.0)
    dyn = ct.FERiccatiDynamics(mesh, ("C", "C"), k3=1e8)
    out_r = dyn.run(lambda t: loads, 0.0015, 30, solver="riccati")
    out_g = dyn.run(lambda t: loads, 0.0015, 30, solver="global")
    peak = np.max(np.abs(out_g["v"]))
    assert peak > 1e-3
    np.testing.assert_allclose(out_r["v"], out_g["v"], atol=1e-9 * peak)
    assert max(out_r["iterations"]) >= 2


# ----------------------------------------------------------------------
# Sec. 4.4 von Karman plate, Examples 4.4 and 4.5
# ----------------------------------------------------------------------
E44 = CH4["4.4"]["data"]
CLAMP = [0, 1, 2, 3, 4]


def _quarter(n_x, n_y, nonlinear=True, **kw):
    """Quarter plate: x = 0 and y = 0 symmetry lines, the other edges clamped
    and immovable."""
    half = 0.5 * E44["a"]
    mesh = ct.VonKarmanPlateMesh(half, half, n_x, n_y, E44["E"], E44["h"],
                                 E44["mu"], E44["rho"], [1, 4], CLAMP,
                                 nonlinear=nonlinear, **kw)
    dyn = ct.RiccatiStripDynamics(mesh, mesh.section_zero([2, 3]),
                                  mesh.section_zero(CLAMP))
    return mesh, dyn


def _peak(t, w):
    """Peak value and time by parabolic interpolation of the sampled
    history."""
    k = int(np.argmax(w))
    y0, y1, y2 = w[k - 1], w[k], w[k + 1]
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    return y1 - 0.25 * (y0 - y2) * d, t[k] + d * (t[1] - t[0])


def _ritz_center(c, h, e_mod, mu, p, nb, n_load=8, nq=32):
    """Independent Ritz solution of the clamped, immovable square plate [-c,
    c]^2 (von Karman).

    w = sum a_ij (1 - x^2)^2 (1 - y^2)^2 x^2i y^2j, u and v odd polynomial
    series vanishing on the edges (x, y scaled by c); Newton on the
    quartic energy with Gauss quadrature. Returns the center deflection.
    """
    d = e_mod * h**3 / (12 * (1 - mu * mu))
    amat = e_mod * h / (1
                        - mu * mu) * np.array([[1, mu, 0], [mu, 1, 0],
                                               [0, 0, (1 - mu) / 2]])
    xg, wg = np.polynomial.legendre.leggauss(nq)
    poly = np.polynomial.Polynomial

    def basis(kind, odd=0):
        out = []
        for i in range(nb):
            if kind == "w":
                p0 = poly([1, 0, -1]) ** 2 * poly([0] * (2 * i) + [1])
                out.append([p0(xg), p0.deriv(1)(xg), p0.deriv(2)(xg)])
            else:
                p0 = poly([1, 0, -1]) * poly([0] * (2 * i + odd) + [1])
                out.append([p0(xg), p0.deriv(1)(xg)])
        return np.array(out)

    fx, po, pe = basis("w"), basis("m", 1), basis("m", 0)
    wq = np.outer(wg, wg).ravel() * c * c

    def tens(a, b):
        return np.einsum("iq,jr->qrij", a, b).reshape(nq * nq, nb * nb)

    w_ = tens(fx[:, 0], fx[:, 0])
    wx, wy = tens(fx[:, 1], fx[:, 0]) / c, tens(fx[:, 0], fx[:, 1]) / c
    wxx, wyy = tens(fx[:, 2], fx[:, 0]) / c**2, tens(fx[:, 0], fx[:, 2]) / c**2
    wxy = tens(fx[:, 1], fx[:, 1]) / c**2
    ux, uy = tens(po[:, 1], pe[:, 0]) / c, tens(po[:, 0], pe[:, 1]) / c
    vx, vy = tens(pe[:, 1], po[:, 0]) / c, tens(pe[:, 0], po[:, 1]) / c
    n = nb * nb

    def quad(a, b, wt=None):
        return np.einsum("q,qi,qj->ij", wq if wt is None else wq * wt, a, b)

    kb = d * (quad(wxx + wyy, wxx + wyy)
              - (1 - mu) * (quad(wxx, wyy) + quad(wyy, wxx) - 2 * quad(wxy,
                                                                       wxy)))
    fq = wq @ w_
    z = np.zeros_like(ux)
    bm = np.stack([np.hstack([ux, z]), np.hstack([z, vy]),
                   np.hstack([uy, vx])], 1)
    x = np.zeros(3 * n)
    for step in range(1, n_load + 1):
        for _ in range(60):
            a, b = x[:n], x[n:]
            gx, gy = wx @ a, wy @ a
            eps = (np.einsum("qkj,j->qk", bm, b)
                   + np.stack([0.5 * gx**2, 0.5 * gy**2, gx * gy], 1))
            nr = eps @ amat
            hg = np.stack([gx[:, None] * wx, gy[:, None] * wy,
                           gy[:, None] * wx + gx[:, None] * wy],
                          1)
            ahg = np.einsum("kl,qlj->qkj", amat, hg)
            abm = np.einsum("kl,qlj->qkj", amat, bm)
            g_a = (kb @ a + np.einsum("q,qk,qkj->j", wq, nr, hg)
                   - p * step / n_load * fq)
            g_b = np.einsum("q,qk,qkj->j", wq, nr, bm)
            kaa = kb + np.einsum("q,qki,qkj->ij", wq, hg, ahg) + quad(wx, wx,
                                                                      nr[:,
                                                                         0]) \
                + quad(wy, wy, nr[:, 1]) + quad(wx, wy, nr[:, 2]) + quad(wy,
                                                                         wx,
                                                                         nr[:,
                                                                            2])
            kab = np.einsum("q,qki,qkj->ij", wq, hg, abm)
            kbb = np.einsum("q,qki,qkj->ij", wq, bm, abm)
            dx = np.linalg.solve(np.block([[kaa, kab], [kab.T, kbb]]),
                                 -np.concatenate([g_a, g_b]))
            x += dx
            if np.max(np.abs(dx[:n])) < 1e-14 * np.max(np.abs(x[:n])):
                break
    return x[0]


def test_von_karman_tangent_is_derivative_of_force():
    """FD: strip and stiffener tangents equal central differences of the
    internal force, 1e-8.

    Random displacements of order 1e-2 m (w/h about 2); central difference
    step 1e-7 gives a truncation error near 1e-11 (measured 4e-11). The
    strip tangent is symmetric to round-off (energy-based, Eq. 4.89).
    """
    mesh, _ = _quarter(3, 3, x_stiffeners=[(0, 5.5e6, 4.5), (1, 11e6, 9.0)],
                       y_stiffeners=[(1, 11e6, 9.0)])
    rng = np.random.default_rng(0)
    nd = mesh.n_dof
    x = rng.normal(size=2 * nd) * 1e-2
    _, k = mesh.strip_force(0, x[:nd], x[nd:])
    _, ks = mesh.section_force(1, x[:nd])
    k_fd = np.zeros_like(k)
    ks_fd = np.zeros_like(ks)
    step = 1e-7
    for j in range(2 * nd):
        dx = np.zeros(2 * nd)
        dx[j] = step
        fp = mesh.strip_force(0, (x + dx)[:nd], (x + dx)[nd:])[0]
        fm = mesh.strip_force(0, (x - dx)[:nd], (x - dx)[nd:])[0]
        k_fd[:, j] = (fp - fm) / (2 * step)
        if j < nd:
            ks_fd[:, j] = (mesh.section_force(1, (x + dx)[:nd])[0]
                           - mesh.section_force(1,
                                                (x - dx)[:nd])[0]) / (2 * step)
    np.testing.assert_allclose(k, k_fd, atol=1e-8 * np.max(np.abs(k)))
    np.testing.assert_allclose(ks, ks_fd, atol=1e-8 * np.max(np.abs(ks)))
    np.testing.assert_allclose(k, k.T, atol=1e-14 * np.max(np.abs(k)))


def test_von_karman_static_vs_ritz():
    """AN + CV: static clamped plate of Example 4.4 data vs the independent
    Ritz solution.

    Ritz (6 x 6 terms in w, u, v) converges to 5 digits: linear coefficient
    0.0012653 (series solution 0.00126532, Timoshenko's table prints the
    truncated 0.00126) and w_c/h = 1.21604 at
    q a^4 / (E h^4) = 151. The FE strip model (quarter 6 x 6 and 12 x 12)
    converges with order 1.0 (+-0.2) in the linear case (measured errors 7.8 %,
    3.8 %; the 9-DOF triangle converges slowly on clamped edges) and the
    large-deflection error falls from 1.6 % to 1.2 % (measured).
    """
    a, h, e_mod, mu, p = E44["a"], E44["h"], E44["E"], E44["mu"], E44["p"]
    d = e_mod * h**3 / (12 * (1 - mu * mu))
    lin5 = _ritz_center(a / 2, h, e_mod, mu, p * 1e-4, 5,
                        n_load=1) / (p * 1e-4 * a**4 / d)
    lin6 = _ritz_center(a / 2, h, e_mod, mu, p * 1e-4, 6,
                        n_load=1) / (p * 1e-4 * a**4 / d)
    assert lin6 == pytest.approx(lin5, rel=1e-5)
    assert lin6 == pytest.approx(0.00126532, rel=1e-5)
    nl5 = _ritz_center(a / 2, h, e_mod, mu, p, 5) / h
    nl6 = _ritz_center(a / 2, h, e_mod, mu, p, 6) / h
    assert nl6 == pytest.approx(nl5, rel=1e-4)
    errs_lin, errs_nl = [], []
    for n in (6, 12):
        mesh, dyn = _quarter(n, n, nonlinear=False)
        v = dyn.static(mesh.consistent_pressure(p), 1)
        errs_lin.append(v[mesh.w_index(0, 0)] / (lin6 * p * a**4 / d) - 1.0)
        mesh, dyn = _quarter(n, n)
        v = dyn.static(mesh.consistent_pressure(p), 5)
        errs_nl.append(v[mesh.w_index(0, 0)] / (nl6 * h) - 1.0)
    assert 0.8 <= math.log2(errs_lin[0] / errs_lin[1]) <= 1.2
    assert 0.0 < errs_nl[1] < errs_nl[0] < 0.017
    assert errs_nl[1] < 0.013


def test_example_4_4_linear_limit():
    """AN: at 1e-6 of the load (w/h about 4e-6) the von Karman response equals
    the linear plate.

    Same mesh and Wilson-theta steps. The bending DOFs change by terms of
    relative order (w/h)^2 = 2e-11 (test: 1e-9 of the peak). The in-plane
    DOFs are zero in the linear plate and of order w^2 / a in the von Karman
    plate (relative 1e-8 of the peak; test: below 1e-7).
    """
    p = E44["p"] * 1e-6
    out = []
    for nonlinear in (True, False):
        mesh, dyn = _quarter(*E44["quarter_mesh"], nonlinear=nonlinear)
        loads = mesh.consistent_pressure(p)
        out.append(dyn.run(lambda t, ld=loads: ld, E44["dt"], 20,
                           solver="global")["v"])
    bend = np.tile([dof % 5 < 3 for dof in mesh.section_dofs],
                   mesh.n_strips + 1)
    peak = np.max(np.abs(out[1]))
    np.testing.assert_allclose(out[0][:, bend], out[1][:, bend],
                               atol=1e-9 * peak)
    assert np.max(np.abs(out[1][:, ~bend])) == 0.0
    assert np.max(np.abs(out[0][:, ~bend])) < 1e-7 * peak


def test_example_4_4_riccati_equals_global_fem():
    """XM: FE-RTMM (Eq. 4.106-4.115) = global FEM solve (Eq. 4.100) for the von
    Karman plate.

    Example 4.4 data, 20 steps (30 ms, up to w/h = 2); identical algorithm,
    so the histories agree to round-off: 1e-11 of the peak (measured 7e-15).
    """
    mesh, dyn = _quarter(*E44["quarter_mesh"])
    loads = mesh.consistent_pressure(E44["p"])
    v_r = dyn.run(lambda t: loads, E44["dt"], 20, solver="riccati")
    v_g = dyn.run(lambda t: loads, E44["dt"], 20, solver="global")
    np.testing.assert_allclose(v_r["v"], v_g["v"],
                               atol=1e-11 * np.max(np.abs(v_g["v"])))
    assert max(v_r["iterations"]) >= 3


def test_example_4_4_figure_4_13():
    """BK Example 4.4, Fig. 4.13 (p. 160): center displacement peak, peak time,
    zero crossing.

    Quarter 6 x 8 strips, dt = 1.5 ms, FE-RTMM. Figure reading tolerances
    (JSON): peak 13.5 +- 0.5 mm, 30 +- 2 ms, zero crossing 60 +- 2 ms.
    Computed: 13.16 mm at 29.2 ms, zero at 59.0 ms.
    """
    ref = CH4["4.4"]["fig_reading"]
    mesh, dyn = _quarter(*E44["quarter_mesh"])
    loads = mesh.consistent_pressure(E44["p"])
    out = dyn.run(lambda t: loads, E44["dt"], 44, solver="riccati")
    w = out["v"][:, mesh.w_index(0, 0)] * 1e3
    t = out["t"] * 1e3
    peak, t_peak = _peak(t, w)
    i = next(i for i in range(int(np.argmax(w)), len(w) - 1) if w[i] > 0
             >= w[i + 1])
    t_zero = t[i] + w[i] / (w[i] - w[i + 1]) * (t[i + 1] - t[i])
    assert peak == pytest.approx(ref["peak_mm"], abs=ref["peak_mm_tol"])
    assert t_peak == pytest.approx(ref["peak_time_ms"],
                                   abs=ref["peak_time_tol_ms"])
    assert t_zero == pytest.approx(ref["zero_time_ms"],
                                   abs=ref["zero_time_tol_ms"])


def _ex45_stiffeners():
    d = CH4["4.5"]["data"]
    ea, ei = d["EA"], d["EI_y"]
    xs = [(0, ea / 2, ei / 2), (1, ea, ei), (2, ea, ei), (3, ea, ei)]
    ys = [(0, ea / 2, ei / 2), (2, ea, ei), (4, ea, ei)]
    return xs, ys


@pytest.mark.xfail(strict=True, reason=(
    "Fig. 4.15 not reproduced: with the printed stiffeners (EA = 11e6 N, EI_y "
    "= 9 N m^2, concentric, 7 + 5 lines) the peak is 12.9 mm at 28.8 ms vs "
    "2.65 mm at 20 ms in the figure. Suspected cause: stiffener data "
    "(eccentricity or units) missing; see the solver-free test."))
def test_example_4_5_figure_4_15():
    """BK Example 4.5, Fig. 4.15 (p. 161): peak 2.65 +- 0.1 mm at 20 +- 1.5
    ms."""
    ref = CH4["4.5"]["fig_reading"]
    xs, ys = _ex45_stiffeners()
    mesh, dyn = _quarter(6, 4, x_stiffeners=xs, y_stiffeners=ys)
    loads = mesh.consistent_pressure(E44["p"])
    out = dyn.run(lambda t: loads, CH4["4.5"]["data"]["dt"], 25,
                  solver="riccati")
    peak, t_peak = _peak(out["t"] * 1e3, out["v"][:, mesh.w_index(0, 0)] * 1e3)
    assert peak == pytest.approx(ref["peak_mm"], abs=ref["peak_mm_tol"])
    assert t_peak == pytest.approx(ref["peak_time_ms"],
                                   abs=ref["peak_time_tol_ms"])


def test_example_4_5_stiffener_data_inconsistent_with_figure():
    """Solver-free: the printed stiffeners are too weak for the 5 x reduction
    of Fig. 4.15 vs 4.13.

    Concentric stiffeners at the spacings of Fig. 4.14 add at most
    EA / (E h s) to the membrane stiffness and EI / (D s) to the bending
    stiffness of the plate strip they cover (s = spacing). Both ratios are
    below 0.1, so no stiffness of the stiffened plate exceeds 1.2 times the
    bare plate (two families), while the figures need a peak ratio of 0.20.
    """
    d = CH4["4.5"]["data"]
    e_mod, h, mu, a = E44["E"], E44["h"], E44["mu"], E44["a"]
    dd = e_mod * h**3 / (12 * (1 - mu * mu))
    s_x, s_y = a / (d["stiffeners_x"] + 1), a / (d["stiffeners_y"] + 1)
    ratios = [d["EA"] / (e_mod * h * s) for s in (s_x, s_y)] + \
        [d["EI_y"] / (dd * s) for s in (s_x, s_y)]
    assert max(ratios) < 0.1
    stiffening_bound = 1.0 + 2 * max(ratios)
    fig_ratio = (CH4["4.5"]["fig_reading"]["peak_mm"]
                 / CH4["4.4"]["fig_reading"]["peak_mm"])
    assert 1.0 / fig_ratio > 4 * stiffening_bound


# ----------------------------------------------------------------------
# Sec. 4.5 Fourier series TMM and Sec. 4.6 finite difference TMM
# ----------------------------------------------------------------------
def test_fourier_closed_form_equals_matrix_exponential():
    """AN: Eq. (4.139) equals e^{A y} of Eq. (4.134), 1e-12 relative (round-off
    of cosh, sinh)."""
    for beta, y, e_mod, mu in ((2.0, 0.7, 3.0, 0.3), (0.5, 3.0, 200.0, 0.25)):
        ref = ric.expm(ct.fourier_system_matrix(beta, e_mod, mu) * y)
        np.testing.assert_allclose(ct.fourier_transfer(beta, y, e_mod, mu),
                                   ref, atol=1e-12 * np.max(np.abs(ref)))


def _airy_strip(beta, height, e_mod, mu, alpha, t0, kt, p_edge, ys):
    """Independent Airy stress function solution phi = f(y) sin(beta x) plus
    thermal particular."""
    b = beta

    def funcs(y):
        ch, sh = math.cosh(b * y), math.sinh(b * y)
        g = np.array([ch, y * ch, sh, y * sh])
        g1 = np.array([b * sh, ch + b * y * sh, b * ch, sh + b * y * ch])
        g2 = np.array([b * b * ch, 2 * b * sh + b * b * y * ch, b * b * sh,
                       2 * b * ch + b * b * y * sh])
        g3 = np.array([b**3 * sh, 3 * b * b * ch + b**3 * y * sh, b**3 * ch,
                       3 * b * b * sh + b**3 * y * ch])
        return g, g1, g2, g3

    def part(y):
        temp = t0 + kt * y
        return np.array([-(1 + mu) * alpha * temp / b,
                         -(1 + mu) * alpha * kt / b**2, -e_mod * alpha * temp,
                         -e_mod * alpha * kt / b])

    def state(cvec, y):
        g, g1, g2, g3 = funcs(y)
        f, f1, f2, f3 = g @ cvec, g1 @ cvec, g2 @ cvec, g3 @ cvec
        hom = np.array([-(f2 + mu * b * b * f) / (e_mod * b),
                        -2 * (1 + mu) * f1 / e_mod
                        + (f3 + mu * b * b * f1) / (e_mod * b * b), -b * b * f,
                        -b * f1])
        return hom + part(y), f2

    mat = np.zeros((4, 4))
    rhs = np.zeros(4)
    for row, (y, target) in enumerate(((0.0, p_edge), (height, 0.0))):
        g, g1, _, _ = funcs(y)
        mat[2 * row] = -b * b * g
        mat[2 * row + 1] = -b * g1
        rhs[2 * row] = target - part(y)[2]
        rhs[2 * row + 1] = -part(y)[3]
    cvec = np.linalg.solve(mat, rhs)
    states, sx = zip(*[state(cvec, y) for y in ys])
    return np.array(states), np.array(sx)


def test_fourier_tmm_vs_airy_solution():
    """AN: Fourier TMM (Eq. 4.131-4.144) vs Airy stress function, edge load
    plus linear temperature.

    Tolerance 1e-10 of the largest stress (exact methods; round-off of the
    exponentials).
    """
    beta, height, e_mod, mu, alpha = math.pi, 0.8, 70.0, 0.3, 0.01
    t0, kt, p_edge = 2.0, -1.5, 0.7
    ys = np.linspace(0.0, height, 9)
    states, sx = ct.fourier_strip_solution(beta, height, e_mod, mu, alpha, t0,
                                           kt, p_edge, ys)
    ref, ref_sx = _airy_strip(beta, height, e_mod, mu, alpha, t0, kt, p_edge,
                              ys)
    scale = np.max(np.abs(ref))
    np.testing.assert_allclose(states, ref, atol=1e-10 * scale)
    np.testing.assert_allclose(sx, ref_sx, atol=1e-10 * scale)


def test_fd_book_scheme_exact_for_linear_states():
    """AN: with D1, D2 of Eq. (4.153)-(4.154) the FD-TMM transfers linear-in-x
    states exactly.

    Uniform tension sigma_y = s0 (u = -mu s0 x / E, v = s0 y / E) and pure
    shear tau = t0 (u = 2 (1 + mu) t0 y / E). Tolerance 100 eps ||e^{A y}||
    ||z_0||: round-off of the matrix exponential (no discretization error).
    """
    n, length, e_mod, mu, y = 6, 1.0, 5.0, 0.3, 0.6
    a, _ = ct.fd_system_matrix(n, length, e_mod, mu, scheme="book")
    x = np.linspace(0.0, length, n + 1)
    one = np.ones(n + 1)
    s0, t0 = 2.0, 0.5
    z0 = np.concatenate([-mu * s0 * x / e_mod, 0 * x, s0 * one, 0 * x])
    zy = np.concatenate([-mu * s0 * x / e_mod, s0 * y / e_mod * one, s0 * one,
                         0 * x])
    u = ric.expm(a * y)
    tol = 100 * np.finfo(float).eps * np.linalg.norm(u, 2)
    np.testing.assert_allclose(u @ z0, zy, atol=tol * np.linalg.norm(z0))
    z0 = np.concatenate([0 * x, 0 * x, 0 * x, t0 * one])
    zy = np.concatenate([2 * (1 + mu) * t0 * y / e_mod * one, 0 * x, 0 * x,
                         t0 * one])
    np.testing.assert_allclose(u @ z0, zy, atol=tol * np.linalg.norm(z0))


def test_fd_tmm_converges_to_fourier_solution():
    """CV: symmetric FD-TMM (Eq. 4.155-4.160) converges to the Fourier TMM with
    order 2 +- 0.2.

    One Fourier term (beta = pi / L): edge load and linear temperature.
    Error: max |sigma_y| difference at y = H/2 over the lines. H = 0.1 keeps
    the growth e^{2 H / dx} of the plain transfer below 1e3 for n = 32
    (with H = 0.5 the n = 32 solve loses all digits).
    """
    length, height, e_mod, mu, alpha = 1.0, 0.1, 1.0, 0.3, 0.02
    beta = math.pi / length
    t0, kt, p_edge = 1.0, 0.5, 0.3
    states, _ = ct.fourier_strip_solution(beta, height, e_mod, mu, alpha, t0,
                                          kt, p_edge, [0.5 * height])
    errs = []
    for n in (8, 16, 32):
        x = np.linspace(0.0, length, n + 1)
        sn = np.sin(beta * x)
        out = ct.fd_strip_solution(n, length, height, e_mod, mu, alpha,
                                   t0 * sn, kt * sn, p_edge * sn,
                                   [0.5 * height])
        errs.append(np.max(np.abs(out["sigma_y"][0] - states[0, 2] * sn)))
    for e1, e2 in zip(errs[:-1], errs[1:]):
        assert 1.8 <= math.log2(e1 / e2) <= 2.2


# ----------------------------------------------------------------------
# Sec. 4.7 / 15.13 grid TMM
# ----------------------------------------------------------------------
def _grid(key, **kw):
    d = CH4[key]["data"]
    return ct.GridPlate(d["a"], d["b"], d["h"], d["E"], d["G"], d["mu"],
                        d["rho"], d["n"], d["m"], **kw)


def _match_book(values, computed, tol):
    for val in np.ravel(values):
        assert np.min(np.abs(computed - val)) <= tol, val


def test_example_4_6_table_4_5():
    """BK Example 4.6, Table 4.5 (p. 179): TMM values within 0.01 (one unit in
    the last digit).

    XM: TMM roots equal the global eigenvalues of the same grid to 1e-8.
    AN: the analytical column is Navier pi^2 (i^2 + j^2) sqrt(D / rho h), 0.01.
    The book prints beta = 0.333; the table needs beta = 0.3333 (with 0.333
    the errors are 0.02-0.08).
    """
    grid = _grid("4.6", beta_t=0.3333)
    ref = grid.global_frequencies()[:11]
    w = grid.chain().natural_frequencies(520.0, omega_min=1.0, n_grid=500,
                                         max_halvings=1)
    np.testing.assert_allclose(w, ref, rtol=1e-8)
    _match_book(CH4["4.6"]["tmm"], w, 0.01)
    d = CH4["4.6"]["data"]
    dd = d["E"] * d["h"] ** 3 / (12 * (1 - d["mu"] ** 2))
    navier = np.array([[math.pi**2 * (i * i + j * j)
                        * math.sqrt(dd / (d["rho"] * d["h"]))
                        for i in (1, 2, 3)] for j in (1, 2, 3)])
    np.testing.assert_allclose(navier, CH4["4.6"]["analytical"], atol=0.01)


@pytest.mark.xfail(strict=True, reason=(
    "Table 4.5 with the printed beta = 0.333: computed 65.173, 153.268, "
    "245.551, 288.258, 378.487, 507.034 vs book 65.19, 153.29, 245.61, "
    "288.29, 378.56, 507.14 (errors 0.02-0.11). Cause: the book computed with "
    "beta = 0.3333 (all entries within 0.01)."))
def test_example_4_6_table_4_5_printed_beta():
    """BK Table 4.5 with beta = 0.333 as printed: values within 0.01."""
    _match_book(CH4["4.6"]["tmm"],
                _grid("4.6", beta_t=0.333).global_frequencies(), 0.01)


def test_example_4_7_table_4_6():
    """BK Example 4.7, Table 4.6 (p. 180): all 12 TMM values within 0.01
    without rotary inertia.

    With the rotary inertia of Eq. (4.168) the first value is 170.84 instead
    of 174.65: the book example drops J_x, J_y (inconsistent with Example 4.6).
    XM: TMM roots = global eigenvalues, 5e-7.
    """
    grid = _grid("4.7", beta_t=0.3333, rotary_inertia=False)
    ref = grid.global_frequencies(("C", "C"))
    w = grid.chain(("C", "C")).natural_frequencies(2200.0, omega_min=1.0,
                                                   n_grid=1500)
    # 5e-7: round-off of the 17-matrix transfer product near 2000 rad/s
    # (measured 1.2e-7)
    np.testing.assert_allclose(w, ref[:len(w)], rtol=5e-7)
    _match_book(CH4["4.7"]["tmm"], w, 0.01)


@pytest.mark.xfail(strict=True, reason=(
    "Table 4.6 with the rotary inertia of Eq. (4.168) (used in Example 4.6): "
    "first value 170.84 vs book 174.65 (2.2 %). Cause: the book example drops "
    "J_x, J_y."))
def test_example_4_7_table_4_6_with_rotary_inertia():
    """BK Table 4.6 with the grid model of Eq. (4.168) including J_x, J_y:
    values within 0.01."""
    w = _grid("4.7", beta_t=0.3333).global_frequencies(("C", "C"))
    _match_book(CH4["4.7"]["tmm"], w, 0.01)


def test_example_4_7_reference_239_vs_levy_exact():
    """BK Table 4.6 reference [239] vs the exact Levy TMM (Sec. 15.11):
    agreement 1 %.

    The reference values differ from the exact plate by up to 0.8 % (for
    example 166.67 vs 165.29); the book's error column is relative to them.
    """
    d = CH4["4.7"]["data"]
    dd = d["E"] * d["h"] ** 3 / (12 * (1 - d["mu"] ** 2))
    ref = np.array(CH4["4.7"]["reference_239"])
    for j in range(3):
        strip = ct.LevyPlateStrip(d["a"], d["b"], j + 1, dd, d["rho"] * d["h"],
                                  d["mu"])
        w = Chain([strip], [0, 1], [0, 1]).natural_frequencies(1500.0,
                                                               omega_min=1.0,
                                                               n_grid=800)
        np.testing.assert_allclose(w[:3], ref[j], rtol=1e-2)


@pytest.mark.xfail(strict=True, reason=(
    "Table 4.6 reference [239] vs the exact Levy plate (rows j = 1..3): "
    "differences up to 0.83 % (166.67 vs 165.29, 517.22 vs 514.35). Suspected "
    "cause: approximate reference solution."))
def test_example_4_7_reference_239_at_book_precision():
    """BK Table 4.6 reference [239], rows j = 1..3, equals the exact Levy plate
    within 0.01."""
    d = CH4["4.7"]["data"]
    dd = d["E"] * d["h"] ** 3 / (12 * (1 - d["mu"] ** 2))
    ref = np.array(CH4["4.7"]["reference_239"])
    for j in range(3):
        strip = ct.LevyPlateStrip(d["a"], d["b"], j + 1, dd, d["rho"] * d["h"],
                                  d["mu"])
        w = Chain([strip], [0, 1], [0, 1]).natural_frequencies(1500.0,
                                                               omega_min=1.0,
                                                               n_grid=800)
        np.testing.assert_allclose(w[:3], ref[j], atol=0.01)


def test_grid_converges_to_grillage_limit_not_kirchhoff():
    """CV: the grid model (Sec. 4.7) converges to its continuum limit, not to
    the Kirchhoff plate.

    Limit of the grid: rho h (1 + h^2 (a^2 + b^2)/12) omega^2 =
    D (a^4 + b^4) + 2 G beta h^3 a^2 b^2 (torsion of both beam families).
    For beta = 1/3 the twisting term is 2 (1 - mu) times the plate value
    2 D a^2 b^2, so omega_11 tends to 66.6 rad/s, not 59.60 (Example 4.6
    data): the 9.4 % error of Table 4.5 is model error, not mesh error.
    Measured order 2 on n = 8, 16, 32; the test asks 2 +- 0.2.
    """
    d = CH4["4.6"]["data"]
    dd = d["E"] * d["h"] ** 3 / (12 * (1 - d["mu"] ** 2))
    beta_t = 1.0 / 3.0
    k = math.pi
    limit = math.sqrt((2 * dd * k**4 + 2 * d["G"] * beta_t * d["h"] ** 3
                       * k**4)
                      / (d["rho"] * d["h"]
                         * (1 + d["h"] ** 2 * 2 * k * k / 12)))
    errs = []
    for n in (8, 16, 32):
        g = ct.GridPlate(d["a"], d["b"], d["h"], d["E"], d["G"], d["mu"],
                         d["rho"], n, n, beta_t=beta_t)
        errs.append(abs(g.global_frequencies()[0] - limit))
    for e1, e2 in zip(errs[:-1], errs[1:]):
        assert 1.8 <= math.log2(e1 / e2) <= 2.2
    assert limit > 1.1 * 59.60


# ----------------------------------------------------------------------
# Sec. 15.11 rectangular plate
# ----------------------------------------------------------------------
def test_levy_closed_forms_vs_expm():
    """AN: Eq. (15.53) and (15.54) (with the corrected entries) equal e^{A x}
    of Eq. (15.51), 1e-11.

    The printed u_14 of (15.53) and u_22, u_33, u_41 of (15.54) fail the
    same check (book errors); the printed forms are evaluated below.
    """
    d, mu, rho, x, beta = 2.0, 0.3, 1.5, 0.8, math.pi
    ref = ct.LevyPlateStrip(x, 1.0, 1, d, 0.0, mu).transfer(0j).real
    got = ct.levy_massless_transfer(x, beta, d, mu)
    np.testing.assert_allclose(got, ref, atol=1e-11 * np.max(np.abs(ref)))
    s, c = math.sinh(beta * x), math.cosh(beta * x)
    printed_u14 = -(s / beta - x * c) / (2 * d * beta**2)
    assert printed_u14 == pytest.approx(-ref[0, 3], rel=1e-10)
    for w in (3.0, 40.0):
        ref = ct.LevyPlateStrip(x, 1.0, 1, d, rho, mu).transfer(1j * w).real
        got = ct.levy_free_vibration_transfer(x, beta, d, mu, rho, w)
        np.testing.assert_allclose(got, ref, atol=1e-11 * np.max(np.abs(ref)))


def test_levy_ssss_navier_with_inplane_pressure():
    """AN: SSSS plate with in-plane compression, rho omega^2 = D (a^2 + b^2)^2
    - P_x a^2 - P_y b^2.

    Levy strip (Eq. 15.51) with w = M_x = 0 at x = 0, L_x; 1e-10 relative.
    """
    d, rho, mu, lx, ly = 3.0, 2.0, 0.3, 1.3, 0.9
    px, py = 4.0, 2.5
    for m in (1, 2):
        strip = ct.LevyPlateStrip(lx, ly, m, d, rho, mu, p_x=px, p_y=py)
        w = Chain([strip], [0, 2], [0, 2]).natural_frequencies(120.0,
                                                               omega_min=0.5,
                                                               n_grid=600)
        bb = m * math.pi / ly
        exact = []
        for i in range(1, 6):
            aa = i * math.pi / lx
            w2 = (d * (aa * aa + bb * bb) ** 2 - px * aa * aa
                  - py * bb * bb) / rho
            if 0.25 < w2 ** 0.5 < 120.0:
                exact.append(w2**0.5)
        np.testing.assert_allclose(w, exact, rtol=1e-10)


# ----------------------------------------------------------------------
# Sec. 15.12 disks
# ----------------------------------------------------------------------
def test_disk_magnus_equals_bessel_and_reciprocity():
    """AN: Magnus integration of Eq. (15.64) = exact J, Y, I, K transfer, 1e-9;
    IV reciprocity.

    Magnus error O(h^4) with 400 steps is below 1e-10. Reciprocity
    r U^T J U = r0 J for the work-conjugate pairs (w, Q_r), (theta, M_r)
    holds to 1e-10 of max|U|^2.
    """
    d, mu, rho_h, w = 2.0, 0.3, 1.5, 7.0
    k = (rho_h * w * w / d) ** 0.25
    jmat = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0],
                     [-1, 0, 0, 0.0]])
    for m in (0, 1, 3):
        u = ct.DiskElement(0.3, 1.2, d, mu, rho_h, m,
                           n_steps=400).transfer(1j * w).real
        ref = ct.disk_bessel_transfer(0.3, 1.2, m, k, d, mu)
        np.testing.assert_allclose(u, ref, atol=1e-9 * np.max(np.abs(ref)))
        np.testing.assert_allclose(1.2 * u.T @ jmat @ u, 0.3 * jmat,
                                   atol=1e-10 * np.max(np.abs(u)) ** 2)


def test_clamped_solid_disk_frequencies():
    """AN: clamped circular plate lambda^2 = omega a^2 sqrt(rho h / D) (Leissa,
    NASA SP-160).

    m = 0 from Eq. (15.70) (det of the w, theta rows at r = a); m = 1, 2
    from the regular Bessel solutions. Literature values have 5 digits:
    10.2158, 39.771 (m = 0), 21.260 (m = 1), 34.877 (m = 2); tolerance 5e-5
    relative.
    """
    d, mu, rho_h, a = 1.0, 0.3, 1.0, 1.0

    def det0(lam):
        u = ct.solid_disk_transfer(a, d, mu, rho_h, lam * lam)
        return u[0, 0] * u[1, 2] - u[0, 2] * u[1, 0]

    r0 = _roots(det0, 1.5, 7.0)
    np.testing.assert_allclose(r0**2, [10.2158, 39.771], rtol=5e-5)
    for m, val in ((1, 21.260), (2, 34.877)):
        def detm(lam, m=m):
            sj = ct.disk_bessel_state(a, m, lam, d, mu, "J")
            si = ct.disk_bessel_state(a, m, lam, d, mu, "I")
            return sj[0] * si[1] - si[0] * sj[1]
        rm = _roots(detm, 2.0, 6.2)
        assert rm[0] ** 2 == pytest.approx(val, rel=5e-5)


def test_solid_disk_closed_forms_are_exact_solutions():
    """AN: Eq. (15.70) columns = Bessel states; Eq. (15.66) columns transfer
    with Eq. (15.64), 1e-9.

    Columns 1, 3 of (15.70) are the states of w = (J_0 + I_0)/2 and
    (J_0 - I_0)/(2 D alpha^2); columns of (15.66c) at r1 equal the Magnus
    transfer (lambda = 0) of the columns at r0.
    """
    d, mu, rho_h, w = 1.5, 0.3, 2.0, 5.0
    al = (rho_h * w * w / d) ** 0.25
    r = 0.8
    u = ct.solid_disk_transfer(r, d, mu, rho_h, w)
    sj = ct.disk_bessel_state(r, 0, al, d, mu, "J")
    si = ct.disk_bessel_state(r, 0, al, d, mu, "I")
    np.testing.assert_allclose(u[:, 0], (sj + si) / 2, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(u[:, 2], (sj - si) / (2 * d * al * al),
                               rtol=1e-12, atol=1e-13)
    for m in (2, 3):
        t = ct.DiskElement(0.4, 0.9, d, mu, 0.0, m,
                           n_steps=400).transfer(0j).real
        u0 = ct.solid_disk_massless_matrix(0.4, m, d, mu)
        u1 = ct.solid_disk_massless_matrix(0.9, m, d, mu)
        np.testing.assert_allclose(t @ u0[:, :2], u1[:, :2],
                                   atol=1e-9 * np.max(np.abs(u1)))
    t = ct.DiskElement(0.4, 0.9, d, mu, 0.0, 0, n_steps=400).transfer(0j).real
    u0 = ct.solid_disk_massless_matrix(0.4, 0, d, mu)
    u1 = ct.solid_disk_massless_matrix(0.9, 0, d, mu)
    np.testing.assert_allclose(t @ u0[:, [0, 2]], u1[:, [0, 2]], atol=1e-10)


def test_annular_disk_frequency_magnus_vs_bessel():
    """XM: annular plate clamped at r = 0.4, free at r = 1, m = 1: TMM root =
    Bessel root, 1e-8."""
    d, mu, rho_h = 1.0, 0.3, 1.0
    el = ct.DiskElement(0.4, 1.0, d, mu, rho_h, 1, n_steps=300)
    w = Chain([el], [0, 1], [2, 3]).natural_frequencies(40.0, omega_min=0.5,
                                                        n_grid=200)

    def det(om):
        u = ct.disk_bessel_transfer(0.4, 1.0, 1, (rho_h * om * om / d) ** 0.25,
                                    d, mu)
        return np.linalg.det(u[2:, 2:])

    ref = _roots(det, 0.5, 40.0, n=200)
    np.testing.assert_allclose(w, ref, rtol=1e-8)
    assert len(w) >= 1


# ----------------------------------------------------------------------
# Sec. 15.14 thick-walled cylinder, 15.15 thin-walled cylinder
# ----------------------------------------------------------------------
def test_thick_cylinder_massless_closed_form():
    """AN: Eq. (15.95) with the Lame lambda = Lame solution of a pressurized
    cylinder, 1e-12.

    XM: (15.95) = Magnus integration of Eq. (15.92) with rho = 0, 1e-10.
    The printed lambda of Eq. (15.93), E mu / (2 (1 - mu^2)), breaks the
    agreement by more than 1 % (book error).
    """
    e_mod, mu, a, b, p = 210.0, 0.3, 1.0, 2.0, 5.0
    g, lam = ct.lame_constants(e_mod, mu)
    u = ct.thick_cylinder_massless_transfer(b, a, lam, g, mu)
    ua = -u[1, 1] * (-p) / u[1, 0]
    c1c2 = np.linalg.solve([[2 * (lam + g), -2 * g / a**2],
                            [2 * (lam + g), -2 * g / b**2]],
                           [-p, 0.0])
    assert ua == pytest.approx(c1c2[0] * a + c1c2[1] / a, rel=1e-12)
    ref = ct.ThickCylinder(a, b, lam, g, 0.0, n_steps=200).transfer(0j).real
    np.testing.assert_allclose(u, ref, atol=1e-10 * np.max(np.abs(ref)))
    lam_printed = e_mod * mu / (2 * (1 - mu * mu))
    bad = ct.thick_cylinder_massless_transfer(b, a, lam_printed, g, mu)
    assert np.max(np.abs(bad - ref) / np.max(np.abs(ref))) > 1e-2
    np.testing.assert_allclose(ct.thick_cylinder_massless_solid(b, lam, mu)[0,
                                                                            1],
                               b / (2 * (lam + g)), rtol=1e-14)


def test_thick_cylinder_bessel_forms_and_thin_ring_limit():
    """AN: Eq. (15.97) and (15.98) vs Magnus integration, 1e-9; thin ring limit
    1e-4.

    Free-free cylinder of thickness t = 0.01 R: first radial frequency vs
    the plane-strain ring sqrt(E / ((1 - mu^2) rho)) / R; the difference is
    O((t/R)^2) = 1e-4 (measured 1.1e-5 relative).
    """
    e_mod, mu, rho = 210e9, 0.3, 7800.0
    g, lam = ct.lame_constants(e_mod, mu)
    a, b, w = 0.5, 1.0, 9000.0
    ref = ct.ThickCylinder(a, b, lam, g, rho,
                           n_steps=200).transfer(1j * w).real
    np.testing.assert_allclose(ct.thick_cylinder_transfer(b, a, lam, g, rho,
                                                          w), ref,
                               rtol=1e-9, atol=1e-9 * np.max(np.abs(ref)))
    r0 = 0.05
    t = ct.ThickCylinder(r0, b, lam, g, rho,
                         n_steps=1500).transfer(1j * w).real
    u0 = ct.thick_cylinder_solid_transfer(r0, lam, g, rho, w)
    u1 = ct.thick_cylinder_solid_transfer(b, lam, g, rho, w)
    np.testing.assert_allclose(t @ u0[:, 1], u1[:, 1], rtol=1e-9)
    radius, thick = 1.0, 0.01
    el = ct.ThickCylinder(radius - thick / 2, radius + thick / 2, lam, g, rho,
                          n_steps=20)
    ring = math.sqrt(e_mod / ((1 - mu * mu) * rho)) / radius
    w = Chain([el], [1], [1]).natural_frequencies(1.5 * ring,
                                                  omega_min=0.5 * ring,
                                                  n_grid=200)
    assert w[0] == pytest.approx(ring, rel=1e-4)


def test_thin_cylinder_simply_supported_frequencies():
    """AN: Eq. (15.99) with v = M = 0 at both ends vs the closed-form
    dispersion relation, 1e-10.

    For v = V sin(k x): k^2 / (E h / R^2 - rho h omega^2) =
    1 / (P_x + rho h r_y^2 omega^2 - D k^2) - 1 / D_Q (shear and rotary
    inertia included). XM: Eq. (15.101) (printed x^3 / (3 D_x) corrected to
    x^3 / (6 D_x)) = e^{A x} without foundation and inertia, 1e-12.
    """
    e_mod, h, mu, radius, rho, length = 70e9, 0.004, 0.33, 0.5, 2700.0, 1.2
    d_q, r_y2, p_x = 2.0e7, h * h / 12, 1.0e3
    cyl = ct.ThinCylinder(length, radius, e_mod, h, mu, rho * h, d_q=d_q,
                          r_y2=r_y2, p_x=p_x)
    w_tmm = Chain([cyl], [0, 2], [0, 2]).natural_frequencies(9000.0,
                                                             omega_min=10.0,
                                                             n_grid=600)
    d = cyl.d
    k_f = e_mod * h / radius**2
    exact = []
    for i in range(1, 12):
        k = i * math.pi / length

        def f(w2, k=k):
            return (k * k * (p_x + rho * h * r_y2 * w2 - d * k * k)
                    - (k_f - rho * h * w2)
                    * (1 - (p_x + rho * h * r_y2 * w2 - d * k * k) / d_q))

        roots = _roots(f, 1.0, 9000.0**2, n=4000)
        exact.extend(np.sqrt(roots))
    exact = np.sort([x for x in exact if 10.0 < x < 9000.0])
    np.testing.assert_allclose(w_tmm, exact, rtol=1e-10)
    bare = ct.ThinCylinder(0.7, radius, e_mod, h, mu, 0.0, d_q=d_q)
    bare.k_f = 0.0
    np.testing.assert_allclose(ct.thin_cylinder_massless_shear(0.7, d, d_q),
                               bare.transfer(0j).real, rtol=1e-12, atol=1e-15)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
