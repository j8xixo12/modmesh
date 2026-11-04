# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests of solvcon.multibody.tmm.flexible: rigid bodies with several ends,
closed-form beams, Riccati mode shapes (Ch. 10, 15) and the multi-flexible-body
TMM (Ch. 6).

Book: Rui, Wang, Zhang, Transfer Matrix Method for Multibody Systems (Wiley
2018). Tolerances: identities between matrix forms 1e-12 relative to the
largest entry (rounding of a few products); eigenfrequencies 1e-10 relative
unless stated; figure readings use the reading tolerance stored in the JSON
file.
"""

import functools
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest


from solvcon import multibody
from solvcon.multibody.tmm import flexible as fx
from solvcon.multibody.tmm import library as lib
from solvcon.multibody.tmm.linear import Chain

DATA = Path(__file__).parent / "data" / "multibody"
with open(DATA / "tmm_ch6_examples.json") as _fh:
    CH6 = json.load(_fh)

RNG = np.random.default_rng(7)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)


def _cross(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


# ----------------------------------------------------------------------
# Rigid body with N inputs and L outputs (Eq. 15.16-15.19)
# ----------------------------------------------------------------------
def _random_body(n_in, n_out):
    m = 2.3
    a = RNG.normal(size=(3, 3))
    j_c = a @ a.T + np.eye(3)
    c = RNG.normal(size=3)
    ins = [RNG.normal(size=3) for _ in range(n_in)]
    outs = [RNG.normal(size=3) for _ in range(n_out)]
    return fx.RigidBodyMultiEnd(m, j_c, c, ins, outs)


def _one_one_reference(body, s):
    """Eq. (15.14) of library.SpatialRigidBody for the body with origin I."""
    ref = body.inputs[0]
    j_i = fx._inertia_about(body.j_c, body.m, body.center - ref)
    return lib.SpatialRigidBody(body.m, j_i, body.outputs[0] - ref,
                                body.center - ref).transfer(s)


def test_one_input_one_output_forms_equal_eq_15_14():
    """XM Eq. (15.17)-(15.19) with N = L = 1 reduce to Eq. (15.14) (1e-12
    relative)."""
    body = _random_body(1, 1)
    s = 1j * 3.7
    u14 = _one_one_reference(body, s)
    assert _rel(body.inputs_matrix(s), u14) < 1e-12
    assert _rel(np.linalg.inv(body.outputs_matrix(s)), u14) < 1e-12
    u_o, u_i = body.general_matrices(s)
    assert _rel(np.linalg.solve(u_o, u_i), u14) < 1e-12


def test_eq_15_17_printed_sign_is_inconsistent():
    """Book erratum Eq. (15.17), p. 614: the printed U_I(3,1) = -m w^2 l_I1C^T
    fails Eq. (15.14).

    The printed sign gives a relative error of order 1 against Eq. (15.14),
    (15.18), (15.19); the corrected block reproduces them (previous test).
    """
    body = _random_body(1, 1)
    s = 1j * 3.7
    u_o, u_i = body.general_matrices(s)
    l_c = _cross(body.center - body.inputs[0])
    printed = u_i.copy()
    printed[6:9, 0:3] = -body.m * (-s * s) * l_c.T
    err = _rel(np.linalg.solve(u_o, printed), _one_one_reference(body, s))
    assert err > 1e-2
    assert CH6["errata"]["15.17"]["page"] == 614


def _star_tree():
    """Body on a ground hinge with two branches (hinge + end body), all
    12-state."""
    m_b, jc_b, c_b = (3.0, np.diag([0.04, 0.05, 0.06]),
                      np.array([0.1, 0.02, -0.03]))
    outs = [np.array([0.3, 0.05, 0.0]), np.array([-0.1, 0.25, 0.08])]
    body = fx.RigidBodyMultiEnd(m_b, jc_b, c_b, [np.zeros(3)], outs)
    k_g, kr_g = (2e4, 3e4, 2.5e4), (400.0, 500.0, 350.0)
    ends = [(1.2, np.diag([0.01, 0.012, 0.009]), np.array([0.05, 0.0, 0.01]),
             (1e4, 1.3e4, 0.8e4), (90.0, 120.0, 70.0)),
            (0.7, np.diag([0.004, 0.006, 0.005]), np.array([0.0, 0.04, -0.02]),
             (0.6e4, 0.9e4, 1.1e4), (50.0, 60.0, 80.0))]
    branches = []
    for m, jc, c, k, kr in ends:
        j_i = fx._inertia_about(jc, m, c)
        branches.append([lib.SpatialElasticHinge(k, kr),
                         lib.SpatialRigidBody(m, j_i, (0, 0, 0), c)])
    tree = fx.StarTree([lib.SpatialElasticHinge(k_g, kr_g)], body, branches)
    return tree, (m_b, jc_b, c_b, outs, k_g, kr_g, ends)


def _star_tree_global(params):
    """Independent M, K model: 6 dof per body ([X, Theta] of its reference
    point)."""
    m_b, jc_b, c_b, outs, k_g, kr_g, ends = params

    def mass(m, jc, c):
        t = np.eye(6)
        t[0:3, 3:6] = -_cross(c)
        blk = np.zeros((6, 6))
        blk[0:3, 0:3] = m * np.eye(3)
        blk[3:6, 3:6] = jc
        return t.T @ blk @ t

    n = 6 * (1 + len(ends))
    mg = np.zeros((n, n))
    kg = np.zeros((n, n))
    mg[0:6, 0:6] = mass(m_b, jc_b, c_b)
    b = np.zeros((6, n))
    b[:, 0:6] = np.eye(6)
    kg += b.T @ np.diag(list(k_g) + list(kr_g)) @ b
    for i, (m, jc, c, k, kr) in enumerate(ends):
        sl = slice(6 + 6 * i, 12 + 6 * i)
        mg[sl, sl] = mass(m, jc, c)
        b = np.zeros((6, n))
        b[:, sl] = np.eye(6)
        b[:, 0:6] -= fx.point_displacement_matrix(np.zeros(3), outs[i])
        kg += b.T @ np.diag(list(k) + list(kr)) @ b
    li = np.linalg.inv(np.linalg.cholesky(mg))
    return np.sqrt(np.sort(np.linalg.eigvalsh(li @ kg @ li.T)))


@pytest.mark.parametrize("form", ["outputs", "general", "inputs"])
def test_star_tree_three_forms_vs_global_eigenproblem(form):
    """XM Eq. (15.17), (15.18), (15.19) in a tree: 18 frequencies vs global M,
    K (1e-9).

    1e-9: root refinement of the scan (xtol) on frequencies up to 3e3 rad/s.
    """
    tree, params = _star_tree()
    exact = _star_tree_global(params)
    found = tree.natural_frequencies(exact[-1] * 1.05, form=form, n_grid=4000)
    assert len(found) == len(exact)
    np.testing.assert_allclose(found, exact, rtol=1e-9)


# ----------------------------------------------------------------------
# Closed-form massless beams (Eq. 15.24, 15.29)
# ----------------------------------------------------------------------
def _exact_series(a, x, terms=90):
    """e^{A x} by the power series of Eq. (10.70) in exact rational
    arithmetic."""
    n = len(a)
    ax = [[Fraction(v) * Fraction(x) for v in row] for row in a]
    term = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    total = [row[:] for row in term]
    for k in range(1, terms):
        term = [[sum(term[i][m] * ax[m][j] for m in range(n)) / k
                 for j in range(n)] for i in range(n)]
        total = [[total[i][j] + term[i][j] for j in range(n)] for i
                 in range(n)]
    return np.array([[float(v) for v in row] for row in total])


@pytest.mark.parametrize("axial", [300.0, -300.0, 4000.0])
def test_eq_15_24_closed_form_vs_first_order_ode(axial):
    """XM/AN Eq. (15.24) (corrected) vs the exact series and
    library.AxialLoadBeam.

    Closed form vs the rational power series of e^{A x}: 1e-14 (rounding).
    library.AxialLoadBeam (Pade expm, Eq. 10.69): 1e-10; its error grows with
    |A x| (measured 6e-11 for P = 4000 N), the closed form does not.
    """
    for inv_gas in (Fraction(0), Fraction(1, 50000)):
        gas = np.inf if inv_gas == 0 else float(1 / inv_gas)
        u = fx.AxialLoadBeamClosedForm(1.3, 200.0, axial,
                                       gas).transfer(0.0).real
        a = [[0, 1, 0, -inv_gas], [0, 0, Fraction(1, 200), 0],
             [0, -Fraction(axial), 0, 1], [0, 0, 0, 0]]
        assert _rel(u, _exact_series(a, Fraction(13, 10))) < 1e-14
        ref = lib.AxialLoadBeam(1.3, 200.0, axial, gas).transfer(0.0).real
        assert _rel(u, ref) < 1e-10


def test_eq_15_24_printed_compression_form_fails():
    """Book erratum Eq. (15.24), p. 617: lambda = sqrt(P/EI), d1 = -lambda^2
    for compression.

    The printed form violates Theta' = M/EI (U(2,3) slope), so it differs
    from the ODE solution by more than 1 relative.
    """
    x, ei, p = 1.3, 200.0, 300.0
    lam = math.sqrt(p / ei)
    d1 = -lam * lam
    sh, ch = math.sinh(lam * x), math.cosh(lam * x)
    printed = np.array([[1, sh / lam, -(1 - ch) / (d1 * ei),
                         -(lam * x - sh) / (d1 * lam * ei)],
                        [0, ch, sh / (lam * ei), -(1 - ch) / (d1 * ei)],
                        [0, d1 * ei * sh / lam, ch, sh / lam], [0, 0, 0, 1]])
    ref = lib.AxialLoadBeam(x, ei, p).transfer(0.0).real
    assert _rel(printed, ref) > 0.5


@pytest.mark.parametrize("c1,c2,length,p", [(1.0, 0.3, 1.0, 500.0),
                                            (1.2, 0.2, 2.0, 2000.0),
                                            (1.0, -0.2, 1.5, 800.0)])
def test_eq_15_29_closed_form_vs_magnus_ode(c1, c2, length, p):
    """XM/CV Eq. (15.29) (corrected closed form) vs the 4th-order Magnus
    solution.

    Error ratio for 100 -> 200 steps is 16 (order 4, accepted 13-19); at 400
    steps the truncation error is below 5e-11 relative (measured up to
    1.1e-11).
    """
    ei = 150.0
    exact = fx.TaperedAxialBeam(length, ei, p, c1, c2).transfer(0.0).real
    errs = [_rel(fx.TaperedAxialBeamODE(length, ei, p, c1, c2,
                                        n).transfer(0.0).real, exact)
            for n in (100, 200, 400)]
    assert 13.0 < errs[0] / errs[1] < 19.0
    assert errs[2] < 5e-11


def test_eq_15_29_closed_form_limit_uniform_beam():
    """AN: for c2 -> 0 the tapered beam tends to the uniform beam with axial
    load (Eq. 15.24).

    c2 = 1e-4 changes EI by 2e-4 over the length; the difference is of that
    order.
    """
    u = fx.TaperedAxialBeam(1.0, 200.0, 300.0, 1.0, 1e-4).transfer(0.0).real
    ref = fx.axial_load_beam_matrix(1.0, 200.0, 300.0)
    assert _rel(u, ref) < 5e-4


def test_eq_15_29_printed_form_errors():
    """Book erratum Eq. (15.29), p. 619: printed form singular at c1 = 1 and
    wrong entries."""
    def printed(length, eii, p, c1, c2):
        beta = math.sqrt(p / (c2**2 * eii) - 0.25)
        d1 = c1 - c2 * length
        d2 = math.sqrt(d1)
        d3, d4 = math.sin(beta * math.log(d1)), math.cos(beta * math.log(d1))
        d5, d6 = math.sin(beta * math.log(c1)), math.cos(beta * math.log(c1))
        d7 = math.tan(beta * math.log(c1))
        d8, d9 = d4 * d5 - d3 * d6, d3 * d5 + d4 * d6
        d10 = 1 / (2 * beta) + 1 / d7
        d11 = (1 - d2 * d3 / (math.sqrt(c1) * d5)) / p
        d12, d13 = d3 / d5 - d9, d4 / d6 - d9
        sc = math.sqrt(c1)
        return np.array([
            [1, -sc * d2 * d8 / (beta * c2), d2 * d8 * d10 / (sc * p) - d11,
             sc * d2 * d8 / (beta * c2 * p) + length / p],
            [0, math.sqrt(c1 / d2) * (d7 - 0.5 * d8 / beta),
             c2 / (sc * d1 * p) * (d12 / 2 + d13 / d7 + d8 * d10 / 2),
             (1 - sc * d9 / d2 + sc * d8 / (2 * beta * d2)) / p],
            [0, -p * math.sqrt(c1 * d1) * d8 / (beta * c2),
             math.sqrt(d1 / c1) * (d3 / d5 + d8 * d10),
             math.sqrt(c1 * d1) * d8 / (beta * c2)],
            [0, 0, 0, 1]])

    with pytest.raises(ZeroDivisionError):
        printed(1.0, 200.0, 500.0, 1.0, 0.3)
    book = printed(2.0, 100.0, 2000.0, 1.2, 0.2)
    exact = fx.TaperedAxialBeam(2.0, 100.0, 2000.0, 1.2,
                                0.2).transfer(0.0).real
    # rows 3 and column 4 agree; u_12, u_13 have flipped signs; u_22, u_23
    # differ
    np.testing.assert_allclose(book[2, 1:], exact[2, 1:], rtol=1e-10)
    np.testing.assert_allclose(book[1:3, 3], exact[1:3, 3], rtol=1e-10)
    np.testing.assert_allclose(book[0, 1:3], -exact[0, 1:3], rtol=1e-10)
    assert abs(book[1, 1] - exact[1, 1]) > 0.1


# ----------------------------------------------------------------------
# Riccati mode shapes (Eq. 10.130, 10.135)
# ----------------------------------------------------------------------
def test_riccati_mode_equals_plain_mode_and_global_vector():
    """XM: Riccati back substitution = plain propagation (Eq. 2.46) where both
    are accurate."""
    els = []
    for _ in range(4):
        els += [lib.PlanarElasticHinge(1e3, 5e4, 4e3),
                lib.PlanarRigidBody(1.0, 5.0 / 12.0 * 0.01, (0.1, 0.0),
                                    (0.05, 0.0))]
    ch = Chain(els, [0, 1, 2], [3, 4, 5])
    w = ch.natural_frequencies(300.0)
    for wk, md in zip(w, ch.modes(w)):
        ric = fx.riccati_mode(ch, wk)
        np.testing.assert_allclose(ric.states, md.states, rtol=0, atol=1e-9)


@pytest.mark.parametrize("k", [3, 8, 16, 19])
def test_riccati_mode_keeps_precision_on_long_cantilever(k):
    """AN/IV Sec. 10.9-10.10: 20-element cantilever, mode k, vs the analytic
    mode shape.

    Riccati: displacement error below 1e-9 of the maximum (measured 2e-10 at
    k = 8, 1e-13 otherwise) and the free-end M, Q below 1e-11 of their
    maximum. Plain propagation of the boundary vector (Eq. 2.46) amplifies
    the rounding by cosh(lambda L): for k >= 16 its free-end M, Q exceed the
    largest interior value.
    """
    n_el = 20
    ch = Chain([lib.EulerBernoulliBeam(1.0 / n_el, 1.0,
                                       1.0) for _ in range(n_el)], [0, 1],
               [2, 3])
    bl = fx.cantilever_beam_roots(k)[-1]
    md = fx.riccati_mode(ch, bl**2)
    shape = fx.cantilever_bending_modes(1.0,
                                        k)(np.linspace(0, 1, n_el + 1))[0][-1]
    y = md.states[:, 0] / md.states[-1, 0] * shape[-1]
    assert np.max(np.abs(y - shape)) < 1e-9 * np.max(np.abs(shape))
    inner = np.max(np.abs(md.states[:, 2:]))
    assert np.max(np.abs(md.states[-1][2:])) < 1e-11 * inner
    if k >= 16:
        plain = ch.propagate(md.states[0], 1j * bl**2, 0.0).real
        assert np.max(np.abs(plain[-1][2:])) > inner


# ----------------------------------------------------------------------
# Haug engine cross-check: lumped masses on massless beams (P5.6 / P7)
# ----------------------------------------------------------------------
def _haug_module():
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    return multibody


def test_haug_bushing_beam_chain_vs_tmm():
    """XM: Haug modal analysis (bushings = massless EB beams) vs TMM
    SpatialBeam(m = 0) chain.

    Two rigid bodies on two massless beams, 12 modes from 72 to 1.7e4
    rad/s. Agreement 1e-8 relative (Haug eigen-solver precision on the
    stiff axial modes).
    """
    mbs = _haug_module()
    from solvcon.multibody.tools.elements import beam_stiffness, bushing_matrix

    e_mod, g_mod, ln, d = 2.1e11, 8.1e10, 0.5, 0.02
    area, i_a = math.pi * d**2 / 4, math.pi * d**4 / 64
    ms = [1.5, 0.8]
    js = [np.diag([2e-3, 3e-3, 4e-3]), np.diag([1e-3, 1.5e-3, 2.5e-3])]
    z = [0.0, 0.0, 0.0]
    sys_h = mbs.DynamicsSystem(1e-3)
    prev = mbs.create_ground(0)
    sys_h.Add(prev)
    k = beam_stiffness(e_mod, g_mod, area, i_a, 1.7 * i_a, 2 * i_a, ln)
    for i, (m, ji) in enumerate(zip(ms, js)):
        b = mbs.create_mobilized_body_quat(i + 1, [ln * (i + 1), 0, 0], z, z,
                                           [1, 0, 0, 0], z, z, m,
                                           list(np.diag(ji)), z, z)
        sys_h.Add(b)
        sys_h.Add(mbs.create_bushing(prev, b, z, z, k, bushing_matrix(z, z)))
        prev = b
    sys_h.init()
    w_haug = (np.sort(np.array(mbs.modal_analysis(sys_h).frequencies))
              * 2 * math.pi)
    els = []
    for m, ji in zip(ms, js):
        els += [lib.SpatialBeam(ln, e_mod * area, e_mod * i_a,
                                1.7 * e_mod * i_a, g_mod * 2 * i_a, 0.0, 0.0),
                lib.SpatialRigidBody(m, ji, (0, 0, 0), (0, 0, 0))]
    w_tmm = Chain(els, list(range(6)),
                  list(range(6, 12))).natural_frequencies(2e4)
    assert len(w_tmm) == len(w_haug) == 12
    np.testing.assert_allclose(w_tmm, w_haug, rtol=1e-8)


# ----------------------------------------------------------------------
# Ch. 6: planar and spatial beams with large motion
# ----------------------------------------------------------------------
_EX = CH6["examples"]["6.6.1"]["data"]
_RA = _EX["rho"] * _EX["A"]
_EA = _EX["E"] * _EX["A"]
_EI = _EX["E"] * _EX["I"]
_L = _EX["l"]


def _hub_beam(n_axial=1, n_bend=3, coupling=True, moment=0.0, tip=0.0,
              spatial=None):
    """Hub-beam chain of Sec. 6.6 (planar, or spatial with the given pin
    axis)."""
    if spatial is None:
        els = [fx.PlanarRigidElement(10.0, 5.0, (0, 0), (1.0, 0),
                                     moment=moment),
               fx.PlanarFlexibleBeam(_L, _EA, _EI, _RA, n_axial, n_bend,
                                     coupling)]
        if tip:
            els.append(fx.PlanarRigidElement(tip, 0.0))
        return fx.FlexibleChain(els), els[1]
    axis = np.asarray(spatial, float)
    if callable(moment):
        def mom(t):
            return axis * moment(t)[0]
    else:
        mom = axis * moment
    n_y, n_z = (n_bend, 0) if abs(axis[2]) > 0.5 else (0, n_bend)
    els = [fx.SpatialRigidElement(10.0, 5.0 * np.eye(3), (0, 0, 0),
                                  (1.0, 0, 0), moment=mom),
           fx.SpatialFlexibleBeam(_L, _EA, _EI, _EI, _RA, 0.0, n_axial, n_y,
                                  n_z, coupling)]
    if tip:
        els.append(fx.SpatialRigidElement(tip, np.zeros((3, 3))))
    return fx.FlexibleChain(els, axis=axis), els[1]


def _lagrange_accel(beam, y, moment, coupling):
    """Independent equations of motion of the planar hub-beam (virtual power,
    Kane form).

    r_P = R(theta) (a e1 + xi e1 + Phi q); M = J_hub + sum dm J^T J,
    f = [M_ext, 0] - sum dm J^T gamma - [0, K q + f_c].
    """
    md = beam.md
    n = beam.n_modes
    th, thd, q, qd = y[0], y[1], y[2:2 + n], y[2 + n:]
    r = fx._rot2(th)
    rp = r @ fx._D2
    mm = np.zeros((n + 1, n + 1))
    f = np.zeros(n + 1)
    mm[0, 0] = 5.0
    f[0] = moment
    for x, w, tab in zip(md.xg, md.wg, md.tab):
        phi = tab[0][:2]
        p = np.array([1.0 + x, 0.0]) + phi @ q
        jac = np.hstack([(rp @ p).reshape(2, 1), r @ phi])
        gam = -r @ p * thd**2 + 2 * rp @ (phi @ qd) * thd
        dm = w * md.rho_a
        mm += dm * jac.T @ jac
        f -= dm * jac.T @ gam
    f[1:] -= md.k @ q
    if coupling:
        f[1:] -= md.axial_coupling(q)
    return np.linalg.solve(mm, f)


def _random_state(n):
    scale_q = np.array([1e-4] + [0.2, 0.05, 0.01, 0.005][:n - 1])
    return np.concatenate([[0.7, 2.3], RNG.normal(size=n) * scale_q,
                           RNG.normal(size=n) * 10 * scale_q])


@pytest.mark.parametrize("coupling", [False, True])
def test_planar_tmm_accelerations_vs_lagrange(coupling):
    """XM Eq. (6.33)-(6.60): TMM accelerations = independent virtual-power
    equations (1e-10)."""
    ch, beam = _hub_beam(coupling=coupling, moment=3.0)
    for _ in range(3):
        y = _random_state(beam.n_modes)
        res = ch.solve(y, 0.0)
        got = np.concatenate([[res["theta_dd"]], res["q_dd"]])
        ref = _lagrange_accel(beam, y, 3.0, coupling)
        assert _rel(got, ref) < 1e-10


@pytest.mark.parametrize("axis", [(0, 0, 1), (0, -1, 0)])
def test_spatial_beam_equals_planar_book_coefficients(axis):
    """XM Eq. (6.87)-(6.113) by quadrature vs the closed forms of Eq.
    (6.35)-(6.60) in the plane.

    Rotation about z with bending in y, and about -y with bending in z (the
    book example). Accelerations and the input boundary state agree to 1e-10.
    """
    ch_p, beam = _hub_beam(moment=3.0)
    ch_s, _ = _hub_beam(moment=3.0, spatial=axis)
    for _ in range(3):
        y = _random_state(beam.n_modes)
        rp, rs = ch_p.solve(y, 0.0), ch_s.solve(y, 0.0)
        assert _rel(np.concatenate([[rs["theta_dd"]], rs["q_dd"]]),
                    np.concatenate([[rp["theta_dd"]], rp["q_dd"]])) < 1e-10


def _spatial_reference(beam, frame, q, qd, z_in):
    """Independent global Newton-Euler of the spatial beam for a known input
    state.

    With clamped modes, R_I = [r_o'', alpha]. The modal equations give q'',
    then q_O and m_O follow from the global force and moment balances of
    all mass points (absolute accelerations), and R_O from the kinematics.
    """
    md = beam.md
    a, om = frame
    rdd, alpha, m_i, q_i = z_in[0:3], z_in[3:6], z_in[6:9], z_in[9:12]
    wg = _cross(om)
    e1g = a[:, 0]
    pts = []
    for x, w, tab in zip(md.xg, md.wg, md.tab):
        phi = tab[0]
        sg = a @ (np.array([x, 0, 0]) + phi @ q)
        known = (rdd + np.cross(alpha, sg) + wg @ wg @ sg + 2 * wg @ a @ phi
                 @ qd)
        pts.append((w * md.rho_a, a @ phi, sg, known))
    n = beam.n_modes
    mq = sum(dm * jp.T @ jp for dm, jp, _, _ in pts)
    rhs = (-sum(dm * jp.T @ kn for dm, jp, _, kn in pts) - md.k @ q
           - md.axial_coupling(q))
    phi0, sl0 = md.tab0[0], md.tab0[1]
    phil, sll = md.tabl[0], md.tabl[1]
    rhs += (a @ phi0).T @ q_i - (a @ sl0).T @ m_i
    # q_O and m_O are unknown and enter the modal equations: solve jointly
    big = np.zeros((n + 6, n + 6))
    vec = np.zeros(n + 6)
    big[:n, :n] = mq
    big[:n, n:n + 3] = -(a @ sll).T
    big[:n, n + 3:] = (a @ phil).T
    vec[:n] = rhs
    # force balance: sum dm r_P'' = q_I - q_O
    big[n:n + 3, :n] = sum(dm * jp for dm, jp, _, _ in pts)
    big[n:n + 3, n + 3:] = np.eye(3)
    vec[n:n + 3] = q_i - sum(dm * kn for dm, _, _, kn in pts)
    # moment balance about o2: sum s x dm r_P'' + J_xi terms = -m_I + m_O + s_I
    # x q_I - s_O x q_O
    s_o = a @ (np.array([beam.length, 0, 0]) + phil @ q)
    s_i = a @ (phi0 @ q)
    jx = beam.j_xi
    row = np.zeros((3, n + 6))
    rvec = np.zeros(3)
    for dm, jp, sg, kn in pts:
        row[:, :n] += dm * _cross(sg) @ jp
        rvec -= dm * np.cross(sg, kn)
    rvec -= jx * (np.outer(e1g, e1g) @ alpha + np.cross(om, e1g) * (e1g @ om))
    row[:, n:n + 3] = -np.eye(3)
    row[:, n + 3:] = _cross(s_o)
    rvec += -m_i + np.cross(s_i, q_i)
    big[n + 3:] = row
    vec[n + 3:] = rvec
    sol = np.linalg.solve(big, vec)
    qdd, m_o, q_o = sol[:n], sol[n:n + 3], sol[n + 3:]
    r_o = (rdd + np.cross(alpha, s_o) + wg @ wg @ s_o + 2 * wg @ a @ phil @ qd
           + a @ phil @ qdd)
    al_o = alpha + a @ sll @ qdd + wg @ a @ sll @ qd
    return qdd, np.concatenate([r_o, al_o, m_o, q_o, [1.0]])


def test_spatial_beam_3d_state_vs_global_newton_euler():
    """XM Eq. (6.104)-(6.113) for a general 3D state (all mode types, J_xi) vs
    Newton-Euler.

    1e-9 relative: the reference solves a different (n + 6) linear system.
    """
    beam = fx.SpatialFlexibleBeam(_L, _EA, _EI, 1.6 * _EI, _RA, 3e-3, 2, 3, 2,
                                  True)
    n = beam.n_modes
    a = fx.rotation_about(RNG.normal(size=3), 0.9)
    om = RNG.normal(size=3)
    q = RNG.normal(size=n) * np.array([1e-4, 1e-5, 0.1, 0.03, 0.01, 0.08,
                                       0.02])
    qd = RNG.normal(size=n) * np.array([1e-3, 1e-4, 0.5, 0.2, 0.1, 0.4, 0.1])
    z_in = np.concatenate([RNG.normal(size=12), [1.0]])
    u, ubar = beam.matrices((a, om), q, qd, 0.0)
    qdd_ref, z_out_ref = _spatial_reference(beam, (a, om), q, qd, z_in)
    assert _rel(ubar @ z_in, qdd_ref) < 1e-9
    assert _rel(u @ z_in, z_out_ref) < 1e-9


def _linear_frequencies(n_bend):
    """Small-vibration frequencies of the FFR hub-beam from the TMM
    accelerations.

    Central differences of the right-hand side at rest (step 1e-7) remove
    the quadratic velocity terms; the axial mode decouples.
    """
    ch, beam = _hub_beam(n_bend=n_bend, coupling=False)
    n = 2 + 2 * beam.n_modes
    jac = np.zeros((n, n))
    h = 1e-7
    for k in range(n):
        dy = np.zeros(n)
        dy[k] = h
        jac[:, k] = (ch.rhs(0.0, dy) - ch.rhs(0.0, -dy)) / (2 * h)
    lam = np.linalg.eigvals(jac)
    w = np.sort(np.abs(lam.imag[lam.imag > 1e-3]))
    return w


def test_hub_beam_small_vibration_converges_to_exact_frequencies():
    """AN/CV: FFR TMM frequencies of the pinned hub-beam -> exact
    Euler-Bernoulli frequencies.

    Exact: linear TMM chain TransverseRigidBody (Eq. 15.13) +
    EulerBernoulliBeam (Eq. 15.20), pinned input (Y = M = 0), free output.
    Cantilever modes are Ritz functions, so the FFR frequencies are upper
    bounds that decrease with the mode count. First two bending frequencies:
    relative error below 2e-4 with 6 modes (measured 6e-5 and 1.4e-4).
    """
    exact_chain = Chain([lib.TransverseRigidBody(10.0, 5.0, 1.0, 0.0),
                         lib.EulerBernoulliBeam(_L, _EI, _RA)], [0, 2], [2, 3])
    exact = exact_chain.natural_frequencies(400.0, omega_min=1e-3)
    errs = []
    for nb in (2, 4, 6):
        w = _linear_frequencies(nb)
        bend = np.sort(w[w < 1000.0])[:2]
        errs.append((bend - exact[:2]) / exact[:2])
    errs = np.array(errs)
    assert np.all(errs > -1e-9)
    assert np.all(np.diff(errs, axis=0) < 0)
    assert np.all(errs[-1] < 2e-4)


def _angular_momentum(ch, beam, y):
    """H = J_hub theta_dot + sum dm (r_P x v_P) about the pin (planar)."""
    md = beam.md
    n = beam.n_modes
    th, thd, q, qd = y[0], y[1], y[2:2 + n], y[2 + n:]
    r = fx._rot2(th)
    h = 5.0 * thd
    for x, w, tab in zip(md.xg, md.wg, md.tab):
        phi = tab[0][:2]
        p = r @ (np.array([1.0 + x, 0.0]) + phi @ q)
        v = fx._D2 @ p * thd + r @ (phi @ qd)
        h += w * md.rho_a * (p[0] * v[1] - p[1] * v[0])
    return h


def test_angular_momentum_theorem_and_energy_rk4_order():
    """IV Sec. 6.6: H(t) = int M dt (exact law) and energy conservation without
    the f term.

    Angular momentum with M = 20 N m: |H(0.4) - 8| < 1e-8 (RK4 error).
    Energy (axial coupling off, no load, initial spin and deformation):
    relative drift below 1e-11 at dt = 1e-4 (measured 1.9e-12) and error
    ratio 32 +- 6 for dt 2e-4 -> 1e-4: the RK4 amplification factor of an
    oscillator is |R(i w h)|^2 = 1 - (w h)^6 / 72, so the energy error over a
    fixed time is O(h^5) (dt must resolve the axial mode at 1.6e3 rad/s).
    """
    ch, beam = _hub_beam(moment=20.0)
    _, ys = ch.simulate(0.4, 1e-3)
    assert abs(_angular_momentum(ch, beam, ys[-1]) - 8.0) < 1e-8

    ch, beam = _hub_beam(coupling=False)
    n = beam.n_modes
    y0 = np.zeros(2 + 2 * n)
    y0[1] = 2.0
    y0[3] = 0.2
    y0[2] = 1e-5

    def energy(y):
        md = beam.md
        th, thd, q, qd = y[0], y[1], y[2:2 + n], y[2 + n:]
        r = fx._rot2(th)
        e = 0.5 * 5.0 * thd**2 + 0.5 * q @ md.k @ q
        for x, w, tab in zip(md.xg, md.wg, md.tab):
            phi = tab[0][:2]
            p = r @ (np.array([1.0 + x, 0.0]) + phi @ q)
            v = fx._D2 @ p * thd + r @ (phi @ qd)
            e += 0.5 * w * md.rho_a * v @ v
        return e

    e0 = energy(y0)
    drift = []
    for dt in (2e-4, 1e-4):
        _, ys = ch.simulate(0.2, dt, y0=y0)
        drift.append(abs(energy(ys[-1]) - e0) / e0)
    assert drift[1] < 1e-11
    assert 26.0 < drift[0] / drift[1] < 38.0


@functools.lru_cache(maxsize=None)
def _run_book_example(tip, t_step=10.0):
    """Sec. 6.6 run: M = 20 N m * 10 / t_step for t < t_step (same impulse),
    1e-3 s RK4."""
    data = CH6["examples"]["6.6.1"]["data"]
    assert data["hub_J"] == 5.0

    def moment(t):
        return np.array([200.0 / t_step if t < t_step else 0.0])

    ch, beam = _hub_beam(moment=moment, tip=tip)
    ts, ys = ch.simulate(t_step + 5.0, 1e-3)
    q = ys[:, 2:2 + beam.n_modes]
    u = q[:, :1] @ beam.ph_end[1][:1]
    v = q[:, 1:] @ beam.vp_end[1]
    late = ts > t_step + 0.5
    early = ts < t_step
    return {"omega_mean_after_10s": ys[late, 1].mean(),
            "u_tip_mean_after_10s": u[late].mean(),
            "v_tip_min_before_10s": v[early].min(),
            "v_tip_max_before_10s": v[early].max(),
            "v_tip_amplitude_after_10s": 0.5 * (v[late].max() - v[late].min())}


def _check_reading(example, tip, key):
    ref = CH6["examples"][example]["figure_readings"][key]
    got = _run_book_example(tip)[key]
    assert abs(got - ref["value"]) <= ref["tol"], (key, got, ref["value"])


@pytest.mark.slow
@pytest.mark.parametrize("key", ["omega_mean_after_10s",
                                 "u_tip_mean_after_10s",
                                 "v_tip_min_before_10s",
                                 "v_tip_max_before_10s"])
def test_example_6_6_1_hub_beam_figures(key):
    """BK Example 6.6.1, Fig. 6.6-6.8 (p. 213-214): readings within the reading
    tolerance."""
    _check_reading("6.6.1", 0.0, key)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=(
    "Fig. 6.8 late tip amplitude: TMM 0.182 m vs 0.16 +- 0.01 read. Suspected "
    "cause: the residual vibration after the moment step depends on the "
    "vibration phase at t = 10 s; moving the step by -0.1/-0.05/+0.05 s (same "
    "impulse) gives 0.278/0.228/0.177 m "
    "(test_late_amplitude_is_phase_sensitive), so a small frequency "
    "difference of the book model shifts the amplitude; the early-phase "
    "readings agree."))
def test_example_6_6_1_late_tip_amplitude():
    """BK Example 6.6.1, Fig. 6.8 after t = 10 s (not reproduced, see
    reason)."""
    _check_reading("6.6.1", 0.0, "v_tip_amplitude_after_10s")


@pytest.mark.slow
def test_late_amplitude_is_phase_sensitive():
    """IV evidence for the xfail reasons: the late amplitude changes by more
    than 20 % when the step time moves by 0.05 s at equal impulse, the mean
    spin rate does not (1e-3 relative, angular momentum theorem)."""
    a = _run_book_example(0.0, 10.0)
    b = _run_book_example(0.0, 9.95)
    assert abs(a["v_tip_amplitude_after_10s"] / b["v_tip_amplitude_after_10s"]
               - 1.0) > 0.2
    assert abs(a["omega_mean_after_10s"] / b["omega_mean_after_10s"]
               - 1.0) < 1e-3


@pytest.mark.slow
def test_example_6_6_2_hub_beam_tip_body_mean_spin():
    """BK Example 6.6.2, Fig. 6.11 (p. 215): omega = 200 / J_total = 7.50
    rad/s."""
    _check_reading("6.6.2", 0.2, "omega_mean_after_10s")


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=(
    "Fig. 6.12 early tip minimum: TMM -0.405 m vs -0.38 +- 0.02 read (6 %). "
    "Suspected cause: the terminal rigid body of Fig. 6.10 has a finite size "
    "but the book gives only its mass (0.2 kg) and J = 0; it is modeled as a "
    "point mass at the beam end, which sets the first frequency and the step "
    "response amplitude."))
def test_example_6_6_2_early_tip_minimum():
    """BK Example 6.6.2, Fig. 6.12 before t = 10 s (not reproduced, see
    reason)."""
    _check_reading("6.6.2", 0.2, "v_tip_min_before_10s")


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=(
    "Fig. 6.12 late tip amplitude: TMM 0.21 m vs 0.155 +- 0.01 read. "
    "Suspected cause: phase sensitivity of the residual vibration (0.168 m "
    "for a step at 10.05 s, 0.281 m at 9.95 s) and the unknown terminal body "
    "geometry (see the early-minimum xfail)."))
def test_example_6_6_2_late_tip_amplitude():
    """BK Example 6.6.2, Fig. 6.12 after t = 10 s (not reproduced, see
    reason)."""
    _check_reading("6.6.2", 0.2, "v_tip_amplitude_after_10s")

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
