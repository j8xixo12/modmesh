# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Discrete time TMM for multi-rigid-flexible-body systems (book Ch. 8, Sec.
15.22-15.29).

Planar beams (Secs. 8.2-8.7, 8.14; library Secs. 15.22, 15.24, 15.26,
15.28): :class:`PlanarBeam`, :class:`PlanarFlexibleChain`. Spatial beams
(Secs. 8.8-8.11, 8.15; library Secs. 15.23, 15.25, 15.27):
:class:`SpatialBeam`, :class:`SpatialFlexibleChain` with ball hinges and a
clamped base. Mode bases (:class:`ModeBasis`): the sine modes of Eq. (8.69)
(the book's choice) and the clamped-free eigenfunctions, which are modes
"that satisfy the boundary condition" (Sec. 8.4) of a clamped beam.

Beam model (Secs. 8.2-8.4). The floating frame of a beam has its origin at
the input end and its x2 axis along the chord to the output end. The
transverse deformation is ``u(x2, t) = sum_k sin(k pi x2 / l) q^k(t)``
(Eq. 8.69); axial deformation and rotary inertia are neglected
(Euler-Bernoulli beam with inextensible chord). The state vector of a beam
end is (Eq. 8.88)::

    z = [x, y, theta, m, q_x, q_y, q^1, ..., q^n, 1]

with the position of the end, the angle theta of the floating frame, the
internal moment and force (sign convention of Ch. 7) and the modal
coordinates, which are equal at both ends.

Equations of motion. With ``r(x2) = r_I + A(theta) [x2, u]^T`` the
equations of the beam are, for the coordinates ``p = [x_I, y_I, theta,
q^1..q^n]``::

    M(p) pdd + h(p, pd) + K p = Q_g + J(0)^T q_I - s(0) m_I - J(l)^T q_O + s(l)
    m_O

where ``M = int mu J^T J dx2``, ``h = int mu J^T Jd pd dx2``,
``K = diag(0, 0, 0, EI (k pi / l)^4 l / 2)``, ``s(x2) = [0, 0, 1,
phi'(x2)]`` is the virtual rotation of the beam axis and ``Q_g`` the
gravity. The first two rows give q_O (Eqs. 8.71-8.76, 8.85-8.87), the
third row gives m_O (Eqs. 8.77-8.84) and the modal rows are the vibration
equations of Eq. (8.93). The book expands the integrals for n = 3 and
drops some coupling terms; this module evaluates the same integrals by
Gauss quadrature, for any n, and keeps all terms of the kinematics, so the
model conserves energy.

Time discretization (Sec. 8.13). Each coordinate uses ``pdd = A p + B``
and ``pd = C p + D`` of a step-by-step method of
:mod:`solvcon.multibody.tmm.dttmm`. The coefficient functions M, h, J(l) and
the rotation of the end positions use the predicted state of Eq. (7.62); the
local error is O(dT^3). Rigid bodies use Eq. (7.101).

Hinges (Secs. 8.5-8.7). A smooth hinge gives ``m = 0`` at the hinge; for
an outboard beam the angle and the modal coordinates are removed with the
moment condition at its output end and its vibration equations
(Eq. 8.94-8.98). A fixed hinge welds the tangent of the outboard body to
the tangent of the inboard body, ``theta_out + u'_out(0) = theta_in +
u'_in(l)``, and transmits the moment. A rotary spring-damper hinge
(Sec. 8.6) adds ``m = k Delta + c dDelta/dt`` with the tangent angle jump
Delta. :class:`PlanarFlexibleChain` assembles the overall transfer equation
of a chain (Eq. 8.236, 8.244) with the affine expressions of
:mod:`solvcon.multibody.tmm.topology`: every angle or modal coordinate that a
hinge cannot remove locally is an unknown of the overall equation, and the
hinge or vibration equation is an equation of it.
"""

import numpy as np

from . import dttmm, rigid
from .topology import Affine, AffineSystem


def rot_exp(phi):
    """Return exp(phi~), the rotation of the rotation vector phi (Rodrigues).

    :rtype: numpy.ndarray
    """
    phi = np.asarray(phi, dtype=float)
    t = np.linalg.norm(phi)
    s = rigid.skew(phi)
    if t < 1e-8:
        return np.eye(3) + s + 0.5 * s @ s
    return np.eye(3) + np.sin(t) / t * s + (1.0 - np.cos(t)) / t ** 2 * s @ s


def rot_log(r):
    """Return the rotation vector of a rotation matrix (angle below pi).

    :rtype: numpy.ndarray
    """
    c = np.clip(0.5 * (np.trace(r) - 1.0), -1.0, 1.0)
    t = np.arccos(c)
    v = 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0],
                        r[1, 0] - r[0, 1]])
    if t < 1e-6:
        return v * (1.0 + t * t / 6.0)
    return v * t / np.sin(t)


def rot_jr(phi):
    """Return the right Jacobian of exp: d exp(phi) = exp(phi) (J_r dphi)~.

    :rtype: numpy.ndarray
    """
    phi = np.asarray(phi, dtype=float)
    t = np.linalg.norm(phi)
    s = rigid.skew(phi)
    if t < 1e-6:
        return np.eye(3) - 0.5 * s + s @ s / 6.0
    return (np.eye(3) - (1.0 - np.cos(t)) / t ** 2 * s
            + (t - np.sin(t)) / t ** 3 * s @ s)


def _cantilever_roots(n):
    """Return beta_k l of the clamped-free beam, cos(b) cosh(b) = -1.

    :rtype: numpy.ndarray
    """
    out = []
    for k in range(1, n + 1):
        b = (k - 0.5) * np.pi
        for _ in range(60):
            # f = cos b + 1/cosh b (scaled form of cos cosh + 1)
            f = np.cos(b) + 1.0 / np.cosh(b)
            df = -np.sin(b) - np.tanh(b) / np.cosh(b)
            step = f / df
            b -= step
            if abs(step) < 1e-15:
                break
        out.append(b)
    return np.array(out)


class ModeBasis:
    """Assumed transverse modes phi_k(x2) of a beam with phi_k(0) = 0.

    ``"sine"``: sin(k pi x2 / l), the modes of Eq. (8.69); the floating
    frame is the chord frame (u(0) = u(l) = 0) and phi_k'(0) is not zero.
    ``"cantilever"``: clamped-free beam eigenfunctions (u(0) = u'(0) = 0),
    the modes "which satisfy the boundary condition" of Sec. 8.4 for a beam
    clamped at its input end; the floating frame is the root tangent frame.

    :ivar kind: basis name
    :vartype kind: str
    :ivar n: number of modes
    :vartype n: int
    """

    def __init__(self, length, n, kind="sine"):
        """Initialize the basis.

        :param length: beam length
        :type length: float
        :param n: number of modes
        :type n: int
        :param kind: 'sine' or 'cantilever'
        :type kind: str
        :raises ValueError: for an unknown kind
        """
        if kind not in ("sine", "cantilever", "polynomial"):
            raise ValueError("unknown mode basis: %s" % kind)
        self.length = float(length)
        self.n = int(n)
        self.kind = kind
        if kind == "sine":
            self.beta = np.arange(1, n + 1) * np.pi / self.length
        elif kind == "polynomial":
            # phi_k'' = P_{k-1}(2 x / l - 1): u(0) = u'(0) = 0, free curvature
            # at both ends (complete H2 basis for beams welded to other
            # bodies).
            from numpy.polynomial import legendre as leg
            self._coef = []
            for k in range(n):
                c = np.zeros(k + 1)
                c[k] = 1.0
                c2 = leg.legint(c, m=2, lbnd=-1.0, k=[0.0, 0.0])
                self._coef.append((c, leg.legint(c, m=1, lbnd=-1.0, k=[0.0]),
                                   c2))
            self.beta = np.ones(n)
        else:
            bl = _cantilever_roots(n)
            self.beta = bl / self.length
            # sigma = (cosh + cos) / (sinh + sin); 1 - sigma and 1 + sigma in
            # a form without cancellation.
            e = np.exp(-bl)
            den = 1.0 + 2.0 * np.sin(bl) * e - e * e
            self.one_minus = 2.0 * (np.sin(bl) - np.cos(bl) - e) * e / den
            self.sigma = 1.0 - self.one_minus
            self.bl = bl

    def values(self, x, der=0):
        """Return phi^(der)(x) as an array of shape (len(x), n).

        :param x: positions in [0, l]
        :type x: array_like
        :param der: derivative order 0, 1 or 2
        :type der: int
        :rtype: numpy.ndarray
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if self.kind == "polynomial":
            from numpy.polynomial import legendre as leg
            xi = 2.0 * x / self.length - 1.0
            h = 0.5 * self.length
            idx = {0: 2, 1: 1, 2: 0}[der]
            return np.column_stack([leg.legval(xi, c[idx]) * h ** (2 - der)
                                    for c in self._coef])
        b = self.beta
        bx = np.outer(x, b)
        if self.kind == "sine":
            base = [np.sin, np.cos, lambda z: -np.sin(z)][der](bx)
            return base * b ** der
        sig = self.sigma
        # phi = cosh - cos - sigma (sinh - sin)
        # = 0.5 (1 - sigma) e^{bx} + 0.5 (1 + sigma) e^{-bx} - cos + sigma sin
        # (1 - sigma) e^{bx} = (one_minus e^{bl}) e^{b(x - l)}; the first
        # factor is of order 1, so no large exponential is formed.
        grow = 0.5 * (self.one_minus * np.exp(self.bl)) * np.exp(bx - self.bl)
        decay = 0.5 * (1.0 + sig) * np.exp(-bx)
        if der == 0:
            out = grow + decay - np.cos(bx) + sig * np.sin(bx)
        elif der == 1:
            out = grow - decay + np.sin(bx) + sig * np.cos(bx)
        else:
            out = grow + decay + np.cos(bx) - sig * np.sin(bx)
        return out * b ** der


class PlanarBeam:
    """Uniform Euler-Bernoulli beam with large planar motion (Sec. 8.4).

    :ivar length: length l
    :vartype length: float
    :ivar mass: total mass (mu l)
    :vartype mass: float
    :ivar ei: bending stiffness EI
    :vartype ei: float
    :ivar n: number of modes (Eq. 8.70 uses 3)
    :vartype n: int
    """

    def __init__(self, length, mass, ei, n_modes=3, n_gauss=None,
                 basis="sine"):
        """Initialize the beam.

        :param length: length
        :type length: float
        :param mass: total mass
        :type mass: float
        :param ei: bending stiffness
        :type ei: float
        :param n_modes: number of sine modes
        :type n_modes: int
        :param n_gauss: Gauss points (default max(2 n + 12, 40))
        :type n_gauss: int or None
        :param basis: mode basis, see :class:`ModeBasis`
        :type basis: str
        """
        self.length = float(length)
        self.mass = float(mass)
        self.ei = float(ei)
        self.n = int(n_modes)
        ng = n_gauss or max(2 * self.n + 12, 40)
        xg, wg = np.polynomial.legendre.leggauss(ng)
        self.xg = 0.5 * self.length * (xg + 1.0)
        self.wg = 0.5 * self.length * wg * self.mass / self.length
        self.basis = ModeBasis(self.length, self.n, basis)
        self.phi_g = self.basis.values(self.xg)  # (ng, n)
        dd = self.basis.values(self.xg, 2)
        self.kmat = self.ei * (dd.T * (0.5 * self.length * wg)) @ dd
        self.dphi0 = self.basis.values(0.0, 1)[0]
        self.dphil = self.basis.values(self.length, 1)[0]
        self.phil = self.basis.values(self.length)[0]
        if basis == "sine":
            self.phil = np.zeros(self.n)
        else:
            self.dphi0 = np.zeros(self.n)

    @property
    def size(self):
        """Return the number of coordinates p = [x, y, theta, q].

        :rtype: int
        """
        return 3 + self.n

    def deflection(self, x2, q):
        """Return u(x2) for modal coordinates q (Eq. 8.69).

        :param x2: positions along the beam
        :type x2: array_like
        :param q: modal coordinates
        :type q: array_like
        :rtype: numpy.ndarray
        """
        return self.basis.values(x2) @ np.asarray(q, dtype=float)

    def integrals(self, theta, theta_d, q, q_d, gravity):
        """Return M, h and Q_g of the beam equations (module docstring).

        :param theta: frame angle
        :type theta: float
        :param theta_d: frame angular velocity
        :type theta_d: float
        :param q: modal coordinates
        :type q: numpy.ndarray
        :param q_d: modal velocities
        :type q_d: numpy.ndarray
        :param gravity: gravity acceleration
        :type gravity: numpy.ndarray
        :return: (M, h, Q_g, Jl) with Jl = dr(l)/dp (2 x size) and the
            velocity term Jd pd at x2 = l appended as Jl[:, -1]... see code
        :rtype: tuple
        """
        r = rigid.rot2(theta)
        dr = rigid.rot2(theta + 0.5 * np.pi)
        u = self.phi_g @ q
        ud = self.phi_g @ q_d
        npt = self.size
        m_mat = np.zeros((npt, npt))
        h = np.zeros(npt)
        qg = np.zeros(npt)
        e2 = r[:, 1]
        for i in range(len(self.xg)):
            s = np.array([self.xg[i], u[i]])
            jac = np.zeros((2, npt))
            jac[:, 0:2] = np.eye(2)
            jac[:, 2] = dr @ s
            jac[:, 3:] = np.outer(e2, self.phi_g[i])
            jdpd = (-theta_d ** 2 * (r @ s)
                    + 2.0 * theta_d * (dr @ np.array([0.0, ud[i]])))
            w = self.wg[i]
            m_mat += w * jac.T @ jac
            h += w * jac.T @ jdpd
            qg += w * jac.T @ gravity
        s_l = np.array([self.length, self.phil @ q])
        jl = np.zeros((2, npt))
        jl[:, 0:2] = np.eye(2)
        jl[:, 2] = dr @ s_l
        jl[:, 3:] = np.outer(e2, self.phil)
        jd_l = (-theta_d ** 2 * (r @ s_l)
                + 2.0 * theta_d * (dr @ np.array([0.0, self.phil @ q_d])))
        return m_mat, h, qg, (jl, jd_l)

    def energy(self, r_i, theta, q, v_i, theta_d, q_d, gravity):
        """Return kinetic plus potential (elastic and gravity) energy.

        :rtype: float
        """
        r = rigid.rot2(theta)
        dr = rigid.rot2(theta + 0.5 * np.pi)
        u = self.phi_g @ q
        ud = self.phi_g @ q_d
        e = 0.5 * float(q @ self.kmat @ q)
        for i in range(len(self.xg)):
            s = np.array([self.xg[i], u[i]])
            pos = np.asarray(r_i) + r @ s
            vel = (np.asarray(v_i) + theta_d * (dr @ s) + r
                   @ np.array([0.0, ud[i]]))
            e += self.wg[i] * (0.5 * vel @ vel - gravity @ pos)
        return e


def _stack(parts):
    """Return the Affine expression stacking scalar/vector Affine parts.

    :param parts: list of Affine
    :type parts: list
    :rtype: Affine
    """
    nu = max(p.m.shape[1] for p in parts)
    return Affine(np.vstack([p.width(nu) for p in parts]),
                  np.concatenate([p.c for p in parts]))


def _rows(expr, idx):
    """Return the rows idx of an Affine expression.

    :rtype: Affine
    """
    return Affine(expr.m[idx], expr.c[idx])


class PlanarFlexibleChain:
    """Planar chain of rigid bodies and large-motion beams (Secs. 8.13, 8.14).

    Hinge k joins body k-1 (the ground for k = 0) to body k. Hinge types:
    ``"smooth"`` (Sec. 8.5), ``"fixed"`` (Sec. 8.7), ``("spring", k, c)``
    (rotary spring-damper, Sec. 8.6) and, for the base hinge only,
    ``("driven", theta0, omega)`` (prescribed angle theta0 + omega t of
    body 0). The output end of the last body is free (``tip=None``) or
    pinned to a ground point (``tip=point``, closed four-bar of Example 8.2).

    :ivar bodies: list of solvcon.multibody.tmm.rigid.PlanarBody or PlanarBeam
    :vartype bodies: list
    :ivar hinges: hinge descriptions
    :vartype hinges: list
    :ivar gravity: gravity acceleration
    :vartype gravity: numpy.ndarray
    :ivar base: ground point of hinge 0
    :vartype base: numpy.ndarray
    :ivar tip: ground point of the tip pin or None
    :vartype tip: numpy.ndarray or None
    """

    def __init__(self, bodies, hinges=None, gravity=(0.0, -9.81),
                 base=(0.0, 0.0), tip=None, redundant=False):
        """Initialize the chain.

        :param bodies: bodies from the base to the tip
        :type bodies: list
        :param hinges: hinge descriptions (default all smooth)
        :type hinges: list or None
        :param gravity: gravity acceleration
        :type gravity: array_like
        :param base: ground point of the base hinge
        :type base: array_like
        :param tip: ground point of a tip pin, or None for a free tip
        :type tip: array_like or None
        :param redundant: accept a redundant constraint (for example the
            axial force of a beam between two ground pins); the minimum-norm
            internal forces are returned
        :type redundant: bool
        """
        self.bodies = list(bodies)
        self.hinges = (list(hinges) if hinges is not None
                       else ["smooth"] * len(bodies))
        self.gravity = np.asarray(gravity, dtype=float)
        self.base = np.asarray(base, dtype=float)
        self.tip = None if tip is None else np.asarray(tip, dtype=float)
        self.redundant = bool(redundant)
        for k, h in enumerate(self.hinges):
            if isinstance(h, tuple) and h[0] == "driven" and k != 0:
                raise ValueError("only the base hinge can be driven")

    @staticmethod
    def _is_beam(b):
        return isinstance(b, PlanarBeam)

    @staticmethod
    def _kind(h):
        return h if isinstance(h, str) else h[0]

    # -- one instant -------------------------------------------------------

    def _beam_relations(self, sysm, beam, p_in, theta, q, m_in, q_in, coef,
                        mode):
        """Add the beam equations and return (p_out, q_out, m_out,
        tangent_out).

        :param sysm: overall equation
        :type sysm: AffineSystem
        :param beam: beam
        :type beam: PlanarBeam
        :param p_in: input-end position (dt) or acceleration (acc)
        :type p_in: Affine
        :param theta: frame angle (dt) or angular acceleration (acc)
        :type theta: Affine
        :param q: modal coordinates (dt) or accelerations (acc)
        :type q: Affine
        :param m_in: input moment
        :type m_in: Affine
        :param q_in: input force
        :type q_in: Affine
        :param coef: coefficients of the instant
        :type coef: dict
        :param mode: 'dt' or 'acc'
        :type mode: str
        :rtype: tuple
        """
        p = _stack([p_in, theta, q])
        m_mat, h, qg = coef["M"], coef["h"], coef["Qg"]
        if mode == "dt":
            pdd = coef["A"] * p + coef["B"]
            kfull = np.zeros((beam.size, beam.size))
            kfull[3:, 3:] = beam.kmat
            kp = p.lmul(kfull)
        else:
            pdd = p
            kp = AffineSystem.const(np.concatenate(
                [np.zeros(3), beam.kmat @ coef["q_val"]]))
        dyn = pdd.lmul(m_mat) + (h - qg)
        dyn = dyn + kp
        # Rows 0-1: q_O = q_I - (M pdd + h - Q_g)_r.
        q_out = q_in - _rows(dyn, [0, 1])
        jl, _ = coef["Jl"]
        # Row 2: m_O = m_I + (M pdd + h + K p - Q_g)_theta + Jl_theta . q_O.
        m_out = m_in + _rows(dyn, [2]) + q_out.lmul(jl[:, 2].reshape(1, 2))
        # Modal rows (Eq. 8.93): ... + m_I phi'(0) - m_O phi'(l) + Jl_q^T q_O =
        # 0.
        modal = (_rows(dyn, list(range(3, beam.size))) + m_in.outer(beam.dphi0)
                 - m_out.outer(beam.dphil) + q_out.lmul(jl[:, 3:].T))
        sysm.equation(modal)
        g_th, g_0, g_q = coef["geo"]
        p_out = p_in + theta.outer(g_th) + g_0 + q.lmul(g_q)
        tangent = theta + q.lmul(beam.dphil.reshape(1, -1))
        return p_out, q_out, m_out, tangent

    def _coefficients(self, k, mode, ctx):
        """Return the per-instant coefficients of body k.

        :rtype: dict
        """
        b = self.bodies[k]
        if not self._is_beam(b):
            return {}
        if mode == "dt":
            st = ctx["states"][k]
            lin = ctx["lin"]
            dt = lin.dt
            th, thd, thdd = (float(st["theta"].x[-1]), float(st["theta"].v),
                             float(st["theta"].a))
            q, qd, qdd = st["q"].x[-1], st["q"].v, st["q"].a
            th_p = th + thd * dt + 0.5 * thdd * dt * dt
            thd_p = thd + thdd * dt
            q_p = q + qd * dt + 0.5 * qdd * dt * dt
            qd_p = qd + qdd * dt
            m_mat, h, qg, jl = b.integrals(th_p, thd_p, q_p, qd_p,
                                           self.gravity)
            a_c, b_r, _, _ = lin.coefficients(st["r_i"])
            _, b_th, _, _ = lin.coefficients(st["theta"])
            _, b_q, _, _ = lin.coefficients(st["q"])
            g_th, g_0, _, _ = dttmm._planar_geometry([b.length, 0.0],
                                                     st["theta"], dt)
            # Deflection of the output end, rotated with the predicted angle.
            g_q = np.outer(rigid.rot2(th_p)[:, 1], b.phil)
            return {"M": m_mat, "h": h, "Qg": qg, "Jl": jl, "A": a_c,
                    "B": np.concatenate([b_r, np.atleast_1d(b_th), b_q]),
                    "geo": (g_th, g_0, g_q)}
        th, w = ctx["theta"][k], ctx["omega"][k]
        q, qd = ctx["q"][k], ctx["qd"][k]
        m_mat, h, qg, jl = b.integrals(th, w, q, qd, self.gravity)
        jmat, jd_l = jl
        return {"M": m_mat, "h": h, "Qg": qg, "Jl": jl, "q_val": q,
                "geo": (jmat[:, 2], jd_l, jmat[:, 3:])}

    def deduce(self, mode, ctx):
        """Deduce and solve the overall transfer equation at one instant.

        :param mode: 'dt' (positions at t_i) or 'acc' (accelerations at t)
        :type mode: str
        :param ctx: per-instant context
        :type ctx: dict
        :return: per body: theta (angle or angular acceleration), p_in, q
            (modal coordinates or accelerations), and the residual
        :rtype: dict
        """
        sysm = AffineSystem()
        n = len(self.bodies)
        h0 = self.hinges[0]
        kind0 = self._kind(h0)
        p = AffineSystem.const(self.base if mode == "dt" else np.zeros(2))
        q_f = sysm.unknown(2)
        m = sysm.unknown(1) if kind0 != "smooth" else AffineSystem.const([0.0])
        tangent = AffineSystem.const([0.0])
        res = {"theta": [], "p_in": [], "q": [], "p_out": None}
        for k in range(n):
            b = self.bodies[k]
            kind = self._kind(self.hinges[k])
            beam = self._is_beam(b)
            qb = sysm.unknown(b.n) if beam else None
            slope0 = (qb.lmul(b.dphi0.reshape(1, -1)) if beam
                      else AffineSystem.const([0.0]))
            if kind == "smooth":
                if k > 0:
                    sysm.equation(m)
                theta = sysm.unknown(1)
            elif kind == "fixed":
                theta = tangent - slope0
            elif kind == "spring":
                kr, cr = self.hinges[k][1], self.hinges[k][2]
                if mode == "dt":
                    var = ctx["springs"][k]
                    _, _, c_c, d_d = ctx["lin"].coefficients(var)
                    # m = k Delta + c (C Delta + D) as an equation of the
                    # overall system (dividing by k would cancel digits for
                    # small dT).
                    delta = sysm.unknown(1)
                    sysm.equation(m - (kr + cr * c_c) * delta
                                  - cr * float(d_d))
                    theta = tangent - slope0 + delta
                else:
                    theta = sysm.unknown(1)
                    sysm.equation(m - ctx["spring_moment"][k])
            else:  # driven base hinge
                th0, om = self.hinges[k][1], self.hinges[k][2]
                val = th0 + om * ctx["time"] if mode == "dt" else 0.0
                theta = AffineSystem.const([val])
            res["theta"].append(theta)
            res["p_in"].append(p)
            res["q"].append(qb)
            if beam:
                coef = self._coefficients(k, mode, ctx)
                p, q_f, m, tangent = self._beam_relations(sysm, b, p, theta,
                                                          qb, m, q_f, coef,
                                                          mode)
            else:
                if mode == "dt":
                    u = dttmm.planar_body_matrix_dt(b, ctx["states"][k],
                                                    ctx["lin"],
                                                    b.mass * self.gravity)
                else:
                    u = rigid.planar_body_matrix(b.mass, b.j_i, b.l_io, b.l_ic,
                                                 ctx["theta"][k],
                                                 ctx["omega"][k],
                                                 b.mass * self.gravity)
                z = _stack([p, theta, m, q_f,
                            AffineSystem.const([1.0])]).lmul(u)
                p, theta_o, m, q_f = (_rows(z, [0, 1]), _rows(z, [2]),
                                      _rows(z, [3]), _rows(z, [4, 5]))
                tangent = theta_o
        if self.tip is None:
            sysm.equation(m)
            sysm.equation(q_f)
        else:
            sysm.equation(m)
            sysm.equation(p - (self.tip if mode == "dt" else np.zeros(2)))
        u, resid = sysm.solve(strict=not self.redundant)
        out = {"theta": np.array([float(t.value(u)[0]) for t in res["theta"]]),
               "p_in": [x.value(u) for x in res["p_in"]],
               "q": [None if x is None else x.value(u) for x in res["q"]],
               "residual": resid, "p_tip": p.value(u)}
        return out

    # -- kinematics, initial accelerations, integration ----------------------

    def kinematics(self, theta, omega, q, qd):
        """Return input-end positions and velocities of every body.

        :rtype: tuple
        """
        p = self.base.copy()
        v = np.zeros(2)
        ps, vs = [], []
        for k, b in enumerate(self.bodies):
            ps.append(p.copy())
            vs.append(v.copy())
            if self._is_beam(b):
                lo = np.array([b.length, b.phil @ q[k]])
                vrel = rigid.rot2(theta[k]) @ np.array([0.0, b.phil @ qd[k]])
            else:
                lo, vrel = b.l_io, np.zeros(2)
            r = rigid.rot2(theta[k]) @ lo
            p = p + r
            v = v + omega[k] * np.array([-r[1], r[0]]) + vrel
        return ps, vs

    def _spring_moments(self, theta, omega, q, qd):
        """Return the moments of the spring hinges and their deflections.

        :rtype: tuple
        """
        mom, dfl = {}, {}
        tan_in, rate_in = 0.0, 0.0
        for k, b in enumerate(self.bodies):
            h = self.hinges[k]
            beam = self._is_beam(b)
            t0 = theta[k] + (b.dphi0 @ q[k] if beam else 0.0)
            r0 = omega[k] + (b.dphi0 @ qd[k] if beam else 0.0)
            if self._kind(h) == "spring":
                d, dd = t0 - tan_in, r0 - rate_in
                mom[k] = h[1] * d + h[2] * dd
                dfl[k] = (d, dd)
            tan_in = theta[k] + (b.dphil @ q[k] if beam else 0.0)
            rate_in = omega[k] + (b.dphil @ qd[k] if beam else 0.0)
        return mom, dfl

    def accelerations(self, theta, omega, q=None, qd=None):
        """Return angular and modal accelerations (acceleration form).

        :param theta: body angles
        :type theta: array_like
        :param omega: body angular velocities
        :type omega: array_like
        :param q: modal coordinates per body (None for rigid bodies)
        :type q: list or None
        :param qd: modal velocities per body
        :type qd: list or None
        :return: result of :meth:`deduce` in mode 'acc'
        :rtype: dict
        """
        q, qd = self._default_q(q, qd)
        mom, _ = self._spring_moments(theta, omega, q, qd)
        ctx = {"theta": np.asarray(theta, float),
               "omega": np.asarray(omega, float), "q": q, "qd": qd,
               "spring_moment": mom}
        return self.deduce("acc", ctx)

    def _default_q(self, q, qd):
        q = [None] * len(self.bodies) if q is None else list(q)
        qd = [None] * len(self.bodies) if qd is None else list(qd)
        for k, b in enumerate(self.bodies):
            if self._is_beam(b):
                q[k] = np.zeros(b.n) if q[k] is None else np.asarray(q[k],
                                                                     float)
                qd[k] = (np.zeros(b.n) if qd[k] is None
                         else np.asarray(qd[k], float))
        return q, qd

    def energy(self, theta, omega, q=None, qd=None):
        """Return the total energy (open chains and closed chains).

        :rtype: float
        """
        q, qd = self._default_q(q, qd)
        ps, vs = self.kinematics(theta, omega, q, qd)
        e = 0.0
        for k, b in enumerate(self.bodies):
            if self._is_beam(b):
                e += b.energy(ps[k], theta[k], q[k], vs[k], omega[k], qd[k],
                              self.gravity)
            else:
                rc = rigid.rot2(theta[k]) @ b.l_ic
                vc = vs[k] + omega[k] * np.array([-rc[1], rc[0]])
                jc = b.j_i - b.mass * b.l_ic @ b.l_ic
                e += 0.5 * b.mass * vc @ vc + 0.5 * jc * omega[k] ** 2
                e -= b.mass * self.gravity @ (ps[k] + rc)
        mom, dfl = self._spring_moments(theta, omega, q, qd)
        for k, (d, _) in dfl.items():
            e += 0.5 * self.hinges[k][1] * d * d
        return e

    def _pack(self, theta, omega, q, qd):
        parts = [np.asarray(theta, float), np.asarray(omega, float)]
        parts += [q[k] for k, b in enumerate(self.bodies) if self._is_beam(b)]
        parts += [qd[k] for k, b in enumerate(self.bodies) if self._is_beam(b)]
        return np.concatenate(parts)

    def _unpack(self, y):
        n = len(self.bodies)
        th, om = y[:n], y[n:2 * n]
        q, qd = [None] * n, [None] * n
        i = 2 * n
        for k, b in enumerate(self.bodies):
            if self._is_beam(b):
                q[k] = y[i:i + b.n]
                i += b.n
        for k, b in enumerate(self.bodies):
            if self._is_beam(b):
                qd[k] = y[i:i + b.n]
                i += b.n
        return th, om, q, qd

    def simulate_rk4(self, theta0, omega0, t_end, dt, q0=None, qd0=None):
        """Integrate with the acceleration form and RK4 (reference solution).

        :return: dict with t, theta, omega, q and qd (per beam body arrays)
        :rtype: dict
        """
        q0, qd0 = self._default_q(q0, qd0)
        n = len(self.bodies)

        def rhs(y, t):
            th, om, q, qd = self._unpack(y)
            for k, h in enumerate(self.hinges):
                if self._kind(h) == "driven":
                    th = th.copy()
                    om = om.copy()
                    th[k] = h[1] + h[2] * t
                    om[k] = h[2]
            res = self.accelerations(th, om, q, qd)
            alpha = res["theta"]
            qdd = [res["q"][k] for k, b in enumerate(self.bodies)
                   if self._is_beam(b)]
            return np.concatenate([om, alpha]
                                  + [x for x in qd if x is not None] + qdd)

        y = self._pack(theta0, omega0, q0, qd0)
        steps = int(round(t_end / dt))
        hist = np.empty((steps + 1, y.size))
        hist[0] = y
        for i in range(steps):
            t = i * dt
            k1 = rhs(y, t)
            k2 = rhs(y + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = rhs(y + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = rhs(y + dt * k3, t + dt)
            y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            hist[i + 1] = y
        return self._history(dt * np.arange(steps + 1), hist, n)

    def _history(self, t, hist, n):
        out = {"t": t, "theta": hist[:, :n], "omega": hist[:, n:2 * n],
               "q": {}, "qd": {}}
        i = 2 * n
        for key in ("q", "qd"):
            for k, b in enumerate(self.bodies):
                if self._is_beam(b):
                    out[key][k] = hist[:, i:i + b.n]
                    i += b.n
        return out

    def simulate_dt(self, theta0, omega0, t_end, dt, q0=None, qd0=None,
                    method="newmark", **kwargs):
        """Integrate with the DT-TMM (algorithm of Sec. 8.13).

        :param theta0: initial body angles (frame angles of the beams)
        :type theta0: array_like
        :param omega0: initial angular velocities
        :type omega0: array_like
        :param t_end: final time
        :type t_end: float
        :param dt: time step
        :type dt: float
        :param q0: initial modal coordinates per body
        :type q0: list or None
        :param qd0: initial modal velocities per body
        :type qd0: list or None
        :param method: linearizer name
        :type method: str
        :return: dict with t, theta, q (per beam body) and residual
        :rtype: dict
        """
        lin = dttmm.make_linearizer(method, dt, **kwargs)
        q0, qd0 = self._default_q(q0, qd0)
        theta0 = np.asarray(theta0, float)
        omega0 = np.asarray(omega0, float)
        for k, h in enumerate(self.hinges):
            if self._kind(h) == "driven":
                theta0[k], omega0[k] = h[1], h[2]
        acc = self.accelerations(theta0, omega0, q0, qd0)
        ps, vs = self.kinematics(theta0, omega0, q0, qd0)
        states = {}
        for k, b in enumerate(self.bodies):
            al, a_i = acc["theta"][k], acc["p_in"][k]
            if self._is_beam(b):
                states[k] = {"theta": dttmm.Variable(theta0[k], omega0[k], al),
                             "r_i": dttmm.Variable(ps[k], vs[k], a_i),
                             "q": dttmm.Variable(q0[k], qd0[k], acc["q"][k])}
            else:
                states[k] = dttmm.PlanarBodyState.from_motion(
                    [b.l_ic], theta0[k], omega0[k], al, ps[k], vs[k], a_i)
        springs = {}
        _, dfl = self._spring_moments(theta0, omega0, q0, qd0)
        for k, (d, dd) in dfl.items():
            # Second derivative of the deflection from the accelerations.
            springs[k] = dttmm.Variable(d, dd, self._delta_acc(k, acc))
        ctx = {"states": states, "lin": lin, "springs": springs, "time": 0.0}
        steps = int(round(t_end / dt))
        n = len(self.bodies)
        th = np.empty((steps + 1, n))
        qh = {k: np.empty((steps + 1, b.n)) for k, b in enumerate(self.bodies)
              if self._is_beam(b)}
        resid = np.zeros(steps + 1)
        th[0] = theta0
        for k in qh:
            qh[k][0] = q0[k]
        for i in range(steps):
            ctx["time"] = (i + 1) * dt
            res = self.deduce("dt", ctx)
            for k, b in enumerate(self.bodies):
                st = states[k]
                thk = res["theta"][k]
                p_i = res["p_in"][k]
                if self._is_beam(b):
                    lin.update(st["theta"], thk)
                    lin.update(st["r_i"], p_i)
                    lin.update(st["q"], res["q"][k])
                    qh[k][i + 1] = res["q"][k]
                else:
                    g_th, g_0, _, _ = dttmm._planar_geometry(b.l_ic, st.theta,
                                                             dt)
                    lin.update(st.theta, thk)
                    lin.update(st.r_i, p_i)
                    lin.update(st.r_c, p_i + g_th * thk + g_0)
                th[i + 1, k] = thk
            for k in springs:
                lin.update(springs[k], self._delta_value(k, res))
            resid[i + 1] = res["residual"]
        return {"t": dt * np.arange(steps + 1), "theta": th, "q": qh,
                "residual": resid}

    def _tangents(self, k, theta, q):
        b_prev = self.bodies[k - 1] if k else None
        t_in = 0.0
        if k:
            t_in = theta[k - 1] + (b_prev.dphil @ q[k - 1]
                                   if self._is_beam(b_prev) else 0.0)
        b = self.bodies[k]
        t_out = theta[k] + (b.dphi0 @ q[k] if self._is_beam(b) else 0.0)
        return t_in, t_out

    def _delta_value(self, k, res):
        t_in, t_out = self._tangents(k, res["theta"], res["q"])
        return t_out - t_in

    def _delta_acc(self, k, acc):
        t_in, t_out = self._tangents(k, acc["theta"], acc["q"])
        return t_out - t_in


def smooth_hinge_beam_matrix(beam_rows, n):
    """Transfer matrix of a smooth hinge whose outboard body is a beam (Eq.
    8.98).

    ``beam_rows`` holds the n vibration equations and the output moment
    of the outboard beam as linear functions of its input state
    ``[x, y, theta, m, q_x, q_y, q^1..q^n, 1]`` (Eq. 8.94). The hinge
    removes theta and q^k with the zero moments at both beam ends.

    :param beam_rows: (n + 1) x (7 + n) matrix, rows [modal..., m_O]
    :type beam_rows: numpy.ndarray
    :param n: number of modes
    :type n: int
    :return: (7 + n) x (7 + n) hinge matrix z_O = U z_I
    :rtype: numpy.ndarray
    """
    size = 7 + n
    unk = [2] + list(range(6, 6 + n))
    known = [0, 1, 4, 5, size - 1]
    a = beam_rows[:, unk]
    b = beam_rows[:, known]
    sol = -np.linalg.solve(a, b)
    u = np.zeros((size, size))
    for i in (0, 1, 4, 5, size - 1):
        u[i, i] = 1.0
    for r, i in enumerate(unk):
        u[i, known] = sol[r]
    return u


# ---------------------------------------------------------------------------
# Spatial large-motion beams (Secs. 8.8-8.12)
# ---------------------------------------------------------------------------

class SpatialBeam:
    """Uniform Euler-Bernoulli beam with large spatial motion (Secs. 8.8, 8.9).

    Floating frame at the input end with the x-y-z angles of Eq. (8.19).
    Transverse deformations ``v = sum phi_k q_v^k`` (along y2) and
    ``w = sum phi_k q_w^k`` (along z2) (Eq. 8.146, mode shapes Y_k, Z_k of
    Eq. 8.171 with one basis per plane); torsion and axial deformation are
    neglected. The cross section adds the rotary inertia ``j_axial`` about
    the beam axis (``J_4`` of Example 8.3). Coordinates
    ``p = [x, y, z, theta_x, theta_y, theta_z, q_v^1..q_v^n, q_w^1..q_w^n]``.

    :ivar length: length
    :vartype length: float
    :ivar mass: mass
    :vartype mass: float
    :ivar n: modes per plane
    :vartype n: int
    """

    def __init__(self, length, mass, ei_y, ei_z=None, n_modes=3, j_axial=0.0,
                 basis="sine", n_gauss=None):
        """Initialize the beam.

        :param length: length
        :type length: float
        :param mass: total mass
        :type mass: float
        :param ei_y: bending stiffness for w (about y2)
        :type ei_y: float
        :param ei_z: bending stiffness for v (about z2), default ei_y
        :type ei_z: float or None
        :param n_modes: modes per plane
        :type n_modes: int
        :param j_axial: rotary inertia of the beam about its axis
        :type j_axial: float
        :param basis: mode basis, see :class:`ModeBasis`
        :type basis: str
        :param n_gauss: Gauss points
        :type n_gauss: int or None
        """
        self.length = float(length)
        self.mass = float(mass)
        self.n = int(n_modes)
        self.j_axial = float(j_axial)
        ei_z = ei_y if ei_z is None else ei_z
        ng = n_gauss or max(2 * self.n + 12, 40)
        xg, wg = np.polynomial.legendre.leggauss(ng)
        self.xg = 0.5 * self.length * (xg + 1.0)
        self.wg = 0.5 * self.length * wg * self.mass / self.length
        self.basis = ModeBasis(self.length, self.n, basis)
        self.phi_g = self.basis.values(self.xg)
        dd = self.basis.values(self.xg, 2)
        k0 = (dd.T * (0.5 * self.length * wg)) @ dd
        z = np.zeros_like(k0)
        self.kmat = np.block([[ei_z * k0, z], [z, ei_y * k0]])
        self.phil = self.basis.values(self.length)[0]
        if basis == "sine":
            self.phil = np.zeros(self.n)
        self.dphi0 = self.basis.values(0.0, 1)[0]
        if basis != "sine":
            self.dphi0 = np.zeros(self.n)
        self.dphil = self.basis.values(self.length, 1)[0]

    @property
    def size(self):
        """Return the number of coordinates.

        :rtype: int
        """
        return 6 + 2 * self.n

    def _shape(self, phi):
        """Return E(x) (3 x 2n): deformation [0, v, w] per modal coordinate."""
        e = np.zeros((3, 2 * self.n))
        e[1, :self.n] = phi
        e[2, self.n:] = phi
        return e

    def rotation_rows(self, dphi):
        """Return B(x) (3 x 2n): body-frame rotation [0, -w', v'] (Eq.
        8.144)."""
        b = np.zeros((3, 2 * self.n))
        b[2, :self.n] = dphi
        b[1, self.n:] = -dphi
        return b

    def deflection(self, x2, q):
        """Return (v, w) at x2 for the modal coordinates q.

        :rtype: tuple
        """
        ph = self.basis.values(x2)
        q = np.asarray(q, dtype=float)
        return ph @ q[:self.n], ph @ q[self.n:]

    def integrals(self, theta, theta_d, q, q_d, gravity):
        """Return M, h, Q_g and the output-end Jacobian data.

        :return: (M, h, Q_g, (Jl, Jdl, A, H)) with Jl = dr(l)/dp and Jdl the
            velocity term of the output-end acceleration
        :rtype: tuple
        """
        a = rigid.rotation_xyz(theta)
        hm = dttmm.h_matrix(theta)
        hd = dttmm.h_dot_matrix(theta, theta_d)
        w = hm @ theta_d
        wt = rigid.skew(w)
        hdt = hd @ theta_d
        npt = self.size
        m_mat = np.zeros((npt, npt))
        h = np.zeros(npt)
        qg = np.zeros(npt)
        v, wv = self.phi_g @ q[:self.n], self.phi_g @ q[self.n:]
        vd, wd = self.phi_g @ q_d[:self.n], self.phi_g @ q_d[self.n:]

        def point(s, sd, e):
            jac = np.zeros((3, npt))
            jac[:, 0:3] = np.eye(3)
            jac[:, 3:6] = -a @ rigid.skew(s) @ hm
            jac[:, 6:] = a @ e
            jd = a @ (wt @ wt @ s + np.cross(hdt, s) + 2.0 * wt @ sd)
            return jac, jd

        for i in range(len(self.xg)):
            s = np.array([self.xg[i], v[i], wv[i]])
            sd = np.array([0.0, vd[i], wd[i]])
            jac, jd = point(s, sd, self._shape(self.phi_g[i]))
            m_mat += self.wg[i] * jac.T @ jac
            h += self.wg[i] * jac.T @ jd
            qg += self.wg[i] * jac.T @ gravity
        if self.j_axial:
            jd_ax = np.diag([self.j_axial, 0.0, 0.0])
            m_mat[3:6, 3:6] += hm.T @ jd_ax @ hm
            h[3:6] += hm.T @ (jd_ax @ hdt + np.cross(w, jd_ax @ w))
        s_l = np.array([self.length, self.phil @ q[:self.n],
                        self.phil @ q[self.n:]])
        sd_l = np.array([0.0, self.phil @ q_d[:self.n],
                         self.phil @ q_d[self.n:]])
        jl, jdl = point(s_l, sd_l, self._shape(self.phil))
        return m_mat, h, qg, (jl, jdl, a, hm)

    def energy(self, r_i, theta, q, v_i, theta_d, q_d, gravity):
        """Return kinetic plus elastic plus gravity energy.

        :rtype: float
        """
        a = rigid.rotation_xyz(theta)
        hm = dttmm.h_matrix(theta)
        w_g = a @ (hm @ theta_d)
        v, wv = self.phi_g @ q[:self.n], self.phi_g @ q[self.n:]
        vd, wd = self.phi_g @ q_d[:self.n], self.phi_g @ q_d[self.n:]
        e = 0.5 * float(q @ self.kmat @ q)
        for i in range(len(self.xg)):
            s = np.array([self.xg[i], v[i], wv[i]])
            pos = np.asarray(r_i) + a @ s
            vel = (np.asarray(v_i) + np.cross(w_g, a @ s) + a
                   @ np.array([0.0, vd[i], wd[i]]))
            e += self.wg[i] * (0.5 * vel @ vel - gravity @ pos)
        wb = hm @ theta_d
        e += 0.5 * self.j_axial * wb[0] ** 2
        return e


class SpatialFlexibleChain:
    """Spatial chain of rigid bodies and large-motion beams (Secs. 8.13, 8.15).

    All bodies use the x-y-z angles of Eq. (7.71). Hinge k joins body k-1
    (the ground for k = 0) to body k: ``"ball"`` is a smooth
    ball-and-socket hinge (Secs. 8.11, 7.7.2); ``"fixed"`` clamps body 0 to
    the ground (base hinge only; a beam needs a basis with phi'(0) = 0,
    Sec. 8.10). The output end of the last body is free.

    :ivar bodies: solvcon.multibody.tmm.rigid.SpatialBody or SpatialBeam
    :vartype bodies: list
    :ivar hinges: hinge descriptions
    :vartype hinges: list
    :ivar gravity: gravity acceleration
    :vartype gravity: numpy.ndarray
    :ivar base: ground point
    :vartype base: numpy.ndarray
    """

    def __init__(self, bodies, hinges=None, gravity=(0.0, -9.81, 0.0),
                 base=(0.0, 0.0, 0.0)):
        """Initialize the chain.

        :param bodies: bodies from the base to the tip
        :type bodies: list
        :param hinges: 'ball' or 'fixed' per body (default all 'ball')
        :type hinges: list or None
        :param gravity: gravity acceleration
        :type gravity: array_like
        :param base: ground point of hinge 0
        :type base: array_like
        :raises ValueError: for an unsupported hinge layout
        """
        self.bodies = list(bodies)
        self.hinges = (list(hinges) if hinges is not None
                       else ["ball"] * len(bodies))
        self.gravity = np.asarray(gravity, dtype=float)
        self.base = np.asarray(base, dtype=float)
        for h in self.hinges:
            if self._kind(h) not in ("ball", "fixed", "spring"):
                raise ValueError("unknown hinge: %s" % (h,))

    @staticmethod
    def _is_beam(b):
        return isinstance(b, SpatialBeam)

    @staticmethod
    def _kind(h):
        return h if isinstance(h, str) else h[0]

    # -- hinge kinematics (Secs. 8.10-8.12) ----------------------------------

    def _end_rot(self, k, q, end):
        """Return the rotation vector alpha of body k at an end (0 input, 1
        output)."""
        b = self.bodies[k] if k >= 0 else None
        if b is None or not self._is_beam(b):
            return np.zeros(3)
        rows = b.rotation_rows(b.dphi0 if end == 0 else b.dphil)
        return rows @ q

    def _end_rows(self, k, end):
        b = self.bodies[k]
        return b.rotation_rows(b.dphi0 if end == 0 else b.dphil)

    @staticmethod
    def compose(theta_in, a, d, b):
        """Outboard angles of A(theta_out) = A(theta_in) E(a) E(d) E(-b).

        ``a`` is the rotation of the inboard tangent frame at its output end,
        ``d`` the hinge rotation (spring, Sec. 8.12; zero for a fixed hinge,
        Sec. 8.10), ``b`` the rotation of the outboard tangent at its input
        end (Eq. 8.144).

        :return: (theta_out, J_theta, J_a, J_d, J_b, matrix A_out)
        :rtype: tuple
        """
        a_in = rigid.rotation_xyz(theta_in)
        e1, e2, e3 = rot_exp(a), rot_exp(d), rot_exp(-np.asarray(b))
        r = a_in @ e1 @ e2 @ e3
        th = rigid.angles_xyz(r)
        hinv = np.linalg.inv(dttmm.h_matrix(th))
        j_th = hinv @ (e1 @ e2 @ e3).T @ dttmm.h_matrix(theta_in)
        j_a = hinv @ (e2 @ e3).T @ rot_jr(a)
        j_d = hinv @ e3.T @ rot_jr(d)
        j_b = -hinv @ rot_jr(-np.asarray(b))
        return th, j_th, j_a, j_d, j_b, r

    def spring_deflection(self, k, theta, q):
        """Return the hinge rotation Delta of hinge k from the body angles.

        :rtype: numpy.ndarray
        """
        th_in = theta[k - 1] if k else np.zeros(3)
        t_in = rigid.rotation_xyz(th_in) @ rot_exp(
            self._end_rot(k - 1, q[k - 1] if k else None, 1) if k
            else np.zeros(3))
        t_out = (rigid.rotation_xyz(theta[k])
                 @ rot_exp(self._end_rot(k, q[k], 0)))
        return rot_log(t_in.T @ t_out)

    def _hinge_dt(self, k, ctx, prev_theta, prev_q, qb, m, kind):
        """Linearized outboard angles of a fixed or spring hinge (DT-TMM)."""
        lin = ctx["lin"]
        dt = lin.dt

        def pred(var):
            return var.x[-1] + var.v * dt + 0.5 * var.a * dt * dt

        def angles(j):
            st = ctx["states"][j]
            return st["theta"] if isinstance(st, dict) else st.theta

        if k:
            th_bar = pred(angles(k - 1))
            th_aff = prev_theta
        else:
            th_bar = np.zeros(3)
            th_aff = AffineSystem.const(np.zeros(3))
        a_bar = np.zeros(3)
        a_aff = AffineSystem.const(np.zeros(3))
        if k and self._is_beam(self.bodies[k - 1]):
            rows = self._end_rows(k - 1, 1)
            a_bar = rows @ pred(ctx["states"][k - 1]["q"])
            a_aff = prev_q.lmul(rows)
        b_bar = np.zeros(3)
        b_aff = AffineSystem.const(np.zeros(3))
        if self._is_beam(self.bodies[k]):
            rows = self._end_rows(k, 0)
            b_bar = rows @ pred(ctx["states"][k]["q"])
            b_aff = qb.lmul(rows)
        d_bar = np.zeros(3)
        d_aff = AffineSystem.const(np.zeros(3))
        if kind == "spring":
            kr, cr = self.hinges[k][1], self.hinges[k][2]
            var = ctx["springs"][k]
            _, _, c_c, d_d = lin.coefficients(var)
            d_bar = pred(var)
            t_bar = rigid.rotation_xyz(th_bar) @ rot_exp(a_bar)
            # m = T (k Delta + c Delta_dot), Delta_dot = C Delta + D (Eq.
            # 7.28).
            d_aff = ctx["sysm"].unknown(3)
            ctx["sysm"].equation(m.lmul(t_bar.T) - (kr + cr * c_c) * d_aff
                                 - cr * d_d)
        th0, j_th, j_a, j_d, j_b, _ = self.compose(th_bar, a_bar, d_bar, b_bar)
        return ((th_aff - th_bar).lmul(j_th) + (a_aff - a_bar).lmul(j_a)
                + (d_aff - d_bar).lmul(j_d) + (b_aff - b_bar).lmul(j_b) + th0)

    def _fixed_acc(self, k, ctx, prev_theta, prev_q, qb):
        """Outboard angle accelerations of a fixed hinge (acceleration
        form)."""
        th, thd = ctx["theta"], ctx["theta_d"]
        q, qd = ctx["q"], ctx["qd"]

        def frames(x, eps):
            th_in = th[k - 1] + eps * thd[k - 1] if k else np.zeros(3)
            a = (self._end_rot(k - 1, q[k - 1] + eps * qd[k - 1], 1)
                 if k and self._is_beam(self.bodies[k - 1]) else np.zeros(3))
            b = (self._end_rot(k, q[k] + eps * qd[k], 0)
                 if self._is_beam(self.bodies[k]) else np.zeros(3))
            a_in = rigid.rotation_xyz(th_in)
            r = a_in @ rot_exp(a) @ rot_exp(-b)
            return a_in @ rot_exp(a) @ rot_jr(a), r @ rot_jr(-b)

        c1, c3 = frames(None, 0.0)
        eps = 1e-6
        c1p, c3p = frames(None, eps)
        c1m, c3m = frames(None, -eps)
        c1d, c3d = (c1p - c1m) / (2 * eps), (c3p - c3m) / (2 * eps)
        ad = (self._end_rows(k - 1, 1) @ qd[k - 1]
              if k and self._is_beam(self.bodies[k - 1]) else np.zeros(3))
        bd = (self._end_rows(k, 0) @ qd[k] if self._is_beam(self.bodies[k])
              else np.zeros(3))
        omega_dd = AffineSystem.const(c1d @ ad - c3d @ bd)
        if k:
            a_in = rigid.rotation_xyz(th[k - 1])
            omega_dd = (omega_dd
                        + prev_theta.lmul(a_in @ dttmm.h_matrix(th[k - 1]))
                        + a_in @ (dttmm.h_dot_matrix(th[k - 1], thd[k - 1])
                                  @ thd[k - 1]))
            if self._is_beam(self.bodies[k - 1]):
                omega_dd = omega_dd + prev_q.lmul(c1
                                                  @ self._end_rows(k - 1, 1))
        if self._is_beam(self.bodies[k]):
            omega_dd = omega_dd - qb.lmul(c3 @ self._end_rows(k, 0))
        a_out = rigid.rotation_xyz(th[k])
        hinv = np.linalg.inv(dttmm.h_matrix(th[k]))
        return (omega_dd.lmul(hinv @ a_out.T)
                - hinv @ (dttmm.h_dot_matrix(th[k], thd[k]) @ thd[k]))

    def _default_q(self, q, qd):
        q = [None] * len(self.bodies) if q is None else list(q)
        qd = [None] * len(self.bodies) if qd is None else list(qd)
        for k, b in enumerate(self.bodies):
            if self._is_beam(b):
                q[k] = np.zeros(2 * b.n) if q[k] is None else np.asarray(q[k],
                                                                         float)
                qd[k] = (np.zeros(2 * b.n) if qd[k] is None
                         else np.asarray(qd[k], float))
        return q, qd

    def _coef(self, k, mode, ctx):
        b = self.bodies[k]
        if mode == "dt":
            st = ctx["states"][k]
            lin = ctx["lin"]
            dt = lin.dt
            th, thd, thdd = st["theta"].x[-1], st["theta"].v, st["theta"].a
            q, qd, qdd = st["q"].x[-1], st["q"].v, st["q"].a
            th_p = th + thd * dt + 0.5 * thdd * dt * dt
            q_p = q + qd * dt + 0.5 * qdd * dt * dt
            m_mat, h, qg, ends = b.integrals(th_p, thd + thdd * dt, q_p,
                                             qd + qdd * dt, self.gravity)
            a_c, b_r, _, _ = lin.coefficients(st["r_i"])
            _, b_th, _, _ = lin.coefficients(st["theta"])
            _, b_q, _, _ = lin.coefficients(st["q"])
            _, at, phi = dttmm.rotation_linearization(th, thd, dt)
            l_x = np.array([b.length, 0.0, 0.0])
            g_th = np.column_stack([m @ l_x for m in at])
            g_q = ends[2] @ b._shape(b.phil)
            return {"M": m_mat, "h": h, "Qg": qg, "ends": ends, "A": a_c,
                    "q_now": q_p, "B": np.concatenate([b_r, b_th, b_q]),
                    "geo": (g_th, phi @ l_x, g_q)}
        th, thd = ctx["theta"][k], ctx["theta_d"][k]
        m_mat, h, qg, ends = b.integrals(th, thd, ctx["q"][k], ctx["qd"][k],
                                         self.gravity)
        jl, jdl = ends[0], ends[1]
        return {"M": m_mat, "h": h, "Qg": qg, "ends": ends,
                "q_val": ctx["q"][k], "q_now": ctx["q"][k],
                "geo": (jl[:, 3:6], jdl, jl[:, 6:])}

    def _beam(self, sysm, b, p_in, theta, q, m_in, q_in, coef, mode):
        """Add the spatial beam equations; return (p_out, q_out, m_out)."""
        p = _stack([p_in, theta, q])
        if mode == "dt":
            pdd = coef["A"] * p + coef["B"]
            kfull = np.zeros((b.size, b.size))
            kfull[6:, 6:] = b.kmat
            kp = p.lmul(kfull)
        else:
            pdd = p
            kp = AffineSystem.const(np.concatenate([np.zeros(6),
                                                    b.kmat @ coef["q_val"]]))
        dyn = pdd.lmul(coef["M"]) + (coef["h"] - coef["Qg"]) + kp
        jl, _, a, hm = coef["ends"]
        q_out = q_in - _rows(dyn, [0, 1, 2])
        # theta rows: dyn_th = H^T A^T (m_O - m_I) - Jl_th^T q_O.
        conv = a @ np.linalg.inv(hm.T)
        m_out = (m_in
                 + (_rows(dyn,
                          [3, 4, 5]) + q_out.lmul(jl[:, 3:6].T)).lmul(conv))
        # Virtual rotation of the end tangent frames A exp(alpha~) (Eq. 8.144):
        # A exp(alpha~) J_r(alpha) B dq.
        qv = coef["q_now"]
        b0 = b.rotation_rows(b.dphi0)
        bl = b.rotation_rows(b.dphil)
        a0, al = b0 @ qv, bl @ qv
        t0 = a @ rot_exp(a0) @ rot_jr(a0) @ b0
        tl = a @ rot_exp(al) @ rot_jr(al) @ bl
        modal = (_rows(dyn, list(range(6, b.size))) + m_in.lmul(t0.T)
                 - m_out.lmul(tl.T) + q_out.lmul(jl[:, 6:].T))
        sysm.equation(modal)
        g_th, g_0, g_q = coef["geo"]
        p_out = p_in + theta.lmul(g_th) + g_0 + q.lmul(g_q)
        return p_out, q_out, m_out

    def deduce(self, mode, ctx):
        """Deduce and solve the overall transfer equation at one instant.

        :param mode: 'dt' or 'acc'
        :type mode: str
        :param ctx: per-instant context
        :type ctx: dict
        :return: per body theta (angles or angle accelerations), p_in, q
        :rtype: dict
        """
        sysm = AffineSystem()
        ctx["sysm"] = sysm
        p = AffineSystem.const(self.base if mode == "dt" else np.zeros(3))
        q_f = sysm.unknown(3)
        m = (AffineSystem.const(np.zeros(3)) if self.hinges[0] == "ball"
             else sysm.unknown(3))
        res = {"theta": [], "p_in": [], "q": []}
        prev_theta, prev_q = None, None
        for k, b in enumerate(self.bodies):
            beam = self._is_beam(b)
            qb = sysm.unknown(2 * b.n) if beam else None
            kind = self._kind(self.hinges[k])
            if kind == "ball":
                if k > 0:
                    sysm.equation(m)
                theta = sysm.unknown(3)
            elif mode == "acc" and kind == "spring":
                theta = sysm.unknown(3)
                sysm.equation(m - ctx["spring_moment"][k])
            elif mode == "acc":
                theta = self._fixed_acc(k, ctx, prev_theta, prev_q, qb)
            else:
                theta = self._hinge_dt(k, ctx, prev_theta, prev_q, qb, m, kind)
            prev_theta, prev_q = theta, qb
            res["theta"].append(theta)
            res["p_in"].append(p)
            res["q"].append(qb)
            if beam:
                coef = self._coef(k, mode, ctx)
                p, q_f, m = self._beam(sysm, b, p, theta, qb, m, q_f, coef,
                                       mode)
            elif mode == "dt":
                u, _, _ = dttmm.spatial_body_matrix_dt(b, ctx["states"][k],
                                                       ctx["lin"],
                                                       b.mass * self.gravity)
                z = _stack([p, theta, m, q_f,
                            AffineSystem.const([1.0])]).lmul(u)
                p, m, q_f = (_rows(z, [0, 1, 2]), _rows(z, [6, 7, 8]),
                             _rows(z, [9, 10, 11]))
            else:
                th, thd = ctx["theta"][k], ctx["theta_d"][k]
                a = rigid.rotation_xyz(th)
                hm = dttmm.h_matrix(th)
                hd = dttmm.h_dot_matrix(th, thd)
                w = a @ (hm @ thd)
                omega_dd = theta.lmul(a @ hm) + a @ (hd @ thd)
                u = rigid.spatial_body_matrix(b.mass, b.j_i, a, b.l_io, b.l_ic,
                                              w, b.mass * self.gravity)
                z = _stack([p, omega_dd, m, q_f,
                            AffineSystem.const([1.0])]).lmul(u)
                p, m, q_f = (_rows(z, [0, 1, 2]), _rows(z, [6, 7, 8]),
                             _rows(z, [9, 10, 11]))
        sysm.equation(m)
        sysm.equation(q_f)
        u, resid = sysm.solve()
        return {"theta": [t.value(u) for t in res["theta"]],
                "p_in": [x.value(u) for x in res["p_in"]],
                "q": [None if x is None else x.value(u) for x in res["q"]],
                "residual": resid}

    def accelerations(self, theta, theta_d, q=None, qd=None):
        """Return angle and modal accelerations (acceleration form).

        :rtype: dict
        """
        q, qd = self._default_q(q, qd)
        theta = np.asarray(theta, float)
        theta_d = np.asarray(theta_d, float)
        return self.deduce("acc", {
            "theta": theta, "theta_d": theta_d, "q": q, "qd": qd,
            "spring_moment": self._spring_moments(theta, theta_d, q, qd)})

    def _tangent(self, k, end, theta, theta_d, q, qd):
        """Return the tangent frame T = A E(alpha) of body k at an end and its
        body rate."""
        if k < 0:
            return np.eye(3), np.zeros(3)
        b = self.bodies[k]
        a_mat = rigid.rotation_xyz(theta[k])
        w_b = dttmm.h_matrix(theta[k]) @ theta_d[k]
        if not self._is_beam(b):
            return a_mat, w_b
        rows = self._end_rows(k, end)
        al, ald = rows @ q[k], rows @ qd[k]
        e = rot_exp(al)
        return a_mat @ e, e.T @ w_b + rot_jr(al) @ ald

    def _spring_state(self, k, theta, theta_d, q, qd):
        """Return (Delta, Delta_dot, T_in) of spring hinge k (Sec. 8.12).

        ``Delta = log(T_in^T T_out)`` and ``Delta_dot = J_r(Delta)^-1 w_rel``.
        """
        t_in, w_in = self._tangent(k - 1, 1, theta, theta_d, q, qd)
        t_out, w_out = self._tangent(k, 0, theta, theta_d, q, qd)
        r = t_in.T @ t_out
        d = rot_log(r)
        dd = np.linalg.solve(rot_jr(d), w_out - r.T @ w_in)
        return d, dd, t_in

    def _spring_moments(self, theta, theta_d, q, qd):
        out = {}
        for k, h in enumerate(self.hinges):
            if self._kind(h) == "spring":
                d, dd, t = self._spring_state(k, theta, theta_d, q, qd)
                out[k] = t @ (h[1] * d + h[2] * dd)
        return out

    def kinematics(self, theta, theta_d, q, qd):
        """Return input-end positions and velocities.

        :rtype: tuple
        """
        p = self.base.copy()
        v = np.zeros(3)
        ps, vs = [], []
        for k, b in enumerate(self.bodies):
            ps.append(p.copy())
            vs.append(v.copy())
            a = rigid.rotation_xyz(theta[k])
            w = a @ (dttmm.h_matrix(theta[k]) @ theta_d[k])
            if self._is_beam(b):
                lo = np.array([b.length, b.phil @ q[k][:b.n],
                               b.phil @ q[k][b.n:]])
                vrel = (a
                        @ np.array([0.0, b.phil @ qd[k][:b.n],
                                    b.phil @ qd[k][b.n:]]))
            else:
                lo, vrel = b.l_io, np.zeros(3)
            r = a @ lo
            p = p + r
            v = v + np.cross(w, r) + vrel
        return ps, vs

    def energy(self, theta, theta_d, q=None, qd=None):
        """Return the total energy.

        :rtype: float
        """
        q, qd = self._default_q(q, qd)
        ps, vs = self.kinematics(theta, theta_d, q, qd)
        e = 0.0
        for k, b in enumerate(self.bodies):
            if self._is_beam(b):
                e += b.energy(ps[k], theta[k], q[k], vs[k], theta_d[k], qd[k],
                              self.gravity)
            else:
                a = rigid.rotation_xyz(theta[k])
                w = a @ (dttmm.h_matrix(theta[k]) @ theta_d[k])
                rc = a @ b.l_ic
                vc = vs[k] + np.cross(w, rc)
                jc = a @ b.inertia_about_c() @ a.T
                e += 0.5 * b.mass * vc @ vc + 0.5 * w @ jc @ w
                e -= b.mass * self.gravity @ (ps[k] + rc)
        for k, h in enumerate(self.hinges):
            if self._kind(h) == "spring":
                d = self.spring_deflection(k, theta, q)
                e += 0.5 * h[1] * d @ d
        return e

    def _sizes(self):
        return [2 * b.n if self._is_beam(b) else 0 for b in self.bodies]

    def _pack(self, theta, theta_d, q, qd):
        parts = [np.asarray(theta, float).ravel(),
                 np.asarray(theta_d, float).ravel()]
        parts += [q[k] for k, b in enumerate(self.bodies) if self._is_beam(b)]
        parts += [qd[k] for k, b in enumerate(self.bodies) if self._is_beam(b)]
        return np.concatenate(parts)

    def _unpack(self, y):
        n = len(self.bodies)
        th = y[:3 * n].reshape(n, 3)
        thd = y[3 * n:6 * n].reshape(n, 3)
        q, qd = [None] * n, [None] * n
        i = 6 * n
        for target in (q, qd):
            for k, b in enumerate(self.bodies):
                if self._is_beam(b):
                    target[k] = y[i:i + 2 * b.n]
                    i += 2 * b.n
        return th, thd, q, qd

    def simulate_rk4(self, theta0, theta_d0, t_end, dt, q0=None, qd0=None):
        """Integrate with the acceleration form and RK4.

        :return: dict with t, theta (N, n, 3), theta_d, q, qd
        :rtype: dict
        """
        q0, qd0 = self._default_q(q0, qd0)

        def rhs(y):
            th, thd, q, qd = self._unpack(y)
            res = self.accelerations(th, thd, q, qd)
            qdd = [res["q"][k] for k, b in enumerate(self.bodies)
                   if self._is_beam(b)]
            return np.concatenate([thd.ravel(), np.concatenate(res["theta"])]
                                  + [x for x in qd if x is not None] + qdd)

        y = self._pack(theta0, theta_d0, q0, qd0)
        steps = int(round(t_end / dt))
        hist = np.empty((steps + 1, y.size))
        hist[0] = y
        for i in range(steps):
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * dt * k1)
            k3 = rhs(y + 0.5 * dt * k2)
            k4 = rhs(y + dt * k3)
            y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            hist[i + 1] = y
        n = len(self.bodies)
        out = {"t": dt * np.arange(steps + 1),
               "theta": hist[:, :3 * n].reshape(-1, n, 3),
               "theta_d": hist[:, 3 * n:6 * n].reshape(-1, n, 3), "q": {},
               "qd": {}}
        i = 6 * n
        for key in ("q", "qd"):
            for k, b in enumerate(self.bodies):
                if self._is_beam(b):
                    out[key][k] = hist[:, i:i + 2 * b.n]
                    i += 2 * b.n
        return out

    def simulate_dt(self, theta0, theta_d0, t_end, dt, q0=None, qd0=None,
                    method="newmark", **kwargs):
        """Integrate with the DT-TMM (Sec. 8.13).

        :return: dict with t, theta (N, n, 3), q and residual
        :rtype: dict
        """
        lin = dttmm.make_linearizer(method, dt, **kwargs)
        q0, qd0 = self._default_q(q0, qd0)
        n = len(self.bodies)
        theta0 = np.asarray(theta0, float).reshape(n, 3)
        theta_d0 = np.asarray(theta_d0, float).reshape(n, 3)
        acc = self.accelerations(theta0, theta_d0, q0, qd0)
        ps, vs = self.kinematics(theta0, theta_d0, q0, qd0)
        states = {}
        for k, b in enumerate(self.bodies):
            thdd = acc["theta"][k]
            a_i = acc["p_in"][k]
            if self._is_beam(b):
                states[k] = {"theta": dttmm.Variable(theta0[k], theta_d0[k],
                                                     thdd),
                             "r_i": dttmm.Variable(ps[k], vs[k], a_i),
                             "q": dttmm.Variable(q0[k], qd0[k], acc["q"][k])}
            else:
                a = rigid.rotation_xyz(theta0[k])
                hm = dttmm.h_matrix(theta0[k])
                w = a @ (hm @ theta_d0[k])
                al = a @ (hm @ thdd + dttmm.h_dot_matrix(theta0[k],
                                                         theta_d0[k])
                          @ theta_d0[k])

                def point(pnt, k=k, a=a, w=w, al=al, a_i=a_i):
                    rp = a @ pnt
                    return dttmm.Variable(ps[k] + rp, vs[k] + np.cross(w, rp),
                                          a_i + np.cross(al, rp)
                                          + np.cross(w, np.cross(w, rp)))

                states[k] = dttmm.SpatialBodyState(dttmm.Variable(theta0[k],
                                                                  theta_d0[k],
                                                                  thdd),
                                                   dttmm.Variable(ps[k], vs[k],
                                                                  a_i),
                                                   point(b.l_ic),
                                                   point(b.l_io))
        springs = {}
        for k, h in enumerate(self.hinges):
            if self._kind(h) == "spring":
                d, dd, _ = self._spring_state(k, theta0, theta_d0, q0, qd0)
                e = 1e-4
                plus = self._advance(theta0, theta_d0, q0, qd0, acc, e)
                minus = self._advance(theta0, theta_d0, q0, qd0, acc, -e)
                ddd = (self.spring_deflection(k, *plus) - 2 * d
                       + self.spring_deflection(k, *minus)) / e ** 2
                springs[k] = dttmm.Variable(d, dd, ddd)
        ctx = {"states": states, "lin": lin, "springs": springs}
        steps = int(round(t_end / dt))
        th = np.empty((steps + 1, n, 3))
        qh = {k: np.empty((steps + 1, 2 * b.n)) for k, b
              in enumerate(self.bodies)
              if self._is_beam(b)}
        resid = np.zeros(steps + 1)
        th[0] = theta0
        for k in qh:
            qh[k][0] = q0[k]
        for i in range(steps):
            data = {k: dttmm.spatial_body_matrix_dt(b, states[k], lin,
                                                    b.mass * self.gravity)
                    for k, b in enumerate(self.bodies) if not self._is_beam(b)}
            res = self.deduce("dt", ctx)
            for k, b in enumerate(self.bodies):
                st = states[k]
                thk, p_i = res["theta"][k], res["p_in"][k]
                if self._is_beam(b):
                    lin.update(st["theta"], thk)
                    lin.update(st["r_i"], p_i)
                    lin.update(st["q"], res["q"][k])
                    qh[k][i + 1] = res["q"][k]
                else:
                    # The output-end history is not used by Eq. (7.174).
                    _, psi_ic, phi_ic = data[k]
                    lin.update(st.theta, thk)
                    lin.update(st.r_i, p_i)
                    lin.update(st.r_c, p_i + psi_ic @ thk + phi_ic)
                th[i + 1, k] = thk
            qs = [None if not self._is_beam(b) else res["q"][k] for k, b
                  in enumerate(self.bodies)]
            for k, var in springs.items():
                lin.update(var, self.spring_deflection(k, res["theta"], qs))
            resid[i + 1] = res["residual"]
        return {"t": dt * np.arange(steps + 1), "theta": th, "q": qh,
                "residual": resid}

    def _advance(self, theta, theta_d, q, qd, acc, e):
        """Return (theta, q) after a Taylor step e with the accelerations
        acc."""
        th = [t + e * td + 0.5 * e * e * a for t, td, a
              in zip(theta, theta_d, acc["theta"])]
        qq = [None if x is None else x + e * xd + 0.5 * e * e * xa
              for x, xd, xa in zip(q, qd, acc["q"])]
        return th, qq

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
