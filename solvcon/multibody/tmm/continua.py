# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Transfer matrix methods for nonlinear and two-dimensional systems (book Ch.
4, 15.11-15.15).

Contents:

* Bessel functions J_n, Y_n, I_n, K_n of integer order (numpy only), used
  by the disk and cylinder elements (Eq. 15.69-15.70, 15.97-15.98).
* ``magnus_transfer`` - transfer matrix of dZ/dr = A(r) Z with variable
  coefficients (fourth-order Magnus method, steps of e^{A h} as in
  Eq. 10.69-10.70).
* Sec. 4.2 incremental TMM for nonlinear spring-mass chains
  (``IncrementalChain``, Eq. 4.1-4.29), with harmonic balance of any order.
* Sec. 4.3 finite element TMM for thin plates (``plate_triangle``,
  ``FEPlateMesh``, ``FEPlateStrip``, Eq. 4.41-4.86).
* Sec. 4.4 finite element Riccati TMM with the Wilson-theta method
  (``FERiccatiDynamics``, Eq. 4.87-4.116) and its geometrically nonlinear
  form for von Karman plates with stiffeners (``VonKarmanPlateMesh``,
  ``RiccatiStripDynamics``; Examples 4.4-4.5).
* Sec. 4.5 Fourier series TMM for the thermal stress of a thin plate
  (Eq. 4.117-4.144) and Sec. 4.6 finite difference TMM (Eq. 4.145-4.160).
* Sec. 4.7 / 15.13 TMM for two-dimensional grid systems (``GridPlate``,
  Eq. 4.161-4.188, 15.71-15.91).
* Sec. 15.11 rectangular plate strips (Levy type), 15.12 disks, 15.14
  thick-walled cylinders, 15.15 thin-walled cylinders.

Sign convention of the strip methods (Sec. 4.3-4.4). A strip has the
dynamic stiffness relation S [d_I; d_O] = [f_I; f_O] where f are the nodal
forces that act on the strip (Eq. 4.69-4.72). The state vector at a section
is Z = [d; Q] with Q = f_O of the strip on the left = -f_I of the strip on
the right (Eq. 4.73-4.75). If an external nodal load P acts at a section,
then Q_right = Q_left - P. Eq. (4.82) prints +P; the test
``test_fe_tmm_static_vs_global_fem`` shows that -P is required when P is
the external load (book sign error).
"""

import math

import numpy as np

from . import riccati as _ric
from .linear import Chain, Element

# ----------------------------------------------------------------------
# Bessel functions of integer order (needed by Eq. 15.69-15.70, 15.97-15.98)
# ----------------------------------------------------------------------
_GL_X, _GL_W = np.polynomial.legendre.leggauss(48)


def _gauss_composite(fun, lo, hi, panels):
    """Composite 48-point Gauss-Legendre rule of fun on [lo, hi]."""
    edges = np.linspace(lo, hi, panels + 1)
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        t = 0.5 * (b - a) * _GL_X + 0.5 * (a + b)
        total += 0.5 * (b - a) * np.dot(_GL_W, fun(t))
    return total


def _scalar_map(fun, n, x):
    x_arr = np.asarray(x, dtype=float)
    out = np.array([fun(int(n), float(v)) for v in x_arr.ravel()])
    return out.reshape(x_arr.shape) if x_arr.shape else float(out[0])


def _trapezoid_periodic(fun, x):
    count = 2 * int(abs(x)) + 64
    tau = np.linspace(0.0, np.pi, count + 1)
    val = fun(tau)
    return (np.sum(val) - 0.5 * (val[0] + val[-1])) / count


def _bj(n, x):
    return _trapezoid_periodic(lambda t: np.cos(n * t - x * np.sin(t)), x)


def _bi(n, x):
    return _trapezoid_periodic(lambda t: np.exp(x * np.cos(t)) * np.cos(n * t),
                               x)


def _upper_limit(n, x, growth):
    t = 1.0
    for _ in range(60):
        t = math.asinh((60.0 + n * t) / x) if growth == "sinh" else math.acosh(
            max(1.0, (60.0 + n * t) / x))
    return max(t, 1.0)


def _by(n, x):
    first = _gauss_composite(lambda t: np.sin(x * np.sin(t) - n * t), 0.0,
                             np.pi, 2 + int(x) // 8) / np.pi
    upper = _upper_limit(n, x, "sinh")
    sign = -1.0 if n % 2 else 1.0
    second = _gauss_composite(
        lambda t: (np.exp(n * t - x * np.sinh(t))
                   + sign * np.exp(-n * t - x * np.sinh(t))),
        0.0, upper, 8) / np.pi
    return first - second


def _bk(n, x):
    upper = _upper_limit(n, x, "cosh")
    return _gauss_composite(lambda t: np.exp(-x * np.cosh(t)) * np.cosh(n * t),
                            0.0, upper, 8)


def bessel_j(n, x):
    """Bessel function of the first kind J_n(x), integer n, real x.

    Bessel's integral J_n(x) = (1/pi) int_0^pi cos(n t - x sin t) dt with the
    trapezoidal rule, which converges geometrically for this periodic
    integrand.

    :param n: Order.
    :type n: int
    :param x: Argument(s).
    :type x: float or numpy.ndarray
    :return: J_n(x).
    :rtype: float or numpy.ndarray
    """
    return _scalar_map(_bj, n, x)


def bessel_i(n, x):
    """Modified Bessel function I_n(x) = (1/pi) int_0^pi e^{x cos t} cos(n t)
    dt.

    :param n: Order.
    :type n: int
    :param x: Argument(s).
    :type x: float or numpy.ndarray
    :return: I_n(x).
    :rtype: float or numpy.ndarray
    """
    return _scalar_map(_bi, n, x)


def bessel_y(n, x):
    """Bessel function of the second kind Y_n(x), integer n, x > 0.

    Y_n(x) = (1/pi) int_0^pi sin(x sin t - n t) dt
    - (1/pi) int_0^inf (e^{n t} + (-1)^n e^{-n t}) e^{-x sinh t} dt,
    composite Gauss-Legendre quadrature.

    :param n: Order.
    :type n: int
    :param x: Argument(s), positive.
    :type x: float or numpy.ndarray
    :return: Y_n(x).
    :rtype: float or numpy.ndarray
    """
    return _scalar_map(_by, n, x)


def bessel_k(n, x):
    """Modified Bessel function K_n(x) = int_0^inf e^{-x cosh t} cosh(n t) dt,
    x > 0.

    :param n: Order.
    :type n: int
    :param x: Argument(s), positive.
    :type x: float or numpy.ndarray
    :return: K_n(x).
    :rtype: float or numpy.ndarray
    """
    return _scalar_map(_bk, n, x)


# ----------------------------------------------------------------------
# Variable coefficient transfer matrices (Eq. 10.68-10.70 applied stepwise)
# ----------------------------------------------------------------------
def magnus_transfer(a_of_r, r0, r1, n_steps=200):
    """Transfer matrix of dZ/dr = A(r) Z from r0 to r1 (fourth-order Magnus
    method).

    Each step uses Omega = h/2 (A1 + A2) + sqrt(3)/12 h^2 (A2 A1 - A1 A2)
    with A at the two Gauss points and U_step = e^Omega (Eq. 10.70). The
    global error is O(h^4).

    :param a_of_r: Callable r -> A(r) (square matrix).
    :type a_of_r: callable
    :param r0: Input coordinate.
    :type r0: float
    :param r1: Output coordinate.
    :type r1: float
    :param n_steps: Number of steps.
    :type n_steps: int
    :return: U(r1, r0).
    :rtype: numpy.ndarray
    """
    h = (r1 - r0) / n_steps
    off = math.sqrt(3.0) / 6.0 * h
    u = None
    for k in range(n_steps):
        mid = r0 + (k + 0.5) * h
        a1 = np.asarray(a_of_r(mid - off))
        a2 = np.asarray(a_of_r(mid + off))
        omega = (0.5 * h * (a1 + a2)
                 + math.sqrt(3.0) / 12.0 * h * h * (a2 @ a1 - a1 @ a2))
        step = _ric.expm(omega)
        u = step if u is None else step @ u
    return u


# ----------------------------------------------------------------------
# Sec. 4.2 Incremental transfer matrix method for nonlinear systems
# ----------------------------------------------------------------------
class NonlinearSpring:
    """Spring-damper of Fig. 4.2a with force k x + K(x) + c xdot (Eq. 4.1).

    :param k: Linear stiffness k_i.
    :type k: float
    :param c: Viscous damping c_i.
    :type c: float
    :param eps: Coefficient of the cubic nonlinearity K(x) = eps x^3 (Example
        4.1).
    :type eps: float
    :param force: Optional callable K(x) (replaces the cubic law).
    :type force: callable or None
    :param tangent: Optional callable dK/dx (needed with ``force``).
    :type tangent: callable or None
    """

    def __init__(self, k, c=0.0, eps=0.0, force=None, tangent=None):
        self.k = float(k)
        self.c = float(c)
        self.eps = float(eps)
        self._force = force
        self._tangent = tangent

    def nl_force(self, x):
        """Nonlinear part K(x)."""
        if self._force is not None:
            return self._force(x)
        return self.eps * x**3

    def nl_tangent(self, x):
        """dK/dx (Eq. 4.12b)."""
        if self._tangent is not None:
            return self._tangent(x)
        return 3.0 * self.eps * x**2


class IncrementalChain:
    """Incremental TMM for the n-DOF chain of Fig. 4.1 (Sec. 4.2, Eq.
    4.1-4.29).

    The chain is wall - spring 1 - mass 1 - spring 2 - ... - mass n - free.
    The excitation of mass i is F_i^c cos(Omega t) + F_i^s sin(Omega t)
    (Eq. 4.2). With tau = Omega t / l (l = 1: harmonic oscillations, l > 1:
    subharmonics of order 1/l, Sec. 4.2 item 2) every displacement and
    internal force is a Fourier series in tau truncated at ``n_harm``
    (Eq. 4.9-4.10). The incremental state vector of a connection point is
    (Eq. 4.13, 4.21)

        [da_0, dg_0, da_1, db_1, dg_1, dh_1, ..., da_N, db_N, dg_N, dh_N,
        dOmega, 1]

    where a, b are the cos and sin coefficients of x and g, h those of the
    internal force q_x. The force q_x is positive in compression
    (Eq. 4.1: q = -k (x_out - x_in) - ...), as in
    ``solvcon.multibody.tmm.library.Spring``.

    The Fourier integrals of Eq. (4.12a-b) and (4.20) are evaluated with the
    trapezoidal rule on ``n_samples`` points, which is exact for
    trigonometric polynomials of degree below ``n_samples``.

    :param masses: Masses m_i.
    :type masses: list
    :param springs: NonlinearSpring objects (spring i connects mass i - 1, or
        the wall, to mass i).
    :type springs: list
    :param forces: Complex numbers F_i^c - i F_i^s (or real F_i^c).
    :type forces: list
    :param n_harm: Highest harmonic N.
    :type n_harm: int
    :param sub_order: l of tau = Omega t / l.
    :type sub_order: int
    :param n_samples: Number of tau samples (default 8 (N + 1) + 32).
    :type n_samples: int or None
    """

    def __init__(self, masses, springs, forces, n_harm=3, sub_order=1,
                 n_samples=None):
        if len(masses) != len(springs) or len(forces) != len(masses):
            raise ValueError("one spring and one force per mass")
        self.masses = [float(m) for m in masses]
        self.springs = list(springs)
        self.forces = [complex(f) for f in forces]
        self.n_harm = int(n_harm)
        self.sub_order = int(sub_order)
        if self.sub_order > self.n_harm:
            raise ValueError("n_harm must include the forcing harmonic "
                             "sub_order")
        self.n_coef = 2 * self.n_harm + 1
        self.size = 2 * self.n_coef + 2
        ns = n_samples if n_samples is not None else 8 * (self.n_harm + 1) + 32
        self.tau = 2.0 * np.pi * np.arange(ns) / ns
        basis = [np.ones(ns)]
        for j in range(1, self.n_harm + 1):
            basis += [np.cos(j * self.tau), np.sin(j * self.tau)]
        self.phi = np.array(basis).T
        self.proj = 2.0 / ns * self.phi.T
        self.proj[0] *= 0.5
        self.dtau = np.zeros((self.n_coef, self.n_coef))
        self.j2 = np.zeros(self.n_coef)
        for j in range(1, self.n_harm + 1):
            ia, ib = 2 * j - 1, 2 * j
            self.dtau[ia, ib] = j
            self.dtau[ib, ia] = -j
            self.j2[ia] = self.j2[ib] = j * j
        # x and q coefficient positions in the incremental state vector (Eq.
        # 4.13)
        self.ix = ([0] + [4 * j - 2 + k for j in range(1, self.n_harm + 1)
                          for k in (0, 1)])
        self.iq = ([1] + [4 * j + k for j in range(1, self.n_harm + 1)
                          for k in (0, 1)])
        self.i_omega = self.size - 2
        self.i_one = self.size - 1
        self.n_points = 2 * len(self.masses) + 1
        self.x = np.zeros((self.n_points, self.n_coef))
        self.q = np.zeros((self.n_points, self.n_coef))
        self.omega = None

    # -- helpers --------------------------------------------------------
    def _load_vector(self, i):
        vec = np.zeros(self.n_coef)
        f = self.forces[i]
        vec[2 * self.sub_order - 1] = f.real
        vec[2 * self.sub_order] = -f.imag
        return vec

    def series(self, coef):
        """Values of a Fourier series at the tau samples."""
        return self.phi @ coef

    def coefficients(self, values):
        """Fourier coefficients (Eq. 4.12a normalization) of sampled values."""
        return self.proj @ values

    def spring_matrix(self, i, x_in, x_out, q_in, omega):
        """Incremental transfer matrix U_i = U_{i,O}^{-1} U_{i,I} of spring i
        (Eq. 4.11-4.16).

        :param i: Spring index.
        :type i: int
        :param x_in: Current x coefficients at the input end.
        :type x_in: numpy.ndarray
        :param x_out: Current x coefficients at the output end.
        :type x_out: numpy.ndarray
        :param q_in: Current q coefficients at the input end.
        :type q_in: numpy.ndarray
        :param omega: Current excitation frequency Omega.
        :type omega: float
        :return: Incremental transfer matrix of order ``size``.
        :rtype: numpy.ndarray
        """
        sp = self.springs[i]
        nu = omega / self.sub_order
        dlt = x_out - x_in
        dlt_t = self.series(dlt)
        kt = self.proj @ (sp.nl_tangent(dlt_t)[:, None] * self.phi)
        res = -q_in - (sp.k * dlt + self.coefficients(sp.nl_force(dlt_t))
                       + sp.c * nu * (self.dtau @ dlt))
        hmat = sp.k * np.eye(self.n_coef) + kt + sp.c * nu * self.dtau
        hinv = np.linalg.inv(hmat)
        u = np.eye(self.size)
        ix, iq = self.ix, self.iq
        u[np.ix_(ix, iq)] = -hinv
        u[ix,
          self.i_omega] = (-hinv @ (sp.c / self.sub_order * (self.dtau @ dlt)))
        u[ix, self.i_one] = hinv @ res
        return u

    def mass_matrix(self, i, x_in, q_in, q_out, omega):
        """Incremental transfer matrix of lumped mass i (Eq. 4.19-4.23).

        :param i: Mass index.
        :type i: int
        :param x_in: Current x coefficients.
        :type x_in: numpy.ndarray
        :param q_in: Current q coefficients at the input end.
        :type q_in: numpy.ndarray
        :param q_out: Current q coefficients at the output end.
        :type q_out: numpy.ndarray
        :param omega: Current excitation frequency Omega.
        :type omega: float
        :return: Incremental transfer matrix of order ``size``.
        :rtype: numpy.ndarray
        """
        m = self.masses[i]
        nu = omega / self.sub_order
        s_res = (-q_in + q_out - m * nu * nu * self.j2 * x_in
                 - self._load_vector(i))
        u = np.eye(self.size)
        ix, iq = self.ix, self.iq
        u[iq, ix] = m * nu * nu * self.j2
        u[iq, self.i_omega] = 2.0 * m * nu / self.sub_order * self.j2 * x_in
        u[iq, self.i_one] = -s_res
        return u

    def element_matrices(self, omega):
        """Incremental transfer matrices of all elements in transfer order (Eq.
        4.26)."""
        mats = []
        for i in range(len(self.masses)):
            p = 2 * i
            mats.append(self.spring_matrix(i, self.x[p], self.x[p + 1],
                                           self.q[p], omega))
            mats.append(self.mass_matrix(i, self.x[p + 1], self.q[p + 1],
                                         self.q[p + 2], omega))
        return mats

    def increment(self, omega, d_omega=0.0):
        """One incremental step: solve Eq. (4.27)-(4.29) and update the
        solution (Eq. 4.5).

        :param omega: Current Omega.
        :type omega: float
        :param d_omega: Prescribed increment dOmega.
        :type d_omega: float
        :return: Max norm of the increments of x and q.
        :rtype: float
        """
        mats = self.element_matrices(omega)
        u = np.eye(self.size)
        for mat in mats:
            u = mat @ u
        # input: da = 0 (wall, Eq. 4.28), dg unknown; output: dg = 0 (free end)
        iq = self.iq
        a = u[np.ix_(iq, iq)]
        rhs = -(u[iq, self.i_omega] * d_omega + u[iq, self.i_one])
        z = np.zeros(self.size)
        z[iq] = np.linalg.solve(a, rhs)
        z[self.i_omega] = d_omega
        z[self.i_one] = 1.0
        change = 0.0
        for p in range(self.n_points):
            if p > 0:
                z = mats[p - 1] @ z
            dx, dq = z[self.ix], z[self.iq]
            self.x[p] += dx
            self.q[p] += dq
            change = max(change, np.max(np.abs(dx)), np.max(np.abs(dq)))
        self.omega = omega + d_omega
        return change

    def solve(self, omega, tol=1e-12, max_iter=50, d_omega=0.0):
        """Improve the current solution until the increments are below tol
        (Sec. 4.2).

        The first increment may change Omega by ``d_omega`` (continuation in
        Omega, Eq. 4.29); the later increments keep Omega fixed.

        :param omega: Current Omega.
        :type omega: float
        :param tol: Absolute tolerance on the Fourier coefficient increments.
        :type tol: float
        :param max_iter: Maximum number of increments.
        :type max_iter: int
        :param d_omega: Frequency increment of the first step.
        :type d_omega: float
        :return: Number of increments used.
        :rtype: int
        :raises RuntimeError: If the iteration does not converge.
        """
        w = omega
        for it in range(max_iter):
            change = self.increment(w, d_omega if it == 0 else 0.0)
            w = self.omega
            if change < tol:
                return it + 1
        raise RuntimeError("incremental TMM did not converge")

    def mass_coefficients(self):
        """Fourier coefficients of the displacement of every mass, shape (n, 2N
        + 1)."""
        return np.array([self.x[2 * i + 1] for i in range(len(self.masses))])

    def amplitudes(self):
        """Amplitude sqrt(a_j^2 + b_j^2) of harmonic j of every mass, shape (n,
        N + 1)."""
        c = self.mass_coefficients()
        amp = np.zeros((len(self.masses), self.n_harm + 1))
        amp[:, 0] = np.abs(c[:, 0])
        for j in range(1, self.n_harm + 1):
            amp[:, j] = np.hypot(c[:, 2 * j - 1], c[:, 2 * j])
        return amp


# ----------------------------------------------------------------------
# Sec. 4.3 Finite element TMM for thin plates
# ----------------------------------------------------------------------
_TRI_U, _TRI_W = np.polynomial.legendre.leggauss(6)


def _tri_points():
    """Collapsed Gauss rule on the unit triangle (exact for degree <= 10)."""
    u = 0.5 * (_TRI_U + 1.0)
    w = 0.5 * _TRI_W
    pts, wts = [], []
    for ui, wi in zip(u, w):
        for vj, wj in zip(u, w):
            pts.append((ui * (1.0 - vj), ui * vj))
            wts.append(wi * wj * ui)
    return np.array(pts), np.array(wts)


_TRI_PTS, _TRI_WTS = _tri_points()


def _poly(x, y):
    """d, d_x, d_y, d_xx, d_yy, d_xy of Eq. (4.45)."""
    z = 0.0 * x
    one = z + 1.0
    d = np.array([one, x, y, x * x, x * y, y * y, x**3, x * x * y + x * y * y,
                  y**3])
    dx = np.array([z, one, z, 2 * x, y, z, 3 * x * x, 2 * x * y + y * y, z])
    dy = np.array([z, z, one, z, x, 2 * y, z, x * x + 2 * x * y, 3 * y * y])
    dxx = np.array([z, z, z, 2 * one, z, z, 6 * x, 2 * y, z])
    dyy = np.array([z, z, z, z, z, 2 * one, z, 2 * x, 6 * y])
    dxy = np.array([z, z, z, z, one, z, z, 2 * x + 2 * y, z])
    return d, dx, dy, dxx, dyy, dxy


def plate_triangle(y2, x3, y3, d_flex, mu, rho_h):
    """Stiffness and consistent mass of the 9-DOF thin plate triangle (Eq.
    4.41-4.59).

    Local frame of Fig. 4.6: node 1 at (0, 0), node 2 at (0, y2) and node 3
    at (x3, y3). Nodal DOFs [w, theta_x = dw/dy, theta_y = -dw/dx]
    (Eq. 4.41). Displacement field Eq. (4.44); K = int B^T D B dA with the
    curvatures [w_xx, w_yy, 2 w_xy] (Eq. 4.52-4.59) and M = rho h int N^T N dA
    (Eq. 4.57).

    :param y2: Local y of node 2.
    :type y2: float
    :param x3: Local x of node 3.
    :type x3: float
    :param y3: Local y of node 3.
    :type y3: float
    :param d_flex: Flexural rigidity D = E h^3 / (12 (1 - mu^2)).
    :type d_flex: float
    :param mu: Poisson's ratio.
    :type mu: float
    :param rho_h: Mass per unit area rho h.
    :type rho_h: float
    :return: (K, M), each 9 x 9.
    :rtype: tuple
    """
    nodes = [(0.0, 0.0), (0.0, y2), (x3, y3)]
    rows = []
    for xn, yn in nodes:
        d, dx, dy, _, _, _ = _poly(np.array(xn), np.array(yn))
        rows += [d, dy, -dx]
    n_inv = np.linalg.inv(np.array(rows, dtype=float))
    area = 0.5 * abs(x3 * y2)
    p1, p2, p3 = (np.array(p) for p in nodes)
    xs = p1[0] + _TRI_PTS[:, 0] * (p2 - p1)[0] + _TRI_PTS[:, 1] * (p3 - p1)[0]
    ys = p1[1] + _TRI_PTS[:, 0] * (p2 - p1)[1] + _TRI_PTS[:, 1] * (p3 - p1)[1]
    wts = _TRI_WTS * 2.0 * area
    d, _, _, dxx, dyy, dxy = _poly(xs, ys)
    dmat = d_flex * np.array([[1.0, mu, 0.0], [mu, 1.0, 0.0],
                              [0.0, 0.0, 0.5 * (1.0 - mu)]])
    k = np.zeros((9, 9))
    m = np.zeros((9, 9))
    for q in range(len(wts)):
        b = np.array([dxx[:, q], dyy[:, q], 2.0 * dxy[:, q]]) @ n_inv
        nvec = d[:, q] @ n_inv
        k += wts[q] * b.T @ dmat @ b
        m += wts[q] * rho_h * np.outer(nvec, nvec)
    return k, m


_BC_EDGE = {"S": None, "C": [0, 1, 2], "F": []}


class FEPlateMesh:
    """Rectangular thin plate divided into strips along x (Fig. 4.7, Sec.
    4.3.4).

    The plate [0, a] x [0, b] has ``n_strips`` strips of width a / n_strips;
    each strip has ``n_cells`` rectangular cells along y, each cut into two
    right triangles with the local y axis along a
    cell side (Fig. 4.7b): (A, D, C) and (C, B, A) with the right angle at
    node 2. The split (A, D, B) with node 3 on the local x axis makes n of
    Eq. (4.49) singular for square cells (the side 2-3 lies on x + y = const,
    where x^2 y + x y^2 is not independent). The
    section DOFs are [w, theta_x, theta_y] of the nodes j = 0 .. n_cells
    (Eq. 4.70). Supports on the edges y = 0 and y = b remove DOFs
    (Sec. 4.3.4): 'S' simply supported (w = theta_y = 0), 'C' clamped,
    'F' free.

    :param a: Length in the transfer direction x.
    :type a: float
    :param b: Width in y.
    :type b: float
    :param n_strips: Number of strips n.
    :type n_strips: int
    :param n_cells: Number of cells along y.
    :type n_cells: int
    :param d_flex: Flexural rigidity D.
    :type d_flex: float
    :param mu: Poisson's ratio.
    :type mu: float
    :param rho_h: Mass per unit area.
    :type rho_h: float
    :param bc_y: Supports at (y = 0, y = b), letters 'S', 'C', 'F'.
    :type bc_y: tuple
    """

    def __init__(self, a, b, n_strips, n_cells, d_flex, mu, rho_h,
                 bc_y=("S", "S")):
        self.a, self.b = float(a), float(b)
        self.n_strips, self.n_cells = int(n_strips), int(n_cells)
        dx = self.a / self.n_strips
        dy = self.b / self.n_cells
        k_loc, m_loc = plate_triangle(dy, dx, dy, d_flex, mu, rho_h)
        flip = np.diag([1.0, -1.0, -1.0] * 3)
        k2, m2 = flip @ k_loc @ flip, flip @ m_loc @ flip
        nn = self.n_cells + 1
        size = 6 * nn
        ks = np.zeros((size, size))
        ms = np.zeros((size, size))

        def node(side, j):
            return 3 * (side * nn + j)

        for c in range(self.n_cells):
            a_, d_, b_, c_ = node(0, c), node(0, c + 1), node(1, c), node(1,
                                                                          c
                                                                          + 1)
            for kk, mm, tri in ((k_loc, m_loc, (a_, d_, c_)),
                                (k2, m2, (c_, b_, a_))):
                dofs = [t + q for t in tri for q in range(3)]
                ks[np.ix_(dofs, dofs)] += kk
                ms[np.ix_(dofs, dofs)] += mm
        removed = set()
        for j, bc in ((0, bc_y[0]), (self.n_cells, bc_y[1])):
            fixed = [0, 2] if bc == "S" else _BC_EDGE[bc]
            removed |= {3 * j + q for q in fixed}
        self.section_dofs = [q for q in range(3 * nn) if q not in removed]
        keep = self.section_dofs + [3 * nn + q for q in self.section_dofs]
        self.k_strip = ks[np.ix_(keep, keep)]
        self.m_strip = ms[np.ix_(keep, keep)]
        self.n_dof = len(self.section_dofs)

    def section_zero(self, bc):
        """Indices of zero state components [d; Q] at an x edge ('S', 'C',
        'F').

        :param bc: Support letter.
        :type bc: str
        :return: Sorted indices into the section state vector.
        :rtype: list
        """
        nd = self.n_dof
        zero = []
        for pos, dof in enumerate(self.section_dofs):
            q = dof % 3
            if bc == "C" or (bc == "S" and q in (0, 1)):
                zero.append(pos)
            if bc == "F" or (bc == "S" and q == 2):
                zero.append(nd + pos)
        return sorted(zero)

    def strips(self):
        """List of FEPlateStrip elements in transfer order (all equal, Example
        4.2)."""
        return [FEPlateStrip(self.k_strip, self.m_strip) for _
                in range(self.n_strips)]

    def chain(self, bc_x=("S", "S"), loads=None):
        """linear.Chain of the strips with x-edge supports (Eq. 4.77-4.81).

        :param bc_x: Supports at (x = 0, x = a).
        :type bc_x: tuple
        :param loads: Optional (n_strips + 1, n_dof) external nodal loads at
            the sections (Eq. 4.82-4.86).
        :type loads: numpy.ndarray or None
        :return: Chain.
        :rtype: solvcon.multibody.tmm.linear.Chain
        """
        els = []
        strips = self.strips()
        for i in range(self.n_strips + 1):
            if loads is not None:
                els.append(SectionLoad(loads[i]))
            if i < self.n_strips:
                els.append(strips[i])
        return Chain(els, self.section_zero(bc_x[0]),
                     self.section_zero(bc_x[1]))

    def global_matrices(self, bc_x=("S", "S")):
        """Global FEM stiffness and mass with all supports (Eq. 4.63-4.66).

        :param bc_x: Supports at (x = 0, x = a).
        :type bc_x: tuple
        :return: (K, M, free) with free the kept indices of the section-major
            DOF vector of length (n_strips + 1) n_dof.
        :rtype: tuple
        """
        nd = self.n_dof
        n_tot = (self.n_strips + 1) * nd
        k = np.zeros((n_tot, n_tot))
        m = np.zeros((n_tot, n_tot))
        for i in range(self.n_strips):
            idx = list(range(i * nd, (i + 2) * nd))
            k[np.ix_(idx, idx)] += self.k_strip
            m[np.ix_(idx, idx)] += self.m_strip
        fixed = set()
        for sec, bc in ((0, bc_x[0]), (self.n_strips, bc_x[1])):
            for pos in self.section_zero(bc):
                if pos < nd:
                    fixed.add(sec * nd + pos)
        free = [q for q in range(n_tot) if q not in fixed]
        return k[np.ix_(free, free)], m[np.ix_(free, free)], free

    def consistent_pressure(self, p):
        """Section loads of a uniform pressure p, consistent nodal forces int
        N^T p dA (Eq. 4.60).

        :param p: Pressure.
        :type p: float
        :return: Array (n_strips + 1, n_dof).
        :rtype: numpy.ndarray
        """
        dx = self.a / self.n_strips
        dy = self.b / self.n_cells
        _, m_unit = plate_triangle(dy, dx, dy, 1.0, 0.3, 1.0)
        # int N dA: the first row of the triangle uses w = 1 basis; use the
        # mass with rho h = 1 applied to the rigid translation w = 1.
        rigid = np.array([1.0, 0.0, 0.0] * 3)
        f_loc = m_unit @ rigid * p
        flip = np.diag([1.0, -1.0, -1.0] * 3)
        f2 = flip @ f_loc
        nn = self.n_cells + 1
        strip_f = np.zeros(6 * nn)
        for c in range(self.n_cells):
            a_, d_, b_, c_ = 3 * c, 3 * (c + 1), 3 * (nn + c), 3 * (nn + c + 1)
            for ff, tri in ((f_loc, (a_, d_, c_)), (f2, (c_, b_, a_))):
                for t_i, t in enumerate(tri):
                    strip_f[t:t + 3] += ff[3 * t_i:3 * t_i + 3]
        keep_l = self.section_dofs
        loads = np.zeros((self.n_strips + 1, self.n_dof))
        for i in range(self.n_strips):
            loads[i] += strip_f[keep_l]
            loads[i + 1] += strip_f[[3 * nn + q for q in keep_l]]
        return loads


def strip_transfer(s_mat, n_dof):
    """Transfer matrix of a strip from its dynamic stiffness matrix (Eq.
    4.72-4.74).

    U11 = -S12^{-1} S11, U12 = -S12^{-1}, U21 = S21 - S22 S12^{-1} S11,
    U22 = -S22 S12^{-1}; state [d; Q] with the input force -Q_{i,i-1}.

    :param s_mat: Strip dynamic stiffness S = K - omega^2 M of order 2 n_dof.
    :type s_mat: numpy.ndarray
    :param n_dof: Number of DOFs of one section.
    :type n_dof: int
    :return: Transfer matrix of order 2 n_dof.
    :rtype: numpy.ndarray
    """
    n = n_dof
    s11, s12 = s_mat[:n, :n], s_mat[:n, n:]
    s21, s22 = s_mat[n:, :n], s_mat[n:, n:]
    s12_inv = np.linalg.inv(s12)
    return np.block([[-s12_inv @ s11, -s12_inv],
                     [s21 - s22 @ s12_inv @ s11, -s22 @ s12_inv]])


class FEPlateStrip(Element):
    """Finite element strip with dynamic stiffness K + s^2 M (Eq. 4.69-4.74).

    :param k_strip: Strip stiffness matrix [input DOFs, output DOFs].
    :type k_strip: numpy.ndarray
    :param m_strip: Strip mass matrix.
    :type m_strip: numpy.ndarray
    """

    def __init__(self, k_strip, m_strip):
        self.k_strip = np.asarray(k_strip, dtype=float)
        self.m_strip = np.asarray(m_strip, dtype=float)
        self.n_dof = self.k_strip.shape[0] // 2
        self.size = 2 * self.n_dof

    def transfer(self, s, eta=1.0):
        """Transfer matrix U_i of Eq. (4.73)-(4.74) at the Laplace variable s.

        :param s: Laplace variable (i omega).
        :type s: complex
        :param eta: Unused.
        :type eta: float
        :return: U.
        :rtype: numpy.ndarray
        """
        dtype = complex if np.iscomplexobj(s) else float
        smat = (self.k_strip + (s * s) * self.m_strip).astype(dtype)
        return strip_transfer(smat, self.n_dof)


class SectionLoad(Element):
    """External nodal load P at a section: d unchanged, Q -> Q - P (Eq. 4.82
    with the sign fix).

    :param load: External nodal loads of the section DOFs.
    :type load: numpy.ndarray
    """

    def __init__(self, load):
        self.load = np.asarray(load, dtype=complex)
        self.size = 2 * len(self.load)

    def transfer(self, s, eta=1.0):
        """Identity."""
        return np.eye(self.size, dtype=complex)

    def forcing(self, s):
        """Load column [0; -P]."""
        return np.concatenate([np.zeros(len(self.load), dtype=complex),
                               -self.load])


# ----------------------------------------------------------------------
# Sec. 4.4 Finite element Riccati TMM with the Wilson-theta method
# ----------------------------------------------------------------------
class _MatrixElement:
    """Element with a stored transfer matrix (input of
    solvcon.multibody.tmm.riccati)."""

    def __init__(self, u, f):
        self.u = u
        self.f = f
        self.size = u.shape[0]

    def transfer(self, s, eta=1.0):
        return self.u

    def forcing(self, s):
        return self.f


class _ChainLike:
    def __init__(self, elements, input_zero, output_zero):
        self.elements = elements
        self.size = elements[0].size
        self.input_zero = list(input_zero)
        self.output_zero = list(output_zero)


class FERiccatiDynamics:
    """Transient response of a strip model by the FE Riccati TMM (Sec. 4.4, Eq.
    4.87-4.116).

    The model has n strips with matrices K_i, M_i, C_i (order 2 n_dof) and
    n + 1 sections. A nodal nonlinear restoring force F_nl(d) acts at every
    section (cubic springs k3 w^3 on the w DOFs in the tests). Every time
    step uses the Wilson-theta method (Eq. 4.91-4.105) and the modified
    Newton-Raphson iteration of Sec. 4.4.2.3 with the tangent stiffness at
    time t. The linear equations H dv = r of each iteration are solved by
    the Riccati transfer (Eq. 4.106-4.115, ``solvcon.multibody.tmm.riccati``)
    or, for comparison, by a direct solve of the global matrix
    (``solver='global'``).

    Residual. The book prints the unbalanced force (Eq. 4.116) with the
    linear term K_T dv, which makes every correction after the first one
    zero. This class uses the residual of the nonlinear equation
    R(t + theta dt) - M a - C v - F(v), which is the modified Newton-Raphson
    method that Sec. 4.4.2.3 describes in words.

    :param mesh: FEPlateMesh (strip matrices and section DOFs).
    :type mesh: FEPlateMesh
    :param bc_x: Supports at (x = 0, x = a).
    :type bc_x: tuple
    :param k3: Coefficient of the cubic node springs on w.
    :type k3: float
    :param alpha_c: Mass-proportional damping C = alpha_c M.
    :type alpha_c: float
    :param theta: Wilson parameter theta >= 1.37 for unconditional stability.
    :type theta: float
    """

    def __init__(self, mesh, bc_x=("C", "C"), k3=0.0, alpha_c=0.0, theta=1.4):
        self.mesh = mesh
        self.nd = mesh.n_dof
        self.ns = mesh.n_strips + 1
        self.bc_x = bc_x
        self.k3 = float(k3)
        self.alpha_c = float(alpha_c)
        self.theta = float(theta)
        self.zero_in = mesh.section_zero(bc_x[0])
        self.zero_out = mesh.section_zero(bc_x[1])
        self.w_mask = np.array([dof % 3 == 0 for dof in mesh.section_dofs])
        k, m, free = mesh.global_matrices(bc_x)
        self.free = free
        n_tot = self.ns * self.nd
        self.k_full = np.zeros((n_tot, n_tot))
        self.m_full = np.zeros((n_tot, n_tot))
        self.k_full[np.ix_(free, free)] = k
        self.m_full[np.ix_(free, free)] = m
        self.fixed_mask = np.ones(n_tot, dtype=bool)
        self.fixed_mask[free] = False

    def nonlinear_force(self, v):
        """Global nodal force K_lin v + F_nl(v) and the nonlinear tangent
        diagonal."""
        w = v.reshape(self.ns, self.nd)
        f_nl = np.where(self.w_mask[None, :], self.k3 * w**3, 0.0).ravel()
        t_nl = np.where(self.w_mask[None, :], 3.0 * self.k3 * w**2,
                        0.0).ravel()
        return self.k_full @ v + f_nl, t_nl

    def _solve_riccati(self, coef_m, coef_c, t_nl, rhs):
        mesh = self.mesh
        nd = self.nd
        h_strip = (mesh.k_strip
                   + (coef_m + coef_c * self.alpha_c) * mesh.m_strip)
        u_strip = strip_transfer(h_strip, nd)
        els = []
        tn = t_nl.reshape(self.ns, nd)
        rr = rhs.reshape(self.ns, nd)
        zeros = np.zeros(2 * nd)
        for i in range(self.ns):
            u_sec = np.eye(2 * nd)
            u_sec[nd:, :nd] = np.diag(tn[i])
            f_sec = np.concatenate([np.zeros(nd), -rr[i]])
            els.append(_MatrixElement(u_sec, f_sec))
            if i < self.ns - 1:
                els.append(_MatrixElement(u_strip, zeros))
        chain = _ChainLike(els, self.zero_in, self.zero_out)
        res = _ric.riccati_steady_state(chain, 0.0, loads=[e.f for e in els])
        return np.concatenate([np.real(res["inputs"][2 * i][:nd])
                               for i in range(self.ns)])

    def _solve_global(self, coef_m, coef_c, t_nl, rhs):
        h = (self.k_full + (coef_m + coef_c * self.alpha_c) * self.m_full
             + np.diag(t_nl))
        out = np.zeros_like(rhs)
        free = self.free
        out[free] = np.linalg.solve(h[np.ix_(free, free)], rhs[free])
        return out

    def run(self, load, dt, n_steps, solver="riccati", tol=1e-10, max_iter=30):
        """Integrate from rest under the section loads load(t) (Eq.
        4.87-4.105).

        :param load: Callable t -> array (n_strips + 1, n_dof) of external
            nodal loads.
        :type load: callable
        :param dt: Time step.
        :type dt: float
        :param n_steps: Number of steps.
        :type n_steps: int
        :param solver: 'riccati' (Eq. 4.110-4.115) or 'global' (Eq. 4.100).
        :type solver: str
        :param tol: Relative tolerance of the unbalanced force.
        :type tol: float
        :param max_iter: Maximum number of Newton iterations per step.
        :type max_iter: int
        :return: dict with ``t``, ``v`` (n_steps + 1, n_dof total) and
            ``iterations``.
        :rtype: dict
        """
        solve = (self._solve_riccati if solver == "riccati"
                 else self._solve_global)
        n_tot = self.ns * self.nd
        cm = self.alpha_c * self.m_full
        v = np.zeros(n_tot)
        vd = np.zeros(n_tot)
        f_int, t_nl = self.nonlinear_force(v)
        r0 = np.asarray(load(0.0), dtype=float).ravel()
        r0[self.fixed_mask] = 0.0
        va = np.zeros(n_tot)
        free = self.free
        va[free] = np.linalg.solve(self.m_full[np.ix_(free, free)],
                                   (r0 - cm @ vd - f_int)[free])
        tau = self.theta * dt
        c1, c2 = 6.0 / tau**2, 3.0 / tau
        hist = [v.copy()]
        iters = []
        for step in range(n_steps):
            t = step * dt
            r_next = np.asarray(load(t + dt), dtype=float).ravel()
            r_now = np.asarray(load(t), dtype=float).ravel()
            r_tau = r_now + self.theta * (r_next - r_now)
            r_tau[self.fixed_mask] = 0.0
            _, t_nl = self.nonlinear_force(v)
            dv = np.zeros(n_tot)
            for it in range(max_iter):
                acc = c1 * dv - 6.0 / tau * vd - 2.0 * va
                vel = c2 * dv - 2.0 * vd - 0.5 * tau * va
                f_int, _ = self.nonlinear_force(v + dv)
                res = r_tau - self.m_full @ acc - cm @ vel - f_int
                res[self.fixed_mask] = 0.0
                scale = max(np.linalg.norm(r_tau), np.linalg.norm(f_int),
                            1e-300)
                if np.linalg.norm(res) <= tol * scale and it > 0:
                    break
                dv = dv + solve(c1, c2, t_nl, res)
            iters.append(it)
            a_new = (6.0 / (self.theta**3 * dt * dt) * dv
                     - 6.0 / (self.theta**2 * dt) * vd
                     + (1.0 - 3.0 / self.theta) * va)
            v = v + dt * vd + dt * dt / 6.0 * (a_new + 2.0 * va)
            vd = vd + 0.5 * dt * (a_new + va)
            va = a_new
            hist.append(v.copy())
        return {"t": dt * np.arange(n_steps + 1), "v": np.array(hist),
                "iterations": iters}


# ----------------------------------------------------------------------
# Sec. 4.5 Fourier series TMM (thermal stress of a thin plate)
# ----------------------------------------------------------------------
def fourier_system_matrix(beta, e_mod, mu):
    """System matrix A of the m-th Fourier term, state [u_m, -v_m, sigma_ym,
    tau_xym] (Eq. 4.134).

    :param beta: beta = m pi / L.
    :type beta: float
    :param e_mod: Young's modulus E.
    :type e_mod: float
    :param mu: Poisson's ratio.
    :type mu: float
    :return: 4 x 4 matrix of dZ/dy = A Z + F (Eq. 4.133).
    :rtype: numpy.ndarray
    """
    return np.array([
        [0.0, beta, 0.0, 2.0 * (1.0 + mu) / e_mod],
        [-mu * beta, 0.0, -(1.0 - mu * mu) / e_mod, 0.0],
        [0.0, 0.0, 0.0, beta],
        [e_mod * beta**2, 0.0, -mu * beta, 0.0],
    ])


def fourier_transfer(beta, y, e_mod, mu):
    """Closed form U_m(y) = e^{A y} (Eq. 4.139), state [u_m, -v_m, sigma_ym,
    tau_xym] (Eq. 4.140).

    The matrix is symmetric about the minor diagonal (persymmetric) as the
    book states after Eq. (4.133); the entries not printed follow from
    U_ij = U_{5-j, 5-i}.

    :param beta: beta = m pi / L.
    :type beta: float
    :param y: Transfer coordinate.
    :type y: float
    :param e_mod: Young's modulus.
    :type e_mod: float
    :param mu: Poisson's ratio.
    :type mu: float
    :return: U_m(y).
    :rtype: numpy.ndarray
    """
    by = beta * y
    sh, ch = math.sinh(by), math.cosh(by)
    p, q = 0.5 * (1.0 + mu), 0.5 * (1.0 - mu)
    r = (3.0 - mu) / (1.0 + mu) / beta
    u = np.zeros((4, 4))
    u[0, 0] = ch + p * by * sh
    u[0, 1] = p * by * ch + q * sh
    p2 = (1.0 + mu) ** 2 / (2.0 * e_mod)
    u[0, 2] = -p2 * y * sh
    u[0, 3] = p2 * (r * sh + y * ch)
    u[1, 0] = q * sh - p * by * ch
    u[1, 1] = ch - p * by * sh
    u[1, 2] = p2 * (y * ch - r * sh)
    u[2, 0] = 0.5 * e_mod * beta * by * sh
    u[2, 1] = 0.5 * e_mod * beta * (by * ch - sh)
    u[3, 0] = 0.5 * e_mod * beta * (by * ch + sh)
    for i in range(4):
        for j in range(4):
            if u[i, j] == 0.0 and u[3 - j, 3 - i] != 0.0:
                u[i, j] = u[3 - j, 3 - i]
    return u


def _linear_load_transfer(a, f0, f1, y):
    """Extended transfer of dZ/dy = A Z + f0 + f1 y (Eq. 4.141 for a linear
    load)."""
    n = a.shape[0]
    aug = np.zeros((n + 2, n + 2))
    aug[:n, :n] = a
    aug[:n, n] = f1
    aug[:n, n + 1] = f0
    aug[n, n + 1] = 1.0
    ext = _ric.expm(aug * y)
    return ext[:n, :n], ext[:n, n + 1]


def fourier_strip_solution(beta, height, e_mod, mu, alpha=0.0, t0=0.0, kt=0.0,
                           p_edge=0.0, ys=None):
    """Fourier term m of the thermal stress problem of Fig. 4.17 (Eq.
    4.131-4.144).

    Strip 0 <= y <= height. Temperature term T_m(y) = T_0m + K_Tm y
    (Eq. 4.142), load column F_m of Eq. (4.143). Boundary conditions:
    sigma_ym = p_edge, tau_xym = 0 at y = 0 and sigma_ym = tau_xym = 0 at
    y = height. Returns the states [u_m, v_m, sigma_ym, tau_xym] (v with its
    own sign) and sigma_xm of Eq. (4.144).

    :param beta: m pi / L.
    :type beta: float
    :param height: Strip height.
    :type height: float
    :param e_mod: Young's modulus.
    :type e_mod: float
    :param mu: Poisson's ratio.
    :type mu: float
    :param alpha: Thermal expansion coefficient.
    :type alpha: float
    :param t0: T_0m.
    :type t0: float
    :param kt: K_Tm.
    :type kt: float
    :param p_edge: sigma_ym at y = 0.
    :type p_edge: float
    :param ys: Output coordinates (default 11 points).
    :type ys: numpy.ndarray or None
    :return: (states (len(ys), 4), sigma_x (len(ys))).
    :rtype: tuple
    """
    a = fourier_system_matrix(beta, e_mod, mu)
    f1 = alpha * kt * np.array([0.0, -(1.0 + mu), 0.0, e_mod * beta])
    f0 = alpha * t0 * np.array([0.0, -(1.0 + mu), 0.0, e_mod * beta])
    u_h, f_h = _linear_load_transfer(a, f0, f1, height)
    z0 = np.zeros(4)
    z0[2] = p_edge
    known = u_h[2:, 2] * p_edge + f_h[2:]
    z0[:2] = np.linalg.solve(u_h[2:, :2], -known)
    ys = (np.linspace(0.0, height, 11) if ys is None
          else np.asarray(ys, dtype=float))
    states = []
    for y in ys:
        u_y, f_y = _linear_load_transfer(a, f0, f1, y)
        states.append(u_y @ z0 + f_y)
    states = np.array(states)
    states[:, 1] *= -1.0
    sigma_x = (-e_mod * beta * states[:, 0] + mu * states[:, 2]
               - alpha * e_mod * (t0 + kt * ys))
    return states, sigma_x


# ----------------------------------------------------------------------
# Sec. 4.6 Finite difference TMM
# ----------------------------------------------------------------------
def fd_difference_matrices(n, dx):
    """First and second difference matrices D1, D2 on n + 1 points (Eq.
    4.146-4.154).

    :param n: Number of intervals.
    :type n: int
    :param dx: Spacing.
    :type dx: float
    :return: (D1, D2), each (n + 1) x (n + 1).
    :rtype: tuple
    """
    d1 = np.zeros((n + 1, n + 1))
    d2 = np.zeros((n + 1, n + 1))
    d1[0, :2] = [-2.0, 2.0]
    d1[n, n - 1:] = [-2.0, 2.0]
    d2[0, :4] = [2.0, -5.0, 4.0, -1.0]
    d2[n, n - 3:] = [-1.0, 4.0, -5.0, 2.0]
    for k in range(1, n):
        d1[k, k - 1:k + 2] = [-1.0, 0.0, 1.0]
        d2[k, k - 1:k + 2] = [1.0, -2.0, 1.0]
    return d1 / (2.0 * dx), d2 / dx**2


def fd_system_matrix(n, length, e_mod, mu, scheme="book"):
    """System matrix and thermal load matrix of the finite difference TMM (Eq.
    4.155-4.157).

    ``scheme='book'``: state [u; v; sigma_y; tau_xy] on all n + 1 lines with
    D1, D2 of Eq. (4.153)-(4.154). ``scheme='symmetric'``: the edges x = 0
    and x = L carry the conditions of the Fourier solution of Sec. 4.5
    (v = sigma_y = 0, u and tau_xy even), imposed by mirror points; v and
    sigma_y are kept on the n - 1 interior lines only, so the state has
    2 (n + 1) + 2 (n - 1) entries [u (n+1); v (n-1); sigma_y (n-1); tau (n+1)].

    :param n: Number of intervals.
    :type n: int
    :param length: Width L.
    :type length: float
    :param e_mod: Young's modulus.
    :type e_mod: float
    :param mu: Poisson's ratio.
    :type mu: float
    :param scheme: 'book' or 'symmetric'.
    :type scheme: str
    :return: (A, G) with dZ/dy = A Z + alpha G t, t the line temperatures
        (n + 1 values, Eq. 4.156).
    :rtype: tuple
    """
    dx = length / n
    if scheme == "book":
        d1, d2 = fd_difference_matrices(n, dx)
        z = np.zeros((n + 1, n + 1))
        i1 = np.eye(n + 1)
        a = np.block([
            [z, -d1, z, 2.0 * (1.0 + mu) / e_mod * i1],
            [-mu * d1, z, (1.0 - mu * mu) / e_mod * i1, z],
            [z, z, z, -d1],
            [-e_mod * d2, z, -mu * d1, z],
        ])
        g = np.vstack([z, (1.0 + mu) * i1, z, e_mod * d1])
        return a, g
    ne, no = n + 1, n - 1
    # odd function g given on interior points 1..n-1, derivative on all points
    d_odd = np.zeros((ne, no))
    for k in range(ne):
        for s_, c_ in ((k + 1, 1.0), (k - 1, -1.0)):
            if 1 <= s_ <= n - 1:
                d_odd[k, s_ - 1] += c_ / (2.0 * dx)
            elif s_ == -1:
                d_odd[k, 0] += c_ * (-1.0) / (2.0 * dx)
            elif s_ == n + 1:
                d_odd[k, n - 2] += c_ * (-1.0) / (2.0 * dx)
    # even function on all points, first derivative on interior points
    d_even = np.zeros((no, ne))
    for k in range(1, n):
        d_even[k - 1, k - 1] = -1.0 / (2.0 * dx)
        d_even[k - 1, k + 1] = 1.0 / (2.0 * dx)
    d2_even = np.zeros((ne, ne))
    for k in range(ne):
        for s_, c_ in ((k - 1, 1.0), (k, -2.0), (k + 1, 1.0)):
            s_ = -s_ if s_ < 0 else (2 * n - s_ if s_ > n else s_)
            d2_even[k, s_] += c_ / dx**2
    a = np.zeros((2 * ne + 2 * no, 2 * ne + 2 * no))
    iu, iv = slice(0, ne), slice(ne, ne + no)
    isg, it = slice(ne + no, ne + 2 * no), slice(ne + 2 * no, 2 * ne + 2 * no)
    a[iu, iv] = -d_odd
    a[iu, it] = 2.0 * (1.0 + mu) / e_mod * np.eye(ne)
    a[iv, iu] = -mu * d_even
    a[iv, isg] = (1.0 - mu * mu) / e_mod * np.eye(no)
    a[isg, it] = -d_even
    a[it, iu] = -e_mod * d2_even
    a[it, isg] = -mu * d_odd
    # temperature is odd (sine series): interior values carry it
    g = np.zeros((2 * ne + 2 * no, ne))
    g[iv, 1:n] = (1.0 + mu) * np.eye(no)
    g[it, 1:n] = e_mod * d_odd
    return a, g


def fd_strip_solution(n, length, height, e_mod, mu, alpha, temp0, temp_k,
                      p_edge, ys):
    """Symmetric-scheme finite difference TMM solution of the strip problem
    (Eq. 4.159-4.160).

    Same boundary conditions as ``fourier_strip_solution``: sigma_y(x, 0) =
    p_edge(x), tau = 0 at y = 0; sigma_y = tau = 0 at y = height. The
    temperature is t(y) = temp0 + temp_k y with line values temp0, temp_k.
    The transfer e^{A y} has growth rates up to about 2 / dx, so the strip
    height should satisfy 2 height / dx < 30 (round-off of the plain TMM).

    :param n: Number of intervals.
    :type n: int
    :param length: Width L.
    :type length: float
    :param height: Strip height.
    :type height: float
    :param e_mod: Young's modulus.
    :type e_mod: float
    :param mu: Poisson's ratio.
    :type mu: float
    :param alpha: Thermal expansion coefficient.
    :type alpha: float
    :param temp0: Line temperatures at y = 0 (n + 1).
    :type temp0: numpy.ndarray
    :param temp_k: Line temperature gradients (n + 1).
    :type temp_k: numpy.ndarray
    :param p_edge: Line values of sigma_y at y = 0 (n + 1).
    :type p_edge: numpy.ndarray
    :param ys: Output coordinates.
    :type ys: numpy.ndarray
    :return: dict with ``u``, ``v``, ``sigma_y``, ``tau`` on all lines (zeros
        at the edges for v, sigma_y) and ``sigma_x`` (Eq. 4.160), each
        (len(ys), n + 1).
    :rtype: dict
    """
    a, g = fd_system_matrix(n, length, e_mod, mu, scheme="symmetric")
    ne, no = n + 1, n - 1
    f0 = alpha * g @ np.asarray(temp0, dtype=float)
    f1 = alpha * g @ np.asarray(temp_k, dtype=float)
    size = a.shape[0]
    known_idx = list(range(ne + no, size))
    unk_idx = list(range(ne + no))
    z0 = np.zeros(size)
    z0[ne + no:ne + 2 * no] = np.asarray(p_edge, dtype=float)[1:n]
    u_h, f_h = _linear_load_transfer(a, f0, f1, height)
    rhs = -(u_h[np.ix_(known_idx, known_idx)] @ z0[known_idx] + f_h[known_idx])
    # u = const is a rigid translation of the discrete strip (the Fourier
    # terms m >= 1 exclude it): the minimum-norm solution removes it.
    z0[unk_idx] = np.linalg.lstsq(u_h[np.ix_(known_idx, unk_idx)], rhs,
                                  rcond=1e-13)[0]
    out = {k: [] for k in ("u", "v", "sigma_y", "tau", "sigma_x")}
    d_dx = np.zeros((ne, ne))
    dxl = length / n
    for k in range(1, n):
        d_dx[k, k - 1], d_dx[k, k + 1] = -0.5 / dxl, 0.5 / dxl
    for y in np.asarray(ys, dtype=float):
        u_y, f_y = _linear_load_transfer(a, f0, f1, y)
        z = u_y @ z0 + f_y
        u = z[:ne]
        v = np.concatenate([[0.0], z[ne:ne + no], [0.0]])
        sg = np.concatenate([[0.0], z[ne + no:ne + 2 * no], [0.0]])
        tau = z[ne + 2 * no:]
        temp = np.asarray(temp0) + np.asarray(temp_k) * y
        out["u"].append(u)
        out["v"].append(v)
        out["sigma_y"].append(sg)
        out["tau"].append(tau)
        out["sigma_x"].append(e_mod * (d_dx @ u) + mu * sg
                              - alpha * e_mod * temp)
    return {k: np.array(v) for k, v in out.items()}


# ----------------------------------------------------------------------
# Sec. 4.7 / 15.13 TMM for two-dimensional grid systems
# ----------------------------------------------------------------------

def _beam_stiffness(ei, gj, length, along_x):
    """Stiffness of a massless beam between two grid nodes, DOFs [Z, Theta_x,
    Theta_y] x 2.

    Theta_x = dZ/dy and Theta_y = -dZ/dx (Eq. 4.41). A beam along x bends
    with the slope -Theta_y and twists with Theta_x; a beam along y bends
    with the slope Theta_x and twists with Theta_y.
    """
    ln = length
    kb = ei / ln**3 * np.array([[12.0, 6 * ln, -12.0, 6 * ln],
                                [6 * ln, 4 * ln * ln, -6 * ln, 2 * ln * ln],
                                [-12.0, -6 * ln, 12.0, -6 * ln],
                                [6 * ln, 2 * ln * ln, -6 * ln, 4 * ln * ln]])
    if along_x:
        tb = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        tt = np.array([0.0, 1.0, 0.0])
    else:
        tb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        tt = np.array([0.0, 0.0, 1.0])
    big = np.zeros((4, 6))
    big[:2, :3] = tb
    big[2:, 3:] = tb
    tor = np.zeros((2, 6))
    tor[0, :3] = tt
    tor[1, 3:] = tt
    return (big.T @ kb @ big + gj / ln * tor.T
            @ np.array([[1.0, -1.0], [-1.0, 1.0]]) @ tor)


def _edge_condense(k, node_dofs, fixed):
    """Condense a beam stiffness to the inner node when the other end is a
    support.

    :param k: 6 x 6 beam stiffness.
    :param node_dofs: slice of the inner node (0:3 or 3:6).
    :param fixed: fixed DOFs (0..2) of the support node.
    """
    inner = list(range(node_dofs.start, node_dofs.stop))
    other = [q for q in range(6) if q not in inner]
    free_sup = [other[q] for q in range(3) if q not in fixed]
    kk = k[np.ix_(inner, inner)]
    if free_sup:
        kc = k[np.ix_(inner, free_sup)]
        kk = kk - kc @ np.linalg.solve(k[np.ix_(free_sup, free_sup)], kc.T)
    return kk


# positions of Q_z, M_x, M_y (conjugate to Z, Theta_x, Theta_y) in the row
# state of Eq. (4.161)
_GRID_FORCE = [5, 3, 4]
_SUPPORT_FIXED_X = {"S": [0, 1], "C": [0, 1, 2], "F": None}
_SUPPORT_FIXED_Y = {"S": [0, 2], "C": [0, 1, 2], "F": None}
_SUPPORT_ZERO = {"S": [0, 1, 4], "C": [0, 1, 2], "F": [3, 4, 5]}


class GridBeamStrip(Element):
    """Strip of m parallel massless beams along x (Eq. 4.164-4.166,
    15.71-15.73).

    Row-wise state of Eq. (4.161)-(4.162): [Z, Theta_x, Theta_y, M_x, M_y,
    Q_z] for rows j = 1 .. m. The beam block follows from the beam stiffness
    by Eq. (4.74) in the sign convention of Eq. (4.168): the force part Q of
    this module is the negative of the strip-method Q of Sec. 4.3.

    :param length: Beam length l_{i,j}.
    :type length: float
    :param ei: Bending stiffness EI_{i,j}.
    :type ei: float
    :param gj: Torsional stiffness (GJ_p)_{i,j}.
    :type gj: float
    :param rows: Number of rows m.
    :type rows: int
    """

    def __init__(self, length, ei, gj, rows):
        self.rows = int(rows)
        self.size = 6 * self.rows
        u = strip_transfer(_beam_stiffness(ei, gj, length, True), 3)
        flip = np.diag([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        perm = np.zeros((6, 6))
        for k_, p_ in enumerate([0, 1, 2] + _GRID_FORCE):
            perm[p_, k_] = 1.0
        self.block = perm @ flip @ u @ flip @ perm.T

    def transfer(self, s, eta=1.0):
        """Block diagonal transfer matrix (Eq. 4.165)."""
        u = np.zeros((self.size, self.size), dtype=complex)
        for j in range(self.rows):
            sl = slice(6 * j, 6 * j + 6)
            u[sl, sl] = self.block
        return u


class GridMassStrip(Element):
    """Strip of m lumped masses joined by beams along y (Eq. 4.167-4.184,
    15.82-15.91).

    Z_{i,i+1} = U_i Z_{i,i-1}: displacements unchanged and
    Q_out = Q_in + (omega^2 J - K_col) d per Eq. (4.168)-(4.172) with J =
    diag(m, J_x, J_y) of every mass and K_col the stiffness of the y beams,
    including the edge beams with the supports of Eq. (4.177)-(4.183).

    :param masses: (m, 3) array [m_ij, J_x, J_y].
    :type masses: numpy.ndarray
    :param k_col: Stiffness of the y beams on [Z, Theta_x, Theta_y] of the m
        nodes.
    :type k_col: numpy.ndarray
    """

    def __init__(self, masses, k_col):
        self.masses = np.asarray(masses, dtype=float)
        self.rows = self.masses.shape[0]
        self.size = 6 * self.rows
        self.k_col = np.asarray(k_col, dtype=float)

    def transfer(self, s, eta=1.0):
        """Transfer matrix of Eq. (4.184)."""
        n = self.rows
        coupling = -(s * s) * np.diag(self.masses.ravel()) - self.k_col
        u = np.eye(self.size, dtype=complex)
        disp = [6 * j + q for j in range(n) for q in range(3)]
        force = [6 * j + _GRID_FORCE[q] for j in range(n) for q in range(3)]
        u[np.ix_(force, disp)] = coupling
        return u


class GridPlate:
    """Rectangular plate as a grid of lumped masses and massless beams (Sec.
    4.7).

    The plate a x b is cut into n strips along x and m grids along y
    (Fig. 4.20). Lumped masses m_ij = rho h a b / (n m) with
    J_x = m_ij (b^2/m^2 + h^2)/12 and J_y = m_ij (a^2/n^2 + h^2)/12
    (Eq. 4.168); beams along x: EI = E (b/m) h^3 / (12 (1 - mu^2)),
    GJ_p = G beta h^3 b/m (Eq. 4.166); beams along y: the same with a/n
    (Eq. 4.172); edge beams have half length (Eq. 4.166, 4.172).

    Reproduction notes (tests/data/multibody/tmm_ch4_examples.json): Table 4.5
    (Example 4.6) is reproduced with beta = 0.3333 and the rotary inertia;
    Table 4.6 (Example 4.7) with beta = 0.3333 and without the rotary
    inertia (J = 0); the printed beta = 0.333 is off by one unit in the
    last digit of most entries.

    :param a: Length along x.
    :type a: float
    :param b: Length along y.
    :type b: float
    :param h: Thickness.
    :type h: float
    :param e_mod: Young's modulus.
    :type e_mod: float
    :param g_mod: Shear modulus.
    :type g_mod: float
    :param mu: Poisson's ratio.
    :type mu: float
    :param rho: Density.
    :type rho: float
    :param n: Number of strips along x.
    :type n: int
    :param m: Number of grids along y.
    :type m: int
    :param beta_t: Torsion constant factor beta.
    :type beta_t: float
    :param rotary_inertia: Include J_x, J_y.
    :type rotary_inertia: bool
    :param bc_y: Supports at y = 0 and y = b ('S', 'C', 'F').
    :type bc_y: tuple
    """

    def __init__(self, a, b, h, e_mod, g_mod, mu, rho, n, m, beta_t=0.3333,
                 rotary_inertia=True, bc_y=("S", "S")):
        self.a, self.b, self.h = float(a), float(b), float(h)
        self.n, self.m = int(n), int(m)
        d_unit = e_mod * h**3 / (12.0 * (1.0 - mu * mu))
        self.ei_x = d_unit * b / m
        self.gj_x = g_mod * beta_t * h**3 * b / m
        self.ei_y = d_unit * a / n
        self.gj_y = g_mod * beta_t * h**3 * a / n
        mass = rho * h * a * b / (n * m)
        if rotary_inertia:
            jx = mass * ((b / m) ** 2 + h * h) / 12.0
            jy = mass * ((a / n) ** 2 + h * h) / 12.0
        else:
            jx = jy = 0.0
        self.node_mass = np.array([mass, jx, jy])
        self.bc_y = bc_y
        self.k_col = self._column_stiffness()

    def _column_stiffness(self):
        m = self.m
        k = np.zeros((3 * m, 3 * m))
        lin = self.b / m
        ledge = self.b / (2 * m)
        for j in range(m - 1):
            kb = _beam_stiffness(self.ei_y, self.gj_y, lin, False)
            idx = list(range(3 * j, 3 * j + 6))
            k[np.ix_(idx, idx)] += kb
        kb = _beam_stiffness(self.ei_y, self.gj_y, ledge, False)
        for node, sl, bc in ((0, slice(3, 6), self.bc_y[0]),
                             (m - 1, slice(0, 3), self.bc_y[1])):
            fixed = _SUPPORT_FIXED_Y[bc]
            if fixed is None:
                continue
            idx = list(range(3 * node, 3 * node + 3))
            k[np.ix_(idx, idx)] += _edge_condense(kb, sl, fixed)
        return k

    def elements(self):
        """Elements in transfer order: beam strip, mass strip, ..., beam strip
        (Eq. 4.185)."""
        els = [GridBeamStrip(self.a / (2 * self.n), self.ei_x, self.gj_x,
                             self.m)]
        masses = np.tile(self.node_mass, (self.m, 1))
        for i in range(self.n):
            els.append(GridMassStrip(masses, self.k_col))
            ln = self.a / (2 * self.n) if i == self.n - 1 else self.a / self.n
            els.append(GridBeamStrip(ln, self.ei_x, self.gj_x, self.m))
        return els

    def chain(self, bc_x=("S", "S")):
        """Chain with the boundary state vectors of Eq. (4.186)-(4.187).

        :param bc_x: Supports at x = 0 and x = a.
        :type bc_x: tuple
        :return: Chain.
        :rtype: solvcon.multibody.tmm.linear.Chain
        """
        zero_in = [6 * j + q for j in range(self.m) for q
                   in _SUPPORT_ZERO[bc_x[0]]]
        zero_out = [6 * j + q for j in range(self.m) for q
                    in _SUPPORT_ZERO[bc_x[1]]]
        return Chain(self.elements(), zero_in, zero_out)

    def global_matrices(self, bc_x=("S", "S")):
        """Global stiffness and mass of the same grid (independent assembly).

        :param bc_x: Supports at x = 0 and x = a.
        :type bc_x: tuple
        :return: (K, M) on the DOFs [Z, Theta_x, Theta_y] of all masses,
            column-major (i, j) -> 3 (i m + j).
        :rtype: tuple
        """
        n, m = self.n, self.m
        nt = 3 * n * m
        k = np.zeros((nt, nt))
        for i in range(n):
            idx = list(range(3 * i * m, 3 * (i + 1) * m))
            k[np.ix_(idx, idx)] += self.k_col
        lin, ledge = self.a / n, self.a / (2 * n)
        kb_in = _beam_stiffness(self.ei_x, self.gj_x, lin, True)
        kb_edge = _beam_stiffness(self.ei_x, self.gj_x, ledge, True)
        for j in range(m):
            for i in range(n - 1):
                idx = list(range(3 * (i * m + j), 3 * (i * m + j) + 3)) + \
                    list(range(3 * ((i + 1) * m + j),
                               3 * ((i + 1) * m + j) + 3))
                k[np.ix_(idx, idx)] += kb_in
            for node, sl, bc in ((j, slice(3, 6), bc_x[0]),
                                 ((n - 1) * m + j, slice(0, 3), bc_x[1])):
                fixed = _SUPPORT_FIXED_X[bc]
                if fixed is None:
                    continue
                idx = list(range(3 * node, 3 * node + 3))
                k[np.ix_(idx, idx)] += _edge_condense(kb_edge, sl, fixed)
        mass = np.tile(self.node_mass, n * m)
        return k, mass

    def global_frequencies(self, bc_x=("S", "S")):
        """Eigenfrequencies of the global model (rad/s); massless DOFs are
        condensed.

        :param bc_x: Supports at x = 0 and x = a.
        :type bc_x: tuple
        :return: Sorted eigenfrequencies.
        :rtype: numpy.ndarray
        """
        k, mass = self.global_matrices(bc_x)
        dyn = np.flatnonzero(mass > 0.0)
        stat = np.flatnonzero(mass == 0.0)
        if stat.size:
            k = k[np.ix_(dyn, dyn)] - k[np.ix_(dyn, stat)] @ np.linalg.solve(
                k[np.ix_(stat, stat)], k[np.ix_(stat, dyn)])
            mass = mass[dyn]
        scale = 1.0 / np.sqrt(mass)
        w2 = np.linalg.eigvalsh(scale[:, None] * k * scale[None, :])
        return np.sqrt(np.abs(w2))


# ----------------------------------------------------------------------
# Sec. 15.11 Rectangular plate strips (Levy type)
# ----------------------------------------------------------------------
class LevyPlateStrip(Element):
    """Plate strip simply supported at y = 0 and y = L_y, term m of Eq. (15.48)
    (Sec. 15.11).

    State [w_m, theta_m, M_xm, Q_xm] (Eq. 15.52), theta = -dw/dx, Q_x the
    Kirchhoff shear. System matrix of Eq. (15.51) for an orthotropic plate
    with foundation K, in-plane pressures P_x, P_y (compression positive)
    and mass per unit area rho; U = e^{A x} (Eq. 10.69). The term
    rho d2w/dt2 becomes rho s^2 w.

    :param length: Strip length in x.
    :type length: float
    :param l_y: Width L_y.
    :type l_y: float
    :param m_index: Fourier index m.
    :type m_index: int
    :param d_x: Flexural stiffness D_x (Eq. 15.44).
    :type d_x: float
    :param rho: Mass per unit area.
    :type rho: float
    :param mu: Poisson's ratio mu_y (isotropic: mu).
    :type mu: float
    :param d_y: D_y (default D_x).
    :type d_y: float or None
    :param d_xy: D_xy (default D_x (1 - mu) / 2).
    :type d_xy: float or None
    :param mu_x: mu_x (default mu).
    :type mu_x: float or None
    :param k_found: Foundation modulus K.
    :type k_found: float
    :param p_x: In-plane pressure P_x.
    :type p_x: float
    :param p_y: In-plane pressure P_y.
    :type p_y: float
    """

    size = 4

    def __init__(self, length, l_y, m_index, d_x, rho, mu, d_y=None, d_xy=None,
                 mu_x=None, k_found=0.0, p_x=0.0, p_y=0.0):
        self.length = float(length)
        self.beta = m_index * np.pi / l_y
        self.d_x = float(d_x)
        self.d_y = self.d_x if d_y is None else float(d_y)
        self.d_xy = (0.5 * self.d_x * (1.0 - mu) if d_xy is None
                     else float(d_xy))
        self.mu_y = float(mu)
        self.mu_x = float(mu) if mu_x is None else float(mu_x)
        self.rho = float(rho)
        self.k_found = float(k_found)
        self.p_x, self.p_y = float(p_x), float(p_y)

    def system_matrix(self, s):
        """A(s) of Eq. (15.51)."""
        b2 = self.beta**2
        a = np.zeros((4, 4), dtype=complex)
        a[0, 1] = -1.0
        a[1, 0] = -self.mu_y * b2
        a[1, 2] = 1.0 / self.d_x
        a[2, 1] = 4.0 * self.d_xy * b2
        a[2, 3] = 1.0
        a[3, 0] = self.d_y * (1.0 - self.mu_x * self.mu_y) * b2 * b2 \
            - (self.p_y
               - self.mu_y * self.p_x) * b2 + self.k_found + self.rho * s * s
        a[3, 2] = -self.p_x / self.d_x + self.mu_x * self.d_y / self.d_x * b2
        return a

    def transfer(self, s, eta=1.0):
        """U = e^{A x} (Eq. 10.69 with Eq. 15.51)."""
        return _ric.expm(self.system_matrix(s) * self.length)


def levy_massless_transfer(x, beta, d_flex, mu):
    """Massless isotropic rectangular plate strip, Eq. (15.53) as printed
    except u_14.

    s = sinh(beta x), c = cosh(beta x) (the book prints cos; the test
    against e^{A x} shows cosh). The printed u_14 = -(s/beta - x c) /
    (2 D beta^2) has the wrong sign (small x gives +x^3/(6D) instead of
    -x^3/(6D)); this function uses +(s/beta - x c) / (2 D beta^2).

    :param x: Coordinate.
    :type x: float
    :param beta: m pi / L_y.
    :type beta: float
    :param d_flex: Flexural rigidity D.
    :type d_flex: float
    :param mu: Poisson's ratio.
    :type mu: float
    :return: U_x.
    :rtype: numpy.ndarray
    """
    b = beta
    s, c = math.sinh(b * x), math.cosh(b * x)
    d = d_flex
    k1 = 0.5 * b * (1.0 - mu) * x * s
    u12 = -0.5 * ((1.0 + mu) * s / b + (1.0 - mu) * x * c)
    u21 = -0.5 * b * b * ((1.0 + mu) * s / b - (1.0 - mu) * x * c)
    u32 = 0.5 * d * b * b * ((3.0 - 2.0 * mu - mu * mu) * s / b
                             + (1.0 - mu) ** 2 * x * c)
    u34 = 0.5 * ((1.0 + mu) * s / b + (1.0 - mu) * x * c)
    u41 = 0.5 * d * b**4 * ((3.0 - 2.0 * mu - mu * mu) * s / b
                            - (1.0 - mu) ** 2 * x * c)
    u43 = 0.5 * b * b * ((1.0 + mu) * s / b - (1.0 - mu) * x * c)
    k3 = 0.5 * d * b**3 * (1.0 - mu) ** 2 * x * s
    return np.array([
        [-k1 + c, u12, -x * s / (2 * d * b),
         (s / b - x * c) / (2 * d * b * b)],
        [u21, k1 + c, (s / b + x * c) / (2 * d), x * s / (2 * d * b)],
        [k3, u32, k1 + c, u34],
        [u41, -k3, u43, -k1 + c],
    ])


def levy_free_vibration_transfer(x, beta, d_flex, mu, rho, omega):
    """Isotropic plate strip in free vibration, Eq. (15.54).

    a^2 = sqrt(rho/D) omega + beta^2, b^2 = sqrt(rho/D) omega - beta^2,
    g = 2 sqrt(rho/D) omega, eta_1,2 = sqrt(rho/D) omega -+ (1 - mu) beta^2;
    s = sinh(a x), S = sin(b x), c = cosh(a x), C = cos(b x). For
    b^2 < 0 the same formula holds with complex b (the result is real).

    Two printed entries disagree with e^{A x} of Eq. (15.51) (test
    ``test_levy_closed_forms_vs_expm``): u_22 = u_33 = (eta_2 c + eta_1 C)/g
    (printed eta_1 c + eta_2 C) and u_41 = -D (eta_1^2 a s + eta_2^2 b S)/g
    (printed with +). This function uses the corrected entries; with them
    every entry agrees with e^{A x}.

    :param x: Coordinate.
    :type x: float
    :param beta: m pi / L_y.
    :type beta: float
    :param d_flex: D.
    :type d_flex: float
    :param mu: Poisson's ratio.
    :type mu: float
    :param rho: Mass per unit area.
    :type rho: float
    :param omega: Circular frequency (> 0).
    :type omega: float
    :return: U_x.
    :rtype: numpy.ndarray
    """
    k = math.sqrt(rho / d_flex) * omega
    b2 = beta * beta
    a = np.sqrt(complex(k + b2))
    b = np.sqrt(complex(k - b2))
    g = 2.0 * k
    e1, e2 = k - (1.0 - mu) * b2, k + (1.0 - mu) * b2
    s, cc = np.sinh(a * x), np.cosh(a * x)
    ss, cb = np.sin(b * x), np.cos(b * x)
    d = d_flex
    sa = s / a
    sb = ss / b if b != 0 else x
    u = np.array([
        [e1 * cc + e2 * cb, -(e2 * sa + e1 * sb), -(cc - cb) / d,
         -(sa - sb) / d],
        [-(e1 * a * s - e2 * b * ss), e2 * cc + e1 * cb, (a * s + b * ss) / d,
         (cc - cb) / d],
        [-d * e1 * e2 * (cc - cb), d * (e2**2 * sa - e1**2 * sb),
         e2 * cc + e1 * cb, e2 * sa + e1 * sb],
        [-d * (e1**2 * a * s + e2**2 * b * ss), d * e1 * e2 * (cc - cb),
         e1 * a * s - e2 * b * ss, e1 * cc + e2 * cb],
    ]) / g
    return np.real(u)


# ----------------------------------------------------------------------
# Sec. 15.12 Disks
# ----------------------------------------------------------------------
def disk_system_matrix(r, m, d_flex, mu, lam):
    """System matrix of an isotropic disk, Fourier term m (Eq. 15.64, isotropic
    case).

    State [w_m, theta_m, M_rm, Q_rm] (Eq. 15.65), theta = -dw/dr, Q_r the
    Kirchhoff shear Q_r - (1/r) dM_rphi/dphi. lam = (rho h omega^2 - K)/D,
    that is lam = -(rho h s^2 + K)/D. The dw, dtheta, dM_r rows agree with
    Eq. (15.64) for D_r = D_phi = D, D_rphi = D (1 - mu)/2. The M_r
    coefficient of dQ_r/dr (+mu m^2 / r^2) is confirmed by the exact Bessel
    solutions and the reciprocity test; the text layer of Eq. (15.64) shows
    a minus sign (not checked on the rendered page).

    :param r: Radius.
    :type r: float
    :param m: Fourier index.
    :type m: int
    :param d_flex: D.
    :type d_flex: float
    :param mu: Poisson's ratio.
    :type mu: float
    :param lam: (rho h omega^2 - K)/D (complex allowed).
    :type lam: complex
    :return: 4 x 4 matrix A(r).
    :rtype: numpy.ndarray
    """
    d, nu, m2 = d_flex, mu, m * m
    a = np.zeros((4, 4), dtype=complex)
    a[0, 1] = -1.0
    a[1, 0] = -nu * m2 / r**2
    a[1, 1] = -nu / r
    a[1, 2] = 1.0 / d
    a[2, 0] = (1.0 - nu) * (3.0 + nu) * d * m2 / r**3
    a[2, 1] = (1.0 - nu) * (1.0 + nu + 2.0 * m2) * d / r**2
    a[2, 2] = -(1.0 - nu) / r
    a[2, 3] = 1.0
    a[3, 0] = -d * lam + (1.0 - nu) * d * m2 * ((1.0 + nu) * m2 + 2.0) / r**4
    a[3, 1] = (1.0 - nu) * (3.0 + nu) * d * m2 / r**3
    a[3, 2] = nu * m2 / r**2
    a[3, 3] = -1.0 / r
    return a


class DiskElement(Element):
    """Annular isotropic disk r_in <= r <= r_out, Fourier term m (Sec. 15.12).

    Transfer matrix of Eq. (15.64)-(15.65) by ``magnus_transfer``.

    :param r_in: Inner radius (> 0).
    :type r_in: float
    :param r_out: Outer radius.
    :type r_out: float
    :param d_flex: D.
    :type d_flex: float
    :param mu: Poisson's ratio.
    :type mu: float
    :param rho_h: Mass per unit area.
    :type rho_h: float
    :param m: Fourier index.
    :type m: int
    :param k_found: Foundation modulus K.
    :type k_found: float
    :param n_steps: Magnus steps.
    :type n_steps: int
    """

    size = 4

    def __init__(self, r_in, r_out, d_flex, mu, rho_h, m=0, k_found=0.0,
                 n_steps=200):
        self.r_in, self.r_out = float(r_in), float(r_out)
        self.d, self.mu, self.rho_h = float(d_flex), float(mu), float(rho_h)
        self.m, self.k_found, self.n_steps = (int(m), float(k_found),
                                              int(n_steps))

    def transfer(self, s, eta=1.0):
        """U(r_out, r_in)."""
        lam = -(self.rho_h * s * s + self.k_found) / self.d
        return magnus_transfer(lambda r: disk_system_matrix(r, self.m, self.d,
                                                            self.mu, lam),
                               self.r_in, self.r_out, self.n_steps)


def _bessel_with_derivative(kind, m, x):
    """Z_m(x) and Z_m'(x) for kind 'J', 'Y', 'I', 'K'."""
    fun = {"J": bessel_j, "Y": bessel_y, "I": bessel_i, "K": bessel_k}[kind]
    z = fun(m, x)
    if m == 0:
        z1 = fun(1, x)
        dz = {"J": -z1, "Y": -z1, "I": z1, "K": -z1}[kind]
    else:
        zm1 = fun(m - 1, x)
        dz = (-zm1 if kind == "K" else zm1) - m / x * z
    return z, dz


def disk_bessel_state(r, m, k, d_flex, mu, kind):
    """State [w, theta, M_r, Q_r] of the exact solution w = Z_m(k r) cos(m phi)
    (Eq. 15.64-15.70).

    Z = J_m, Y_m (nabla^2 w = -k^2 w) or I_m, K_m (nabla^2 w = +k^2 w),
    k^4 = rho h omega^2 / D.

    :param r: Radius.
    :type r: float
    :param m: Fourier index.
    :type m: int
    :param k: Wavenumber k.
    :type k: float
    :param d_flex: D.
    :type d_flex: float
    :param mu: Poisson's ratio.
    :type mu: float
    :param kind: 'J', 'Y', 'I' or 'K'.
    :type kind: str
    :return: State vector.
    :rtype: numpy.ndarray
    """
    z, dz = _bessel_with_derivative(kind, m, k * r)
    w, w1 = z, k * dz
    sign = -1.0 if kind in ("J", "Y") else 1.0
    w2 = -w1 / r + (sign * k * k + m * m / r**2) * w
    l1 = sign * k * k * w1
    d = d_flex
    mr = -d * (w2 + mu * (w1 / r - m * m * w / r**2))
    qr = -d * l1 + (1.0 - mu) * d * m * m * (w1 / r**2 - w / r**3)
    return np.array([w, -w1, mr, qr])


def disk_bessel_transfer(r0, r1, m, k, d_flex, mu):
    """Exact annular disk transfer matrix B(r1) B(r0)^{-1} from J, Y, I, K
    solutions.

    :param r0: Input radius.
    :type r0: float
    :param r1: Output radius.
    :type r1: float
    :param m: Fourier index.
    :type m: int
    :param k: Wavenumber, k^4 = rho h omega^2 / D.
    :type k: float
    :param d_flex: D.
    :type d_flex: float
    :param mu: Poisson's ratio.
    :type mu: float
    :return: 4 x 4 transfer matrix.
    :rtype: numpy.ndarray
    """
    def basis(r):
        return np.array([disk_bessel_state(r, m, k, d_flex, mu,
                                           kd) for kd in "JYIK"]).T

    return basis(r1) @ np.linalg.inv(basis(r0))


def solid_disk_massless_matrix(r, m, d_flex, mu):
    """Massless solid isotropic disk, Eq. (15.66a-c).

    m = 0: transfer from the center state [w_0, theta_0, M_0, Q_0];
    m = 1 and m >= 2: matrix of the regular static solutions (columns
    w = r^m, r^{m+2} for m >= 2) as printed in the book.

    :param r: Radius.
    :type r: float
    :param m: Fourier index.
    :type m: int
    :param d_flex: D.
    :type d_flex: float
    :param mu: Poisson's ratio.
    :type mu: float
    :return: 4 x 4 matrix.
    :rtype: numpy.ndarray
    """
    d = d_flex
    if m == 0:
        return np.array([[1.0, 0.0, -r * r / (2 * d * (1 + mu)), 0.0],
                         [0.0, 0.0, r / (d * (1 + mu)), 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0]])
    if m == 1:
        return np.array([[0.0, -r, 0.0, -r**3 / (2 * d * (3 + mu))],
                         [0.0, 1.0, 0.0, 3 * r * r / (2 * d * (3 + mu))],
                         [0.0, 0.0, 0.0, r],
                         [0.0, 0.0, 0.0, 1.0]])
    u = np.zeros((4, 4))
    u[:, 0] = [r**m, -m * r ** (m - 1),
               -d * m * (m - 1) * (1 - mu) * r ** (m - 2),
               d * m * m * (m - 1) * (1 - mu) * r ** (m - 3)]
    u[:, 1] = [r ** (m + 2), -(m + 2) * r ** (m + 1),
               -d * (1 + m) * (m + 2 - mu * m + 2 * mu) * r**m,
               d * m * (1 + m) * (m * (1 - mu) - 4) * r ** (m - 1)]
    return u


def solid_disk_transfer(r, d_flex, mu, rho_h, omega):
    """Symmetrical (m = 0) solid disk, Eq. (15.70), alpha^4 = rho h omega^2 /
    D.

    Columns 1 and 3 multiply the center deflection w_0 and the center
    variable -D nabla^2 w(0) = 2 M_r(0) / (1 + mu); the printed alpha^2 =
    omega^2 P / D stands for alpha^4 = rho h omega^2 / D.

    :param r: Radius.
    :type r: float
    :param d_flex: D.
    :type d_flex: float
    :param mu: Poisson's ratio.
    :type mu: float
    :param rho_h: Mass per unit area.
    :type rho_h: float
    :param omega: Circular frequency.
    :type omega: float
    :return: 4 x 4 matrix.
    :rtype: numpy.ndarray
    """
    d = d_flex
    al = (rho_h * omega * omega / d) ** 0.25
    x = al * r
    j0, j1, i0, i1 = (bessel_j(0, x), bessel_j(1, x), bessel_i(0, x),
                      bessel_i(1, x))
    u31 = d * al * (al * (j0 - i0) / 2.0 - (1 - mu) * (j1 - i1) / (2.0 * r))
    u33 = (j0 + i0) / 2.0 - (1 - mu) * (j1 + i1) / (2.0 * x)
    return np.array([
        [(j0 + i0) / 2.0, 0.0, (j0 - i0) / (2.0 * d * al * al), 0.0],
        [al * (j1 - i1) / 2.0, 0.0, (j1 + i1) / (2.0 * d * al), 0.0],
        [u31, 0.0, u33, 0.0],
        [-d * al**3 * (j1 + i1) / 2.0, 0.0, -al * (j1 - i1) / 2.0, 0.0],
    ])


# ----------------------------------------------------------------------
# Sec. 15.14 Thick-walled cylinder (plane radial motion)
# ----------------------------------------------------------------------
def lame_constants(e_mod, mu):
    """Shear modulus G and Lame constant lambda = E mu / ((1 + mu)(1 - 2 mu)).

    Eq. (15.93) prints lambda = E mu / (2 (1 - mu^2)); the closed forms
    (15.95)-(15.96) agree with the Lame solution only for the Lame constant
    returned here (test ``test_thick_cylinder_massless_closed_form``).

    :param e_mod: Young's modulus.
    :type e_mod: float
    :param mu: Poisson's ratio.
    :type mu: float
    :return: (G, lambda).
    :rtype: tuple
    """
    return (e_mod / (2.0 * (1.0 + mu)),
            e_mod * mu / ((1.0 + mu) * (1.0 - 2.0 * mu)))


def thick_cylinder_system_matrix(r, lam, g_mod, rho, s=0.0):
    """A(r) of Eq. (15.92), state [u, sigma_r] (Eq. 15.94).

    :param r: Radius.
    :type r: float
    :param lam: Lame constant lambda.
    :type lam: float
    :param g_mod: Shear modulus G.
    :type g_mod: float
    :param rho: Density.
    :type rho: float
    :param s: Laplace variable (i omega).
    :type s: complex
    :return: 2 x 2 matrix.
    :rtype: numpy.ndarray
    """
    c = lam + 2.0 * g_mod
    return np.array([[-lam / (r * c), 1.0 / c],
                     [4.0 * g_mod * (g_mod + lam) / (r * r * c) + rho * s * s,
                      -2.0 * g_mod / (r * c)]], dtype=complex)


def thick_cylinder_massless_transfer(r, a_k, lam, g_mod, mu):
    """Massless cylinder from the inner radius a_k to r, Eq. (15.95).

    :param r: Radius.
    :type r: float
    :param a_k: Inner radius.
    :type a_k: float
    :param lam: Lame constant.
    :type lam: float
    :param g_mod: Shear modulus.
    :type g_mod: float
    :param mu: Poisson's ratio.
    :type mu: float
    :return: 2 x 2 matrix.
    :rtype: numpy.ndarray
    """
    q = 1.0 / (1.0 - mu)
    return np.array([
        [q * (g_mod * mu / lam * r / a_k + 0.5 * a_k / r),
         mu / (2.0 * (1.0 - mu) * lam) * (r * r - a_k * a_k) / r],
        [g_mod / (1.0 - mu) * (r * r - a_k * a_k) / (a_k * r * r),
         q * (0.5 + g_mod * mu / lam * (a_k / r) ** 2)],
    ])


def thick_cylinder_massless_solid(r, lam, mu):
    """Massless cylinder without a center hole, Eq. (15.96): U = [[0, mu r /
    lambda], [0, 1]].

    :param r: Radius.
    :type r: float
    :param lam: Lame constant.
    :type lam: float
    :param mu: Poisson's ratio.
    :type mu: float
    :return: 2 x 2 matrix from the center state [u(0), sigma_r(0)].
    :rtype: numpy.ndarray
    """
    return np.array([[0.0, mu * r / lam], [0.0, 1.0]])


def _cyl_basis(r, beta, lam, g_mod):
    """Columns [J_1, sigma_r(J_1)], [Y_1, sigma_r(Y_1)] of Eq. (15.97)."""
    c = lam + 2.0 * g_mod
    x = beta * r
    j1, j2 = bessel_j(1, x), bessel_j(2, x)
    y1, y2 = bessel_y(1, x), bessel_y(2, x)
    e2 = c / r * (2.0 * (lam + g_mod) / c * j1 - x * j2)
    e3 = c / r * (2.0 * (lam + g_mod) / c * y1 - x * y2)
    return np.array([[j1, y1], [e2, e3]])


def thick_cylinder_transfer(r, a_k, lam, g_mod, rho, omega):
    """Isotropic thick-walled cylinder in radial vibration, Eq. (15.97).

    beta^2 = rho omega^2 / (lambda + 2 G); the matrix equals B(r) B(a_k)^{-1}
    with the columns of ``_cyl_basis`` (u = J_1(beta r), Y_1(beta r)).

    :param r: Radius.
    :type r: float
    :param a_k: Inner radius.
    :type a_k: float
    :param lam: Lame constant.
    :type lam: float
    :param g_mod: Shear modulus.
    :type g_mod: float
    :param rho: Density.
    :type rho: float
    :param omega: Circular frequency (> 0).
    :type omega: float
    :return: 2 x 2 matrix.
    :rtype: numpy.ndarray
    """
    beta = omega * math.sqrt(rho / (lam + 2.0 * g_mod))
    ba = _cyl_basis(a_k, beta, lam, g_mod)
    br = _cyl_basis(r, beta, lam, g_mod)
    e1 = ba[1, 1] * ba[0, 0] - ba[1, 0] * ba[0, 1]
    return np.array([
        [(ba[1, 1] * br[0, 0] - ba[1, 0] * br[0, 1]) / e1,
         (ba[0, 0] * br[0, 1] - ba[0, 1] * br[0, 0]) / e1],
        [(ba[1, 1] * br[1, 0] - ba[1, 0] * br[1, 1]) / e1,
         (ba[0, 0] * br[1, 1] - ba[0, 1] * br[1, 0]) / e1],
    ])


def thick_cylinder_solid_transfer(r, lam, g_mod, rho, omega):
    """Cylinder without a center hole, Eq. (15.98): [u, sigma_r](r) = U [u,
    sigma_r](0).

    u = sigma_r(0) J_1(beta r) / ((lambda + G) beta); sigma_r(0) =
    (lambda + G) beta u'(0) for u = J_1(beta r).

    :param r: Radius.
    :type r: float
    :param lam: Lame constant.
    :type lam: float
    :param g_mod: Shear modulus.
    :type g_mod: float
    :param rho: Density.
    :type rho: float
    :param omega: Circular frequency (> 0).
    :type omega: float
    :return: 2 x 2 matrix.
    :rtype: numpy.ndarray
    """
    c = lam + 2.0 * g_mod
    beta = omega * math.sqrt(rho / c)
    x = beta * r
    j1, j2 = bessel_j(1, x), bessel_j(2, x)
    return np.array([[0.0, j1 / ((lam + g_mod) * beta)],
                     [0.0,
                      c / ((lam + g_mod) * beta)
                      * (2.0 * (lam + g_mod) / c * j1 - x * j2) / r]])


class ThickCylinder(Element):
    """Thick-walled cylinder a_k <= r <= r_out in plane radial motion (Eq.
    15.92-15.94).

    Transfer matrix by ``magnus_transfer`` of Eq. (15.92).

    :param r_in: Inner radius (> 0).
    :type r_in: float
    :param r_out: Outer radius.
    :type r_out: float
    :param lam: Lame constant.
    :type lam: float
    :param g_mod: Shear modulus.
    :type g_mod: float
    :param rho: Density.
    :type rho: float
    :param n_steps: Magnus steps.
    :type n_steps: int
    """

    size = 2

    def __init__(self, r_in, r_out, lam, g_mod, rho, n_steps=200):
        self.r_in, self.r_out = float(r_in), float(r_out)
        self.lam, self.g_mod, self.rho = float(lam), float(g_mod), float(rho)
        self.n_steps = int(n_steps)

    def transfer(self, s, eta=1.0):
        """U(r_out, r_in), integrated for the balanced state [u, sigma_r /
        (lambda + 2 G)]."""
        c = self.lam + 2.0 * self.g_mod
        sc = np.diag([1.0, 1.0 / c])
        sc_inv = np.diag([1.0, c])
        u = magnus_transfer(
            lambda r: sc
            @ thick_cylinder_system_matrix(r, self.lam, self.g_mod, self.rho,
                                           s)
            @ sc_inv, self.r_in, self.r_out, self.n_steps)
        return sc_inv @ u @ sc


# ----------------------------------------------------------------------
# Sec. 15.15 Thin-walled cylinder (axisymmetric bending)
# ----------------------------------------------------------------------
class ThinCylinder(Element):
    """Thin-walled isotropic cylinder, axisymmetric radial and bending motion
    (Eq. 15.99-15.102).

    State [v, theta, M, Q] (Eq. 15.100). dv/dx = -theta + Q/D_Q,
    dtheta/dx = M/D, dM/dx = Q - P_x theta + rho r_y^2 theta_tt,
    dQ/dx = (E h / R^2) v + rho v_tt. Eq. (15.99) prints K = E h^3 /
    (12 (1 - mu^2)) for the extension stiffness and rho d2theta/dt2 in the
    last row; the extension stiffness is K = E h / (1 - mu^2), so that
    K (1 - mu^2) / R^2 = E h / R^2, and the inertia term acts on v (the
    frequency test uses this form). rho is the mass per unit area.

    :param length: Length L.
    :type length: float
    :param radius: Radius R.
    :type radius: float
    :param e_mod: Young's modulus.
    :type e_mod: float
    :param h: Wall thickness.
    :type h: float
    :param mu: Poisson's ratio.
    :type mu: float
    :param rho_h: Mass per unit area.
    :type rho_h: float
    :param d_q: Shear stiffness D_Q (numpy.inf: no shear deformation).
    :type d_q: float
    :param r_y2: Squared radius of gyration r_y^2 (0: no rotary inertia).
    :type r_y2: float
    :param p_x: Axial pressure P_x.
    :type p_x: float
    """

    size = 4

    def __init__(self, length, radius, e_mod, h, mu, rho_h, d_q=np.inf,
                 r_y2=0.0, p_x=0.0):
        self.length = float(length)
        self.d = e_mod * h**3 / (12.0 * (1.0 - mu * mu))
        self.k_f = e_mod * h / radius**2
        self.rho_h = float(rho_h)
        self.inv_dq = 0.0 if d_q == np.inf else 1.0 / d_q
        self.r_y2 = float(r_y2)
        self.p_x = float(p_x)

    def system_matrix(self, s):
        """A(s) of Eq. (15.99)."""
        a = np.zeros((4, 4), dtype=complex)
        a[0, 1] = -1.0
        a[0, 3] = self.inv_dq
        a[1, 2] = 1.0 / self.d
        a[2, 1] = -self.p_x + self.rho_h * self.r_y2 * s * s
        a[2, 3] = 1.0
        a[3, 0] = self.k_f + self.rho_h * s * s
        return a

    def transfer(self, s, eta=1.0):
        """U = e^{A x} (Eq. 15.102 in closed form)."""
        return _ric.expm(self.system_matrix(s) * self.length)


def thin_cylinder_massless_shear(x, d_x, d_q):
    """Massless thin-walled cylinder with shear deformation, Eq. (15.101) (no
    foundation term).

    The printed u_14 = -x^3 / (3 D_x) + x / D_Q differs from the integral
    of the massless equations, -x^3 / (6 D_x) + x / D_Q; this function
    returns the integral.

    :param x: Coordinate.
    :type x: float
    :param d_x: Flexural stiffness.
    :type d_x: float
    :param d_q: Shear stiffness.
    :type d_q: float
    :return: 4 x 4 matrix.
    :rtype: numpy.ndarray
    """
    return np.array([
        [1.0, -x, -x * x / (2 * d_x), -x**3 / (6 * d_x) + x / d_q],
        [0.0, 1.0, x / d_x, x * x / (2 * d_x)],
        [0.0, 0.0, 1.0, x],
        [0.0, 0.0, 0.0, 1.0],
    ])


# ----------------------------------------------------------------------
# Sec. 4.4 geometrically nonlinear (von Karman) plate strips
# ----------------------------------------------------------------------
def _vk_triangle_data(y2, x3, y3):
    """Quadrature data of the von Karman triangle: w gradients G (nq, 2, 9),
    CST B_m, weights."""
    nodes = [(0.0, 0.0), (0.0, y2), (x3, y3)]
    rows = []
    for xn, yn in nodes:
        d, dx, dy, _, _, _ = _poly(np.array(xn), np.array(yn))
        rows += [d, dy, -dx]
    n_inv = np.linalg.inv(np.array(rows, dtype=float))
    area = 0.5 * abs(x3 * y2)
    p1, p2, p3 = (np.array(p) for p in nodes)
    xs = p1[0] + _TRI_PTS[:, 0] * (p2 - p1)[0] + _TRI_PTS[:, 1] * (p3 - p1)[0]
    ys = p1[1] + _TRI_PTS[:, 0] * (p2 - p1)[1] + _TRI_PTS[:, 1] * (p3 - p1)[1]
    wts = _TRI_WTS * 2.0 * area
    _, dxs, dys, _, _, _ = _poly(xs, ys)
    grad = np.stack([dxs.T @ n_inv, dys.T @ n_inv], axis=1)
    coef = np.linalg.inv(np.array([[1.0, xn, yn] for xn, yn in nodes]))
    b_x, c_y = coef[1], coef[2]
    b_m = np.zeros((3, 6))
    b_m[0, 0::2] = b_x
    b_m[1, 1::2] = c_y
    b_m[2, 0::2] = c_y
    b_m[2, 1::2] = b_x
    return grad, b_m, wts, area


def _vk_beam(ea, ei, length, d_loc, nonlinear):
    """Stiffener on a mesh line: force and tangent on [w1, phi1, s1, w2, phi2,
    s2].

    phi = dw/ds, s the axial displacement; strain e = s' + w'^2 / 2
    (von Karman), energy EA/2 int e^2 + EI/2 int w''^2, 4-point Gauss rule.
    """
    gx, gw = np.polynomial.legendre.leggauss(4)
    ln = length
    f = np.zeros(6)
    k = np.zeros((6, 6))
    wdof = [0, 1, 3, 4]
    sdof = [2, 5]
    for xi, wq in zip(gx, gw):
        t = 0.5 * (xi + 1.0)
        wt = 0.5 * ln * wq
        # Hermite first and second derivatives with respect to s
        h1 = np.array([6 * t * t - 6 * t, ln * (3 * t * t - 4 * t + 1),
                       -6 * t * t + 6 * t, ln * (3 * t * t - 2 * t)]) / ln
        h2 = np.array([12 * t - 6, ln * (6 * t - 4), -12 * t + 6,
                       ln * (6 * t - 2)]) / ln**2
        l1 = np.array([-1.0, 1.0]) / ln
        wp = h1 @ d_loc[wdof]
        e = l1 @ d_loc[sdof] + (0.5 * wp * wp if nonlinear else 0.0)
        n_ax = ea * e
        de_w = h1 * wp if nonlinear else np.zeros(4)
        f[sdof] += wt * n_ax * l1
        f[wdof] += wt * (n_ax * de_w + ei * (h2 @ d_loc[wdof]) * h2)
        k[np.ix_(sdof, sdof)] += wt * ea * np.outer(l1, l1)
        k[np.ix_(sdof, wdof)] += wt * ea * np.outer(l1, de_w)
        k[np.ix_(wdof, sdof)] += wt * ea * np.outer(de_w, l1)
        k[np.ix_(wdof, wdof)] += wt * (ea * np.outer(de_w, de_w)
                                       + ei * np.outer(h2, h2))
        if nonlinear:
            k[np.ix_(wdof, wdof)] += wt * n_ax * np.outer(h1, h1)
    return f, k


class VonKarmanPlateMesh:
    """Rectangular plate with von Karman strains in strips along x (Sec. 4.4,
    Example 4.4-4.5).

    Node DOFs [w, theta_x = dw/dy, theta_y = -dw/dx, u, v]. Bending: the
    9-DOF triangle of Eq. (4.44)-(4.59); membrane: linear (CST) u, v with
    the von Karman strains e_x = u_x + w_x^2/2, e_y = v_y + w_y^2/2,
    g_xy = u_y + v_x + w_x w_y, integrated with the 36-point rule. The
    internal force and the tangent stiffness K_T of Eq. (4.89) follow from
    the strain energy. Consistent mass of w and of u, v; no rotary inertia.
    The cell split is the one of ``FEPlateMesh``.

    Stiffeners (Example 4.5): concentric beams on mesh lines with axial
    stiffness EA (von Karman strain) and out-of-plane bending EI; in-plane
    bending and torsion of the stiffeners are not modeled (the plate nodes
    have no drilling rotation). A stiffener on a symmetry line carries half
    of EA and EI.

    :param a: Length along x (transfer direction).
    :type a: float
    :param b: Width along y.
    :type b: float
    :param n_strips: Number of strips.
    :type n_strips: int
    :param n_cells: Number of cells along y.
    :type n_cells: int
    :param e_mod: Young's modulus.
    :type e_mod: float
    :param h: Thickness.
    :type h: float
    :param mu: Poisson's ratio.
    :type mu: float
    :param rho: Density.
    :type rho: float
    :param fixed_y0: Fixed node DOFs (0..4) on y = 0.
    :type fixed_y0: list
    :param fixed_y1: Fixed node DOFs on y = b.
    :type fixed_y1: list
    :param nonlinear: False drops the quadratic strain terms (linear plate).
    :type nonlinear: bool
    :param x_stiffeners: Rows j with a stiffener parallel to x, as (j, ea, ei).
    :type x_stiffeners: list
    :param y_stiffeners: Sections i with a stiffener parallel to y, as (i, ea,
        ei).
    :type y_stiffeners: list
    """

    def __init__(self, a, b, n_strips, n_cells, e_mod, h, mu, rho, fixed_y0,
                 fixed_y1, nonlinear=True, x_stiffeners=(), y_stiffeners=()):
        self.a, self.b = float(a), float(b)
        self.n_strips, self.n_cells = int(n_strips), int(n_cells)
        self.nonlinear = bool(nonlinear)
        self.dx = self.a / self.n_strips
        self.dy = self.b / self.n_cells
        d_flex = e_mod * h**3 / (12.0 * (1.0 - mu * mu))
        rho_h = rho * h
        self.a_mat = e_mod * h / (1.0 - mu * mu) * np.array(
            [[1.0, mu, 0.0], [mu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - mu)]])
        k_b, m_b = plate_triangle(self.dy, self.dx, self.dy, d_flex, mu, rho_h)
        _, m_unit = plate_triangle(self.dy, self.dx, self.dy, 1.0, mu, 1.0)
        self.grad, self.b_m, self.wts, area = _vk_triangle_data(self.dy,
                                                                self.dx,
                                                                self.dy)
        self.k_b = k_b
        self.wdx = [5 * k + q for k in range(3) for q in range(3)]
        self.mdx = [5 * k + 3 + q for k in range(3) for q in range(2)]
        k_mm = np.einsum("p,ai,ab,bj->ij", self.wts, self.b_m, self.a_mat,
                         self.b_m)
        self.k_mm = k_mm
        m15 = np.zeros((15, 15))
        m15[np.ix_(self.wdx, self.wdx)] = m_b
        cst = rho_h * area / 12.0 * np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0],
                                              [1.0, 1.0, 2.0]])
        for q in range(2):
            idx = [5 * k + 3 + q for k in range(3)]
            m15[np.ix_(idx, idx)] = cst
        self.flip = np.array([1.0, -1.0, -1.0, -1.0, -1.0] * 3)
        self.load_w = m_unit @ np.array([1.0, 0.0, 0.0] * 3)
        nn = self.n_cells + 1
        self.nn = nn
        self.tris = []
        for c in range(self.n_cells):
            a_, d_, b_, c_ = c, c + 1, nn + c, nn + c + 1
            self.tris.append(((a_, d_, c_), False))
            self.tris.append(((c_, b_, a_), True))
        self.tri_dofs = np.array([[5 * nd_ + q for nd_ in tri
                                   for q in range(5)]
                                  for tri, _ in self.tris])
        self.tri_flip = np.array([self.flip if fl else np.ones(15)
                                  for _, fl in self.tris])
        removed = {5 * 0 + q for q in fixed_y0} | {5 * self.n_cells + q for q
                                                   in fixed_y1}
        self.section_dofs = [q for q in range(5 * nn) if q not in removed]
        self.n_dof = len(self.section_dofs)
        self.strip_keep = (self.section_dofs
                           + [5 * nn + q for q in self.section_dofs])
        m_strip = np.zeros((10 * nn, 10 * nn))
        for dofs, fl in zip(self.tri_dofs, self.tri_flip):
            m_strip[np.ix_(dofs, dofs)] += fl[:, None] * m15 * fl[None, :]
        self.m_strip = m_strip[np.ix_(self.strip_keep, self.strip_keep)]
        self.x_stiffeners = list(x_stiffeners)
        self.y_stiffeners = list(y_stiffeners)

    # -- element level --------------------------------------------------
    def _triangle_forces(self, q):
        """Internal forces (t, 15) and tangents (t, 15, 15) for local triangle
        DOFs q (t, 15)."""
        qw = q[:, self.wdx]
        qm = q[:, self.mdx]
        nt = q.shape[0]
        f = np.zeros((nt, 15))
        k = np.zeros((nt, 15, 15))
        wd, md = self.wdx, self.mdx
        f[:, wd] = qw @ self.k_b.T
        e_lin = qm @ self.b_m.T
        if not self.nonlinear:
            f[:, md] = e_lin @ self.a_mat @ self.b_m * np.sum(self.wts)
            k[np.ix_(range(nt), wd, wd)] = self.k_b
            k[np.ix_(range(nt), md, md)] = self.k_mm
            return f, k
        g = np.einsum("pkj,tj->tpk", self.grad, qw)
        eps = e_lin[:, None, :] + np.stack(
            [0.5 * g[..., 0] ** 2, 0.5 * g[..., 1] ** 2,
             g[..., 0] * g[..., 1]], axis=-1)
        n_res = eps @ self.a_mat
        hm = np.zeros(g.shape[:2] + (3, 2))
        hm[..., 0, 0] = g[..., 0]
        hm[..., 1, 1] = g[..., 1]
        hm[..., 2, 0] = g[..., 1]
        hm[..., 2, 1] = g[..., 0]
        hg = np.einsum("tpik,pkj->tpij", hm, self.grad)
        w = self.wts
        f[:, md] = np.einsum("p,tpi,ij->tj", w, n_res, self.b_m)
        f[:, wd] += np.einsum("p,tpi,tpij->tj", w, n_res, hg)
        ahg = np.einsum("ab,tpbj->tpaj", self.a_mat, hg)
        k_mw = np.einsum("p,ai,tpaj->tij", w, self.b_m, ahg)
        nmat = np.stack([np.stack([n_res[..., 0], n_res[..., 2]], -1),
                         np.stack([n_res[..., 2], n_res[..., 1]], -1)], -2)
        k_ww = np.einsum("p,tpai,tpaj->tij", w, hg, ahg) + np.einsum(
            "p,pki,tpkl,plj->tij", w, self.grad, nmat, self.grad) + self.k_b
        for i_, row in enumerate(md):
            k[:, row, md] = self.k_mm[i_]
            k[:, row, wd] = k_mw[:, i_]
            k[:, wd, row] = k_mw[:, i_]
        k[np.ix_(range(nt), wd, wd)] = k_ww
        return f, k

    def _beam_forces(self, ea, ei, length, d_nodes, along_x):
        """Stiffener beam between two nodes with DOFs (2, 5)."""
        if along_x:
            loc = np.array([d_nodes[0, 0], -d_nodes[0, 2], d_nodes[0, 3],
                            d_nodes[1, 0], -d_nodes[1, 2], d_nodes[1, 3]])
            pos = [0, 2, 3, 5, 7, 8]
            sgn = np.array([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
        else:
            loc = np.array([d_nodes[0, 0], d_nodes[0, 1], d_nodes[0, 4],
                            d_nodes[1, 0], d_nodes[1, 1], d_nodes[1, 4]])
            pos = [0, 1, 4, 5, 6, 9]
            sgn = np.ones(6)
        f_l, k_l = _vk_beam(ea, ei, length, loc, self.nonlinear)
        f = np.zeros(10)
        k = np.zeros((10, 10))
        f[pos] = sgn * f_l
        k[np.ix_(pos, pos)] = sgn[:, None] * k_l * sgn[None, :]
        return f, k

    # -- strip and section level ---------------------------------------
    def _full(self, d_sec):
        full = np.zeros(5 * self.nn)
        full[self.section_dofs] = d_sec
        return full

    def strip_force(self, i, d_left, d_right):
        """Internal force and tangent of strip i on its retained DOFs [left;
        right].

        :param i: Strip index.
        :type i: int
        :param d_left: Section displacements at the input side.
        :type d_left: numpy.ndarray
        :param d_right: Section displacements at the output side.
        :type d_right: numpy.ndarray
        :return: (f, K_T) of order 2 n_dof.
        :rtype: tuple
        """
        q_strip = np.concatenate([self._full(d_left), self._full(d_right)])
        q = q_strip[self.tri_dofs] * self.tri_flip
        f_t, k_t = self._triangle_forces(q)
        f_t *= self.tri_flip
        k_t *= self.tri_flip[:, :, None] * self.tri_flip[:, None, :]
        size = 10 * self.nn
        f = np.zeros(size)
        k = np.zeros((size, size))
        np.add.at(f, self.tri_dofs, f_t)
        np.add.at(k, (self.tri_dofs[:, :, None], self.tri_dofs[:, None, :]),
                  k_t)
        for j, ea, ei in self.x_stiffeners:
            idx = (list(range(5 * j, 5 * j + 5))
                   + list(range(5 * (self.nn + j), 5 * (self.nn + j) + 5)))
            fb, kb = self._beam_forces(ea, ei, self.dx,
                                       q_strip[idx].reshape(2, 5), True)
            f[idx] += fb
            k[np.ix_(idx, idx)] += kb
        keep = self.strip_keep
        return f[keep], k[np.ix_(keep, keep)]

    def section_force(self, i, d_sec):
        """Internal force and tangent of the stiffeners on section line i
        (parallel to y).

        :param i: Section index.
        :type i: int
        :param d_sec: Section displacements.
        :type d_sec: numpy.ndarray
        :return: (f, K_T) of order n_dof.
        :rtype: tuple
        """
        full = self._full(d_sec)
        f = np.zeros(5 * self.nn)
        k = np.zeros((5 * self.nn, 5 * self.nn))
        for sec, ea, ei in self.y_stiffeners:
            if sec != i:
                continue
            for j in range(self.n_cells):
                idx = list(range(5 * j, 5 * j + 10))
                fb, kb = self._beam_forces(ea, ei, self.dy,
                                           full[idx].reshape(2, 5), False)
                f[idx] += fb
                k[np.ix_(idx, idx)] += kb
        keep = self.section_dofs
        return f[keep], k[np.ix_(keep, keep)]

    def section_zero(self, fixed):
        """State indices [d; Q] that vanish at an x edge with the fixed node
        DOFs ``fixed``.

        :param fixed: Fixed node DOFs (0..4); the forces of the other DOFs are
            zero.
        :type fixed: list
        :return: Sorted indices.
        :rtype: list
        """
        nd = self.n_dof
        zero = []
        for pos, dof in enumerate(self.section_dofs):
            zero.append(pos if dof % 5 in fixed else nd + pos)
        return sorted(zero)

    def consistent_pressure(self, p):
        """Section loads (n_strips + 1, n_dof) of a uniform pressure p on w
        (Eq. 4.60)."""
        loads = np.zeros((self.n_strips + 1, 5 * self.nn))
        for (tri, _fl) in self.tris:
            for k_, node in enumerate(tri):
                side, j = divmod(node, self.nn)
                for i in range(self.n_strips):
                    loads[i + side,
                          5 * j:5 * j + 3] += p * self.load_w[3 * k_:3 * k_
                                                              + 3] * \
                        np.array([1.0, -1.0, -1.0] if _fl else [1.0, 1.0, 1.0])
        return loads[:, self.section_dofs]

    def w_index(self, section, node):
        """Position of w of ``node`` in the global section-major vector."""
        return section * self.n_dof + self.section_dofs.index(5 * node)


class RiccatiStripDynamics:
    """Wilson-theta transient analysis of a nonlinear strip model (Sec. 4.4,
    Eq. 4.87-4.116).

    Generalizes ``FERiccatiDynamics`` to strips with a nonlinear internal
    force (``VonKarmanPlateMesh``). Each step forms the tangent strips
    K_T,i(t) (Eq. 4.89), the strip matrices H_i = K_T,i + 6/(theta dt)^2 M_i
    + 3/(theta dt) C_i (Eq. 4.101), and iterates the modified Newton-Raphson
    method with the residual R(t + theta dt) - M a - C v - F(v) (see the
    note on Eq. 4.116 in ``FERiccatiDynamics``). The linear systems are
    solved by the Riccati transfer of Eq. (4.106)-(4.115) or by the global
    matrix (``solver='global'``).

    :param mesh: Strip model with ``strip_force``, ``section_force``,
        ``m_strip``, ``n_dof``, ``n_strips``.
    :type mesh: VonKarmanPlateMesh
    :param zero_in: Zero state indices at x = 0.
    :type zero_in: list
    :param zero_out: Zero state indices at x = a.
    :type zero_out: list
    :param theta: Wilson parameter.
    :type theta: float
    :param alpha_c: Mass-proportional damping.
    :type alpha_c: float
    """

    def __init__(self, mesh, zero_in, zero_out, theta=1.4, alpha_c=0.0):
        self.mesh = mesh
        self.nd = mesh.n_dof
        self.ns = mesh.n_strips + 1
        self.zero_in, self.zero_out = list(zero_in), list(zero_out)
        self.theta, self.alpha_c = float(theta), float(alpha_c)
        n_tot = self.ns * self.nd
        self.fixed_mask = np.zeros(n_tot, dtype=bool)
        for pos in self.zero_in:
            if pos < self.nd:
                self.fixed_mask[pos] = True
        for pos in self.zero_out:
            if pos < self.nd:
                self.fixed_mask[(self.ns - 1) * self.nd + pos] = True
        self.free = np.flatnonzero(~self.fixed_mask)
        self.m_full = np.zeros((n_tot, n_tot))
        for i in range(mesh.n_strips):
            sl = slice(i * self.nd, (i + 2) * self.nd)
            self.m_full[sl, sl] += mesh.m_strip

    def internal(self, v):
        """Global internal force, global tangent and the strip/section
        tangents."""
        nd, ns = self.nd, self.ns
        n_tot = ns * nd
        f = np.zeros(n_tot)
        k = np.zeros((n_tot, n_tot))
        ks, kc = [], []
        vv = v.reshape(ns, nd)
        for i in range(ns - 1):
            fi, ki = self.mesh.strip_force(i, vv[i], vv[i + 1])
            sl = slice(i * nd, (i + 2) * nd)
            f[sl] += fi
            k[sl, sl] += ki
            ks.append(ki)
        for i in range(ns):
            fi, ki = self.mesh.section_force(i, vv[i])
            sl = slice(i * nd, (i + 1) * nd)
            f[sl] += fi
            k[sl, sl] += ki
            kc.append(ki)
        return f, k, ks, kc

    def _solve_riccati(self, coef, ks, kc, rhs):
        nd = self.nd
        els = []
        rr = rhs.reshape(self.ns, nd)
        zeros = np.zeros(2 * nd)
        for i in range(self.ns):
            u_sec = np.eye(2 * nd)
            u_sec[nd:, :nd] = kc[i]
            els.append(_MatrixElement(u_sec,
                                      np.concatenate([np.zeros(nd), -rr[i]])))
            if i < self.ns - 1:
                u_strip = strip_transfer(ks[i] + coef * self.mesh.m_strip, nd)
                els.append(_MatrixElement(u_strip, zeros))
        chain = _ChainLike(els, self.zero_in, self.zero_out)
        res = _ric.riccati_steady_state(chain, 0.0, loads=[e.f for e in els])
        return np.concatenate([np.real(res["inputs"][2 * i][:nd])
                               for i in range(self.ns)])

    def _solve_global(self, coef, k_glob, rhs):
        h = k_glob + coef * self.m_full
        out = np.zeros_like(rhs)
        fr = self.free
        out[fr] = np.linalg.solve(h[np.ix_(fr, fr)], rhs[fr])
        return out

    def static(self, load, n_load_steps=10, tol=1e-10, max_iter=50):
        """Static large-deflection solution by load stepping and Newton (global
        matrix).

        :param load: Section loads (n_strips + 1, n_dof) of the final load.
        :type load: numpy.ndarray
        :param n_load_steps: Number of equal load increments.
        :type n_load_steps: int
        :param tol: Relative tolerance of the residual.
        :type tol: float
        :param max_iter: Newton iterations per increment.
        :type max_iter: int
        :return: Global displacement vector.
        :rtype: numpy.ndarray
        :raises RuntimeError: If Newton does not converge.
        """
        r_full = np.asarray(load, dtype=float).ravel().copy()
        r_full[self.fixed_mask] = 0.0
        v = np.zeros(self.ns * self.nd)
        fr = self.free
        for step in range(1, n_load_steps + 1):
            r = r_full * step / n_load_steps
            for _ in range(max_iter):
                f, k, _, _ = self.internal(v)
                res = r - f
                res[self.fixed_mask] = 0.0
                if np.linalg.norm(res) <= tol * np.linalg.norm(r):
                    break
                v[fr] += np.linalg.solve(k[np.ix_(fr, fr)], res[fr])
            else:
                raise RuntimeError("static Newton did not converge")
        return v

    def run(self, load, dt, n_steps, solver="riccati", tol=1e-10, max_iter=60):
        """Integrate from rest under the section loads load(t) (Eq. 4.87-4.105,
        4.116).

        :param load: Callable t -> (n_strips + 1, n_dof) external loads.
        :type load: callable
        :param dt: Time step.
        :type dt: float
        :param n_steps: Number of steps.
        :type n_steps: int
        :param solver: 'riccati' or 'global'.
        :type solver: str
        :param tol: Relative tolerance of the unbalanced force.
        :type tol: float
        :param max_iter: Maximum modified Newton iterations per step.
        :type max_iter: int
        :return: dict with ``t``, ``v``, ``iterations``.
        :rtype: dict
        :raises RuntimeError: If a step does not converge.
        """
        n_tot = self.ns * self.nd
        mf = self.m_full
        cm = self.alpha_c * mf
        fr = self.free
        v = np.zeros(n_tot)
        vd = np.zeros(n_tot)
        f_int, _, _, _ = self.internal(v)
        r0 = np.asarray(load(0.0), dtype=float).ravel().copy()
        r0[self.fixed_mask] = 0.0
        va = np.zeros(n_tot)
        va[fr] = np.linalg.solve(mf[np.ix_(fr, fr)],
                                 (r0 - cm @ vd - f_int)[fr])
        tau = self.theta * dt
        c1, c2 = 6.0 / tau**2, 3.0 / tau
        coef = c1 + c2 * self.alpha_c
        hist, iters = [v.copy()], []
        for step in range(n_steps):
            t = step * dt
            r_now = np.asarray(load(t), dtype=float).ravel()
            r_next = np.asarray(load(t + dt), dtype=float).ravel()
            r_tau = r_now + self.theta * (r_next - r_now)
            r_tau[self.fixed_mask] = 0.0
            _, k_glob, ks, kc = self.internal(v)
            dv = np.zeros(n_tot)
            for it in range(max_iter):
                acc = c1 * dv - 6.0 / tau * vd - 2.0 * va
                vel = c2 * dv - 2.0 * vd - 0.5 * tau * va
                f_int, _, _, _ = self.internal(v + dv)
                res = r_tau - mf @ acc - cm @ vel - f_int
                res[self.fixed_mask] = 0.0
                scale = max(np.linalg.norm(r_tau), np.linalg.norm(f_int),
                            1e-300)
                if it > 0 and np.linalg.norm(res) <= tol * scale:
                    break
                if solver == "riccati":
                    dv = dv + self._solve_riccati(coef, ks, kc, res)
                else:
                    dv = dv + self._solve_global(coef, k_glob, res)
            else:
                raise RuntimeError("modified Newton did not converge in step "
                                   "%d" % step)
            iters.append(it)
            a_new = (6.0 / (self.theta**3 * dt * dt) * dv
                     - 6.0 / (self.theta**2 * dt) * vd
                     + (1.0 - 3.0 / self.theta) * va)
            v = v + dt * vd + dt * dt / 6.0 * (a_new + 2.0 * va)
            vd = vd + 0.5 * dt * (a_new + va)
            va = a_new
            hist.append(v.copy())
        return {"t": dt * np.arange(n_steps + 1), "v": np.array(hist),
                "iterations": iters}

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
