# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Discrete time transfer matrix method for multi-rigid-body systems (Ch. 7).

The state variables are positions, angles, internal moments and internal
forces at the time instant ``t_i``:

* lumped-mass chain (Eq. 7.9): ``[x, q, 1]``,
* planar state (Eq. 7.21): ``[x, y, theta, m, q_x, q_y, 1]``,
* spatial state (Eq. 7.20): ``[x, y, z, theta_x, theta_y, theta_z, m_x,
  m_y, m_z, q_x, q_y, q_z, 1]`` with space-fixed x-y-z angles
  (Eq. 7.71).

A step-by-step integration method gives the velocity and the acceleration
at ``t_i`` as linear functions of the position at ``t_i`` (Eqs. 7.27,
7.28)::

    xdd(t_i) = A x(t_i) + B,     xd(t_i) = C x(t_i) + D.

Trigonometric functions and products are linearized with Eqs. (7.61) to
(7.66) and the rotation matrix with Eqs. (7.83), (7.84). Then every element
relation is linear in the state vector at ``t_i`` and the overall transfer
equation is solved once per time step (algorithm of Sec. 7.8).

Coefficient functions of the unknown angle (for example ``x_IO`` in
Eq. 7.102) use the predicted values of Eq. (7.62); the local error of every
linearization is then ``O(dT^3)`` and the Newmark average acceleration
method keeps its second order of accuracy.

Sign convention (Eqs. 7.90-7.93, 7.152-7.156): ``q`` at a connection point
is the force on the outboard element, ``m`` is the moment on the inboard
element. Hinge elimination and the ground state follow the module
docstring of :mod:`solvcon.multibody.tmm.rigid`.
"""

import numpy as np

from . import rigid

# ---------------------------------------------------------------------------
# Linearization of velocity and acceleration (Sec. 7.3.1, 7.3.2, Table 7.2)
# ---------------------------------------------------------------------------


class Variable:
    """History of one coordinate (scalar or vector) for a linearizer.

    :ivar x: positions, newest last (at most four entries)
    :vartype x: list
    :ivar v: velocity at the newest time instant
    :vartype v: numpy.ndarray
    :ivar a: acceleration at the newest time instant
    :vartype a: numpy.ndarray
    :ivar a_old: acceleration at the time instant before the newest one
    :vartype a_old: numpy.ndarray or None
    :ivar steps: number of completed time steps
    :vartype steps: int
    """

    def __init__(self, x0, v0, a0):
        """Initialize the history at t_0.

        :param x0: initial position
        :type x0: array_like
        :param v0: initial velocity
        :type v0: array_like
        :param a0: initial acceleration
        :type a0: array_like
        """
        self.x = [np.array(x0, dtype=float)]
        self.v = np.array(v0, dtype=float)
        self.a = np.array(a0, dtype=float)
        self.a_old = None
        self.steps = 0

    def push(self, x, v, a):
        """Store the solution at the new time instant.

        :param x: position
        :type x: array_like
        :param v: velocity
        :type v: array_like
        :param a: acceleration
        :type a: array_like
        """
        self.x.append(np.array(x, dtype=float))
        if len(self.x) > 4:
            self.x.pop(0)
        self.a_old = self.a
        self.v = np.array(v, dtype=float)
        self.a = np.array(a, dtype=float)
        self.steps += 1


class Linearizer:
    """Base class of the step-by-step integration methods (Table 7.2)."""

    def __init__(self, dt):
        """Initialize the method.

        :param dt: time step size
        :type dt: float
        """
        self.dt = float(dt)

    def coefficients(self, var):
        """Return (A, B, C, D) with xdd = A x + B and xd = C x + D.

        :param var: history of the coordinate
        :type var: Variable
        :rtype: tuple
        """
        raise NotImplementedError

    def update(self, var, x_new):
        """Append the solution at t_i and its velocity and acceleration.

        :param var: history of the coordinate
        :type var: Variable
        :param x_new: position at t_i
        :type x_new: array_like
        """
        a_c, b, c_c, d = self.coefficients(var)
        x_new = np.asarray(x_new, dtype=float)
        var.push(x_new, c_c * x_new + d, a_c * x_new + b)


class ForwardEuler(Linearizer):
    """Forward-Euler linearization (Eqs. 7.29-7.32), first order."""

    def coefficients(self, var):
        """Return (A, B, C, D) of Eq. (7.32).

        :param var: history of the coordinate
        :type var: Variable
        :rtype: tuple
        """
        dt = self.dt
        x, v = var.x[-1], var.v
        a_c = 2.0 / dt ** 2
        c_c = 2.0 / dt
        return a_c, -a_c * (x + v * dt), c_c, -c_c * x - v


class Newmark(Linearizer):
    """Newmark-beta linearization (Eqs. 7.33-7.39).

    ``beta = 1/4, gamma = 1/2`` is the average acceleration method,
    which is unconditionally stable and second-order accurate.

    :ivar beta: Newmark parameter beta
    :vartype beta: float
    :ivar gamma: Newmark parameter gamma
    :vartype gamma: float
    """

    def __init__(self, dt, beta=0.25, gamma=0.5):
        """Initialize the method.

        :param dt: time step size
        :type dt: float
        :param beta: parameter beta
        :type beta: float
        :param gamma: parameter gamma
        :type gamma: float
        """
        super().__init__(dt)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def coefficients(self, var):
        """Return (A, B, C, D) of Eq. (7.39).

        :param var: history of the coordinate
        :type var: Variable
        :rtype: tuple
        """
        dt, beta, gamma = self.dt, self.beta, self.gamma
        x, v, a = var.x[-1], var.v, var.a
        a_c = 1.0 / (beta * dt ** 2)
        b = -a_c * (x + dt * v + (0.5 - beta) * dt ** 2 * a)
        c_c = gamma / (beta * dt)
        d = v + dt * (1.0 - gamma) * a + gamma * dt * b
        return a_c, b, c_c, d


class Houbolt(Linearizer):
    """Houbolt linearization with its starting procedure (Eqs. 7.47-7.60).

    Step 1 uses the one-step method (linear acceleration, Eq. 7.60),
    step 2 the two-step method (Eq. 7.56) and later steps the four-point
    formula (Eq. 7.50). The one-step velocity uses the trapezoidal rule
    of the linear acceleration assumption,
    ``D = -(6 x + 4 xd dT + xdd dT^2) / (2 dT)``.
    """

    def coefficients(self, var):
        """Return (A, B, C, D) of Table 7.2 for the current step number.

        :param var: history of the coordinate
        :type var: Variable
        :rtype: tuple
        """
        dt = self.dt
        if var.steps == 0:
            x, v, a = var.x[-1], var.v, var.a
            a_c = 6.0 / dt ** 2
            b = -2.0 / dt ** 2 * (3.0 * x + 3.0 * dt * v + dt ** 2 * a)
            c_c = 3.0 / dt
            d = -(6.0 * x + 4.0 * dt * v + dt ** 2 * a) / (2.0 * dt)
            return a_c, b, c_c, d
        if var.steps == 1:
            xn, xn1 = var.x[-1], var.x[-2]
            an1 = var.a_old
            a_c = 2.0 / dt ** 2
            b = -(4.0 * xn - 2.0 * xn1 + an1 * dt ** 2) / dt ** 2
            c_c = 11.0 / (6.0 * dt)
            d = -(16.0 * xn - 5.0 * xn1 + 2.0 * an1 * dt ** 2) / (6.0 * dt)
            return a_c, b, c_c, d
        xn, xn1, xn2 = var.x[-1], var.x[-2], var.x[-3]
        a_c = 2.0 / dt ** 2
        b = -(5.0 * xn - 4.0 * xn1 + xn2) / dt ** 2
        c_c = 11.0 / (6.0 * dt)
        d = -(18.0 * xn - 9.0 * xn1 + 2.0 * xn2) / (6.0 * dt)
        return a_c, b, c_c, d


def wilson_theta_coefficients(x, v, a, dt, theta=1.4):
    """Return the Wilson-theta coefficients of Eq. (7.46).

    The coefficients relate the acceleration and velocity at
    ``t + theta dT`` to the position at ``t + theta dT``.

    :param x: position at t
    :type x: array_like
    :param v: velocity at t
    :type v: array_like
    :param a: acceleration at t
    :type a: array_like
    :param dt: time step size
    :type dt: float
    :param theta: Wilson parameter (theta >= 1)
    :type theta: float
    :return: (A, B, C, D)
    :rtype: tuple
    """
    x, v, a = (np.asarray(u, dtype=float) for u in (x, v, a))
    td = theta * dt
    a_c = 6.0 / td ** 2
    b = -a_c * (x + td * v + td ** 2 / 3.0 * a)
    c_c = 3.0 / td
    d = -3.0 / td * x - 2.0 * v - 0.5 * td * a
    return a_c, b, c_c, d


def make_linearizer(name, dt, **kwargs):
    """Return a linearizer by name.

    :param name: 'newmark', 'houbolt' or 'euler'
    :type name: str
    :param dt: time step size
    :type dt: float
    :return: linearizer instance
    :rtype: Linearizer
    :raises ValueError: for an unknown name
    """
    if name == "newmark":
        return Newmark(dt, **kwargs)
    if name == "houbolt":
        return Houbolt(dt)
    if name == "euler":
        return ForwardEuler(dt)
    raise ValueError("unknown linearizer: %s" % name)


# ---------------------------------------------------------------------------
# Linearization of nonlinear functions (Sec. 7.3.3, 7.3.4)
# ---------------------------------------------------------------------------

def trig_linearization(theta, theta_d, theta_dd, dt):
    """Linearize sin and cos of the angle at t_i (Eqs. 7.61-7.64).

    :param theta: angle at t_{i-1}
    :type theta: float or numpy.ndarray
    :param theta_d: angular velocity at t_{i-1}
    :type theta_d: float or numpy.ndarray
    :param theta_dd: angular acceleration at t_{i-1}
    :type theta_dd: float or numpy.ndarray
    :param dt: time step size
    :type dt: float
    :return: (s_bar, c_bar, G1, G2) with sin(theta_i) ~ s_bar (explicit,
        Eq. 7.62), sin(theta_i) ~ cos(theta) theta_i + G2 and
        cos(theta_i) ~ -sin(theta) theta_i + G1 (Eqs. 7.63, 7.64)
    :rtype: tuple
    """
    s, c = np.sin(theta), np.cos(theta)
    h = 1.0 - 0.5 * (theta_d * dt) ** 2
    p = theta_d * dt + 0.5 * theta_dd * dt ** 2
    s_bar = s * h + c * p
    c_bar = c * h - s * p
    g2 = s * h - theta * c
    g1 = c * h + theta * s
    return s_bar, c_bar, g1, g2


def product_linearization(a_prev, b_prev, ad_prev, bd_prev, dt):
    """Return the coefficients of the linearized product (Eq. 7.65).

    ``a(t_i) b(t_i) ~ ca * a(t_i) + cb * b(t_i) + c0``.

    :param a_prev: a at t_{i-1}
    :type a_prev: float
    :param b_prev: b at t_{i-1}
    :type b_prev: float
    :param ad_prev: time derivative of a at t_{i-1}
    :type ad_prev: float
    :param bd_prev: time derivative of b at t_{i-1}
    :type bd_prev: float
    :param dt: time step size
    :type dt: float
    :return: (ca, cb, c0)
    :rtype: tuple
    """
    return b_prev, a_prev, -a_prev * b_prev + ad_prev * bd_prev * dt ** 2


def h_matrix(theta):
    """Return H of Eq. (7.144): body angular velocity = H theta_dot.

    :param theta: x-y-z angles
    :type theta: array_like
    :rtype: numpy.ndarray
    """
    sx, sy = np.sin(theta[0]), np.sin(theta[1])
    cx, cy = np.cos(theta[0]), np.cos(theta[1])
    return np.array([[1.0, 0.0, -sy], [0.0, cx, sx * cy], [0.0, -sx, cx * cy]])


def h_dot_matrix(theta, theta_d):
    """Return the time derivative of H (Eq. 7.149).

    :param theta: x-y-z angles
    :type theta: array_like
    :param theta_d: angle rates
    :type theta_d: array_like
    :rtype: numpy.ndarray
    """
    sx, sy = np.sin(theta[0]), np.sin(theta[1])
    cx, cy = np.cos(theta[0]), np.cos(theta[1])
    tx, ty = theta_d[0], theta_d[1]
    return np.array([
        [0.0, 0.0, -cy * ty],
        [0.0, -sx * tx, cx * cy * tx - sx * sy * ty],
        [0.0, -cx * tx, -sx * cy * tx - cx * sy * ty],
    ])


def t_vectors(theta):
    """Return T1, T2, T3 (Eq. 7.77) and T12, T23, T13 (Eq. 7.163).

    :param theta: x-y-z angles
    :type theta: array_like
    :return: (T1, T2, T3, T12, T23, T13)
    :rtype: tuple
    """
    sx, sy = np.sin(theta[0]), np.sin(theta[1])
    cx, cy = np.cos(theta[0]), np.cos(theta[1])
    t1 = np.array([1.0, 0.0, 0.0])
    t2 = np.array([0.0, cx, -sx])
    t3 = np.array([-sy, sx * cy, cx * cy])
    t12 = np.array([0.0, -sx, -cx])
    t23 = np.array([-cy, -sx * sy, -cx * sy])
    t13 = np.array([0.0, cx * cy, -sx * cy])
    return t1, t2, t3, t12, t23, t13


def rotation_linearization(theta, theta_d, dt):
    """Linearize the rotation matrix at t_i (Eqs. 7.73-7.84).

    ``A(t_i) ~ sum_k A T~_k theta_k(t_i) + Phi`` with all factors at
    t_{i-1}.

    :param theta: angles at t_{i-1}
    :type theta: array_like
    :param theta_d: angle rates at t_{i-1}
    :type theta_d: array_like
    :param dt: time step size
    :type dt: float
    :return: (A, [A T~1, A T~2, A T~3], Phi)
    :rtype: tuple
    """
    a = rigid.rotation_xyz(theta)
    t1, t2, t3 = (rigid.skew(t) for t in t_vectors(theta)[:3])
    tx, ty, tz = theta_d
    th = np.asarray(theta, dtype=float)
    second = (0.5 * (t1 @ t1 * tx ** 2 + t2 @ t2 * ty ** 2 + t3 @ t3 * tz ** 2)
              + t2 @ t1 * ty * tx + t3 @ t2 * tz * ty + t3 @ t1 * tz * tx)
    phi = a @ (np.eye(3) - t1 * th[0] - t2 * th[1] - t3 * th[2]
               + second * dt ** 2)
    return a, [a @ t1, a @ t2, a @ t3], phi


# ---------------------------------------------------------------------------
# Lumped mass and spring (Sec. 7.2)
# ---------------------------------------------------------------------------

def spring_matrix_1d(stiffness, length):
    """Transfer matrix of a massless linear spring (Eq. 7.11).

    :param stiffness: spring stiffness K
    :type stiffness: float
    :param length: free length l
    :type length: float
    :return: 3x3 matrix for the state [x, q, 1]
    :rtype: numpy.ndarray
    """
    return np.array([[1.0, -1.0 / stiffness, length], [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]])


def lumped_mass_matrix_1d(mass, a_c, b, force=0.0):
    """Transfer matrix of a lumped mass (Eqs. 7.12-7.14).

    :param mass: mass m
    :type mass: float
    :param a_c: linearization coefficient A
    :type a_c: float
    :param b: linearization coefficient B of the mass position
    :type b: float
    :param force: external force f
    :type force: float
    :return: 3x3 matrix for the state [x, q, 1]
    :rtype: numpy.ndarray
    """
    return np.array([[1.0, 0.0, 0.0], [-mass * a_c, 1.0, -mass * b + force],
                     [0.0, 0.0, 1.0]])


class SpringMassChain:
    """Chain of springs and lumped masses moving along x (Fig. 7.1).

    Spring k joins mass k-1 (the wall for k = 0) to mass k. The wall is at
    ``x = 0`` and the last mass is free (boundary conditions after
    Eq. 7.19).

    :ivar stiffness: spring stiffnesses
    :vartype stiffness: numpy.ndarray
    :ivar length: spring free lengths
    :vartype length: numpy.ndarray
    :ivar mass: lumped masses
    :vartype mass: numpy.ndarray
    :ivar force: constant external forces on the masses
    :vartype force: numpy.ndarray
    """

    def __init__(self, stiffness, length, mass, force):
        """Initialize the chain.

        :param stiffness: spring stiffnesses
        :type stiffness: array_like
        :param length: spring free lengths
        :type length: array_like
        :param mass: masses
        :type mass: array_like
        :param force: external forces
        :type force: array_like
        """
        self.stiffness = np.asarray(stiffness, dtype=float)
        self.length = np.asarray(length, dtype=float)
        self.mass = np.asarray(mass, dtype=float)
        self.force = np.asarray(force, dtype=float)

    def accelerations(self, x):
        """Return the exact accelerations for given positions.

        :param x: mass positions
        :type x: numpy.ndarray
        :rtype: numpy.ndarray
        """
        n = len(self.mass)
        left = np.concatenate(([0.0], x[:-1]))
        tension = self.stiffness * (x - left - self.length)
        acc = (-tension + self.force) / self.mass
        acc[:n - 1] += tension[1:] / self.mass[:n - 1]
        return acc

    def simulate(self, x0, v0, t_end, dt, method="newmark", solver="riccati",
                 **kwargs):
        """Integrate with the DT-TMM (Eqs. 7.15-7.19).

        ``solver="product"`` forms the overall transfer matrix (Eq. 7.19) and
        propagates the base state. Its entries grow like
        ``(m / (beta dT^2 K))^n`` and the propagation loses about that many
        digits for small time steps. ``solver="riccati"`` keeps the relation
        ``x = S q + e`` along the chain (Riccati transfer, Sec. 10.9) and is
        stable for any time step.

        :param x0: initial positions
        :type x0: array_like
        :param v0: initial velocities
        :type v0: array_like
        :param t_end: final time
        :type t_end: float
        :param dt: time step
        :type dt: float
        :param method: linearizer name (see :func:`make_linearizer`)
        :type method: str
        :param solver: 'riccati' or 'product'
        :type solver: str
        :return: (t, x) with x of shape (N, n)
        :rtype: tuple
        """
        lin = make_linearizer(method, dt, **kwargs)
        x0 = np.asarray(x0, dtype=float)
        hist = Variable(x0, v0, self.accelerations(x0))
        steps = int(round(t_end / dt))
        out = [x0.copy()]
        n = len(self.mass)
        for _ in range(steps):
            a_c, b, _, _ = lin.coefficients(hist)
            mats = []
            for k in range(n):
                us = spring_matrix_1d(self.stiffness[k], self.length[k])
                um = lumped_mass_matrix_1d(self.mass[k], a_c, b[k],
                                           self.force[k])
                mats.append(um @ us)
            x = self._solve(mats, solver)
            lin.update(hist, x)
            out.append(x)
        return dt * np.arange(steps + 1), np.array(out)

    @staticmethod
    def _solve(mats, solver):
        """Solve the chain for z_0 = [0, q0, 1] and q_n = 0.

        :param mats: subsystem matrices U_{j+1} U_j (Eq. 7.16)
        :type mats: list
        :param solver: 'riccati' or 'product'
        :type solver: str
        :return: mass positions
        :rtype: numpy.ndarray
        """
        n = len(mats)
        if solver == "product":
            total = np.eye(3)
            for u in mats:
                total = u @ total
            z = np.array([0.0, -total[1, 2] / total[1, 1], 1.0])
            x = np.empty(n)
            for k, u in enumerate(mats):
                z = u @ z
                x[k] = z[0]
            return x
        # Riccati transfer: x_k = s_k q_k + e_k, x_0 = 0.
        s, e = 0.0, 0.0
        coef = []
        for u in mats:
            a, bb, f = u[0]
            c, d, g = u[1]
            den = c * s + d
            coef.append((s, e, c, den, g))
            s, e = ((a * s + bb) / den,
                    a * e + f - (a * s + bb) * (c * e + g) / den)
        x = np.empty(n)
        q = 0.0
        x[n - 1] = s * q + e
        for k in range(n - 1, 0, -1):
            s_k, e_k, c, den, g = coef[k]
            q = (q - c * e_k - g) / den
            x[k - 1] = s_k * q + e_k
        return x


# ---------------------------------------------------------------------------
# Planar elements (Secs. 7.4, 7.6)
# ---------------------------------------------------------------------------

class PlanarBodyState:
    """Linearization histories of a planar rigid body.

    :ivar theta: body angle
    :vartype theta: Variable
    :ivar r_i: input-end position
    :vartype r_i: Variable
    :ivar r_c: mass-center position
    :vartype r_c: Variable
    :ivar r_o: output-end positions, one per output end
    :vartype r_o: list
    """

    def __init__(self, theta, r_i, r_c, r_o):
        """Initialize the histories.

        :param theta: angle history
        :type theta: Variable
        :param r_i: input-end history
        :type r_i: Variable
        :param r_c: mass-center history
        :type r_c: Variable
        :param r_o: output-end histories
        :type r_o: list
        """
        self.theta = theta
        self.r_i = r_i
        self.r_c = r_c
        self.r_o = list(r_o)

    @classmethod
    def from_motion(cls, points, theta, omega, alpha, r_i, v_i, a_i):
        """Build the histories from the exact motion at t_0.

        :param points: body-frame points [C, O1, O2, ...] relative to I
        :type points: list
        :param theta: angle
        :type theta: float
        :param omega: angular velocity
        :type omega: float
        :param alpha: angular acceleration
        :type alpha: float
        :param r_i: input-end position
        :type r_i: array_like
        :param v_i: input-end velocity
        :type v_i: array_like
        :param a_i: input-end acceleration
        :type a_i: array_like
        :rtype: PlanarBodyState
        """
        r = rigid.rot2(theta)
        dr = rigid.rot2(theta + 0.5 * np.pi)
        vars_ = []
        for p in points:
            rp = r @ np.asarray(p, dtype=float)
            drp = dr @ np.asarray(p, dtype=float)
            vars_.append(Variable(np.asarray(r_i) + rp,
                                  np.asarray(v_i) + omega * drp,
                                  np.asarray(a_i) + alpha * drp
                                  - omega ** 2 * rp))
        return cls(Variable(theta, omega, alpha), Variable(r_i, v_i, a_i),
                   vars_[0], vars_[1:])


def _planar_geometry(point, theta_var, dt):
    """Return the linearized position of a body point (Eqs. 7.94-7.95, 7.106).

    ``r_P = r_I + g_theta * theta(t_i) + g_0``.

    :param point: body-frame point relative to I
    :type point: array_like
    :param theta_var: angle history
    :type theta_var: Variable
    :param dt: time step size
    :type dt: float
    :return: (g_theta, g_0, p_bar, p_prev) where p_bar is the predicted
        offset (Eq. 7.62) and p_prev the offset at t_{i-1}
    :rtype: tuple
    """
    th = float(theta_var.x[-1])
    s_bar, c_bar, g1, g2 = trig_linearization(th, float(theta_var.v),
                                              float(theta_var.a), dt)
    b1, b2 = point
    p_prev = rigid.rot2(th) @ np.asarray(point, dtype=float)
    g_theta = np.array([-p_prev[1], p_prev[0]])
    g_0 = np.array([b1 * g1 - b2 * g2, b1 * g2 + b2 * g1])
    p_bar = np.array([b1 * c_bar - b2 * s_bar, b1 * s_bar + b2 * c_bar])
    return g_theta, g_0, p_bar, p_prev


def planar_body_matrix_dt(body, state, lin, f_c=(0.0, 0.0), m_c=0.0, out=0):
    """DT-TMM transfer matrix of a planar rigid body (Eqs. 7.85-7.103).

    :param body: body with mass, j_i, l_io, l_ic (and optionally a list
        ``outputs`` of body-frame output points)
    :type body: solvcon.multibody.tmm.rigid.PlanarBody
    :param state: linearization histories of the body
    :type state: PlanarBodyState
    :param lin: step-by-step integration method
    :type lin: Linearizer
    :param f_c: external force at C
    :type f_c: array_like
    :param m_c: external moment
    :type m_c: float
    :param out: index of the output end (0 for l_io)
    :type out: int
    :return: 7x7 matrix U with z_O = U z_I
    :rtype: numpy.ndarray
    """
    dt = lin.dt
    m = body.mass
    a_c, b_th, _, _ = lin.coefficients(state.theta)
    _, b_ri, _, _ = lin.coefficients(state.r_i)
    _, b_rc, _, _ = lin.coefficients(state.r_c)
    l_io = body_outputs(body)[out]
    go_th, go_0, p_io, _ = _planar_geometry(l_io, state.theta, dt)
    gc_th, gc_0, p_ic, _ = _planar_geometry(body.l_ic, state.theta, dt)
    x_io, y_io = p_io
    x_ic, y_ic = p_ic
    # x_IC(t_{i-1}), y_IC(t_{i-1}) of Eq. (7.103): gc_th = [-y_IC, x_IC].
    x_ic0, y_ic0 = gc_th[1], -gc_th[0]
    fx, fy = f_c
    b_th = float(b_th)
    u57 = fx - m * a_c * gc_0[0] - m * b_rc[0]
    u67 = fy - m * a_c * gc_0[1] - m * b_rc[1]
    u = np.zeros((7, 7))
    u[0, 0] = u[1, 1] = u[2, 2] = u[3, 3] = u[4, 4] = u[5, 5] = u[6, 6] = 1.0
    u[0:2, 2] = go_th
    u[0:2, 6] = go_0
    u[3, 0] = m * a_c * (y_io - y_ic)
    u[3, 1] = -m * a_c * (x_io - x_ic)
    u[3, 2] = -m * a_c * x_ic0 * x_io - m * a_c * y_ic0 * y_io + body.j_i * a_c
    u[3, 4] = -y_io
    u[3, 5] = x_io
    u[3, 6] = (-m_c + u67 * x_io - u57 * y_io + body.j_i * b_th
               + (m * b_ri[1] - fy) * x_ic + (fx - m * b_ri[0]) * y_ic)
    u[4, 0] = -m * a_c
    u[4, 2] = m * a_c * y_ic0
    u[4, 6] = u57
    u[5, 1] = -m * a_c
    u[5, 2] = -m * a_c * x_ic0
    u[5, 6] = u67
    return u


def body_outputs(body):
    """Return the list of body-frame output points of a planar body.

    :param body: planar body
    :type body: solvcon.multibody.tmm.rigid.PlanarBody
    :rtype: list
    """
    outs = getattr(body, "outputs", None)
    return list(outs) if outs else [body.l_io]


def planar_smooth_hinge_matrix_dt(u_out):
    """DT-TMM transfer matrix of a smooth planar hinge (Eqs. 7.216-7.219).

    The angle of the outboard body follows from the zero moment at its
    outboard end (Eq. 7.218). The input moment is kept with ``u_44 = 1``
    (the alternative named after Eq. 7.219); it is zero in a chain.

    :param u_out: 7x7 DT-TMM matrix of the outboard body
    :type u_out: numpy.ndarray
    :return: 7x7 hinge matrix
    :rtype: numpy.ndarray
    """
    u = np.eye(7)
    row = u_out[3]
    u[2, :] = -row / row[2]
    u[2, 2] = 0.0
    return u


def planar_spring_damper_hinge_matrix_dt(k, c, c_c, d_in, d_out):
    """DT-TMM matrix of a linear planar spring-damper hinge (Eqs. 7.197-7.215).

    Translational parts: ``-q = k (r_O - r_I) + c (rd_O - rd_I)``.
    Rotary part: ``m = k_r (theta_O - theta_I) + c_r (thd_O - thd_I)``.
    The velocities use ``rd = C r + D`` (Eq. 7.28).

    :param k: stiffnesses [k_x, k_y, k_r]
    :type k: array_like
    :param c: damping coefficients [c_x, c_y, c_r]
    :type c: array_like
    :param c_c: velocity coefficient C of the integration method (Eq. 7.28)
    :type c_c: float
    :param d_in: velocity coefficients D of [x_I, y_I, theta_I]
    :type d_in: array_like
    :param d_out: velocity coefficients D of [x_O, y_O, theta_O]
    :type d_out: array_like
    :return: 7x7 hinge matrix
    :rtype: numpy.ndarray
    """
    k = np.asarray(k, dtype=float)
    c = np.asarray(c, dtype=float)
    dd = np.asarray(d_out, dtype=float) - np.asarray(d_in, dtype=float)
    den = k + c * c_c
    u = np.eye(7)
    u[0, 4] = -1.0 / den[0]
    u[1, 5] = -1.0 / den[1]
    u[0, 6] = -c[0] * dd[0] / den[0]
    u[1, 6] = -c[1] * dd[1] / den[1]
    u[2, 3] = 1.0 / den[2]
    u[2, 6] = -c[2] * dd[2] / den[2]
    return u


class PlanarChainDT:
    """Planar chain of rigid bodies solved with the DT-TMM (Sec. 7.9.2).

    Hinge 0 joins the ground point ``base`` to body 0; hinge k joins body
    k-1 to body k; the last output end is free. A hinge is ``None`` for a
    smooth hinge (Eq. 7.219) or a tuple ``(k, c)`` of stiffness and damping
    vectors for a spring-damper hinge (Eqs. 7.208, 7.215).

    :ivar bodies: bodies from the base to the tip
    :vartype bodies: list
    :ivar hinges: hinge descriptions
    :vartype hinges: list
    :ivar gravity: gravity acceleration
    :vartype gravity: numpy.ndarray
    :ivar base: ground point
    :vartype base: numpy.ndarray
    """

    def __init__(self, bodies, gravity=(0.0, -9.81), base=(0.0, 0.0),
                 hinges=None):
        """Initialize the chain.

        :param bodies: list of solvcon.multibody.tmm.rigid.PlanarBody
        :type bodies: list
        :param gravity: gravity acceleration
        :type gravity: array_like
        :param base: ground point of hinge 0
        :type base: array_like
        :param hinges: hinge descriptions (default: all smooth)
        :type hinges: list or None
        """
        self.bodies = list(bodies)
        self.gravity = np.asarray(gravity, dtype=float)
        self.base = np.asarray(base, dtype=float)
        self.hinges = (list(hinges) if hinges is not None
                       else [None] * len(bodies))
        for k in range(len(self.hinges) - 1):
            if self.hinges[k] is None and self.hinges[k + 1] is not None:
                raise ValueError("a smooth hinge needs a zero moment at the "
                                 "outboard end of its outboard body (Sec. "
                                 "7.6.3.1)")

    def _initial_states(self, theta0, omega0):
        """Return the body histories at t_0 with exact accelerations.

        :param theta0: initial angles
        :type theta0: array_like
        :param omega0: initial angular velocities
        :type omega0: array_like
        :rtype: list
        """
        ch = rigid.PlanarChain(self.bodies, self.gravity, self.base)
        # Spring-damper hinges start undeflected in translation; their rotary
        # moment acts on the two bodies as an external moment (Sec. 5.8).
        m_ext = np.zeros(len(self.bodies))
        for k, h in enumerate(self.hinges):
            if h is None:
                continue
            kk, cc = h
            dth = theta0[k] - (theta0[k - 1] if k else 0.0)
            dw = omega0[k] - (omega0[k - 1] if k else 0.0)
            mom = kk[2] * dth + cc[2] * dw
            m_ext[k] -= mom
            if k:
                m_ext[k - 1] += mom
        states, alpha = ch.transfer(theta0, omega0, m_ext)
        r_in, _, _ = ch.positions(theta0)
        out = []
        v = np.zeros(2)
        for k, b in enumerate(self.bodies):
            a_i = states[k][0][0:2]
            st = PlanarBodyState.from_motion([b.l_ic, b.l_io], theta0[k],
                                             omega0[k], alpha[k], r_in[k], v,
                                             a_i)
            out.append(st)
            v = st.r_o[0].v
        return out

    def step(self, states, lin):
        """Advance all body histories by one time step.

        :param states: body histories
        :type states: list
        :param lin: step-by-step integration method
        :type lin: Linearizer
        :return: (z_in, z_out) state vectors per body at t_i
        :rtype: list
        """
        n = len(self.bodies)
        umats = [planar_body_matrix_dt(b, s, lin, b.mass * self.gravity)
                 for b, s in zip(self.bodies, states)]
        hmats = []
        for k in range(n):
            h = self.hinges[k]
            if h is None:
                hmats.append(planar_smooth_hinge_matrix_dt(umats[k]))
            else:
                kk, cc = h
                if k == 0:
                    d_in = np.zeros(3)
                else:
                    sp = states[k - 1]
                    d_in = np.concatenate([
                        lin.coefficients(sp.r_o[0])[3],
                        np.atleast_1d(lin.coefficients(sp.theta)[3])])
                s = states[k]
                d_out = np.concatenate([
                    lin.coefficients(s.r_i)[3],
                    np.atleast_1d(lin.coefficients(s.theta)[3])])
                c_c = lin.coefficients(s.theta)[2]
                hmats.append(planar_spring_damper_hinge_matrix_dt(kk, cc, c_c,
                                                                  d_in, d_out))
        total = np.eye(7)
        for u, h in zip(umats, hmats):
            total = u @ h @ total
        # Ground state [x0, y0, 0, m, q_x, q_y, 1] with m = 0 for a smooth
        # base hinge; tip m = q = 0.
        z0 = np.zeros(7)
        z0[0:2] = self.base
        z0[6] = 1.0
        rows = [3, 4, 5]
        cols = [4, 5] if self.hinges[0] is None else [3, 4, 5]
        q0, *_ = np.linalg.lstsq(total[np.ix_(rows, cols)],
                                 -(total[rows] @ z0), rcond=None)
        z = z0.copy()
        z[cols] = q0
        result = []
        for k, (u, h) in enumerate(zip(umats, hmats)):
            z_in = h @ z
            z_out = u @ z_in
            result.append((z_in, z_out))
            z = z_out
        for (z_in, z_out), b, s in zip(result, self.bodies, states):
            gc_th, gc_0, _, _ = _planar_geometry(b.l_ic, s.theta, lin.dt)
            r_c = z_in[0:2] + gc_th * z_in[2] + gc_0
            lin.update(s.theta, z_in[2])
            lin.update(s.r_i, z_in[0:2])
            lin.update(s.r_c, r_c)
            lin.update(s.r_o[0], z_out[0:2])
        return result

    def simulate(self, theta0, omega0, t_end, dt, method="newmark", **kwargs):
        """Integrate the motion with the DT-TMM.

        :param theta0: initial angles
        :type theta0: array_like
        :param omega0: initial angular velocities
        :type omega0: array_like
        :param t_end: final time
        :type t_end: float
        :param dt: time step
        :type dt: float
        :param method: linearizer name
        :type method: str
        :return: dict with t, theta (N, n), omega (N, n), r_in (N, n, 2)
        :rtype: dict
        """
        lin = make_linearizer(method, dt, **kwargs)
        theta0 = np.asarray(theta0, dtype=float)
        omega0 = np.asarray(omega0, dtype=float)
        states = self._initial_states(theta0, omega0)
        steps = int(round(t_end / dt))
        n = len(self.bodies)
        th = np.empty((steps + 1, n))
        om = np.empty((steps + 1, n))
        ri = np.empty((steps + 1, n, 2))

        def record(i):
            for k, s in enumerate(states):
                th[i, k] = s.theta.x[-1]
                om[i, k] = s.theta.v
                ri[i, k] = s.r_i.x[-1]

        record(0)
        for i in range(steps):
            self.step(states, lin)
            record(i + 1)
        return {"t": dt * np.arange(steps + 1), "theta": th, "omega": om,
                "r_in": ri}


# ---------------------------------------------------------------------------
# Spatial elements (Secs. 7.5, 7.7)
# ---------------------------------------------------------------------------

class SpatialBodyState:
    """Linearization histories of a spatial rigid body.

    :ivar theta: x-y-z angles
    :vartype theta: Variable
    :ivar r_i: input-end position
    :vartype r_i: Variable
    :ivar r_c: mass-center position
    :vartype r_c: Variable
    :ivar r_o: output-end position
    :vartype r_o: Variable
    """

    def __init__(self, theta, r_i, r_c, r_o):
        """Initialize the histories.

        :param theta: angle history
        :type theta: Variable
        :param r_i: input-end history
        :type r_i: Variable
        :param r_c: mass-center history
        :type r_c: Variable
        :param r_o: output-end history
        :type r_o: Variable
        """
        self.theta = theta
        self.r_i = r_i
        self.r_c = r_c
        self.r_o = r_o


def angle_rates(theta, omega_global):
    """Return x-y-z angle rates from the inertial angular velocity (Eq. 7.144).

    :param theta: x-y-z angles
    :type theta: array_like
    :param omega_global: angular velocity, inertial frame
    :type omega_global: array_like
    :rtype: numpy.ndarray
    """
    a = rigid.rotation_xyz(theta)
    return np.linalg.solve(h_matrix(theta),
                           a.T @ np.asarray(omega_global, dtype=float))


def angle_accelerations(theta, theta_d, alpha_global):
    """Return x-y-z angle accelerations (Eq. 7.148).

    :param theta: x-y-z angles
    :type theta: array_like
    :param theta_d: angle rates
    :type theta_d: array_like
    :param alpha_global: angular acceleration, inertial frame
    :type alpha_global: array_like
    :rtype: numpy.ndarray
    """
    a = rigid.rotation_xyz(theta)
    wd_body = a.T @ np.asarray(alpha_global, dtype=float)
    return np.linalg.solve(h_matrix(theta),
                           wd_body - h_dot_matrix(theta, theta_d) @ theta_d)


def spatial_body_matrix_dt(body, state, lin, f_c=(0.0, 0.0, 0.0),
                           m_c=(0.0, 0.0, 0.0)):
    """DT-TMM transfer matrix of a spatial rigid body (Eqs. 7.150-7.174).

    :param body: body with mass, j_i (about I, body frame), l_io, l_ic
    :type body: solvcon.multibody.tmm.rigid.SpatialBody
    :param state: linearization histories of the body
    :type state: SpatialBodyState
    :param lin: step-by-step integration method
    :type lin: Linearizer
    :param f_c: external force at C
    :type f_c: array_like
    :param m_c: external moment
    :type m_c: array_like
    :return: 13x13 matrix U with z_O = U z_I
    :rtype: numpy.ndarray
    """
    dt = lin.dt
    m = body.mass
    jb = body.j_i
    f_c = np.asarray(f_c, dtype=float)
    m_c = np.asarray(m_c, dtype=float)
    th0 = state.theta.x[-1]
    thd0 = state.theta.v
    thdd0 = state.theta.a
    a_c, b_th, c_c, d_th = lin.coefficients(state.theta)
    _, b_ri, _, _ = lin.coefficients(state.r_i)
    _, b_rc, _, _ = lin.coefficients(state.r_c)
    # Linearized rotation matrix (Eqs. 7.83, 7.84) and Psi (Eq. 7.158).
    _, at, phi = rotation_linearization(th0, thd0, dt)
    psi_io = np.column_stack([m_ @ body.l_io for m_ in at])
    psi_ic = np.column_stack([m_ @ body.l_ic for m_ in at])
    # Coefficient functions at the predicted angles (Eq. 7.62).
    th_bar = th0 + thd0 * dt + 0.5 * thdd0 * dt ** 2
    a_bar = rigid.rotation_xyz(th_bar)
    r_io = a_bar @ body.l_io
    r_ic = a_bar @ body.l_ic
    t1, t2, t3, t12, t23, t13 = t_vectors(th_bar)
    s1, s2, s3 = rigid.skew(t1), rigid.skew(t2), rigid.skew(t3)
    hb = h_matrix(th_bar)
    h1 = a_bar @ np.column_stack([s1 @ jb @ t1, s2 @ jb @ t2, s3 @ jb @ t3])
    h2 = a_bar @ np.column_stack([jb @ t12 + s1 @ jb @ t2 + s2 @ jb @ t1,
                                  jb @ t23 + s2 @ jb @ t3 + s3 @ jb @ t2,
                                  jb @ t13 + s1 @ jb @ t3 + s3 @ jb @ t1])
    tx, ty, tz = thd0
    kappa1 = 2.0 * np.diag(thd0)
    kappa2 = np.array([[ty, tx, 0.0], [0.0, tz, ty], [tz, 0.0, tx]])
    kappa3 = d_th - 0.5 * thd0
    ax, ay, az = thdd0
    g = h1 @ kappa1 + h2 @ kappa2
    kappa4 = a_bar @ jb @ hb * a_c + g * c_c
    kappa5 = (a_bar @ jb @ hb @ b_th + g @ kappa3
              + h1 @ np.array([ax * ax, ay * ay, az * az]) * dt ** 2
              + h2 @ np.array([ax * ay, ay * az, az * ax]) * dt ** 2)
    u45 = f_c - m * a_c * (phi @ body.l_ic) - m * b_rc
    sio = rigid.skew(r_io)
    sic = rigid.skew(r_ic)
    u = np.zeros((13, 13))
    u[0:3, 0:3] = np.eye(3)
    u[0:3, 3:6] = psi_io
    u[0:3, 12] = phi @ body.l_io
    u[3:6, 3:6] = np.eye(3)
    u[6:9, 0:3] = m * a_c * (sic - sio)
    u[6:9, 3:6] = kappa4 - m * a_c * sio @ psi_ic
    u[6:9, 6:9] = np.eye(3)
    u[6:9, 9:12] = sio
    u[6:9, 12] = kappa5 - m_c + sic @ (m * b_ri - f_c) + sio @ u45
    u[9:12, 0:3] = -m * a_c * np.eye(3)
    u[9:12, 3:6] = -m * a_c * psi_ic
    u[9:12, 9:12] = np.eye(3)
    u[9:12, 12] = u45
    u[12, 12] = 1.0
    return u, psi_ic, phi @ body.l_ic


def ball_hinge_matrix_dt(u_out):
    """DT-TMM matrix of a smooth ball-and-socket hinge (Eqs. 7.226-7.230).

    The input moment is kept (``U_33`` term, zero in a chain) and passed on.

    :param u_out: 13x13 DT-TMM matrix of the outboard body
    :type u_out: numpy.ndarray
    :return: 13x13 hinge matrix
    :rtype: numpy.ndarray
    """
    u = np.eye(13)
    g = np.linalg.inv(u_out[6:9, 3:6])
    u[3:6, :] = -g @ u_out[6:9, :]
    u[3:6, 3:6] = 0.0
    return u


class SpatialChainDT:
    """Spatial chain with smooth ball-and-socket hinges (Sec. 7.9.3).

    :ivar bodies: bodies from the base to the tip
    :vartype bodies: list
    :ivar gravity: gravity acceleration
    :vartype gravity: numpy.ndarray
    :ivar base: ground point
    :vartype base: numpy.ndarray
    """

    def __init__(self, bodies, gravity=(0.0, -9.81, 0.0),
                 base=(0.0, 0.0, 0.0)):
        """Initialize the chain.

        :param bodies: list of solvcon.multibody.tmm.rigid.SpatialBody
        :type bodies: list
        :param gravity: gravity acceleration
        :type gravity: array_like
        :param base: ground point of hinge 0
        :type base: array_like
        """
        self.bodies = list(bodies)
        self.gravity = np.asarray(gravity, dtype=float)
        self.base = np.asarray(base, dtype=float)

    def _initial_states(self, theta0, theta_d0):
        """Return the body histories at t_0 with exact accelerations.

        :param theta0: initial x-y-z angles, shape (n, 3)
        :type theta0: numpy.ndarray
        :param theta_d0: initial angle rates, shape (n, 3)
        :type theta_d0: numpy.ndarray
        :rtype: list
        """
        mats = [rigid.rotation_xyz(t) for t in theta0]
        omegas = [a @ (h_matrix(t) @ td) for a, t, td
                  in zip(mats, theta0, theta_d0)]
        quats = [rigid.matrix_to_quat(a) for a in mats]
        ch = rigid.SpatialChain(self.bodies,
                                [rigid.BallHinge() for _ in self.bodies],
                                self.gravity, self.base)
        _, alpha, accel = ch.transfer(quats, omegas)
        r_in, _, _ = ch.positions(quats)
        out = []
        v = np.zeros(3)
        for k, b in enumerate(self.bodies):
            a, w, al = mats[k], omegas[k], alpha[k]
            thdd = angle_accelerations(theta0[k], theta_d0[k], al)

            def point(p):
                rp = a @ p
                return Variable(r_in[k] + rp, v + np.cross(w, rp),
                                accel[k] + np.cross(al, rp)
                                + np.cross(w, np.cross(w, rp)))

            st = SpatialBodyState(Variable(theta0[k], theta_d0[k], thdd),
                                  Variable(r_in[k], v, accel[k]),
                                  point(b.l_ic), point(b.l_io))
            out.append(st)
            v = st.r_o.v
        return out

    def step(self, states, lin):
        """Advance all body histories by one time step.

        :param states: body histories
        :type states: list
        :param lin: step-by-step integration method
        :type lin: Linearizer
        :return: (z_in, z_out) per body at t_i
        :rtype: list
        """
        data = [spatial_body_matrix_dt(b, s, lin, b.mass * self.gravity)
                for b, s in zip(self.bodies, states)]
        umats = [d[0] for d in data]
        hmats = [ball_hinge_matrix_dt(u) for u in umats]
        total = np.eye(13)
        for u, h in zip(umats, hmats):
            total = u @ h @ total
        z0 = np.zeros(13)
        z0[0:3] = self.base
        z0[12] = 1.0
        rows = list(range(6, 12))
        cols = [9, 10, 11]
        q0, *_ = np.linalg.lstsq(total[np.ix_(rows, cols)],
                                 -(total[rows] @ z0), rcond=None)
        z = z0.copy()
        z[9:12] = q0
        result = []
        for u, h in zip(umats, hmats):
            z_in = h @ z
            z_out = u @ z_in
            result.append((z_in, z_out))
            z = z_out
        for (z_in, z_out), d, s in zip(result, data, states):
            _, psi_ic, phi_ic = d
            r_c = z_in[0:3] + psi_ic @ z_in[3:6] + phi_ic
            lin.update(s.theta, z_in[3:6])
            lin.update(s.r_i, z_in[0:3])
            lin.update(s.r_c, r_c)
            lin.update(s.r_o, z_out[0:3])
        return result

    def simulate(self, theta0, theta_d0, t_end, dt, method="newmark",
                 **kwargs):
        """Integrate the motion with the DT-TMM.

        :param theta0: initial x-y-z angles, shape (n, 3)
        :type theta0: array_like
        :param theta_d0: initial angle rates, shape (n, 3)
        :type theta_d0: array_like
        :param t_end: final time
        :type t_end: float
        :param dt: time step
        :type dt: float
        :param method: linearizer name
        :type method: str
        :return: dict with t, theta (N, n, 3), theta_d (N, n, 3) and
            r_in (N, n, 3)
        :rtype: dict
        """
        lin = make_linearizer(method, dt, **kwargs)
        n = len(self.bodies)
        theta0 = np.asarray(theta0, dtype=float).reshape(n, 3)
        theta_d0 = np.asarray(theta_d0, dtype=float).reshape(n, 3)
        states = self._initial_states(theta0, theta_d0)
        steps = int(round(t_end / dt))
        th = np.empty((steps + 1, n, 3))
        thd = np.empty((steps + 1, n, 3))
        ri = np.empty((steps + 1, n, 3))

        def record(i):
            for k, s in enumerate(states):
                th[i, k] = s.theta.x[-1]
                thd[i, k] = s.theta.v
                ri[i, k] = s.r_i.x[-1]

        record(0)
        for i in range(steps):
            self.step(states, lin)
            record(i + 1)
        return {"t": dt * np.arange(steps + 1), "theta": th, "theta_d": thd,
                "r_in": ri}

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
