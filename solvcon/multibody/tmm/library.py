# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Transfer matrix library of linear elements (book Ch. 2, 10 and 15.2-15.10).

Every element maps the input state vector to the output state vector,
Z_O = U(s) Z_I + f(s). The Laplace variable s replaces i*omega of the book:
m omega^2 is written -m s^2 (Table 10.1), viscous damping C enters as C s
(Eq. 2.92) and structural damping as K (1 + i g) (Eq. 2.74). The argument
eta of ``transfer`` scales all damping parameters (eta = 0 is the undamped
system used for eigenfrequencies and augmented eigenvectors).

Sign conventions (book Sec. 1.7, Ch. 2). Longitudinal and transverse
forces: X_O = X_I - Q/K for a spring (Eq. 2.13) and
Q_O = Q_I + m omega^2 X + F for a mass (Eq. 2.111). Moments and angles:
Theta_O = Theta_I + M/K' for a rotary spring (Eq. 10.38) and
M_O = M_I - J omega^2 Theta - M_ext for a disk (Eq. 2.25, Example 2.14).
Beams: Y' = Theta - Q/GA_s, Theta' = M/EI, M' = Q, Q' = m omega^2 Y
(Eq. 10.54).

Harmonic loads are complex amplitudes: F^c cos(Omega t) + F^s sin(Omega t)
has the amplitude F^c - i F^s (Example 2.14, Eq. a-b).
"""

import math

import numpy as np

from . import riccati as _num
from .linear import Element

_SERIES_TERMS = 18
# 1/(4n+k)! for k = 0..4 (beam functions) and 1/(2n+k)! for k = 0..2 (rod
# functions)
_BEAM_COEF = np.array([[1.0 / math.factorial(4 * n + k)
                        for n in range(_SERIES_TERMS)] for k in range(5)])
_ROD_COEF = np.array([[1.0 / math.factorial(2 * n + k)
                       for n in range(_SERIES_TERMS)] for k in range(3)])


def krylov(z):
    """Krylov functions S, T, U, V (Eq. 10.62, 15.20).

    :param z: Argument lambda * x (real or complex).
    :type z: complex or numpy.ndarray
    :return: (S, T, U, V).
    :rtype: tuple
    """
    ch, co = np.cosh(z), np.cos(z)
    sh, si = np.sinh(z), np.sin(z)
    return (ch + co) / 2.0, (sh + si) / 2.0, (ch - co) / 2.0, (sh - si) / 2.0


def beam_functions(a, x):
    """Entire functions d_k = sum_n a^n x^(4n+k) / (4n+k)!, k = 0..4.

    With lambda^4 = a: d0 = S(lambda x), d1 = T(lambda x)/lambda,
    d2 = U(lambda x)/lambda^2, d3 = V(lambda x)/lambda^3 and
    d4 = (S(lambda x) - 1)/lambda^4 (Eq. 10.61-10.62, 10.77). The power series
    is used for |lambda x| < 2, which removes the 0/0 limits at
    omega = 0 (static response, massless beams); the Krylov closed form is
    used otherwise.

    :param a: lambda^4 (complex).
    :type a: complex
    :param x: Coordinate.
    :type x: float
    :return: Array [d0, d1, d2, d3, d4] (complex).
    :rtype: numpy.ndarray
    """
    a = complex(a)
    lam = a**0.25 if a != 0 else 0.0
    if abs(lam * x) < 2.0:
        powers = np.cumprod(np.concatenate(
            ([1.0 + 0.0j], np.full(_SERIES_TERMS - 1, a * x**4))))
        return (_BEAM_COEF @ powers) * x ** np.arange(5)
    s_, t_, u_, v_ = krylov(lam * x)
    return np.array([s_, t_ / lam, u_ / lam**2, v_ / lam**3, (s_ - 1.0) / a])


def rod_functions(b, x):
    """Entire functions e0 = cos(beta x), e1 = sin(beta x)/beta, e2 = (1 -
    cos(beta x))/beta^2.

    beta^2 = b (Eq. 10.46b, 10.66, Table 10.2). Power series for |beta x| < 1.

    :param b: beta^2 (complex).
    :type b: complex
    :param x: Coordinate.
    :type x: float
    :return: Array [e0, e1, e2].
    :rtype: numpy.ndarray
    """
    b = complex(b)
    beta = np.sqrt(b)
    if abs(beta * x) < 1.0:
        powers = np.cumprod(np.concatenate(
            ([1.0 + 0.0j], np.full(_SERIES_TERMS - 1, -b * x**2))))
        return (_ROD_COEF @ powers) * x ** np.arange(3)
    return np.array([np.cos(beta * x), np.sin(beta * x) / beta,
                     (1.0 - np.cos(beta * x)) / b])


def _inv(k):
    """1/k with 1/inf = 0 (rigid connection)."""
    if k is None or k == np.inf:
        return 0.0
    return 1.0 / k


def _impedance(k, c, g, s, eta):
    """Complex impedance K (1 + i g) + C s (Eq. 2.73-2.74, 2.92b)."""
    return k * (1.0 + 1j * g * eta) + c * eta * s


def _cross(v):
    """Cross-product matrix of a 3-vector (Eq. 10.13)."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


# ----------------------------------------------------------------------
# Springs, rotary springs, elastic hinges (Sec. 2.2.3.1, 2.7.3, 10.2.2-10.2.3,
# 15.2-15.4) #
# ----------------------------------------------------------------------
class Spring(Element):
    """Massless spring-damper in longitudinal vibration, state [X..., Q...].

    Eq. (2.13), (2.92b), (15.1)-(15.3).

    U = [[I, -diag(1/z)], [0, I]] with z = K (1 + i g) + C s per direction.
    dims = 1, 2, 3 gives Eq. (15.1), (15.2), (15.3).

    :param k: Stiffness per direction (scalar or sequence of length dims).
    :type k: float or list
    :param c: Viscous damping coefficient(s) in parallel (Eq. 2.89-2.92).
    :type c: float or list
    :param g: Structural damping factor(s) (Eq. 2.74).
    :type g: float or list
    :param dims: Number of directions.
    :type dims: int
    """

    def __init__(self, k, c=0.0, g=0.0, dims=1):
        self.dims = dims
        self.size = 2 * dims
        self.k = np.broadcast_to(np.asarray(k, dtype=float), (dims,)).copy()
        self.c = np.broadcast_to(np.asarray(c, dtype=float), (dims,)).copy()
        self.g = np.broadcast_to(np.asarray(g, dtype=float), (dims,)).copy()

    def transfer(self, s, eta=1.0):
        """Transfer matrix (Eq. 2.13, 2.92b, 2.141).

        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor.
        :type eta: float
        :return: U.
        :rtype: numpy.ndarray
        """
        n = self.dims
        u = np.eye(self.size, dtype=complex)
        for i in range(n):
            u[i, n + i] = -1.0 / _impedance(self.k[i], self.c[i], self.g[i], s,
                                            eta)
        return u

    def stiffness_weight(self):
        """Strain energy Q^2/K on the input state (Eq. 3.77)."""
        n = self.dims
        w = np.zeros((self.size, self.size))
        for i in range(n):
            w[n + i, n + i] = 1.0 / self.k[i]
        return w

    def damping_weight(self):
        """Damper term C dX^2 = C Q^2/K^2 on the input state (Eq. 3.154)."""
        n = self.dims
        w = np.zeros((self.size, self.size))
        for i in range(n):
            w[n + i, n + i] = self.c[i] / self.k[i] ** 2
        return w


class Damper(Element):
    """Massless longitudinal damper with complex stiffness C s (Eq. 2.144).

    :param c: Viscous damping coefficient.
    :type c: float
    """

    size = 2

    def __init__(self, c):
        self.c = float(c)

    def transfer(self, s, eta=1.0):
        """Transfer matrix [[1, -1/(C s)], [0, 1]] (Eq. 2.144).

        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor.
        :type eta: float
        :return: U.
        :rtype: numpy.ndarray
        """
        return np.array([[1.0, -1.0 / (self.c * eta * s)], [0.0, 1.0]],
                        dtype=complex)


class RotarySpring(Element):
    """Massless rotary spring, state [Theta..., M...] (Eq. 10.38-10.39,
    15.4-15.6).

    U = [[I, diag(1/z)], [0, I]] with z = K' (1 + i g) + C s. With k = 0
    and c > 0 the element is a rotary damper coupling (Example 2.14,
    element 4).

    :param k: Torsional stiffness per axis.
    :type k: float or list
    :param c: Viscous damping coefficient(s).
    :type c: float or list
    :param g: Structural damping factor(s) (Example 2.14, element 6).
    :type g: float or list
    :param dims: Number of axes (1: Eq. 15.4; 2: Eq. 15.5; 3: Eq. 15.6).
    :type dims: int
    """

    def __init__(self, k, c=0.0, g=0.0, dims=1):
        self.dims = dims
        self.size = 2 * dims
        self.k = np.broadcast_to(np.asarray(k, dtype=float), (dims,)).copy()
        self.c = np.broadcast_to(np.asarray(c, dtype=float), (dims,)).copy()
        self.g = np.broadcast_to(np.asarray(g, dtype=float), (dims,)).copy()

    def transfer(self, s, eta=1.0):
        """Transfer matrix (Eq. 10.38).

        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor.
        :type eta: float
        :return: U.
        :rtype: numpy.ndarray
        """
        n = self.dims
        u = np.eye(self.size, dtype=complex)
        for i in range(n):
            u[i, n + i] = 1.0 / _impedance(self.k[i], self.c[i], self.g[i], s,
                                           eta)
        return u

    def stiffness_weight(self):
        """Strain energy M^2/K' on the input state (Eq. 3.77)."""
        n = self.dims
        w = np.zeros((self.size, self.size))
        for i in range(n):
            if self.k[i] > 0:
                w[n + i, n + i] = 1.0 / self.k[i]
        return w


class PlanarElasticHinge(Element):
    """Planar elastic hinge, state [X, Y, Theta_z, M_z, Q_x, Q_y] (Ex. 2.6 Eq.
    a, Eq. 2.57, 15.7).

    U = [[I3, U12], [0, I3]], U12 = [[0, -1/K_x, 0], [0, 0, -1/K_y], [1/K'_z,
    0, 0]]. A stiffness of numpy.inf (or None) is a rigid direction (Example
    2.10, where the x displacement is neglected).

    :param kx: Spring stiffness in x.
    :type kx: float
    :param ky: Spring stiffness in y.
    :type ky: float
    :param kz: Rotary stiffness about z.
    :type kz: float
    """

    size = 6

    def __init__(self, kx, ky, kz):
        self.kx, self.ky, self.kz = kx, ky, kz

    def transfer(self, s, eta=1.0):
        """Transfer matrix (Example 2.6 Eq. a).

        :param s: Laplace variable (unused).
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U.
        :rtype: numpy.ndarray
        """
        u = np.eye(6, dtype=complex)
        u[0, 4] = -_inv(self.kx)
        u[1, 5] = -_inv(self.ky)
        u[2, 3] = _inv(self.kz)
        return u

    def stiffness_weight(self):
        """Strain energy Q_x^2/K_x + Q_y^2/K_y + M_z^2/K'_z (Example 3.4)."""
        w = np.zeros((6, 6))
        w[4, 4] = _inv(self.kx)
        w[5, 5] = _inv(self.ky)
        w[3, 3] = _inv(self.kz)
        return w


class SpatialElasticHinge(Element):
    """Spatial elastic hinge, 12-state of Eq. (7.20) (Eq. 10.40, 15.8).

    :param k: Spring stiffnesses (K_x, K_y, K_z); numpy.inf is rigid.
    :type k: list
    :param kr: Rotary stiffnesses (K'_x, K'_y, K'_z); numpy.inf is rigid.
    :type kr: list
    """

    size = 12

    def __init__(self, k, kr):
        self.k = list(k)
        self.kr = list(kr)

    def transfer(self, s, eta=1.0):
        """Transfer matrix (Eq. 10.40).

        :param s: Laplace variable (unused).
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U.
        :rtype: numpy.ndarray
        """
        u = np.eye(12, dtype=complex)
        for i in range(3):
            u[i, 9 + i] = -_inv(self.k[i])
            u[3 + i, 6 + i] = _inv(self.kr[i])
        return u

    def stiffness_weight(self):
        """Strain energy sum Q^2/K + M^2/K'."""
        w = np.zeros((12, 12))
        for i in range(3):
            w[9 + i, 9 + i] = _inv(self.k[i])
            w[6 + i, 6 + i] = _inv(self.kr[i])
        return w


class BeamHinge(Element):
    """Transverse spring and rotary spring in series on a beam, state [Y,
    Theta_z, M_z, Q_y].

    Y_O = Y_I - Q/K and Theta_O = Theta_I + M/K' (Example 3.9 U_1; the
    transverse and rotary rows of the planar elastic hinge, Eq. 15.7).
    numpy.inf (or None) is a rigid direction.

    :param k: Transverse stiffness K.
    :type k: float
    :param kr: Rotary stiffness K'.
    :type kr: float
    """

    size = 4

    def __init__(self, k=np.inf, kr=np.inf):
        self.k, self.kr = k, kr

    def transfer(self, s, eta=1.0):
        """Transfer matrix (Example 3.9, U_1).

        :param s: Laplace variable (unused).
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U.
        :rtype: numpy.ndarray
        """
        u = np.eye(4, dtype=complex)
        u[0, 3] = -_inv(self.k)
        u[1, 2] = _inv(self.kr)
        return u

    def stiffness_weight(self):
        """Strain energy Q^2/K + M^2/K'."""
        return np.diag([0.0, 0.0, _inv(self.kr), _inv(self.k)])


# ----------------------------------------------------------------------
# Lumped masses, disks, point supports (Sec. 2.2.3, 2.8.1, 15.5, 15.7.7,
# 15.8.6, 15.9.3) #
# ----------------------------------------------------------------------
class LumpedMass(Element):
    """Lumped mass in longitudinal vibration, state [X..., Q...].

    Eq. (2.19), (2.93), (2.111), (15.9)-(15.11).

    Q_O = Q_I + (m omega^2 - K_b - i C Omega) X + F per direction, that is
    U = [[I, 0], [-(m s^2 + C s + K_b) I, I]] with an optional elastic and
    viscous support to the ground (Eq. 15.38).

    :param m: Mass.
    :type m: float
    :param dims: Number of directions (1: Eq. 15.9; 2: 15.10; 3: 15.11).
    :type dims: int
    :param force: Complex harmonic force amplitude(s) (Eq. 2.111).
    :type force: complex or list
    :param k_ground: Stiffness of a support to the ground (Eq. 15.38).
    :type k_ground: float
    :param c_ground: Viscous damping to the ground.
    :type c_ground: float
    """

    def __init__(self, m, dims=1, force=0.0, k_ground=0.0, c_ground=0.0):
        self.m = float(m)
        self.dims = dims
        self.size = 2 * dims
        self.force = np.broadcast_to(np.asarray(force, dtype=complex),
                                     (dims,)).copy()
        self.k_ground = float(k_ground)
        self.c_ground = float(c_ground)

    def transfer(self, s, eta=1.0):
        """Transfer matrix (Eq. 2.19, 2.93b).

        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor.
        :type eta: float
        :return: U.
        :rtype: numpy.ndarray
        """
        n = self.dims
        u = np.eye(self.size, dtype=complex)
        val = -(self.m * s * s + self.c_ground * eta * s + self.k_ground)
        for i in range(n):
            u[n + i, i] = val
        return u

    def forcing(self, s):
        """Load column [0, F] (Eq. 2.111)."""
        f = np.zeros(self.size, dtype=complex)
        f[self.dims:] = self.force
        return f

    def mass_weight(self):
        """Mass matrix m I on the displacements (Eq. 3.3-3.7)."""
        w = np.zeros((self.size, self.size))
        for i in range(self.dims):
            w[i, i] = self.m
        return w

    def stiffness_weight(self):
        """Support energy K_b X^2 (Eq. 15.38)."""
        if self.k_ground == 0:
            return None
        w = np.zeros((self.size, self.size))
        for i in range(self.dims):
            w[i, i] = self.k_ground
        return w

    def damping_weight(self):
        """Ground damper C X^2."""
        if self.c_ground == 0:
            return None
        w = np.zeros((self.size, self.size))
        for i in range(self.dims):
            w[i, i] = self.c_ground
        return w


class Disk(Element):
    """Rigid disk in torsional vibration, state [Theta_x, M_x] (Eq. 2.25,
    15.35, Example 2.14).

    M_O = M_I + (J s^2 + C s + K_b) Theta - M_ext, with rotary inertia J, a
    rotary damper C to the ground (Example 2.14, element 3) and a flexible
    support K_b (Eq. 15.35).

    :param j: Moment of inertia.
    :type j: float
    :param c_ground: Viscous rotary damping to the ground.
    :type c_ground: float
    :param k_ground: Torsional support stiffness.
    :type k_ground: float
    :param torque: Complex harmonic torque amplitude M_ext.
    :type torque: complex
    """

    size = 2

    def __init__(self, j, c_ground=0.0, k_ground=0.0, torque=0.0):
        self.j = float(j)
        self.c_ground = float(c_ground)
        self.k_ground = float(k_ground)
        self.torque = complex(torque)

    def transfer(self, s, eta=1.0):
        """Transfer matrix [[1, 0], [J s^2 + C s + K_b, 1]] (Eq. 2.25).

        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor.
        :type eta: float
        :return: U.
        :rtype: numpy.ndarray
        """
        val = self.j * s * s + self.c_ground * eta * s + self.k_ground
        return np.array([[1.0, 0.0], [val, 1.0]], dtype=complex)

    def forcing(self, s):
        """Load column [0, -M_ext] (Example 2.14, U_5 and U_7)."""
        return np.array([0.0, -self.torque], dtype=complex)

    def mass_weight(self):
        """Rotary inertia J on Theta."""
        return np.diag([self.j, 0.0])

    def stiffness_weight(self):
        """Support energy K_b Theta^2."""
        return np.diag([self.k_ground, 0.0]) if self.k_ground else None

    def damping_weight(self):
        """Ground damper C Theta^2."""
        return np.diag([self.c_ground, 0.0]) if self.c_ground else None


class BeamPointMass(Element):
    """Point mass, rotary inertia and supports on a beam, state [Y, Theta_z,
    M_z, Q_y].

    Example 2.1 Eq. (d) (lumped mass), Eq. (15.27) (spring, rotary spring and
    elastic hinge supports) and Eq. (15.28) (supported lumped mass):
    M_O = M_I + (K' + J s^2) Theta - M_ext, Q_O = Q_I - (m s^2 + C s + K) Y +
    F.

    :param m: Mass.
    :type m: float
    :param j: Rotary inertia.
    :type j: float
    :param k: Translational support stiffness.
    :type k: float
    :param kr: Rotary support stiffness.
    :type kr: float
    :param c: Translational damping to the ground.
    :type c: float
    :param force: Complex harmonic force amplitude F (or static force).
    :type force: complex
    :param moment: Complex harmonic moment amplitude M_ext.
    :type moment: complex
    """

    size = 4

    def __init__(self, m=0.0, j=0.0, k=0.0, kr=0.0, c=0.0, force=0.0,
                 moment=0.0):
        self.m, self.j, self.k = float(m), float(j), float(k)
        self.kr, self.c = float(kr), float(c)
        self.force = complex(force)
        self.moment = complex(moment)

    def transfer(self, s, eta=1.0):
        """Transfer matrix (Example 2.1 Eq. d, Eq. 15.27-15.28).

        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor.
        :type eta: float
        :return: U.
        :rtype: numpy.ndarray
        """
        u = np.eye(4, dtype=complex)
        u[2, 1] = self.kr + self.j * s * s
        u[3, 0] = -(self.m * s * s + self.c * eta * s + self.k)
        return u

    def forcing(self, s):
        """Load column [0, 0, -M_ext, F] (Example 2.13 U_2)."""
        return np.array([0.0, 0.0, -self.moment, self.force], dtype=complex)

    def mass_weight(self):
        """diag(m, J) on (Y, Theta)."""
        return np.diag([self.m, self.j, 0.0, 0.0])

    def stiffness_weight(self):
        """Support energy K Y^2 + K' Theta^2."""
        if self.k == 0 and self.kr == 0:
            return None
        return np.diag([self.k, self.kr, 0.0, 0.0])

    def damping_weight(self):
        """Ground damper C Y^2."""
        return np.diag([self.c, 0.0, 0.0, 0.0]) if self.c else None

    def displacement_index(self):
        """Displacements Y and Theta_z."""
        return [0, 1]


# ----------------------------------------------------------------------
# Rigid bodies (Sec. 10.2.1, 15.6)
# ----------------------------------------------------------------------
class PlanarRigidBody(Element):
    """Rigid body with one input and one output end vibrating in a plane (Eq.
    10.10, 15.12).

    State [X, Y, Theta_z, M_z, Q_x, Q_y]. Body-fixed coordinates with origin
    at the input end I: output end O = (b1, b2), mass center C = (cc1, cc2).
    A harmonic force F at D = (d1, d2) and a moment M give the load column
    f = [0, 0, 0, -M + (l_DO x F)_z, F_x, F_y] (planar form of Eq. 10.21).

    :param m: Mass.
    :type m: float
    :param j_i: Moment of inertia about the input end I.
    :type j_i: float
    :param b: Output end coordinates (b1, b2).
    :type b: tuple
    :param cc: Mass center coordinates (cc1, cc2).
    :type cc: tuple
    :param force: Complex force amplitude (F_x, F_y).
    :type force: tuple
    :param moment: Complex moment amplitude.
    :type moment: complex
    :param d: Point of the force (d1, d2).
    :type d: tuple
    """

    size = 6

    def __init__(self, m, j_i, b, cc, force=(0.0, 0.0), moment=0.0,
                 d=(0.0, 0.0)):
        self.m = float(m)
        self.j_i = float(j_i)
        self.b = tuple(float(v) for v in b)
        self.cc = tuple(float(v) for v in cc)
        self.force = tuple(complex(v) for v in force)
        self.moment = complex(moment)
        self.d = tuple(float(v) for v in d)

    def transfer(self, s, eta=1.0):
        """Transfer matrix (Eq. 10.10).

        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U.
        :rtype: numpy.ndarray
        """
        m, j = self.m, self.j_i
        b1, b2 = self.b
        c1, c2 = self.cc
        w2 = -s * s
        u = np.eye(6, dtype=complex)
        u[0, 2] = -b2
        u[1, 2] = b1
        u[3, 0] = -m * w2 * (b2 - c2)
        u[3, 1] = m * w2 * (b1 - c1)
        u[3, 2] = -w2 * (j - m * (b2 * c2 + b1 * c1))
        u[3, 4] = -b2
        u[3, 5] = b1
        u[4, 0] = m * w2
        u[4, 2] = -m * w2 * c2
        u[5, 1] = m * w2
        u[5, 2] = m * w2 * c1
        return u

    def forcing(self, s):
        """Load column (planar Eq. 10.21)."""
        fx, fy = self.force
        b1, b2 = self.b
        d1, d2 = self.d
        mz = -self.moment + (b1 - d1) * fy - (b2 - d2) * fx
        return np.array([0, 0, 0, mz, fx, fy], dtype=complex)

    def mass_weight(self):
        """Mass matrix [[m, 0, -m cc2], [0, m, m cc1], [-m cc2, m cc1, J_I]]
        (Eq. 3.74)."""
        m, j = self.m, self.j_i
        c1, c2 = self.cc
        w = np.zeros((6, 6))
        w[:3, :3] = [[m, 0.0, -m * c2], [0.0, m, m * c1], [-m * c2, m * c1, j]]
        return w


class TransverseRigidBody(Element):
    """Rigid body in transverse and rotational vibration, state [Y, Theta_z,
    M_z, Q_y] (Eq. 15.13).

    :param m: Mass.
    :type m: float
    :param j_i: Moment of inertia about the input end.
    :type j_i: float
    :param b1: Output end coordinate.
    :type b1: float
    :param cc1: Mass center coordinate.
    :type cc1: float
    :param force: Complex transverse force amplitude.
    :type force: complex
    :param moment: Complex moment amplitude.
    :type moment: complex
    :param d1: Point of the force.
    :type d1: float
    """

    size = 4

    def __init__(self, m, j_i, b1, cc1, force=0.0, moment=0.0, d1=0.0):
        self.m, self.j_i = float(m), float(j_i)
        self.b1, self.cc1 = float(b1), float(cc1)
        self.force = complex(force)
        self.moment = complex(moment)
        self.d1 = float(d1)

    def transfer(self, s, eta=1.0):
        """Transfer matrix (Eq. 15.13).

        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U.
        :rtype: numpy.ndarray
        """
        m, j, b1, c1 = self.m, self.j_i, self.b1, self.cc1
        w2 = -s * s
        return np.array([
            [1.0, b1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [m * w2 * (b1 - c1), -w2 * (j - m * b1 * c1), 1.0, b1],
            [m * w2, m * w2 * c1, 0.0, 1.0],
        ], dtype=complex)

    def forcing(self, s):
        """Load column [0, 0, -M + (b1 - d1) F, F]."""
        f = self.force
        return np.array([0.0, 0.0, -self.moment + (self.b1 - self.d1) * f, f],
                        dtype=complex)

    def mass_weight(self):
        """[[m, m cc1], [m cc1, J_I]] on (Y, Theta)."""
        w = np.zeros((4, 4))
        w[:2, :2] = [[self.m, self.m * self.cc1],
                     [self.m * self.cc1, self.j_i]]
        return w

    def displacement_index(self):
        """Displacements Y and Theta_z."""
        return [0, 1]


class SpatialRigidBody(Element):
    """Rigid body with one input and one output end vibrating in space (Eq.
    10.19, 15.14).

    State of Eq. (7.20) without the trailing 1:
    [X, Y, Z, Theta_x, Theta_y, Theta_z, M_x, M_y, M_z, Q_x, Q_y, Q_z].

    :param m: Mass.
    :type m: float
    :param j_i: Inertia matrix about the input end I (body frame), 3 x 3.
    :type j_i: numpy.ndarray
    :param b: Output end coordinates (b1, b2, b3).
    :type b: tuple
    :param cc: Mass center coordinates.
    :type cc: tuple
    :param force: Complex force amplitude (3).
    :type force: tuple
    :param moment: Complex moment amplitude (3).
    :type moment: tuple
    :param d: Point of the force (3).
    :type d: tuple
    """

    size = 12

    def __init__(self, m, j_i, b, cc, force=(0, 0, 0), moment=(0, 0, 0),
                 d=(0, 0, 0)):
        self.m = float(m)
        self.j_i = np.asarray(j_i, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.cc = np.asarray(cc, dtype=float)
        self.force = np.asarray(force, dtype=complex)
        self.moment = np.asarray(moment, dtype=complex)
        self.d = np.asarray(d, dtype=float)

    def transfer(self, s, eta=1.0):
        """Transfer matrix (Eq. 10.19).

        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U.
        :rtype: numpy.ndarray
        """
        m = self.m
        w2 = -s * s
        l_io = _cross(self.b)
        l_ic = _cross(self.cc)
        l_co = l_io - l_ic
        i3 = np.eye(3)
        u = np.zeros((12, 12), dtype=complex)
        u[0:3, 0:3] = i3
        u[0:3, 3:6] = -l_io
        u[3:6, 3:6] = i3
        u[6:9, 0:3] = m * w2 * l_co
        u[6:9, 3:6] = -w2 * (m * l_io @ l_ic + self.j_i)
        u[6:9, 6:9] = i3
        u[6:9, 9:12] = l_io
        u[9:12, 0:3] = m * w2 * i3
        u[9:12, 3:6] = -m * w2 * l_ic
        u[9:12, 9:12] = i3
        return u

    def forcing(self, s):
        """Load column [0, 0, -M + l_DO F, F] (Eq. 10.21)."""
        f = np.zeros(12, dtype=complex)
        f[6:9] = -self.moment + _cross(self.b - self.d) @ self.force
        f[9:12] = self.force
        return f

    def mass_weight(self):
        """Mass matrix [[m I, -m l_IC], [m l_IC, J_I]] on (r, theta) (Eq.
        3.14)."""
        w = np.zeros((12, 12))
        l_ic = _cross(self.cc)
        w[0:3, 0:3] = self.m * np.eye(3)
        w[0:3, 3:6] = -self.m * l_ic
        w[3:6, 0:3] = self.m * l_ic
        w[3:6, 3:6] = self.j_i
        return w


class RigidBeamOnFoundation(Element):
    """Homogeneous rigid beam on an elastic foundation, state [Y, Theta_z, M_z,
    Q_y].

    Sec. 10.6.3, Eq. (15.26).

    :param length: Length l.
    :type length: float
    :param m: Mass per unit length.
    :type m: float
    :param k_found: Winkler modulus K*.
    :type k_found: float
    :param kr_found: Rotary foundation modulus K'*.
    :type kr_found: float
    :param r: Radius of gyration of the cross-section.
    :type r: float
    """

    size = 4

    def __init__(self, length, m, k_found=0.0, kr_found=0.0, r=0.0):
        self.ln, self.m, self.kf = float(length), float(m), float(k_found)
        self.krf, self.r = float(kr_found), float(r)

    def transfer(self, s, eta=1.0):
        """Transfer matrix (Eq. 15.26).

        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U.
        :rtype: numpy.ndarray
        """
        ln, m = self.ln, self.m
        w2 = -s * s
        q = m * w2 - self.kf
        return np.array([
            [1.0, ln, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [ln**2 / 2 * q,
             ln**3 / 6 * q + ln * (self.krf - self.r**2 * m * w2), 1.0, ln],
            [ln * q, ln**2 / 2 * q, 0.0, 1.0],
        ], dtype=complex)

    def mass_weight(self):
        """Kinetic energy of y(x) = Y + Theta x plus rotary inertia."""
        ln, m = self.ln, self.m
        w = np.zeros((4, 4))
        w[:2,
          :2] = m * np.array([[ln, ln**2 / 2],
                              [ln**2 / 2, ln**3 / 3 + self.r**2 * ln]])
        return w

    def stiffness_weight(self):
        """Foundation energy of y(x) = Y + Theta x."""
        ln = self.ln
        w = np.zeros((4, 4))
        w[:2,
          :2] = self.kf * np.array([[ln, ln**2 / 2], [ln**2 / 2, ln**3 / 3]])
        w[1, 1] += self.krf * ln
        return w

    def displacement_index(self):
        """Displacements Y and Theta_z."""
        return [0, 1]


# ----------------------------------------------------------------------
# Continuous elements (Sec. 10.3-10.4, 15.7-15.10)
# ----------------------------------------------------------------------
class ContinuousElement(Element):
    """Base class of distributed elements (beam, rod, shaft).

    :ivar length: Element length l.
    :vartype length: float
    """

    continuous = True

    def transfer(self, s, eta=1.0):
        """Transfer matrix U(l).

        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor.
        :type eta: float
        :return: U.
        :rtype: numpy.ndarray
        """
        return self.transfer_at(self.length, s, eta)

    def forcing(self, s):
        """Load column f(l)."""
        return self.forcing_at(self.length, s)

    def forcing_at(self, x, s):
        """Load column f(x) of a uniform distributed load (Eq. 10.80)."""
        return np.zeros(self.size, dtype=complex)

    def transfer_at(self, x, s, eta=1.0):
        """Transfer matrix from the input end to the coordinate x."""
        raise NotImplementedError

    def mass_density(self):
        """Mass density weight on the state z(x)."""
        return None

    def stiffness_density(self):
        """Strain energy density weight on the state z(x)."""
        return None

    def damping_density(self):
        """Damping density weight on the state z(x)."""
        return None

    def wavenumber(self, s):
        """Largest wavenumber of the element at s (sets the quadrature
        order)."""
        return 0.0

    def quadrature_points(self, s):
        """Gauss-Legendre order for products of two modes on the element."""
        return 16 + int(2.0 * abs(self.wavenumber(s)) * self.length)


class EulerBernoulliBeam(ContinuousElement):
    """Euler-Bernoulli beam in transverse vibration, state [Y, Theta_z, M_z,
    Q_y].

    Transfer matrix of Eq. (10.61) / (15.20) written with the functions of
    ``beam_functions``; m = 0 gives the massless beam of Eq. (2.31) / (15.22);
    a Winkler foundation k* gives Eq. (15.25). lambda^4 = (m omega^2 - k*)/EI.
    A Kelvin-Voigt internal damping C' replaces EI by EI + C' s (Example 3.10,
    3.11) and an external viscous damping c per unit length adds c s to
    m s^2. A uniform load p (force per length) and q_m (moment per length)
    give the load column of Example 10.4 Eq. (e) (massless limit: Eq. 2.123b).

    :param length: Length l.
    :type length: float
    :param ei: Bending stiffness EI.
    :type ei: float
    :param m: Mass per unit length.
    :type m: float
    :param k_found: Winkler foundation modulus k*.
    :type k_found: float
    :param c_internal: Internal (Kelvin-Voigt) damping C'.
    :type c_internal: float
    :param c_external: External viscous damping per unit length.
    :type c_external: float
    :param p: Complex amplitude of a uniform transverse load per length.
    :type p: complex
    :param q_m: Complex amplitude of a uniform moment per length.
    :type q_m: complex
    """

    size = 4

    def __init__(self, length, ei, m=0.0, k_found=0.0, c_internal=0.0,
                 c_external=0.0, p=0.0, q_m=0.0):
        self.length = float(length)
        self.ei = float(ei)
        self.m = float(m)
        self.k_found = float(k_found)
        self.c_internal = float(c_internal)
        self.c_external = float(c_external)
        self.p = complex(p)
        self.q_m = complex(q_m)

    def _params(self, s, eta):
        ei = self.ei + self.c_internal * eta * s
        a = -(self.m * s * s + self.c_external * eta * s + self.k_found) / ei
        return ei, a

    def transfer_at(self, x, s, eta=1.0):
        """Transfer matrix U(x) (Eq. 10.61, 10.77, 15.20).

        :param x: Coordinate from the input end.
        :type x: float
        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor.
        :type eta: float
        :return: U(x).
        :rtype: numpy.ndarray
        """
        ei, a = self._params(s, eta)
        d0, d1, d2, d3, _ = beam_functions(a, x)
        return np.array([
            [d0, d1, d2 / ei, d3 / ei],
            [a * d3, d0, d1 / ei, d2 / ei],
            [ei * a * d2, ei * a * d3, d0, d1],
            [ei * a * d1, ei * a * d2, a * d3, d0],
        ], dtype=complex)

    def forcing_at(self, x, s):
        """Load column of a uniform load p and moment q_m (Example 10.4 Eq. e,
        Eq. 2.123b)."""
        ei, a = self._params(s, 1.0)
        _, d1, d2, d3, d4 = beam_functions(a, x)
        col_q = np.array([d4 / ei, d3 / ei, d2, d1])
        col_m = np.array([d3 / ei, d2 / ei, d1, a * d4])
        return self.p * col_q - self.q_m * col_m

    def mass_density(self):
        """m on Y (Eq. 3.20)."""
        return np.diag([self.m, 0.0, 0.0, 0.0])

    def stiffness_density(self):
        """Strain energy density M^2/EI + k* Y^2 (Eq. 3.161)."""
        return np.diag([self.k_found, 0.0, 1.0 / self.ei, 0.0])

    def damping_density(self):
        """Damping density c Y^2 + C' M^2/EI^2 (Eq. 3.160)."""
        return np.diag([self.c_external, 0.0, self.c_internal / self.ei**2,
                        0.0])

    def wavenumber(self, s):
        """|lambda| with lambda^4 = (m omega^2 - k*)/EI."""
        _, a = self._params(s, 0.0)
        return abs(a) ** 0.25

    def displacement_index(self):
        """Displacement Y (the rotation is the derivative of Y)."""
        return [0]


class TimoshenkoBeam(ContinuousElement):
    """Timoshenko beam in transverse vibration, state [Y, Theta_z, M_z, Q_y].

    Eq. (10.47)-(10.60), (15.21).

    The first-order equations (Eq. 10.54) are
    Y' = Theta - Q/GA_s, Theta' = M/EI, M' = Q - m rho_z^2 omega^2 Theta,
    Q' = m omega^2 Y. The transfer matrix is B(x) B(0)^{-1} with the
    fundamental matrix of Eq. (10.55) (Eq. 10.59-10.60). Near omega = 0 and
    near the cut-off frequency, where B(0) is singular, the matrix
    exponential of Eq. (10.69) is used.

    :param length: Length l.
    :type length: float
    :param ei: Bending stiffness EI_z.
    :type ei: float
    :param m: Mass per unit length.
    :type m: float
    :param gas: Shear stiffness GA_s = GA/kappa_s.
    :type gas: float
    :param rho_z: Radius of gyration of the cross-section.
    :type rho_z: float
    :param p: Complex amplitude of a uniform transverse load per length.
    :type p: complex
    """

    size = 4

    def __init__(self, length, ei, m, gas, rho_z, p=0.0):
        self.length = float(length)
        self.ei = float(ei)
        self.m = float(m)
        self.gas = float(gas)
        self.rho_z = float(rho_z)
        self.p = complex(p)

    def system_matrix(self, s):
        """System matrix A of Eq. (10.54) with omega^2 = -s^2.

        :param s: Laplace variable.
        :type s: complex
        :return: A (4 x 4).
        :rtype: numpy.ndarray
        """
        w2 = -s * s
        return np.array([
            [0.0, 1.0, 0.0, -1.0 / self.gas],
            [0.0, 0.0, 1.0 / self.ei, 0.0],
            [0.0, -self.m * self.rho_z**2 * w2, 0.0, 1.0],
            [self.m * w2, 0.0, 0.0, 0.0],
        ], dtype=complex)

    def _lambdas(self, s):
        w2 = -s * s
        sigma = self.m * w2 / self.gas
        tau = self.m * self.rho_z**2 * w2 / self.ei
        beta4 = self.m * w2 / self.ei
        root = np.sqrt(complex(beta4 + (sigma - tau) ** 2 / 4.0))
        lam1 = np.sqrt(complex(root - (sigma + tau) / 2.0))
        lam2 = np.sqrt(complex(root + (sigma + tau) / 2.0))
        return sigma, lam1, lam2

    def fundamental(self, x, s):
        """Fundamental matrix B(x) of Eq. (10.55).

        :param x: Coordinate.
        :type x: float
        :param s: Laplace variable.
        :type s: complex
        :return: B(x).
        :rtype: numpy.ndarray
        """
        w2 = -s * s
        sigma, l1, l2 = self._lambdas(s)
        c1 = self.ei * (sigma + l1**2)
        c2 = self.ei * (sigma - l2**2)
        ch, sh = np.cosh(l1 * x), np.sinh(l1 * x)
        co, si = np.cos(l2 * x), np.sin(l2 * x)
        mw = self.m * w2
        return np.array([
            [ch, sh, co, si],
            [(sigma + l1**2) / l1 * sh, (sigma + l1**2) / l1 * ch,
             (sigma - l2**2) / l2 * si, -(sigma - l2**2) / l2 * co],
            [c1 * ch, c1 * sh, c2 * co, c2 * si],
            [mw / l1 * sh, mw / l1 * ch, mw / l2 * si, -mw / l2 * co],
        ], dtype=complex)

    def transfer_at(self, x, s, eta=1.0):
        """Transfer matrix U(x) = e^{A x} (Eq. 10.69, 10.73-10.74).

        The eigenvalues of A are +-lambda_1 and +-i lambda_2 (Eq. 10.53), so
        the Cayley-Hamilton coefficients of Eq. (10.74) have the closed form
        c0 = (l2^2 cosh(l1 x) + l1^2 cos(l2 x))/d, c1 = (l2^2 sinh(l1 x)/l1 +
        l1^2 sin(l2 x)/l2)/d, c2 = (cosh(l1 x) - cos(l2 x))/d,
        c3 = (sinh(l1 x)/l1 - sin(l2 x)/l2)/d with d = l1^2 + l2^2, and
        U = c0 I + c1 A + c2 A^2 + c3 A^3 (Eq. 10.73). This form is well
        conditioned when lambda_1 is close to lambda_2, where B(0) of
        Eq. (10.57) is nearly singular. For max(|l1|, |l2|) x < 0.5 the
        matrix exponential of the dimensionless system (Sec. 10.6.4) is used.

        :param x: Coordinate.
        :type x: float
        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U(x).
        :rtype: numpy.ndarray
        """
        a = self.system_matrix(s)
        _, l1, l2 = self._lambdas(s)
        if x == 0:
            return np.eye(4, dtype=complex)
        if max(abs(l1), abs(l2)) * x < 0.5:
            scale = np.diag([1.0, 1.0 / x, self.ei / x**2, self.ei / x**3])
            inv = np.diag([1.0, x, x**2 / self.ei, x**3 / self.ei])
            return scale @ _num.expm(inv @ a @ scale * x) @ inv
        d = l1**2 + l2**2
        ch, co = np.cosh(l1 * x), np.cos(l2 * x)
        shl = (np.sinh(l1 * x) / l1 if abs(l1 * x) > 1e-4
               else x * (1.0 + (l1 * x) ** 2 / 6.0))
        sil = (np.sin(l2 * x) / l2 if abs(l2 * x) > 1e-4
               else x * (1.0 - (l2 * x) ** 2 / 6.0))
        c0 = (l2**2 * ch + l1**2 * co) / d
        c1 = (l2**2 * shl + l1**2 * sil) / d
        c2 = (ch - co) / d
        c3 = (shl - sil) / d
        a2 = a @ a
        return c0 * np.eye(4) + c1 * a + c2 * a2 + c3 * (a2 @ a)

    def transfer_fundamental(self, x, s):
        """Transfer matrix B(x) B(0)^{-1} with the fundamental matrix of Eq.
        (10.55).

        Eq. (10.59)-(10.60).

        This is the derivation of Sec. 10.3.2. It loses accuracy when
        lambda_1 is close to lambda_2; ``transfer_at`` is used in computations.

        :param x: Coordinate.
        :type x: float
        :param s: Laplace variable.
        :type s: complex
        :return: U(x).
        :rtype: numpy.ndarray
        """
        return _num.transfer_from_fundamental(lambda xx: self.fundamental(xx,
                                                                          s),
                                              x)

    def forcing_at(self, x, s):
        """Load column of a uniform load p from Eq. (10.80)."""
        if self.p == 0:
            return np.zeros(4, dtype=complex)
        f = np.array([0.0, 0.0, 0.0, self.p], dtype=complex)
        ext = _num.extended_from_first_order(self.system_matrix(s), f, x)
        return ext[:4, 4]

    def mass_density(self):
        """m on Y and m rho_z^2 on Theta."""
        return np.diag([self.m, self.m * self.rho_z**2, 0.0, 0.0])

    def stiffness_density(self):
        """Strain energy density M^2/EI + Q^2/GA_s."""
        return np.diag([0.0, 0.0, 1.0 / self.ei, 1.0 / self.gas])

    def wavenumber(self, s):
        """max(|lambda_1|, |lambda_2|)."""
        _, l1, l2 = self._lambdas(s)
        return max(abs(l1), abs(l2))

    def displacement_index(self):
        """Displacement Y and section rotation Theta_z."""
        return [0, 1]


class MasslessShearBeam(Element):
    """Massless beam with shear deformation, state [Y, Theta_z, M_z, Q_y] (Eq.
    15.23).

    :param length: Length.
    :type length: float
    :param ei: Bending stiffness.
    :type ei: float
    :param gas: Shear stiffness GA_s.
    :type gas: float
    """

    size = 4

    def __init__(self, length, ei, gas):
        self.ln, self.ei, self.gas = float(length), float(ei), float(gas)

    def transfer(self, s, eta=1.0):
        """Transfer matrix (Eq. 15.23).

        :param s: Laplace variable (unused).
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U.
        :rtype: numpy.ndarray
        """
        x, ei = self.ln, self.ei
        return np.array([
            [1.0, x, x**2 / (2 * ei), x**3 / (6 * ei) - x / self.gas],
            [0.0, 1.0, x / ei, x**2 / (2 * ei)],
            [0.0, 0.0, 1.0, x],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=complex)


class AxialLoadBeam(ContinuousElement):
    """Massless beam with a constant axial force and shear deformation.

    State [Y, Theta_z, M_z, Q_y].

    Derived from n first-order equations (Sec. 10.4, Eq. 10.68-10.69) with
    Y' = Theta - Q/GA_s, Theta' = M/EI, M' = Q - P Theta, Q' = 0, where
    P > 0 is a compressive force. This is the element of Sec. 15.7.4; the
    closed form of Eq. (15.24) is not transcribed.

    :param length: Length.
    :type length: float
    :param ei: Bending stiffness.
    :type ei: float
    :param axial: Compressive axial force P (negative for tension).
    :type axial: float
    :param gas: Shear stiffness (numpy.inf neglects shear).
    :type gas: float
    """

    size = 4

    def __init__(self, length, ei, axial, gas=np.inf):
        self.length = float(length)
        self.ei = float(ei)
        self.axial = float(axial)
        self.gas = gas

    def system_matrix(self):
        """System matrix A."""
        return np.array([
            [0.0, 1.0, 0.0, -_inv(self.gas)],
            [0.0, 0.0, 1.0 / self.ei, 0.0],
            [0.0, -self.axial, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ])

    def transfer_at(self, x, s, eta=1.0):
        """Transfer matrix e^{A x} (Eq. 10.69).

        :param x: Coordinate.
        :type x: float
        :param s: Laplace variable (unused).
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U(x).
        :rtype: numpy.ndarray
        """
        return _num.transfer_from_first_order(self.system_matrix(),
                                              x).astype(complex)


class Shaft(ContinuousElement):
    """Elastic shaft in torsional vibration, state [Theta_x, M_x] (Eq. 10.46b,
    15.30-15.32, 15.34).

    gamma^2 = (rho J_p omega^2 - K_t)/GJ_p; U = [[cos gamma x, sin(gamma
    x)/(gamma GJ)], [-gamma GJ sin gamma x, cos gamma x]]. rho_jp = 0 gives the
    massless shaft of Eq. (2.22) / (15.31). A uniform external torque t per
    length gives the load column f = -t [(1 - cos gamma x)/(gamma^2 GJ),
    sin(gamma x)/gamma].

    :param length: Length l.
    :type length: float
    :param gj: Torsional stiffness GJ_p.
    :type gj: float
    :param rho_jp: Polar mass moment per unit length rho J_p.
    :type rho_jp: float
    :param k_found: Torsional foundation modulus K_t.
    :type k_found: float
    :param torque: Complex amplitude of a uniform torque per length.
    :type torque: complex
    """

    size = 2

    def __init__(self, length, gj, rho_jp=0.0, k_found=0.0, torque=0.0):
        self.length = float(length)
        self.gj = float(gj)
        self.rho_jp = float(rho_jp)
        self.k_found = float(k_found)
        self.torque = complex(torque)

    def _b(self, s):
        return -(self.rho_jp * s * s + self.k_found) / self.gj

    def transfer_at(self, x, s, eta=1.0):
        """Transfer matrix U(x) (Eq. 10.46b).

        :param x: Coordinate.
        :type x: float
        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U(x).
        :rtype: numpy.ndarray
        """
        b = self._b(s)
        e0, e1, _ = rod_functions(b, x)
        return np.array([[e0, e1 / self.gj], [-self.gj * b * e1, e0]],
                        dtype=complex)

    def forcing_at(self, x, s):
        """Load column of a uniform torque (Eq. 10.80)."""
        _, e1, e2 = rod_functions(self._b(s), x)
        return -self.torque * np.array([e2 / self.gj, e1])

    def mass_density(self):
        """rho J_p on Theta."""
        return np.diag([self.rho_jp, 0.0])

    def stiffness_density(self):
        """Strain energy density M^2/GJ + K_t Theta^2."""
        return np.diag([self.k_found, 1.0 / self.gj])

    def wavenumber(self, s):
        """|gamma|."""
        return abs(self._b(s)) ** 0.5


class RigidShaft(Element):
    """Rigid shaft on an elastic torsional foundation, state [Theta_x, M_x]
    (Eq. 15.33).

    :param length: Length.
    :type length: float
    :param rho_jp: Polar mass moment per unit length.
    :type rho_jp: float
    :param k_found: Foundation modulus K_t.
    :type k_found: float
    """

    size = 2

    def __init__(self, length, rho_jp, k_found=0.0):
        self.ln, self.rho_jp = float(length), float(rho_jp)
        self.k_found = float(k_found)

    def transfer(self, s, eta=1.0):
        """Transfer matrix [[1, 0], [l (K_t - rho J omega^2), 1]] (Eq. 15.33).

        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U.
        :rtype: numpy.ndarray
        """
        return np.array([[1.0, 0.0],
                         [self.ln * (self.k_found + self.rho_jp * s * s),
                          1.0]],
                        dtype=complex)

    def mass_weight(self):
        """Inertia rho J_p l on Theta."""
        return np.diag([self.rho_jp * self.ln, 0.0])

    def stiffness_weight(self):
        """Foundation energy K_t l Theta^2."""
        return np.diag([self.k_found * self.ln, 0.0]) if self.k_found else None


class Rod(ContinuousElement):
    """Rod in longitudinal vibration, state [X, Q_x] (Eq. 10.66, 15.36-15.37,
    Table 10.2).

    beta^2 = (m omega^2 - k*)/EA, U = [[cos beta x, -sin(beta x)/(beta EA)],
    [beta EA sin beta x, cos beta x]]. A uniform load p per length gives
    f = p [-(1 - cos beta x)/(beta^2 EA), sin(beta x)/beta] (Table 10.2).

    :param length: Length l.
    :type length: float
    :param ea: Tensile stiffness EA.
    :type ea: float
    :param m: Mass per unit length.
    :type m: float
    :param k_found: Axial foundation modulus.
    :type k_found: float
    :param p: Complex amplitude of a uniform axial load per length.
    :type p: complex
    """

    size = 2

    def __init__(self, length, ea, m=0.0, k_found=0.0, p=0.0):
        self.length = float(length)
        self.ea = float(ea)
        self.m = float(m)
        self.k_found = float(k_found)
        self.p = complex(p)

    def _b(self, s):
        return -(self.m * s * s + self.k_found) / self.ea

    def transfer_at(self, x, s, eta=1.0):
        """Transfer matrix U(x) (Eq. 10.66).

        :param x: Coordinate.
        :type x: float
        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U(x).
        :rtype: numpy.ndarray
        """
        b = self._b(s)
        e0, e1, _ = rod_functions(b, x)
        return np.array([[e0, -e1 / self.ea], [self.ea * b * e1, e0]],
                        dtype=complex)

    def forcing_at(self, x, s):
        """Load column of a uniform load (Example 10.3, Table 10.2)."""
        _, e1, e2 = rod_functions(self._b(s), x)
        return self.p * np.array([-e2 / self.ea, e1])

    def mass_density(self):
        """m on X."""
        return np.diag([self.m, 0.0])

    def stiffness_density(self):
        """Strain energy density Q^2/EA + k* X^2."""
        return np.diag([self.k_found, 1.0 / self.ea])

    def wavenumber(self, s):
        """|beta|."""
        return abs(self._b(s)) ** 0.5


def _embed(target, block, idx):
    """Place a square block into target at the index set idx."""
    target[np.ix_(idx, idx)] = block


class PlanarBeam(ContinuousElement):
    """Euler-Bernoulli beam in longitudinal and transverse vibration (Eq.
    15.39, 2.59).

    State [X, Y, Theta_z, M_z, Q_x, Q_y]. The rod block acts on (X, Q_x) and
    the beam block on (Y, Theta_z, M_z, Q_y). ``ea=numpy.inf`` gives the
    rigid axial motion of Example 2.10 Eq. (2.59): X_O = X_I and
    Q_x,O = Q_x,I + m omega^2 x X_I.

    :param length: Length l.
    :type length: float
    :param ei: Bending stiffness EI_z.
    :type ei: float
    :param m: Mass per unit length.
    :type m: float
    :param ea: Tensile stiffness (numpy.inf for rigid axial motion).
    :type ea: float
    """

    size = 6
    _ROD = [0, 4]
    _BEAM = [1, 2, 3, 5]

    def __init__(self, length, ei, m, ea=np.inf):
        self.length = float(length)
        self.ei = float(ei)
        self.m = float(m)
        self.ea = ea
        self.beam = EulerBernoulliBeam(length, ei, m)
        self.rod = None if ea == np.inf else Rod(length, ea, m)

    def _rod_matrix(self, x, s):
        if self.rod is None:
            return np.array([[1.0, 0.0], [-self.m * s * s * x, 1.0]],
                            dtype=complex)
        return self.rod.transfer_at(x, s)

    def transfer_at(self, x, s, eta=1.0):
        """Transfer matrix U(x) (Eq. 15.39 / 2.59).

        :param x: Coordinate.
        :type x: float
        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U(x).
        :rtype: numpy.ndarray
        """
        u = np.zeros((6, 6), dtype=complex)
        _embed(u, self._rod_matrix(x, s), self._ROD)
        _embed(u, self.beam.transfer_at(x, s), self._BEAM)
        return u

    def mass_density(self):
        """m on X and Y (Eq. 3.105)."""
        return np.diag([self.m, self.m, 0.0, 0.0, 0.0, 0.0])

    def stiffness_density(self):
        """Strain energy density M^2/EI + Q_x^2/EA."""
        return np.diag([0.0, 0.0, 0.0, 1.0 / self.ei, _inv(self.ea), 0.0])

    def wavenumber(self, s):
        """Largest of the beam and rod wavenumbers."""
        rod = self.rod.wavenumber(s) if self.rod is not None else 0.0
        return max(self.beam.wavenumber(s), rod)

    def displacement_index(self):
        """Displacements X and Y."""
        return [0, 1]


class SpatialTorsionBeam(ContinuousElement):
    """Beam in torsion about x and transverse vibration in z (Eq. 15.40).

    State [Z, Theta_x, Theta_y, M_x, M_y, Q_z].

    The z bending block uses the same signs as Eq. (15.20), as printed in
    Eq. (15.40).

    :param length: Length l.
    :type length: float
    :param ei_y: Bending stiffness EI_y.
    :type ei_y: float
    :param m: Mass per unit length.
    :type m: float
    :param gj: Torsional stiffness GJ_p.
    :type gj: float
    :param rho_jp: Polar mass moment per unit length rho J_p.
    :type rho_jp: float
    """

    size = 6
    _SHAFT = [1, 3]
    _BEAM = [0, 2, 4, 5]

    def __init__(self, length, ei_y, m, gj, rho_jp):
        self.length = float(length)
        self.m = float(m)
        self.rho_jp = float(rho_jp)
        self.beam = EulerBernoulliBeam(length, ei_y, m)
        self.shaft = Shaft(length, gj, rho_jp)

    def transfer_at(self, x, s, eta=1.0):
        """Transfer matrix U(x) (Eq. 15.40).

        :param x: Coordinate.
        :type x: float
        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U(x).
        :rtype: numpy.ndarray
        """
        u = np.zeros((6, 6), dtype=complex)
        _embed(u, self.shaft.transfer_at(x, s), self._SHAFT)
        _embed(u, self.beam.transfer_at(x, s), self._BEAM)
        return u

    def mass_density(self):
        """m on Z, rho J_p on Theta_x."""
        return np.diag([self.m, self.rho_jp, 0.0, 0.0, 0.0, 0.0])

    def stiffness_density(self):
        """M_x^2/GJ + M_y^2/EI_y."""
        return np.diag([0.0, 0.0, 0.0, 1.0 / self.shaft.gj, 1.0 / self.beam.ei,
                        0.0])

    def wavenumber(self, s):
        """Largest wavenumber."""
        return max(self.beam.wavenumber(s), self.shaft.wavenumber(s))

    def displacement_index(self):
        """Displacement Z and torsion angle Theta_x."""
        return [0, 1]


class SpatialBeam(ContinuousElement):
    """Euler-Bernoulli beam in torsion, longitudinal and two transverse
    directions.

    Eq. (15.41)-(15.42).

    12-state [X, Y, Z, Theta_x, Theta_y, Theta_z, M_x, M_y, M_z, Q_x, Q_y,
    Q_z]. Blocks: rod (X, Q_x), shaft (Theta_x, M_x), y bending
    (Y, Theta_z, M_z, Q_y) as Eq. (15.20), z bending (Z, Theta_y, M_y, Q_z)
    with the signs of Eq. (15.41) (D U D with D = diag(1, -1, -1, 1)).
    ``rigid_axial=True`` uses Eq. (15.42) for the rod and shaft blocks with
    distributed inertia: u_10,1 = m omega^2 x and u_7,4 = -rho J_p omega^2 x.

    :param length: Length l.
    :type length: float
    :param ea: Tensile stiffness.
    :type ea: float
    :param ei_y: Bending stiffness about y.
    :type ei_y: float
    :param ei_z: Bending stiffness about z.
    :type ei_z: float
    :param gj: Torsional stiffness.
    :type gj: float
    :param m: Mass per unit length.
    :type m: float
    :param rho_jp: Polar mass moment per unit length.
    :type rho_jp: float
    :param rigid_axial: Rigid longitudinal and torsional motion (Eq. 15.42).
    :type rigid_axial: bool
    """

    size = 12
    _ROD = [0, 9]
    _SHAFT = [3, 6]
    _BEAM_Y = [1, 5, 8, 10]
    _BEAM_Z = [2, 4, 7, 11]
    _FLIP = np.diag([1.0, -1.0, -1.0, 1.0])

    def __init__(self, length, ea, ei_y, ei_z, gj, m, rho_jp,
                 rigid_axial=False):
        self.length = float(length)
        self.m = float(m)
        self.rho_jp = float(rho_jp)
        self.rigid_axial = rigid_axial
        self.rod = Rod(length, ea, m)
        self.shaft = Shaft(length, gj, rho_jp)
        self.beam_y = EulerBernoulliBeam(length, ei_z, m)
        self.beam_z = EulerBernoulliBeam(length, ei_y, m)

    def transfer_at(self, x, s, eta=1.0):
        """Transfer matrix U(x) (Eq. 15.41).

        :param x: Coordinate.
        :type x: float
        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U(x).
        :rtype: numpy.ndarray
        """
        u = np.zeros((12, 12), dtype=complex)
        if self.rigid_axial:
            w2 = -s * s
            _embed(u, np.array([[1.0, 0.0], [self.m * w2 * x, 1.0]]),
                   self._ROD)
            _embed(u, np.array([[1.0, 0.0], [-self.rho_jp * w2 * x, 1.0]]),
                   self._SHAFT)
        else:
            _embed(u, self.rod.transfer_at(x, s), self._ROD)
            _embed(u, self.shaft.transfer_at(x, s), self._SHAFT)
        _embed(u, self.beam_y.transfer_at(x, s), self._BEAM_Y)
        _embed(u, self._FLIP @ self.beam_z.transfer_at(x, s) @ self._FLIP,
               self._BEAM_Z)
        return u

    def mass_density(self):
        """m on X, Y, Z and rho J_p on Theta_x."""
        return np.diag([self.m, self.m, self.m, self.rho_jp] + [0.0] * 8)

    def stiffness_density(self):
        """Strain energy density of the four blocks."""
        w = np.zeros(12)
        w[6] = 1.0 / self.shaft.gj
        w[7] = 1.0 / self.beam_z.ei
        w[8] = 1.0 / self.beam_y.ei
        if not self.rigid_axial:
            w[9] = 1.0 / self.rod.ea
        else:
            w[6] = 0.0
        return np.diag(w)

    def wavenumber(self, s):
        """Largest wavenumber."""
        return max(self.beam_y.wavenumber(s), self.beam_z.wavenumber(s),
                   self.rod.wavenumber(s), self.shaft.wavenumber(s))

    def displacement_index(self):
        """Displacements X, Y, Z and torsion angle Theta_x."""
        return [0, 1, 2, 3]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
