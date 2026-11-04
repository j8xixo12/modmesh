# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Controlled multibody systems, mixed TMM, FE-TMM and finite segment TMM (book
Ch. 9).

Contents:

* Linear controlled chains in the frequency domain (Sec. 9.5, library
  Sec. 15.30): real-time control (Eq. 9.12-9.30) and delay control
  (Eq. 9.40-9.47). ``ControlledChain`` gives the overall transfer matrix of
  Eq. (9.26), the steady-state response, the closed-loop characteristic
  determinant and the closed-loop eigenvalues.
* Discrete time TMM of controlled chains (Sec. 9.6, library Sec. 15.31):
  Newmark linearized transfer matrices (Eq. 9.48-9.62) with real-time
  control (Eq. 9.53-9.56) and with delay control (Eq. 9.81-9.83),
  ``DTControlledChain``.
* Planar rigid bodies in the acceleration form of Ch. 5 (state
  [xdd, ydd, Omega, m, q_x, q_y, 1], Eq. 5.4), used by the mixed TMM
  (Sec. 9.2), the FE-TMM for multibody systems (Sec. 9.3) and the finite
  segment TMM (Sec. 9.4): ``planar_body_transfer``, ``propagate_rods``,
  ``MixedSystem``, ``BeamRodsSystem``, ``FiniteSegmentBeam``.

Sign conventions.

* Longitudinal chains (Sec. 9.5-9.6): state [X, Q_x] with the library
  convention of ``solvcon.multibody.tmm.library``: a spring gives X_O = X_I -
  Q/K and a mass gives Q_O = Q_I - m s^2 X + F. Q is the force that the inboard
  element applies to the outboard element.
* Control law (Eq. 9.13): F_c = -K_a xdd_k - K_v xd_k - K_d x_k. In the
  frequency domain F_c = g(s) X_k with g(s) = -(K_a s^2 + K_v s + K_d)
  e^{-s tau} (Eq. 9.17, 9.43 with s = i Omega).
* A control force on a mass enters the force row (Eq. 9.19). A control
  force in parallel with a spring (Example 9.7) pushes the outboard mass
  in +x and the inboard mass in -x, so X_O = X_I - (Q - F_c)/(K + C s)
  (Example 9.7 Eq. c).
* Planar bodies: at a connection point q is the force that the inboard
  element applies to the outboard element and m is the moment that the
  outboard element applies to the inboard element (book Sec. 6.4: q_I and
  m_O are positive along the axes, q_O and m_I along the negative axes).
  The force on a body is q_I - q_O, the moment on a body is m_O - m_I.

Solution rule for loops and hinges. A control loop (Eq. 9.23-9.24) and a
hinge with an unknown relative angular acceleration (Sec. 5.6-5.8) add one
scalar unknown each. The state vectors are propagated as affine functions
of all unknowns (boundary unknowns, sensor signal, hinge accelerations);
each loop or hinge adds one scalar equation (sensor identity, hinge moment
condition). This is algebraically the same as the elimination of
Eq. (9.24) and of the hinge transfer matrices, and it does not divide by
the loop factor det(I - W U_c), so the closed-loop characteristic
determinant has no poles.
"""

import numpy as np

from . import library as _lib
from . import riccati as _num
from .linear import Chain

# ----------------------------------------------------------------------
# Frequency-domain controlled chain (Sec. 9.5, 15.30)
# ----------------------------------------------------------------------


def control_gain(s, ka, kv, kd, delay=0.0):
    """Frequency response g(s) of the control law F_c = g(s) X_k (Eq. 9.17,
    9.43).

    :param s: Laplace variable (s = i Omega gives K_a Omega^2 - i K_v Omega -
        K_d).
    :type s: complex
    :param ka: Acceleration gain K_a.
    :type ka: float
    :param kv: Velocity gain K_v.
    :type kv: float
    :param kd: Displacement gain K_d.
    :type kd: float
    :param delay: Time delay tau (s).
    :type delay: float
    :return: g(s) = -(K_a s^2 + K_v s + K_d) exp(-s tau).
    :rtype: complex
    """
    return -(ka * s * s + kv * s + kd) * np.exp(-s * delay)


def controlled_mass_matrices(m, s, force=0.0, gains=(0.0, 0.0, 0.0),
                             delay=0.0):
    """Extended matrices U_p and U_p,c of a controlled lumped mass (Eq. 9.19,
    15.203).

    Z_{p,p+1} = U_p Z_{p,p-1} + U_p,c Z_{k,k-1} with Z = [X, Q_x, 1].

    :param m: Mass m_p.
    :type m: float
    :param s: Laplace variable.
    :type s: complex
    :param force: Complex force amplitude F_p.
    :type force: complex
    :param gains: (K_a, K_v, K_d).
    :type gains: tuple
    :param delay: Time delay tau (Eq. 9.43).
    :type delay: float
    :return: (U_p, U_p,c), 3 x 3 complex.
    :rtype: tuple
    """
    u = np.array([[1.0, 0.0, 0.0], [-m * s * s, 1.0, force], [0.0, 0.0, 1.0]],
                 dtype=complex)
    uc = np.zeros((3, 3), dtype=complex)
    uc[1, 0] = control_gain(s, *gains, delay=delay)
    return u, uc


def controlled_spring_matrices(k, c, s, gains=(0.0, 0.0, 0.0), delay=0.0):
    """Extended matrices U_p and U_p,c of a spring-damper with a parallel
    actuator.

    Example 9.7 Eq. (c): U_3,c[0, 0] = (K_a Omega^2 - i K_v Omega - K_d)/(K_3 +
    i C_3 Omega).

    :param k: Spring stiffness.
    :type k: float
    :param c: Damping coefficient.
    :type c: float
    :param s: Laplace variable.
    :type s: complex
    :param gains: (K_a, K_v, K_d).
    :type gains: tuple
    :param delay: Time delay tau.
    :type delay: float
    :return: (U_p, U_p,c), 3 x 3 complex.
    :rtype: tuple
    """
    z = k + c * s
    u = np.array([[1.0, -1.0 / z, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                 dtype=complex)
    uc = np.zeros((3, 3), dtype=complex)
    uc[0, 0] = control_gain(s, *gains, delay=delay) / z
    return u, uc


def _actuator_column(el, s, eta):
    """State change per unit control force for a mass (Eq. 9.19) or a spring
    (Example 9.7)."""
    n = el.size
    col = np.zeros(n + 1, dtype=complex)
    half = n // 2
    if isinstance(el, _lib.LumpedMass):
        col[half] = 1.0
    elif isinstance(el, _lib.Spring):
        col[0] = 1.0 / _lib._impedance(el.k[0], el.c[0], el.g[0], s, eta)
    else:
        raise TypeError("actuator element must be a LumpedMass or a Spring")
    return col


class ControlledChain:
    """Linear chain with one feedback loop (Sec. 9.5.1-9.5.2, Fig. 9.26).

    The sensor reads component ``sensor_component`` of the state vector at
    the input end of element ``sensor`` (Z_{k,k-1} of the book). The control
    force F_c = g(s) X_k (Eq. 9.17, 9.43) acts on element ``actuator``
    (a ``LumpedMass``: Eq. 9.19; a ``Spring``: Example 9.7). The sensor can
    be on either side of the actuator.

    :param elements: Elements of ``solvcon.multibody.tmm.library`` with state
        [X, Q_x], in transfer order.
    :type elements: list
    :param sensor: Index of the element whose input state is measured.
    :type sensor: int
    :param actuator: Index of the controlled element.
    :type actuator: int
    :param gains: (K_a, K_v, K_d) of Eq. (9.13).
    :type gains: tuple
    :param delay: Time delay tau (Eq. 9.41); 0 is real-time control.
    :type delay: float
    :param input_zero: Zero components of the input boundary state.
    :type input_zero: list
    :param output_zero: Zero components of the output boundary state.
    :type output_zero: list
    :param sensor_component: Index of the measured displacement.
    :type sensor_component: int
    """

    def __init__(self, elements, sensor, actuator, gains, delay=0.0,
                 input_zero=(0,), output_zero=(1,), sensor_component=0):
        self.elements = list(elements)
        self.size = self.elements[0].size
        self.sensor = int(sensor)
        self.actuator = int(actuator)
        self.gains = tuple(float(g) for g in gains)
        self.delay = float(delay)
        self.input_zero = sorted(int(i) for i in input_zero)
        self.output_zero = sorted(int(i) for i in output_zero)
        self.input_unknown = [i for i in range(self.size) if i
                              not in self.input_zero]
        self.sensor_component = int(sensor_component)
        if len(self.output_zero) != len(self.input_unknown):
            raise ValueError("boundary conditions must give a square system")

    def _gain(self, s, eta):
        ka, kv, kd = self.gains
        return eta * control_gain(s, ka, kv, kd, self.delay)

    # ------------------------------------------------------------------
    # Book form of the overall transfer matrix (Eq. 9.23-9.26)
    # ------------------------------------------------------------------
    def control_matrix(self, s, eta=1.0):
        """Extended control matrix U_p,c of the actuator (Eq. 9.19, Example 9.7
        Eq. c).

        :param s: Laplace variable.
        :type s: complex
        :param eta: Scale factor of damping and control gains.
        :type eta: float
        :return: Matrix of order size + 1 with U_p,c[:, sensor_component] set.
        :rtype: numpy.ndarray
        """
        n = self.size
        uc = np.zeros((n + 1, n + 1), dtype=complex)
        col = _actuator_column(self.elements[self.actuator], s, eta)
        uc[:, self.sensor_component] = col * self._gain(s, eta)
        return uc

    def _product(self, lo, hi, s, eta):
        """U_{hi-1} ... U_lo (extended); identity if hi <= lo."""
        u = np.eye(self.size + 1, dtype=complex)
        for el in self.elements[lo:hi]:
            u = el.extended(s, eta) @ u
        return u

    def overall_extended(self, s, eta=1.0):
        """Overall extended transfer matrix of the closed loop (Eq. 9.24-9.26).

        For a sensor after the actuator (k > p) this is Eq. (9.26),
        U = U_tail (I - W U_c)^{-1} W U_p V; for k <= p the sensor state is
        Z_k = U_{k-1} ... U_1 Z_0 and U = U_tail (U_p W + U_c) V.

        :param s: Laplace variable.
        :type s: complex
        :param eta: Scale factor of damping and control gains.
        :type eta: float
        :return: Matrix of order size + 1.
        :rtype: numpy.ndarray
        """
        p, k = self.actuator, self.sensor
        n_el = len(self.elements)
        uc = self.control_matrix(s, eta)
        up = self.elements[p].extended(s, eta)
        if k > p:
            v = self._product(0, p, s, eta)
            w = self._product(p + 1, k, s, eta)
            ident = np.eye(self.size + 1)
            zk = np.linalg.solve(ident - w @ uc, w @ up @ v)
            return self._product(k, n_el, s, eta) @ zk
        v = self._product(0, k, s, eta)
        w = self._product(k, p, s, eta)
        return self._product(p + 1, n_el, s, eta) @ (up @ w + uc) @ v

    # ------------------------------------------------------------------
    # Augmented propagation (module docstring)
    # ------------------------------------------------------------------
    def _augmented(self, s, eta, loads):
        """Coefficients of every state vector over [input unknowns, X_k, 1]."""
        n = self.size
        nu = len(self.input_unknown)
        cols = nu + 2
        state = np.zeros((n + 1, cols), dtype=complex)
        for j, i in enumerate(self.input_unknown):
            state[i, j] = 1.0
        state[n, cols - 1] = 1.0
        states = []
        sensor_row = None
        for idx, el in enumerate(self.elements):
            if idx == self.sensor:
                sensor_row = state[self.sensor_component].copy()
            states.append(state)
            ext = el.extended(s, eta)
            if not loads:
                ext[:n, n] = 0.0
            state = ext @ state
            if idx == self.actuator:
                col = _actuator_column(el, s, eta) * self._gain(s, eta)
                state = state.copy()
                state[:, nu] += col
        if self.sensor == len(self.elements):
            sensor_row = state[self.sensor_component].copy()
        states.append(state)
        sensor_row[nu] -= 1.0
        mat = np.vstack([state[self.output_zero], sensor_row[None, :]])
        return mat[:, :-1], -mat[:, -1], states

    def characteristic(self, s, eta=1.0):
        """Closed-loop characteristic determinant D(s) as (phase, logabs).

        D(s) = -det(I - W U_c) times the plain determinant of Eq. (9.28)
        (module docstring); its zeros are the closed-loop eigenvalues and it
        has no poles.

        :param s: Laplace variable.
        :type s: complex
        :param eta: Scale factor of damping and control gains.
        :type eta: float
        :return: (phase, logabs).
        :rtype: tuple
        """
        mat, _, _ = self._augmented(s, eta, loads=False)
        return _num.log_determinant(mat)

    def steady_state(self, omega, eta=1.0):
        """Steady-state complex amplitudes at the input end of every element
        (Eq. 9.27-9.30).

        x(t) = Re(X e^{i Omega t}) = X^r cos(Omega t) - X^i sin(Omega t) (Eq.
        9.30).

        :param omega: Excitation frequency Omega (rad/s).
        :type omega: float
        :param eta: Scale factor of damping and control gains.
        :type eta: float
        :return: Array (n_elements + 1, size): input states, then the output
            boundary.
        :rtype: numpy.ndarray
        """
        mat, rhs, states = self._augmented(1j * omega, eta, loads=True)
        sol = np.linalg.solve(mat, rhs)
        vec = np.concatenate([sol, [1.0]])
        return np.array([st[:self.size] @ vec for st in states])

    def steady_state_book(self, omega):
        """Steady-state response with the book overall matrix (Eq. 9.25-9.29).

        :param omega: Excitation frequency (rad/s).
        :type omega: float
        :return: Array (n_elements + 1, size) of complex input states.
        :rtype: numpy.ndarray
        """
        s = 1j * omega
        u = self.overall_extended(s)
        n = self.size
        a = u[np.ix_(self.output_zero, self.input_unknown)]
        z0 = np.zeros(n + 1, dtype=complex)
        z0[n] = 1.0
        z0[self.input_unknown] = np.linalg.solve(a, -u[self.output_zero, n])
        # sensor state first (Eq. 9.24), then forward transfer (Eq. 9.20-9.21)
        p, k = self.actuator, self.sensor
        if k > p:
            v = self._product(0, p, s, 1.0)
            w = self._product(p + 1, k, s, 1.0)
            uc = self.control_matrix(s)
            zk = np.linalg.solve(np.eye(n + 1) - w @ uc,
                                 w @ self.elements[p].extended(s) @ v @ z0)
        else:
            zk = self._product(0, k, s, 1.0) @ z0
        uc = self.control_matrix(s)
        states = [z0]
        z = z0
        for idx, el in enumerate(self.elements):
            z = el.extended(s) @ z
            if idx == p:
                z = z + uc @ zk
            states.append(z)
        return np.array(states)[:, :n]

    def eigenvalues(self, omega_max, n_steps=8, tol=1e-12):
        """Closed-loop complex eigenvalues by continuation (Sec. 2.7.3, 10.7).

        eta scales all damping parameters and the control gains. At
        eta = 0 the roots are i omega_k of the undamped open-loop chain
        (``linear.Chain.natural_frequencies``); each root is continued to
        eta = 1 with the complex secant method (Eq. 10.104). A step is
        halved when the secant iteration fails or the root moves by more
        than half its modulus.

        :param omega_max: Upper bound of the open-loop frequencies (rad/s).
        :type omega_max: float
        :param n_steps: Number of continuation steps.
        :type n_steps: int
        :param tol: Relative tolerance of the secant iteration.
        :type tol: float
        :return: Eigenvalues with positive imaginary part, sorted by it.
        :rtype: numpy.ndarray
        """
        chain = Chain(self.elements, self.input_zero, self.output_zero)
        starts = [w for w in chain.natural_frequencies(omega_max) if w > 0]
        roots = []
        for w in starts:
            lam = 1j * w
            eta = 0.0
            step = 1.0 / n_steps
            while eta < 1.0:
                eta_new = min(1.0, eta + step)

                def fun(x, e=eta_new):
                    return self.characteristic(x, e)

                guess = lam * (1.0 + 1e-6) + 1e-6 * max(1.0, abs(lam))
                new, ok = _num.complex_secant_root(fun, lam, guess, tol=tol)
                if ok and abs(new - lam) < 0.5 * abs(lam):
                    lam, eta = new, eta_new
                else:
                    step *= 0.5
                    if step < 1e-6:
                        raise RuntimeError("closed-loop continuation failed "
                                           "near %g" % w)
            roots.append(lam.conjugate() if lam.imag < 0 else lam)
        return np.array(sorted(roots, key=lambda z: z.imag))


# ----------------------------------------------------------------------
# Discrete time TMM of a controlled chain (Sec. 9.6, 15.31)
# ----------------------------------------------------------------------
def newmark_coefficients(x, v, a, dt, beta=0.25, gamma=0.5):
    """Linearization xdd = A x + B, xd = C x + D of the Newmark method (Sec.
    7.3.2).

    :param x: Displacement at t_{i-1}.
    :type x: float or numpy.ndarray
    :param v: Velocity at t_{i-1}.
    :type v: float or numpy.ndarray
    :param a: Acceleration at t_{i-1}.
    :type a: float or numpy.ndarray
    :param dt: Time step.
    :type dt: float
    :param beta: Newmark beta.
    :type beta: float
    :param gamma: Newmark gamma.
    :type gamma: float
    :return: (A, B, C, D).
    :rtype: tuple
    """
    a_c = 1.0 / (beta * dt * dt)
    b_c = -(a_c * x + v / (beta * dt) + (0.5 / beta - 1.0) * a)
    c_c = gamma * dt * a_c
    d_c = v + dt * (1.0 - gamma) * a + gamma * dt * b_c
    return a_c, b_c, c_c, d_c


def dt_spring_matrix(k, c, cn, d_in, d_out, free_length=0.0):
    """Discrete time transfer matrix of a spring-damper (Eq. 9.51).

    State z = [x, q, 1]; q = -(K (x_O - x_I - l) + C (xd_O - xd_I)).

    :param k: Stiffness K.
    :type k: float
    :param c: Damping C.
    :type c: float
    :param cn: Newmark coefficient C (xd = C x + D).
    :type cn: float
    :param d_in: Newmark coefficient D of the input point.
    :type d_in: float
    :param d_out: Newmark coefficient D of the output point.
    :type d_out: float
    :param free_length: Free length l.
    :type free_length: float
    :return: 3 x 3 matrix.
    :rtype: numpy.ndarray
    """
    z = k + c * cn
    return np.array([[1.0, -1.0 / z,
                      (c * (d_in - d_out) + k * free_length) / z],
                     [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def dt_mass_matrix(m, a_c, b_c, force=0.0):
    """Discrete time transfer matrix of a lumped mass (Eq. 9.52).

    :param m: Mass.
    :type m: float
    :param a_c: Newmark coefficient A.
    :type a_c: float
    :param b_c: Newmark coefficient B of the mass.
    :type b_c: float
    :param force: External force at t_i.
    :type force: float
    :return: 3 x 3 matrix.
    :rtype: numpy.ndarray
    """
    return np.array([[1.0, 0.0, 0.0], [-m * a_c, 1.0, -m * b_c + force],
                     [0.0, 0.0, 1.0]])


def dt_control_matrix(gains, a_c, b_k, c_c, d_k):
    """Discrete time control matrix U_p,c of a controlled mass (Eq. 9.56,
    15.214).

    F_c = -(K_a A + K_v C + K_d) x_k - K_a B_k - K_v D_k (Eq. 9.53).

    :param gains: (K_a, K_v, K_d).
    :type gains: tuple
    :param a_c: Newmark A.
    :type a_c: float
    :param b_k: Newmark B of the sensor mass.
    :type b_k: float
    :param c_c: Newmark C.
    :type c_c: float
    :param d_k: Newmark D of the sensor mass.
    :type d_k: float
    :return: 3 x 3 matrix acting on z_{k,k-1}.
    :rtype: numpy.ndarray
    """
    ka, kv, kd = gains
    uc = np.zeros((3, 3))
    uc[1, 0] = -(ka * a_c + kv * c_c + kd)
    uc[1, 2] = -ka * b_k - kv * d_k
    return uc


def dt_delay_control_matrix(gains, x_k, v_k, a_k):
    """Discrete time control matrix of a delay controlled mass (Eq. 9.82,
    15.216).

    :param gains: (K_a, K_v, K_d).
    :type gains: tuple
    :param x_k: Sensor displacement at t - tau.
    :type x_k: float
    :param v_k: Sensor velocity at t - tau.
    :type v_k: float
    :param a_k: Sensor acceleration at t - tau.
    :type a_k: float
    :return: 3 x 3 matrix; only u_23 = F_c is nonzero.
    :rtype: numpy.ndarray
    """
    ka, kv, kd = gains
    uc = np.zeros((3, 3))
    uc[1, 2] = -ka * a_k - kv * v_k - kd * x_k
    return uc


def _riccati_states(mats, loads):
    """Input states [x, q] of every element of a wall-to-free chain by the
    Riccati TMM.

    x = S q + e at every point (Eq. 10.127), S_1 = e_1 = 0 at the wall
    (Eq. 10.133), recursion of Eq. (10.131)-(10.132), output q = 0
    (Eq. 10.134) and back substitution of Eq. (10.135).
    """
    n = len(mats)
    s_l, e_l, g_l = [0.0], [0.0], []
    s_v, e_v = 0.0, 0.0
    for u, f in zip(mats, loads):
        g = u[1, 0] * s_v + u[1, 1]
        s_n = (u[0, 0] * s_v + u[0, 1]) / g
        e_v = u[0, 0] * e_v + f[0] - s_n * (u[1, 0] * e_v + f[1])
        s_v = s_n
        s_l.append(s_v)
        e_l.append(e_v)
        g_l.append(g)
    out = np.zeros((n, 2))
    q_out = 0.0
    for i in range(n - 1, -1, -1):
        u, f = mats[i], loads[i]
        q_in = (q_out - u[1, 0] * e_l[i] - f[1]) / g_l[i]
        out[i] = (s_l[i] * q_in + e_l[i], q_in)
        q_out = q_in
    return out


def _plain_states(mats, loads):
    """Input states of every element by the plain overall transfer equation
    (Eq. 9.61-9.62)."""
    u = np.eye(3)
    for m, f in zip(mats, loads):
        e = np.eye(3)
        e[:2, :2] = m
        e[:2, 2] = f
        u = e @ u
    q0 = -u[1, 2] / u[1, 1]
    z = np.array([0.0, q0, 1.0])
    out = []
    for m, f in zip(mats, loads):
        out.append(z[:2].copy())
        z = np.concatenate([m @ z[:2] + f, [1.0]])
    return np.array(out)


class DTControlledChain:
    """Discrete time TMM of a controlled wall-spring-mass chain (Sec. 9.6, Fig.
    9.37).

    Topology: wall - spring 0 - mass 0 - spring 1 - mass 1 - ... - mass n-1 -
    free end. Elements in transfer order: spring j then mass j. The control
    force F_c = -K_a xdd_k - K_v xd_k - K_d x_k of mass ``sensor`` acts on
    mass ``actuator`` (``actuator_kind='mass'``, Eq. 9.55-9.56) or in
    parallel with spring ``actuator`` (``'spring'``, Examples 9.7, 9.11).
    With ``delay_steps`` = d > 0 the control force uses the sensor motion at
    t_i - d dt and enters the load column (Eq. 9.81-9.83); before t = 0 the
    system rests in the initial state.

    :param masses: Masses m_j.
    :type masses: list
    :param stiffness: Spring stiffnesses K_j.
    :type stiffness: list
    :param damping: Damping coefficients C_j.
    :type damping: list
    :param force: Callable t -> array of external forces on the masses.
    :type force: callable
    :param sensor: Index of the measured mass.
    :type sensor: int
    :param actuator: Index of the controlled mass or spring.
    :type actuator: int
    :param gains: (K_a, K_v, K_d).
    :type gains: tuple
    :param dt: Time step.
    :type dt: float
    :param actuator_kind: 'mass' or 'spring'.
    :type actuator_kind: str
    :param delay_steps: Delay in time steps (tau = delay_steps * dt).
    :type delay_steps: int
    :param beta: Newmark beta.
    :type beta: float
    :param gamma: Newmark gamma.
    :type gamma: float
    :param solver: 'transfer' solves the overall transfer equation of
        Eq. (9.57)-(9.62) (products of the element matrices). 'riccati'
        (default) uses the Riccati transfer of Sec. 10.9. The plain products
        grow like (m A / K)^n with A = 1/(beta dt^2), so for small dt the
        overall matrix loses all digits (for Example 9.10 with dt = 1e-3:
        (10 * 4e6 / 300)^3 = 2e15); the Riccati form stays accurate.
    :type solver: str
    """

    def __init__(self, masses, stiffness, damping, force, sensor, actuator,
                 gains, dt, actuator_kind="mass", delay_steps=0, beta=0.25,
                 gamma=0.5, solver="riccati"):
        self.m = np.asarray(masses, dtype=float)
        self.k = np.asarray(stiffness, dtype=float)
        self.c = np.asarray(damping, dtype=float)
        self.force = force
        self.sensor = int(sensor)
        self.actuator = int(actuator)
        self.gains = tuple(float(g) for g in gains)
        self.dt = float(dt)
        if actuator_kind not in ("mass", "spring"):
            raise ValueError("actuator_kind must be 'mass' or 'spring'")
        self.kind = actuator_kind
        self.delay_steps = int(delay_steps)
        self.beta, self.gamma = float(beta), float(gamma)
        if solver not in ("riccati", "transfer"):
            raise ValueError("solver must be 'riccati' or 'transfer'")
        self.solver = solver

    def _control_force(self, xs, vs, as_):
        ka, kv, kd = self.gains
        j = self.sensor
        return -ka * as_[j] - kv * vs[j] - kd * xs[j]

    def initial_acceleration(self, x0, v0, t0=0.0):
        """Accelerations at t0 from the force balance of every mass.

        Spring forces follow from x0, v0. With real-time control and K_a != 0
        the accelerations of the sensor and the actuator are coupled; the
        coupled balance is solved as a small linear system.

        :param x0: Initial displacements.
        :type x0: numpy.ndarray
        :param v0: Initial velocities.
        :type v0: numpy.ndarray
        :param t0: Initial time.
        :type t0: float
        :return: Initial accelerations.
        :rtype: numpy.ndarray
        """
        n = len(self.m)
        xe = np.concatenate([[0.0], x0])
        ve = np.concatenate([[0.0], v0])
        tension = self.k * (xe[1:] - xe[:-1]) + self.c * (ve[1:] - ve[:-1])
        net = np.asarray(self.force(t0), dtype=float).copy()
        net -= tension
        net[:-1] += tension[1:]
        mat = np.diag(self.m)
        ka, kv, kd = self.gains
        delayed = self.delay_steps > 0
        fc_known = (0.0 if delayed else -kv * v0[self.sensor]
                    - kd * x0[self.sensor])
        p = self.actuator
        targets = ([(p, 1.0)] if self.kind == "mass"
                   else [(p, 1.0), (p - 1, -1.0)])
        for i, sign in targets:
            if i < 0:
                continue
            net[i] += sign * fc_known
            if not delayed:
                mat[i, self.sensor] += sign * ka
        if delayed:
            fc = self._control_force(x0, np.zeros(n), np.zeros(n))
            for i, sign in targets:
                if i >= 0:
                    net[i] += sign * fc
        return np.linalg.solve(mat, net)

    def step(self, t, x, v, a, delayed_state=None):
        """One time step t_{i-1} -> t_i = t (Eq. 9.48-9.62).

        :param t: New time t_i.
        :type t: float
        :param x: Displacements at t_{i-1}.
        :type x: numpy.ndarray
        :param v: Velocities at t_{i-1}.
        :type v: numpy.ndarray
        :param a: Accelerations at t_{i-1}.
        :type a: numpy.ndarray
        :param delayed_state: (x, v, a) at t_i - tau for delay control.
        :type delayed_state: tuple or None
        :return: (x, v, a, q) at t_i; q are the spring forces q_{j-1,j}.
        :rtype: tuple
        """
        n = len(self.m)
        dt, beta, gamma = self.dt, self.beta, self.gamma
        a_c, b_c, c_c, d_c = newmark_coefficients(x, v, a, dt, beta, gamma)
        dvec, bvec = d_c, b_c
        fext = np.asarray(self.force(t), dtype=float)
        ka, kv, kd = self.gains
        j_k = self.sensor
        if self.delay_steps > 0:
            fc_coef, fc_const = 0.0, self._control_force(*delayed_state)
        else:
            fc_coef = -(ka * a_c + kv * c_c + kd)
            fc_const = -ka * bvec[j_k] - kv * dvec[j_k]
        mats, cols = [], []
        for j in range(n):
            d_in = 0.0 if j == 0 else dvec[j - 1]
            mats.append(dt_spring_matrix(self.k[j], self.c[j], c_c, d_in,
                                         dvec[j]))
            col = np.zeros(2)
            if self.kind == "spring" and j == self.actuator:
                col[0] = 1.0 / (self.k[j] + self.c[j] * c_c)
            cols.append(col)
            mats.append(dt_mass_matrix(self.m[j], a_c, bvec[j], fext[j]))
            col = np.zeros(2)
            if self.kind == "mass" and j == self.actuator:
                col[1] = 1.0
            cols.append(col)
        loads = [u[:2, 2] for u in mats]
        blocks = [u[:2, :2] for u in mats]
        states_of = (_riccati_states if self.solver == "riccati"
                     else _plain_states)
        # superposition: states = R(loads) + F_c R(unit control force); the
        # control force F_c (not the sensor signal) is the loop unknown, so the
        # large terms K_a A x and K_a B of Eq. (9.53) cancel in a scalar of
        # physical size (Eq. 9.53-9.58, 10.123-10.135)
        r0 = states_of(blocks, loads)
        r1 = states_of(blocks, cols)
        idx = 2 * j_k + 1  # mass j has its input state at element 2 j + 1
        f_c = (fc_coef * r0[idx, 0] + fc_const) / (1.0 - fc_coef * r1[idx, 0])
        states = r0 + f_c * r1
        y = states[1:2 * n:2, 0]
        q_new = states[1:2 * n:2, 1]
        return y, c_c * y + dvec, a_c * y + bvec, q_new

    def simulate(self, t_end, x0=None, v0=None):
        """Time history from t = 0 to t_end (Sec. 9.6.1-9.6.2).

        :param t_end: Final time.
        :type t_end: float
        :param x0: Initial displacements (default 0).
        :type x0: numpy.ndarray or None
        :param v0: Initial velocities (default 0).
        :type v0: numpy.ndarray or None
        :return: (t, x, v, a) arrays with one row per time step.
        :rtype: tuple
        """
        n = len(self.m)
        x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float)
        v = np.zeros(n) if v0 is None else np.asarray(v0, dtype=float)
        a = self.initial_acceleration(x, v)
        n_steps = int(round(t_end / self.dt))
        ts = [0.0]
        xs, vs, as_ = [x], [v], [a]
        for i in range(1, n_steps + 1):
            t = i * self.dt
            delayed = None
            if self.delay_steps > 0:
                j = i - self.delay_steps
                rest = (xs[0], np.zeros(n), np.zeros(n))
                delayed = (xs[j], vs[j], as_[j]) if j >= 0 else rest
            x, v, a, _ = self.step(t, x, v, a, delayed)
            ts.append(t)
            xs.append(x)
            vs.append(v)
            as_.append(a)
        return np.array(ts), np.array(xs), np.array(vs), np.array(as_)


# ----------------------------------------------------------------------
# Planar rigid bodies in the acceleration form (Ch. 5; used by Sec. 9.2-9.4)
# ----------------------------------------------------------------------
_PERP = np.array([[0.0, -1.0], [1.0, 0.0]])


def rot2(theta):
    """Planar rotation matrix (body to inertial frame).

    :param theta: Angle (rad).
    :type theta: float
    :return: 2 x 2 matrix.
    :rtype: numpy.ndarray
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def planar_body_transfer(mass, j_c, c_body, b_body, theta, omega,
                         force=(0.0, 0.0), moment=0.0):
    """Transfer matrix of a planar rigid body with one input and one output
    (Ch. 5, Eq. 5.4).

    State z = [xdd, ydd, Omega, m, q_x, q_y, 1] in the inertial frame. With
    r_IC = A c, r_IO = A b, r_CO = r_IO - r_IC and P r = k x r:

    * a_O = a_I + P r_IO Omega - omega^2 r_IO (rigid-body kinematics),
    * q_O = q_I + f - m a_C with a_C = a_I + P r_IC Omega - omega^2 r_IC,
    * m_O = m_I + J_C Omega + r_IC x q_I + r_CO x q_O - M (moment about C).

    The equations are exact at the given angle and angular velocity (no
    linearization in time, Sec. 5.2-5.3).

    :param mass: Mass m.
    :type mass: float
    :param j_c: Moment of inertia about the mass center.
    :type j_c: float
    :param c_body: Mass center relative to the input point, body frame.
    :type c_body: tuple
    :param b_body: Output point relative to the input point, body frame.
    :type b_body: tuple
    :param theta: Body angle.
    :type theta: float
    :param omega: Body angular velocity.
    :type omega: float
    :param force: External force at the mass center (inertial frame), for
        example m g.
    :type force: tuple
    :param moment: External moment.
    :type moment: float
    :return: 7 x 7 matrix.
    :rtype: numpy.ndarray
    """
    a = rot2(theta)
    r_ic = a @ np.asarray(c_body, dtype=float)
    r_io = a @ np.asarray(b_body, dtype=float)
    r_co = r_io - r_ic
    f = np.asarray(force, dtype=float)
    p_ic, p_io, p_co = _PERP @ r_ic, _PERP @ r_io, _PERP @ r_co
    w2 = omega * omega
    u = np.eye(7)
    u[0:2, 2] = p_io
    u[0:2, 6] = -w2 * r_io
    q_const = f + mass * w2 * r_ic
    u[4:6, 0:2] = -mass * np.eye(2)
    u[4:6, 2] = -mass * p_ic
    u[4:6, 6] = q_const
    # m_O = m_I + J Omega + (P r_IC).q_I + (P r_CO).q_O - M
    u[3, 0:2] = -mass * p_co
    u[3, 2] = j_c - mass * p_co @ p_ic
    u[3, 4:6] = p_ic + p_co
    u[3, 6] = p_co @ q_const - moment
    return u


class PlanarRod:
    """Planar rigid body with input point I and output point O (body frame,
    origin at I).

    :param mass: Mass.
    :type mass: float
    :param j_c: Moment of inertia about the mass center.
    :type j_c: float
    :param c_body: Mass center (body frame).
    :type c_body: tuple
    :param b_body: Output point (body frame).
    :type b_body: tuple
    """

    def __init__(self, mass, j_c, c_body, b_body):
        self.mass = float(mass)
        self.j_c = float(j_c)
        self.c = np.asarray(c_body, dtype=float)
        self.b = np.asarray(b_body, dtype=float)

    @classmethod
    def uniform(cls, mass, length, j_c=None):
        """Uniform rod along the body x axis, I at 0, O at length.

        :param mass: Mass.
        :type mass: float
        :param length: Length.
        :type length: float
        :param j_c: Moment of inertia about the center (default m l^2 / 12).
        :type j_c: float or None
        :return: PlanarRod.
        :rtype: PlanarRod
        """
        jc = mass * length**2 / 12.0 if j_c is None else j_c
        return cls(mass, jc, (0.5 * length, 0.0), (length, 0.0))


def propagate_rods(rods, joints, thetas, omegas, z_root, hinge_index,
                   gravity=(0.0, 0.0)):
    """Affine state propagation along a chain of planar rods (Sec. 5.4-5.8,
    9.2).

    ``z_root`` is the state at the input of rod 0 (after the root joint) as an
    affine function of the unknown vector u: an array (7, n_u + 1) whose
    last column is the constant part (row 6 is [0, ..., 0, 1]). Joint j
    (j = 1 .. n-1) connects rod j-1 and rod j: 'fixed' keeps Omega,
    ('pin',) or ('spring', k, c) adds the unknown relative angular
    acceleration u[hinge_index[j]] and one equation for the hinge moment:
    m = 0 (smooth hinge) or m = k (theta_j - theta_{j-1}) + c (omega_j -
    omega_{j-1}) (elastic hinge). ``joints[0]`` is ignored.

    :param rods: PlanarRod objects.
    :type rods: list
    :param joints: Joint descriptions (see above), one per rod.
    :type joints: list
    :param thetas: Rod angles.
    :type thetas: list
    :param omegas: Rod angular velocities.
    :type omegas: list
    :param z_root: Affine root state (7, n_u + 1).
    :type z_root: numpy.ndarray
    :param hinge_index: Unknown index of each hinge (entry 0 unused).
    :type hinge_index: list
    :param gravity: Gravity acceleration vector.
    :type gravity: tuple
    :return: (inputs, outputs, equations): affine input and output states
        of every rod and the list of hinge equation rows (each row r means
        r[:-1] . u + r[-1] = 0).
    :rtype: tuple
    """
    g = np.asarray(gravity, dtype=float)
    z = np.asarray(z_root, dtype=float).copy()
    inputs, outputs, eqs = [], [], []
    for j, rod in enumerate(rods):
        if j > 0:
            joint = joints[j]
            kind = joint[0] if isinstance(joint, tuple) else joint
            if kind != "fixed":
                row = z[3].copy()
                if kind == "spring":
                    _, k_s, c_s = joint
                    row[-1] -= (k_s * (thetas[j] - thetas[j - 1])
                                + c_s * (omegas[j] - omegas[j - 1]))
                eqs.append(row)
                z = z.copy()
                z[2, hinge_index[j]] += 1.0
        inputs.append(z)
        u = planar_body_transfer(rod.mass, rod.j_c, rod.c, rod.b, thetas[j],
                                 omegas[j], force=rod.mass * g)
        z = u @ z
        outputs.append(z)
    return inputs, outputs, eqs


def rod_points(rods, thetas, root):
    """Input points of every rod and the tip point of a rod chain.

    :param rods: PlanarRod objects.
    :type rods: list
    :param thetas: Rod angles.
    :type thetas: list
    :param root: Input point of rod 0.
    :type root: numpy.ndarray
    :return: Array (n + 1, 2).
    :rtype: numpy.ndarray
    """
    pts = [np.asarray(root, dtype=float)]
    for rod, th in zip(rods, thetas):
        pts.append(pts[-1] + rot2(th) @ rod.b)
    return np.array(pts)


def _affine(n_u, const=None):
    """Zero affine state (7, n_u + 1) with the trailing 1 of the state
    vector."""
    z = np.zeros((7, n_u + 1))
    z[6, -1] = 1.0
    if const is not None:
        z[:6, -1] = const
    return z


def _solve_affine(rows):
    """Solve rows r[:-1] . u = -r[-1]."""
    mat = np.array(rows)
    return np.linalg.solve(mat[:, :-1], -mat[:, -1])


def rk4_step(fun, t, y, h):
    """Classical fourth-order Runge-Kutta step (Sec. 5.9).

    :param fun: Right-hand side f(t, y).
    :type fun: callable
    :param t: Time.
    :type t: float
    :param y: State.
    :type y: numpy.ndarray
    :param h: Step.
    :type h: float
    :return: State at t + h.
    :rtype: numpy.ndarray
    """
    k1 = fun(t, y)
    k2 = fun(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = fun(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = fun(t + h, y + h * k3)
    return y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ----------------------------------------------------------------------
# Finite segment TMM (Sec. 9.4, Example 9.5)
# ----------------------------------------------------------------------
def segment_stiffness(ei_a, l_a, ei_b, l_b):
    """Torsional stiffness between two rigid segments (Eq. 9.2-9.4).

    K = 2 EI_a EI_b / (l_b EI_a + l_a EI_b); the same form with EA gives the
    linear spring of Eq. (9.7).

    :param ei_a: Bending stiffness of segment a.
    :type ei_a: float
    :param l_a: Length of segment a.
    :type l_a: float
    :param ei_b: Bending stiffness of segment b.
    :type ei_b: float
    :param l_b: Length of segment b.
    :type l_b: float
    :return: Stiffness.
    :rtype: float
    """
    return 2.0 * ei_a * ei_b / (l_b * ei_a + l_a * ei_b)


class FiniteSegmentBeam:
    """Finite segment model of a beam on a rotating base (Sec. 9.4, Fig.
    9.19-9.20).

    N equal rigid segments with axis along the body x axis; segment i is
    joined to segment i+1 by a torsional spring K = EI/l (Eq. 9.4) and to
    the base by a half-segment spring 2 EI/l (Eq. 9.3). The base point is
    fixed at the origin and has the prescribed angle theta_b(t). The axial
    springs of Eq. (9.6)-(9.7) are not modeled: the segments are rigid in
    the axial direction and the tip foreshortening follows from the
    segment rotations.

    Generalized coordinates: absolute segment angles theta_i. Boundary
    states (Eq. 9.9-9.10): root z = [0, 0, Omega_1, m, q_x, q_y, 1] with
    m = 2 EI/l (theta_1 - theta_b) + c (omega_1 - omega_b), tip m = q = 0.

    :param length: Beam length l.
    :type length: float
    :param ei: Bending stiffness EI.
    :type ei: float
    :param rho_a: Mass per unit length rho A.
    :type rho_a: float
    :param n_segments: Number of segments N.
    :type n_segments: int
    :param base: Callable t -> (theta_b, omega_b, alpha_b) of the base.
    :type base: callable
    :param damping: Viscous coefficient of every torsional spring.
    :type damping: float
    """

    def __init__(self, length, ei, rho_a, n_segments, base, damping=0.0):
        self.n = int(n_segments)
        self.l_seg = length / self.n
        m_seg = rho_a * self.l_seg
        self.rods = [PlanarRod.uniform(m_seg, self.l_seg) for _
                     in range(self.n)]
        k_in = segment_stiffness(ei, self.l_seg, ei, self.l_seg)
        self.k_root = 2.0 * ei / self.l_seg
        self.c = float(damping)
        self.joints = ["fixed"] + [("spring", k_in, self.c)] * (self.n - 1)
        self.base = base

    def solve(self, t, thetas, omegas):
        """Angular accelerations and root loads at one instant (Eq. 9.11).

        Unknowns: Omega_1, q_x, q_y at the root and the N - 1 hinge relative
        accelerations; equations: N - 1 hinge moments and m = q = 0 at the tip.

        :param t: Time.
        :type t: float
        :param thetas: Segment angles.
        :type thetas: numpy.ndarray
        :param omegas: Segment angular velocities.
        :type omegas: numpy.ndarray
        :return: (alphas, m_root, q_root): segment angular accelerations,
            root (driving) moment and root force.
        :rtype: tuple
        """
        th_b, om_b, _ = self.base(t)
        n_u = 3 + self.n - 1
        m_root = self.k_root * (thetas[0] - th_b) + self.c * (omegas[0] - om_b)
        z = _affine(n_u, [0.0, 0.0, 0.0, m_root, 0.0, 0.0])
        z[2, 0] = 1.0
        z[4, 1] = 1.0
        z[5, 2] = 1.0
        hinge = [None] + list(range(3, n_u))
        inputs, outputs, eqs = propagate_rods(self.rods, self.joints, thetas,
                                              omegas, z, hinge)
        tip = outputs[-1]
        rows = eqs + [tip[3], tip[4], tip[5]]
        u = _solve_affine(rows)
        vec = np.concatenate([u, [1.0]])
        alphas = np.array([zi[2] @ vec for zi in inputs])
        return alphas, m_root, u[1:3]

    def rhs(self, t, y):
        """First-order right-hand side for y = [theta, omega].

        :param t: Time.
        :type t: float
        :param y: State.
        :type y: numpy.ndarray
        :return: dy/dt.
        :rtype: numpy.ndarray
        """
        n = self.n
        alphas, _, _ = self.solve(t, y[:n], y[n:])
        return np.concatenate([y[n:], alphas])

    def tip_deformation(self, t, thetas):
        """Tip deformation in the base frame: (axial u, transverse v).

        :param t: Time.
        :type t: float
        :param thetas: Segment angles.
        :type thetas: numpy.ndarray
        :return: (u, v) with u = x_tip - l (foreshortening, negative).
        :rtype: tuple
        """
        th_b = self.base(t)[0]
        tip = rod_points(self.rods, thetas, np.zeros(2))[-1]
        local = rot2(th_b).T @ tip
        return local[0] - self.n * self.l_seg, local[1]

    def simulate(self, t_end, h, thetas0=None, omegas0=None):
        """RK4 time history (Sec. 5.9).

        :param t_end: Final time.
        :type t_end: float
        :param h: Step.
        :type h: float
        :param thetas0: Initial angles (default: undeformed on the base).
        :type thetas0: numpy.ndarray or None
        :param omegas0: Initial angular velocities (default: base rate).
        :type omegas0: numpy.ndarray or None
        :return: (t, y) with y = [theta, omega] per row.
        :rtype: tuple
        """
        th_b, om_b, _ = self.base(0.0)
        th = (np.full(self.n, th_b) if thetas0 is None
              else np.asarray(thetas0, dtype=float))
        om = (np.full(self.n, om_b) if omegas0 is None
              else np.asarray(omegas0, dtype=float))
        y = np.concatenate([th, om])
        n_steps = int(round(t_end / h))
        ts, ys = [0.0], [y]
        for i in range(n_steps):
            y = rk4_step(self.rhs, i * h, y, h)
            ts.append((i + 1) * h)
            ys.append(y)
        return np.array(ts), np.array(ys)


def spin_up_base(omega_ss, t_acc):
    """Base motion of Example 9.5 (Eq. 9.8): omega = Omega/T (t - T/(2 pi)
    sin(2 pi t/T)).

    :param omega_ss: Steady angular velocity Omega.
    :type omega_ss: float
    :param t_acc: Acceleration time T.
    :type t_acc: float
    :return: Callable t -> (theta, omega, alpha).
    :rtype: callable
    """
    def base(t):
        if t <= t_acc:
            w = 2.0 * np.pi / t_acc
            th = omega_ss / t_acc * (0.5 * t * t
                                     + (np.cos(w * t) - 1.0) / (w * w))
            om = omega_ss / t_acc * (t - np.sin(w * t) / w)
            al = omega_ss / t_acc * (1.0 - np.cos(w * t))
            return th, om, al
        th_t = omega_ss / t_acc * (0.5 * t_acc * t_acc)
        return th_t + omega_ss * (t - t_acc), omega_ss, 0.0
    return base


# ----------------------------------------------------------------------
# FE-TMM for multibody systems (Sec. 9.3, Example 9.4)
# ----------------------------------------------------------------------
def hermite_beam_matrices(ei, rho_a, length):
    """Stiffness and consistent mass matrix of a Hermite beam element (Example
    9.4 Eq. e).

    DOF order [v_1, v_1', v_2, v_2'].

    :param ei: Bending stiffness.
    :type ei: float
    :param rho_a: Mass per unit length.
    :type rho_a: float
    :param length: Element length l.
    :type length: float
    :return: (k_e, m_e).
    :rtype: tuple
    """
    lv = length
    k = ei / lv**3 * np.array([[12, 6 * lv, -12, 6 * lv],
                               [6 * lv, 4 * lv**2, -6 * lv, 2 * lv**2],
                               [-12, -6 * lv, 12, -6 * lv],
                               [6 * lv, 2 * lv**2, -6 * lv, 4 * lv**2]],
                              dtype=float)
    m = rho_a * lv / 420.0 * np.array([[156, 22 * lv, 54, -13 * lv],
                                       [22 * lv, 4 * lv**2, 13 * lv,
                                        -3 * lv**2],
                                       [54, 13 * lv, 156, -22 * lv],
                                       [-13 * lv, -3 * lv**2, -22 * lv,
                                        4 * lv**2]],
                                      dtype=float)
    return k, m


def cantilever_fe(ei, rho_a, length, n_elements, gravity=0.0):
    """Global M, K and gravity load of a cantilever clamped at x = 0 (Example
    9.4 Eq. g-h).

    The clamped DOFs of node 0 are removed; the free DOFs are
    [v_1, v_1', ..., v_n, v_n'].

    :param ei: Bending stiffness.
    :type ei: float
    :param rho_a: Mass per unit length.
    :type rho_a: float
    :param length: Beam length.
    :type length: float
    :param n_elements: Number of elements.
    :type n_elements: int
    :param gravity: Transverse gravity acceleration (for example -9.81).
    :type gravity: float
    :return: (M, K, P).
    :rtype: tuple
    """
    n = int(n_elements)
    le = length / n
    ke, me = hermite_beam_matrices(ei, rho_a, le)
    pe = rho_a * gravity * le * np.array([0.5, le / 12.0, 0.5, -le / 12.0])
    size = 2 * (n + 1)
    mg = np.zeros((size, size))
    kg = np.zeros((size, size))
    pg = np.zeros(size)
    for e in range(n):
        idx = slice(2 * e, 2 * e + 4)
        mg[idx, idx] += me
        kg[idx, idx] += ke
        pg[idx] += pe
    return mg[2:, 2:], kg[2:, 2:], pg[2:]


class BeamRodsSystem:
    """Cantilever FE beam carrying a chain of pinned rigid rods (Sec. 9.3, Fig.
    9.10).

    Subsystem 1 (FEM, Example 9.4 Eq. g): an Euler-Bernoulli cantilever
    along +x, clamped at x = 0, transverse DOFs only (rigid in the axial
    direction). Subsystem 2 (TMM): rods joined by smooth hinges; the input of
    rod 0 is pinned to the beam tip (L, v_tip). The beam tip force is the
    reaction -q_y of the rod chain (Eq. m-n). Generalized coordinates:
    beam DOFs delta and the rod angles.

    :param beam: (EI, rho A, L, n_elements).
    :type beam: tuple
    :param rods: PlanarRod objects of subsystem 2.
    :type rods: list
    :param gravity: Gravity acceleration (negative y).
    :type gravity: float
    :param beam_gravity: Apply the beam weight as consistent nodal loads.
    :type beam_gravity: bool
    """

    def __init__(self, beam, rods, gravity=9.81, beam_gravity=True):
        ei, rho_a, length, n_el = beam
        self.length = float(length)
        self.m_b, self.k_b, self.p_b = cantilever_fe(ei, rho_a, length, n_el,
                                                     -gravity if beam_gravity
                                                     else 0.0)
        self.n_b = self.m_b.shape[0]
        self.rods = list(rods)
        self.joints = ["fixed"] + [("pin",)] * (len(self.rods) - 1)
        self.g = np.array([0.0, -gravity])

    def solve(self, delta, delta_d, thetas, omegas):
        """Accelerations and hinge-2 force at one instant (Example 9.4 Eq.
        i-n).

        Unknowns: beam accelerations, Omega of rod 0, (q_x, q_y) of hinge 2 and
        the hinge relative accelerations; equations: beam rows, hinge
        moments and the free tip (m = q = 0).

        :param delta: Beam DOFs.
        :type delta: numpy.ndarray
        :param delta_d: Beam DOF velocities (unused; no beam damping).
        :type delta_d: numpy.ndarray
        :param thetas: Rod angles.
        :type thetas: numpy.ndarray
        :param omegas: Rod angular velocities.
        :type omegas: numpy.ndarray
        :return: (delta_dd, alphas, q_hinge).
        :rtype: tuple
        """
        nb = self.n_b
        n_r = len(self.rods)
        n_u = nb + 3 + (n_r - 1)
        i_alpha, i_qx, i_qy = nb, nb + 1, nb + 2
        z = _affine(n_u)
        # rod input acceleration = beam tip transverse acceleration
        z[1, nb - 2] = 1.0
        z[2, i_alpha] = 1.0
        z[4, i_qx] = 1.0
        z[5, i_qy] = 1.0
        hinge = [None] + list(range(nb + 3, n_u))
        inputs, outputs, eqs = propagate_rods(self.rods, self.joints, thetas,
                                              omegas, z, hinge, gravity=self.g)
        rows = []
        resid = self.k_b @ delta - self.p_b
        for i in range(nb):
            row = np.zeros(n_u + 1)
            row[:nb] = self.m_b[i]
            row[-1] = resid[i]
            if i == nb - 2:
                row[i_qy] += 1.0  # the rods pull the tip with -q_y
            rows.append(row)
        tip = outputs[-1]
        rows += eqs + [tip[3], tip[4], tip[5]]
        u = _solve_affine(rows)
        vec = np.concatenate([u, [1.0]])
        alphas = np.array([zi[2] @ vec for zi in inputs])
        return u[:nb], alphas, u[[i_qx, i_qy]]

    def rhs(self, t, y):
        """Right-hand side for y = [delta, theta, delta_d, omega].

        :param t: Time.
        :type t: float
        :param y: State.
        :type y: numpy.ndarray
        :return: dy/dt.
        :rtype: numpy.ndarray
        """
        nq = self.n_b + len(self.rods)
        q, qd = y[:nq], y[nq:]
        nb = self.n_b
        dd, al, _ = self.solve(q[:nb], qd[:nb], q[nb:], qd[nb:])
        return np.concatenate([qd, dd, al])

    def energy(self, y):
        """Kinetic + strain + gravity energy (IV check).

        :param y: State [delta, theta, delta_d, omega].
        :type y: numpy.ndarray
        :return: Energy.
        :rtype: float
        """
        nb = self.n_b
        nq = nb + len(self.rods)
        q, qd = y[:nq], y[nq:]
        d, dd = q[:nb], qd[:nb]
        e = 0.5 * dd @ self.m_b @ dd + 0.5 * d @ self.k_b @ d - self.p_b @ d
        r_in = np.array([self.length, d[-2]])
        v_in = np.array([0.0, dd[-2]])
        for rod, th, om in zip(self.rods, q[nb:], qd[nb:]):
            a = rot2(th)
            v_c = v_in + om * (_PERP @ (a @ rod.c))
            r_c = r_in + a @ rod.c
            e += (0.5 * rod.mass * v_c @ v_c + 0.5 * rod.j_c * om * om
                  - rod.mass * self.g @ r_c)
            v_in = v_in + om * (_PERP @ (a @ rod.b))
            r_in = r_in + a @ rod.b
        return e


# ----------------------------------------------------------------------
# Mixed TMM: rod chains (TMM) + particle-spring subsystem (Newton-Euler), Sec.
# 9.2
# ----------------------------------------------------------------------
class MixedSystem:
    """Rod chains (TMM) coupled with a particle-spring subsystem
    (Newton-Euler), Sec. 9.2.

    Each chain is a list of rods joined by smooth hinges. Its root (input of
    rod 0) is pinned to a ground point or to a particle; its tip is free or
    pinned to a particle. A particle on a chain tip is a dependent particle
    (its position follows the chain); every other particle is independent.
    The subsystem boundary forces (Example 9.1 Eq. i, Example 9.3 Eq. a)
    and the hinge accelerations are the unknowns of one linear system per
    instant (Sec. 9.2 steps 7-9): chains give transfer equations, particles
    give Newton's equations.

    Generalized coordinates q: the absolute angles of the rods of chain 0,
    chain 1, ..., then the positions of the independent particles in index
    order.

    Unknowns per chain: Omega of rod 0, the root force (q_x, q_y) applied by
    the root point to rod 0, and the hinge relative accelerations; then the
    accelerations of the independent particles. Equations per chain: hinge
    moments m = 0; at a free tip m = q = 0; at a particle tip m = 0 and the
    particle's Newton equation with a = tip acceleration. Independent
    particles: Newton's equation; a chain rooted at a particle pulls it with
    -q_root.

    :param masses: Particle masses.
    :type masses: list
    :param springs: Springs (i, j, k, free_length) between particles i and j.
    :type springs: list
    :param chains: Chains (rods, root, tip): root is ('ground', (x, y)) or
        ('particle', i); tip is ('free',) or ('particle', i).
    :type chains: list
    :param gravity: Gravity acceleration (negative y).
    :type gravity: float
    """

    def __init__(self, masses, springs, chains, gravity=9.81):
        self.m = np.asarray(masses, dtype=float)
        self.springs = [(int(i), int(j), float(k), float(l0)) for i, j, k, l0
                        in springs]
        self.chains = [(list(rods), root, tip) for rods, root, tip in chains]
        self.g = np.array([0.0, -gravity])
        self.tip_of = {}
        for c, (_, _, tip) in enumerate(self.chains):
            if tip[0] == "particle":
                self.tip_of[int(tip[1])] = c
        self.free_particles = [i for i in range(len(self.m)) if i
                               not in self.tip_of]
        self.order = self._chain_order()
        self.angle_slices = []
        n = 0
        for rods, _, _ in self.chains:
            self.angle_slices.append(slice(n, n + len(rods)))
            n += len(rods)
        self.n_angles = n
        self.n_q = n + 2 * len(self.free_particles)

    def _chain_order(self):
        """Chains in an order where every root point is known before the
        chain."""
        done, order = set(), []
        while len(order) < len(self.chains):
            progress = False
            for c, (_, root, _) in enumerate(self.chains):
                if c in done:
                    continue
                if root[0] == "particle" and int(root[1]) in self.tip_of and \
                        self.tip_of[int(root[1])] not in done:
                    continue
                order.append(c)
                done.add(c)
                progress = True
            if not progress:
                raise ValueError("chain topology has a cycle")
        return order

    def kinematics(self, q, qd):
        """Positions and velocities of all particles and the rod input points.

        :param q: Coordinates.
        :type q: numpy.ndarray
        :param qd: Velocities.
        :type qd: numpy.ndarray
        :return: (pos, vel, roots): particle positions and velocities, and
            for every chain the (position, velocity) of its root point.
        :rtype: tuple
        """
        n_p = len(self.m)
        pos = np.zeros((n_p, 2))
        vel = np.zeros((n_p, 2))
        base = self.n_angles
        for k, i in enumerate(self.free_particles):
            pos[i] = q[base + 2 * k:base + 2 * k + 2]
            vel[i] = qd[base + 2 * k:base + 2 * k + 2]
        roots = [None] * len(self.chains)
        for c in self.order:
            rods, root, tip = self.chains[c]
            if root[0] == "ground":
                r, v = np.asarray(root[1], dtype=float), np.zeros(2)
            else:
                r, v = pos[int(root[1])].copy(), vel[int(root[1])].copy()
            roots[c] = (r.copy(), v.copy())
            th, om = q[self.angle_slices[c]], qd[self.angle_slices[c]]
            for rod, t_j, w_j in zip(rods, th, om):
                arm = rot2(t_j) @ rod.b
                r = r + arm
                v = v + w_j * (_PERP @ arm)
            if tip[0] == "particle":
                pos[int(tip[1])] = r
                vel[int(tip[1])] = v
        return pos, vel, roots

    def spring_forces(self, pos):
        """Spring force on every particle (Example 9.1 Eq. d-h)."""
        f = np.zeros_like(pos)
        for i, j, k, l0 in self.springs:
            d = pos[j] - pos[i]
            dist = np.linalg.norm(d)
            t = k * (dist - l0) * d / dist
            f[i] += t
            f[j] -= t
        return f

    def solve(self, q, qd):
        """Accelerations of all generalized coordinates at one instant (Sec.
        9.2 steps 7-9).

        :param q: Coordinates.
        :type q: numpy.ndarray
        :param qd: Velocities.
        :type qd: numpy.ndarray
        :return: Accelerations (same layout as q).
        :rtype: numpy.ndarray
        """
        pos, _, _ = self.kinematics(q, qd)
        fs = self.spring_forces(pos)
        offsets, n_u = [], 0
        for rods, _, _ in self.chains:
            offsets.append(n_u)
            n_u += 3 + len(rods) - 1
        p_index = {i: n_u + 2 * k for k, i in enumerate(self.free_particles)}
        n_u += 2 * len(self.free_particles)
        acc_part = {}
        for i in self.free_particles:
            z = np.zeros((2, n_u + 1))
            z[0, p_index[i]] = 1.0
            z[1, p_index[i] + 1] = 1.0
            acc_part[i] = z
        rows, inputs_all = [], [None] * len(self.chains)
        tip_force = {}
        for c in self.order:
            rods, root, tip = self.chains[c]
            o = offsets[c]
            z = _affine(n_u)
            if root[0] == "particle":
                z[0:2] = acc_part[int(root[1])]
            z[2, o] = 1.0
            z[4, o + 1] = 1.0
            z[5, o + 2] = 1.0
            joints = ["fixed"] + [("pin",)] * (len(rods) - 1)
            hinge = [None] + list(range(o + 3, o + 3 + len(rods) - 1))
            th, om = q[self.angle_slices[c]], qd[self.angle_slices[c]]
            inputs, outputs, eqs = propagate_rods(rods, joints, th, om, z,
                                                  hinge, gravity=self.g)
            inputs_all[c] = inputs
            rows += eqs
            out = outputs[-1]
            if tip[0] == "free":
                rows += [out[3], out[4], out[5]]
            else:
                rows.append(out[3])
                acc_part[int(tip[1])] = out[0:2].copy()
                tip_force[int(tip[1])] = out[4:6].copy()
        # Newton's equation of every particle: m a - q_tip - F_s - m g + sum
        # q_root = 0
        for i in range(len(self.m)):
            for d in range(2):
                row = self.m[i] * acc_part[i][d]
                if i in tip_force:
                    row = row - tip_force[i][d]
                row = row.copy()
                row[-1] -= fs[i, d] + self.m[i] * self.g[d]
                for c, (_, root, _) in enumerate(self.chains):
                    if root[0] == "particle" and int(root[1]) == i:
                        row[offsets[c] + 1 + d] += 1.0
                rows.append(row)
        u = _solve_affine(rows)
        vec = np.concatenate([u, [1.0]])
        acc = np.zeros(self.n_q)
        for c in range(len(self.chains)):
            acc[self.angle_slices[c]] = [zi[2] @ vec for zi in inputs_all[c]]
        for k, i in enumerate(self.free_particles):
            acc[self.n_angles + 2 * k:self.n_angles + 2 * k
                + 2] = u[p_index[i]:p_index[i] + 2]
        return acc

    def rhs(self, t, y):
        """Right-hand side for y = [q, qd].

        :param t: Time.
        :type t: float
        :param y: State.
        :type y: numpy.ndarray
        :return: dy/dt.
        :rtype: numpy.ndarray
        """
        n = self.n_q
        return np.concatenate([y[n:], self.solve(y[:n], y[n:])])


def mixed_example_9_3(rod, particle_mass, spring_k, spacing, n_particles=5,
                      gravity=9.81):
    """Model of Example 9.3 (Fig. 9.8): rod - particle chain - rod, horizontal
    at rest.

    Rod A is pinned to the ground at the origin, the particles lie at
    x = l + i s (particle 0 on the tip of rod A), rod B starts at the last
    particle. The spacing s is not printed in the book.

    :param rod: PlanarRod used for both rods.
    :type rod: PlanarRod
    :param particle_mass: Particle mass.
    :type particle_mass: float
    :param spring_k: Spring stiffness.
    :type spring_k: float
    :param spacing: Particle spacing s (spring free length).
    :type spacing: float
    :param n_particles: Number of particles.
    :type n_particles: int
    :param gravity: Gravity acceleration.
    :type gravity: float
    :return: (MixedSystem, q0).
    :rtype: tuple
    """
    n = int(n_particles)
    springs = [(i, i + 1, spring_k, spacing) for i in range(n - 1)]
    chains = [([rod], ("ground", (0.0, 0.0)), ("particle", 0)),
              ([rod], ("particle", n - 1), ("free",))]
    sys_ = MixedSystem([particle_mass] * n, springs, chains, gravity)
    x0 = rod.b[0]
    q0 = np.concatenate([[0.0, 0.0],
                         np.ravel([[x0 + i * spacing, 0.0]
                                   for i in range(1, n)])])
    return sys_, q0


def mixed_example_9_1(rod, particle_mass, spring_k, spacing, gravity=9.81):
    """Model of Example 9.1 (Fig. 9.1-9.2): 5 x 5 particle-spring mesh and four
    rods.

    Rod 1 is pinned to the ground at the origin and its tip carries
    m_{1,1}; rods 3, 4 and 5 start at m_{1,5}, m_{5,5} and m_{5,1} and have
    free tips; rods 1, 3, 4 point in +x and rod 5 in -x (Fig. 9.1 arrows).
    Particle m_{i,j} (index 5 (i - 1) + j - 1) lies at
    (l + (j - 1) s, -(i - 1) s). The mesh spacing s is not printed.

    :param rod: PlanarRod used for the four rods.
    :type rod: PlanarRod
    :param particle_mass: Particle mass.
    :type particle_mass: float
    :param spring_k: Spring stiffness.
    :type spring_k: float
    :param spacing: Mesh spacing s (spring free length).
    :type spacing: float
    :param gravity: Gravity acceleration.
    :type gravity: float
    :return: (MixedSystem, q0).
    :rtype: tuple
    """
    def idx(i, j):
        return 5 * (i - 1) + (j - 1)

    springs = []
    for i in range(1, 6):
        for j in range(1, 6):
            if j < 5:
                springs.append((idx(i, j), idx(i, j + 1), spring_k, spacing))
            if i < 5:
                springs.append((idx(i, j), idx(i + 1, j), spring_k, spacing))
    chains = [([rod], ("ground", (0.0, 0.0)), ("particle", idx(1, 1))),
              ([rod], ("particle", idx(1, 5)), ("free",)),
              ([rod], ("particle", idx(5, 5)), ("free",)),
              ([rod], ("particle", idx(5, 1)), ("free",))]
    sys_ = MixedSystem([particle_mass] * 25, springs, chains, gravity)
    x0 = rod.b[0]
    q0 = np.zeros(sys_.n_q)
    q0[3] = np.pi
    for k, p in enumerate(sys_.free_particles):
        i, j = p // 5 + 1, p % 5 + 1
        q0[4 + 2 * k:6 + 2 * k] = (x0 + (j - 1) * spacing, -(i - 1) * spacing)
    return sys_, q0

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
