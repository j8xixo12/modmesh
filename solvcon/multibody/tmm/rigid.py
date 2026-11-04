# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Transfer matrix method for multi-rigid-body systems (book Ch. 5).

The state variables of this formulation are accelerations, angular
accelerations, internal moments and internal forces (Sec. 5.2):

* planar state (Eq. 5.4): ``[xdd, ydd, Omega_z, m_z, q_x, q_y, 1]``
  (7 entries),
* spatial state (Eq. 5.1): ``[xdd, ydd, zdd, Omega_x, Omega_y, Omega_z,
  m_x, m_y, m_z, q_x, q_y, q_z, 1]`` (13 entries).

All vectors are in the inertial frame ``oxyz``. ``Omega`` is the angular
acceleration. Sign convention, read from Eqs. (5.11), (5.12), (5.20) and
(5.21): ``q`` at a connection point is the force that the inboard element
applies to the outboard element, and ``m`` is the moment that the outboard
element applies to the inboard element. The boundary conditions
``m = 0`` and ``q = 0`` do not depend on this choice.

At each time instant the element transfer matrices are known functions of
positions and velocities. The boundary conditions give the unknown boundary
state, the transfer equations give every state vector, and an explicit
integrator advances positions and velocities (algorithm of Sec. 5.9). This
module uses the classical fourth-order Runge-Kutta method.

Hinge elimination. The hinge matrices of Secs. 5.6-5.8 remove the unknown
angular acceleration of the outboard body with the zero-moment condition at
the outboard end of that body. This module uses the complete elimination
(Eq. 5.47, including the ``u_33 m`` term) and starts every chain at the
ground side of the base hinge, with the ground state ``r = 0, Omega = 0``.
The book states the boundary vector after the base hinge (Sec. 5.10); with
the complete elimination that form leaves the overall equation rank
deficient, because the tip moment rows become identically zero. Starting at
the ground state removes the deficiency and gives the same dynamics.
"""

import numpy as np

PLANAR_SIZE = 7
SPATIAL_SIZE = 13


def skew(v):
    """Return the skew-symmetric matrix of a 3-vector (book tilde operator).

    :param v: vector [vx, vy, vz]
    :type v: array_like
    :return: matrix S with S @ w == cross(v, w)
    :rtype: numpy.ndarray
    """
    vx, vy, vz = v
    return np.array([[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]])


def rot2(theta):
    """Return the planar rotation matrix of angle ``theta``.

    :param theta: rotation angle (rad)
    :type theta: float
    :return: 2x2 rotation matrix, body to inertial frame
    :rtype: numpy.ndarray
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def rotation_xyz(theta):
    """Return ``A = A_z A_y A_x`` for space-fixed x-y-z angles (Eq. 7.71).

    :param theta: angles [theta_x, theta_y, theta_z] (rad)
    :type theta: array_like
    :return: 3x3 matrix, body to inertial frame
    :rtype: numpy.ndarray
    """
    sx, sy, sz = np.sin(theta)
    cx, cy, cz = np.cos(theta)
    return np.array([
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy, sx * cy, cx * cy],
    ])


def angles_xyz(a):
    """Return the space-fixed x-y-z angles of a rotation matrix (Eq. 7.71).

    :param a: 3x3 rotation matrix, body to inertial frame
    :type a: numpy.ndarray
    :return: angles [theta_x, theta_y, theta_z] with theta_y in
        [-pi/2, pi/2]
    :rtype: numpy.ndarray
    """
    theta_y = -np.arcsin(np.clip(a[2, 0], -1.0, 1.0))
    theta_x = np.arctan2(a[2, 1], a[2, 2])
    theta_z = np.arctan2(a[1, 0], a[0, 0])
    return np.array([theta_x, theta_y, theta_z])


def quat_to_matrix(e):
    """Return the rotation matrix of Euler parameters (Haug convention).

    :param e: Euler parameters [e0, e1, e2, e3]
    :type e: array_like
    :return: 3x3 matrix, body to inertial frame
    :rtype: numpy.ndarray
    """
    e0 = e[0]
    ev = np.asarray(e[1:], dtype=float)
    return ((2.0 * e0 * e0 - 1.0) * np.eye(3)
            + 2.0 * (np.outer(ev, ev) + e0 * skew(ev)))


def matrix_to_quat(a):
    """Return Euler parameters [e0, e1, e2, e3] of a rotation matrix.

    :param a: 3x3 rotation matrix, body to inertial frame
    :type a: numpy.ndarray
    :return: unit quaternion with e0 >= 0
    :rtype: numpy.ndarray
    """
    tr = np.trace(a)
    candidates = np.array([tr, a[0, 0], a[1, 1], a[2, 2]])
    k = int(np.argmax(candidates))
    if k == 0:
        e0 = 0.5 * np.sqrt(1.0 + tr)
        e = [e0, (a[2, 1] - a[1, 2]) / (4 * e0),
             (a[0, 2] - a[2, 0]) / (4 * e0), (a[1, 0] - a[0, 1]) / (4 * e0)]
    elif k == 1:
        e1 = 0.5 * np.sqrt(1.0 + 2 * a[0, 0] - tr)
        e = [(a[2, 1] - a[1, 2]) / (4 * e1), e1,
             (a[0, 1] + a[1, 0]) / (4 * e1), (a[0, 2] + a[2, 0]) / (4 * e1)]
    elif k == 2:
        e2 = 0.5 * np.sqrt(1.0 + 2 * a[1, 1] - tr)
        e = [(a[0, 2] - a[2, 0]) / (4 * e2), (a[0, 1] + a[1, 0]) / (4 * e2),
             e2, (a[1, 2] + a[2, 1]) / (4 * e2)]
    else:
        e3 = 0.5 * np.sqrt(1.0 + 2 * a[2, 2] - tr)
        e = [(a[1, 0] - a[0, 1]) / (4 * e3), (a[0, 2] + a[2, 0]) / (4 * e3),
             (a[1, 2] + a[2, 1]) / (4 * e3), e3]
    e = np.array(e)
    return e if e[0] >= 0 else -e


def quat_rate(e, omega):
    """Return the time derivative of Euler parameters.

    :param e: Euler parameters [e0, e1, e2, e3]
    :type e: numpy.ndarray
    :param omega: angular velocity in the inertial frame
    :type omega: numpy.ndarray
    :return: de/dt = 0.5 * (0, omega) * e (quaternion product)
    :rtype: numpy.ndarray
    """
    e0, ev = e[0], e[1:]
    return 0.5 * np.concatenate(([-omega @ ev],
                                 e0 * omega + np.cross(omega, ev)))


# ---------------------------------------------------------------------------
# Element transfer matrices
# ---------------------------------------------------------------------------

def planar_body_matrix(mass, j_i, l_io, l_ic, theta, omega, f_c=(0.0, 0.0),
                       m_c=0.0):
    """Transfer matrix of a planar rigid body (Eqs. 5.14-5.18).

    :param mass: body mass m
    :type mass: float
    :param j_i: moment of inertia about the input end I
    :type j_i: float
    :param l_io: output end O in the body frame (relative to I)
    :type l_io: array_like
    :param l_ic: mass center C in the body frame (relative to I)
    :type l_ic: array_like
    :param theta: body angle (rad)
    :type theta: float
    :param omega: body angular velocity (rad/s)
    :type omega: float
    :param f_c: external force at C, inertial frame
    :type f_c: array_like
    :param m_c: external moment about C
    :type m_c: float
    :return: 7x7 matrix U with z_O = U z_I
    :rtype: numpy.ndarray
    """
    r = rot2(theta)
    x_io, y_io = r @ np.asarray(l_io, dtype=float)
    x_ic, y_ic = r @ np.asarray(l_ic, dtype=float)
    fx, fy = f_c
    w2 = omega * omega
    e1 = np.array([-y_io, x_io])
    e2 = np.array([-x_io * w2, -y_io * w2])
    e3 = -mass * np.eye(2)
    e4 = np.array([mass * y_ic, -mass * x_ic])
    e5 = np.array([mass * x_ic * w2 + fx, mass * y_ic * w2 + fy])
    e6 = np.array([-y_io, x_io])
    e7 = np.array([-mass * y_ic, mass * x_ic])
    e8 = j_i
    e9 = -m_c + y_ic * fx - x_ic * fy
    u = np.zeros((7, 7))
    u[0:2, 0:2] = np.eye(2)
    u[0:2, 2] = e1
    u[0:2, 6] = e2
    u[2, 2] = 1.0
    u[3, 0:2] = e6 @ e3 + e7
    u[3, 2] = e6 @ e4 + e8
    u[3, 3] = 1.0
    u[3, 4:6] = e6
    u[3, 6] = e6 @ e5 + e9
    u[4:6, 0:2] = e3
    u[4:6, 2] = e4
    u[4:6, 4:6] = np.eye(2)
    u[4:6, 6] = e5
    u[6, 6] = 1.0
    return u


def spatial_body_matrix(mass, j_i, a, l_io, l_ic, omega, f_c=(0.0, 0.0, 0.0),
                        m_c=(0.0, 0.0, 0.0)):
    """Transfer matrix of a spatial rigid body (Eqs. 5.23-5.27).

    :param mass: body mass m
    :type mass: float
    :param j_i: 3x3 inertia tensor about I in the body frame
    :type j_i: array_like
    :param a: 3x3 rotation matrix A_I, body to inertial frame
    :type a: numpy.ndarray
    :param l_io: output end O in the body frame (relative to I)
    :type l_io: array_like
    :param l_ic: mass center C in the body frame (relative to I)
    :type l_ic: array_like
    :param omega: angular velocity in the inertial frame
    :type omega: array_like
    :param f_c: external force at C, inertial frame
    :type f_c: array_like
    :param m_c: external moment, inertial frame
    :type m_c: array_like
    :return: 13x13 matrix U with z_O = U z_I
    :rtype: numpy.ndarray
    """
    a = np.asarray(a, dtype=float)
    w = np.asarray(omega, dtype=float)
    f_c = np.asarray(f_c, dtype=float)
    m_c = np.asarray(m_c, dtype=float)
    r_io = a @ np.asarray(l_io, dtype=float)
    r_ic = a @ np.asarray(l_ic, dtype=float)
    j_glob = a @ np.asarray(j_i, dtype=float) @ a.T
    wt = skew(w)
    e1 = -skew(r_io)
    e2 = wt @ wt @ r_io
    e3 = -mass * np.eye(3)
    e4 = mass * skew(r_ic)
    e5 = f_c - mass * (wt @ wt @ r_ic)
    e6 = skew(r_io)
    e7 = mass * skew(r_ic)
    e8 = j_glob
    # E9 = -m_C - r_IC x f_C + A (w_b x J w_b) = ... + w x (A J A^T) w.
    e9 = -m_c - np.cross(r_ic, f_c) + np.cross(w, j_glob @ w)
    u = np.zeros((13, 13))
    u[0:3, 0:3] = np.eye(3)
    u[0:3, 3:6] = e1
    u[0:3, 12] = e2
    u[3:6, 3:6] = np.eye(3)
    u[6:9, 0:3] = e6 @ e3 + e7
    u[6:9, 3:6] = e6 @ e4 + e8
    u[6:9, 6:9] = np.eye(3)
    u[6:9, 9:12] = e6
    u[6:9, 12] = e6 @ e5 + e9
    u[9:12, 0:3] = e3
    u[9:12, 3:6] = e4
    u[9:12, 9:12] = np.eye(3)
    u[9:12, 12] = e5
    u[12, 12] = 1.0
    return u


def _blocks(u, n):
    """Split a body matrix into the partitioned blocks of Eq. (5.34)/(5.45).

    :param u: body transfer matrix (7x7 or 13x13)
    :type u: numpy.ndarray
    :param n: block size (1 planar rotation, 3 spatial)
    :type n: int
    :return: dict with keys (row, col) in 1..5, book numbering
    :rtype: dict
    """
    if n == 1:
        sl = [slice(0, 2), slice(2, 3), slice(3, 4), slice(4, 6), slice(6, 7)]
    else:
        sl = [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12),
              slice(12, 13)]
    return {(i + 1, k + 1): u[sl[i], sl[k]] for i in range(5) for k
            in range(5)}


def planar_pin_hinge_matrix(u_out):
    """Transfer matrix of a smooth planar pin hinge (Eqs. 5.28-5.38).

    The angular acceleration of the outboard body follows from the zero
    moment at the outboard end of that body (Eq. 5.36). The ``u_33 m`` term
    of Eq. (5.47) is kept, so the matrix is exact for any input moment.

    :param u_out: 7x7 transfer matrix of the outboard body
    :type u_out: numpy.ndarray
    :return: 7x7 hinge matrix U_j with z_{j+1,j} = U_j z_{j-1,j}
    :rtype: numpy.ndarray
    """
    b = _blocks(u_out, 1)
    inv = 1.0 / b[3, 2][0, 0]
    u = np.eye(7)
    u[2, :] = 0.0
    u[2, 0:2] = -inv * b[3, 1][0]
    u[2, 3] = -inv * b[3, 3][0, 0]
    u[2, 4:6] = -inv * b[3, 4][0]
    u[2, 6] = -inv * b[3, 5][0, 0]
    return u


def spatial_hinge_matrix(u_out, n_free, s_rows, omega_in, omega_out):
    """Transfer matrix of a smooth spatial hinge, general form.

    The outboard angular acceleration is
    ``Omega_out = Omega_in + omega_in x (omega_out - omega_in) + N alpha``
    (time derivative of Eq. 5.52) where the columns of ``N`` span the free
    relative rotation directions. The free coordinates ``alpha`` follow
    from ``S m_{j+1,j+2} = 0`` (Eqs. 5.43, 5.55) at the outboard end of the
    outboard body, with ``m_{j+1,j+2}`` from Eq. (5.46). For ``N = S = I``
    this is the ball-and-socket hinge of Eq. (5.49); for one axis it is the
    pin hinge of Eqs. (5.59)-(5.61).

    :param u_out: 13x13 transfer matrix of the outboard body
    :type u_out: numpy.ndarray
    :param n_free: 3xk matrix of free rotation directions (inertial frame)
    :type n_free: numpy.ndarray
    :param s_rows: kx3 matrix that selects the zero moment components at
        the outboard end of the outboard body
    :type s_rows: numpy.ndarray
    :param omega_in: angular velocity of the inboard body
    :type omega_in: array_like
    :param omega_out: angular velocity of the outboard body
    :type omega_out: array_like
    :return: 13x13 hinge matrix U_j
    :rtype: numpy.ndarray
    """
    b = _blocks(u_out, 3)
    n_free = np.asarray(n_free, dtype=float).reshape(3, -1)
    s_rows = np.asarray(s_rows, dtype=float).reshape(-1, 3)
    w_in = np.asarray(omega_in, dtype=float)
    w_rel = np.asarray(omega_out, dtype=float) - w_in
    g = np.linalg.inv(s_rows @ b[3, 2] @ n_free)
    # Omega_out = (I - N g S u32) (Omega_in + w_in x w_rel) - N g S (u31 r +
    # u33 m + u34 q + u35)
    p = np.eye(3) - n_free @ g @ s_rows @ b[3, 2]
    k = n_free @ g @ s_rows
    u = np.eye(13)
    u[3:6, :] = 0.0
    u[3:6, 0:3] = -k @ b[3, 1]
    u[3:6, 3:6] = p
    u[3:6, 6:9] = -k @ b[3, 3]
    u[3:6, 9:12] = -k @ b[3, 4]
    u[3:6, 12] = p @ np.cross(w_in, w_rel) - (k @ b[3, 5]).ravel()
    return u


def ball_hinge_matrix(u_out):
    """Transfer matrix of a smooth ball-and-socket hinge (Eqs. 5.39-5.49).

    :param u_out: 13x13 transfer matrix of the outboard body
    :type u_out: numpy.ndarray
    :return: 13x13 hinge matrix
    :rtype: numpy.ndarray
    """
    return spatial_hinge_matrix(u_out, np.eye(3), np.eye(3), np.zeros(3),
                                np.zeros(3))


def spatial_pin_hinge_matrix(u_out, axis, axis_next, omega_in, omega_out):
    """Transfer matrix of a smooth pin hinge moving in space (Eqs. 5.50-5.61).

    The two rows ``H_1 A^T`` of ``E_0`` (Eq. 5.59) fix the angular
    acceleration normal to the axis; the row ``H_2 A^T u_32`` uses the zero
    axial moment at the next pin. The book prints ``O_33`` in ``E_3``; this
    function uses ``u_33``, which is required when the axes are not parallel.

    :param u_out: 13x13 transfer matrix of the outboard body
    :type u_out: numpy.ndarray
    :param axis: unit axis of this hinge, inertial frame
    :type axis: array_like
    :param axis_next: unit axis of the zero moment at the outboard end of
        the outboard body, inertial frame
    :type axis_next: array_like
    :param omega_in: angular velocity of the inboard body
    :type omega_in: array_like
    :param omega_out: angular velocity of the outboard body
    :type omega_out: array_like
    :return: 13x13 hinge matrix
    :rtype: numpy.ndarray
    """
    n = np.asarray(axis, dtype=float).reshape(3, 1)
    s = np.asarray(axis_next, dtype=float).reshape(1, 3)
    return spatial_hinge_matrix(u_out, n, s, omega_in, omega_out)


def acceleration_hinge_matrix(u_out):
    """Transfer matrix of an acceleration hinge (Eqs. 5.62-5.74).

    The hinge transmits no internal force or moment; elastic and damping
    forces of the hinge act as external forces on the connected bodies.
    The acceleration and angular acceleration of the outboard body follow
    from ``q = 0`` and ``m = 0`` at its outboard end (Eqs. 5.66, 5.67). The
    two block rows of Eq. (5.74) are solved together, so the outboard
    accelerations do not depend on the inboard accelerations.

    :param u_out: transfer matrix of the outboard body (7x7 or 13x13)
    :type u_out: numpy.ndarray
    :return: hinge matrix of the same size
    :rtype: numpy.ndarray
    """
    size = u_out.shape[0]
    n = 1 if size == PLANAR_SIZE else 3
    nr = 2 if n == 1 else 3
    b = _blocks(u_out, n)
    # [q_out; m_out] = K [r; Omega] + L [m; q] + c = 0
    kmat = np.block([[b[4, 1], b[4, 2]], [b[3, 1], b[3, 2]]])
    lmat = np.block([[b[4, 3], b[4, 4]], [b[3, 3], b[3, 4]]])
    cvec = np.concatenate([b[4, 5].ravel(), b[3, 5].ravel()])
    kinv = np.linalg.inv(kmat)
    nk = nr + n
    u = np.eye(size)
    u[0:nk, :] = 0.0
    u[0:nk, nk:size - 1] = -kinv @ lmat
    u[0:nk, size - 1] = -kinv @ cvec
    return u


# ---------------------------------------------------------------------------
# Chain systems
# ---------------------------------------------------------------------------

class PlanarBody:
    """Planar rigid body with one input end and one output end (Fig. 5.1).

    :ivar mass: mass
    :vartype mass: float
    :ivar j_i: moment of inertia about the input end I
    :vartype j_i: float
    :ivar l_io: output end in the body frame (relative to I)
    :vartype l_io: numpy.ndarray
    :ivar l_ic: mass center in the body frame (relative to I)
    :vartype l_ic: numpy.ndarray
    """

    def __init__(self, mass, j_i, l_io, l_ic):
        """Initialize the body.

        :param mass: mass
        :type mass: float
        :param j_i: moment of inertia about I
        :type j_i: float
        :param l_io: output end, body frame
        :type l_io: array_like
        :param l_ic: mass center, body frame
        :type l_ic: array_like
        """
        self.mass = float(mass)
        self.j_i = float(j_i)
        self.l_io = np.asarray(l_io, dtype=float)
        self.l_ic = np.asarray(l_ic, dtype=float)


def uniform_bar(mass, length, j_c=None):
    """Return a planar bar along the body x axis with C at mid length.

    :param mass: bar mass
    :type mass: float
    :param length: bar length
    :type length: float
    :param j_c: moment of inertia about C (default m l^2 / 12)
    :type j_c: float or None
    :return: planar body with J_I = J_C + m (l/2)^2
    :rtype: PlanarBody
    """
    if j_c is None:
        j_c = mass * length ** 2 / 12.0
    return PlanarBody(mass, j_c + mass * (0.5 * length) ** 2, [length, 0.0],
                      [0.5 * length, 0.0])


class PlanarChain:
    """Planar chain of rigid bodies joined by smooth pin hinges (Ch. 5).

    Hinge 0 joins the ground point ``base`` to the input end of body 0.
    Hinge k joins the output end of body k-1 to the input end of body k.
    The output end of the last body is free.

    :ivar bodies: bodies from the base to the tip
    :vartype bodies: list
    :ivar gravity: gravity acceleration vector
    :vartype gravity: numpy.ndarray
    :ivar base: ground point of hinge 0
    :vartype base: numpy.ndarray
    """

    def __init__(self, bodies, gravity=(0.0, -9.81), base=(0.0, 0.0)):
        """Initialize the chain.

        :param bodies: list of PlanarBody
        :type bodies: list
        :param gravity: gravity acceleration (inertial frame)
        :type gravity: array_like
        :param base: ground point of the base hinge
        :type base: array_like
        """
        self.bodies = list(bodies)
        self.gravity = np.asarray(gravity, dtype=float)
        self.base = np.asarray(base, dtype=float)

    @property
    def n(self):
        """Return the number of bodies.

        :rtype: int
        """
        return len(self.bodies)

    def transfer(self, theta, omega, m_ext=None):
        """Solve the overall transfer equation at one time instant.

        Overall transfer equation (Eqs. 5.8, 5.9) from the ground state
        ``[0, 0, 0, 0, q_x, q_y, 1]`` to the free tip ``m = q = 0``.

        :param theta: body angles
        :type theta: array_like
        :param omega: body angular velocities
        :type omega: array_like
        :param m_ext: external moments on the bodies (default zero)
        :type m_ext: array_like or None
        :return: (states, alpha) where states[k] = (z_in, z_out) of body k
            and alpha holds the angular accelerations
        :rtype: tuple
        """
        if m_ext is None:
            m_ext = np.zeros(self.n)
        mats = [planar_body_matrix(b.mass, b.j_i, b.l_io, b.l_ic, th, w,
                                   b.mass * self.gravity, mc)
                for b, th, w, mc in zip(self.bodies, theta, omega, m_ext)]
        chain = []
        total = np.eye(7)
        for u in mats:
            h = planar_pin_hinge_matrix(u)
            chain.append((total.copy(), h, u))
            total = u @ h @ total
        # Unknowns q_x, q_y of the ground state (entries 4, 5).
        rows = [3, 4, 5]
        a = total[np.ix_(rows, [4, 5])]
        rhs = -total[rows, 6]
        q0, *_ = np.linalg.lstsq(a, rhs, rcond=None)
        z = np.zeros(7)
        z[4:6] = q0
        z[6] = 1.0
        states = []
        alpha = np.zeros(self.n)
        for k, (_, h, u) in enumerate(chain):
            z_in = h @ z
            z_out = u @ z_in
            states.append((z_in, z_out))
            alpha[k] = z_in[2]
            z = z_out
        return states, alpha

    def positions(self, theta):
        """Return the input-end and mass-center positions of every body.

        :param theta: body angles
        :type theta: array_like
        :return: (r_in, r_c, r_out) arrays of shape (n, 2)
        :rtype: tuple
        """
        r_in, r_c, r_out = [], [], []
        p = self.base.copy()
        for b, th in zip(self.bodies, theta):
            r = rot2(th)
            r_in.append(p.copy())
            r_c.append(p + r @ b.l_ic)
            p = p + r @ b.l_io
            r_out.append(p.copy())
        return np.array(r_in), np.array(r_c), np.array(r_out)

    def simulate(self, theta0, omega0, t_end, dt):
        """Integrate the motion with the RK4 method (Sec. 5.9, step 6).

        :param theta0: initial body angles
        :type theta0: array_like
        :param omega0: initial angular velocities
        :type omega0: array_like
        :param t_end: final time
        :type t_end: float
        :param dt: time step
        :type dt: float
        :return: (t, theta, omega) arrays, theta and omega of shape (N, n)
        :rtype: tuple
        """
        n = self.n

        def rhs(y):
            _, alpha = self.transfer(y[:n], y[n:])
            return np.concatenate([y[n:], alpha])

        y = np.concatenate([np.asarray(theta0, float),
                            np.asarray(omega0, float)])
        steps = int(round(t_end / dt))
        out = np.empty((steps + 1, 2 * n))
        out[0] = y
        for i in range(steps):
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * dt * k1)
            k3 = rhs(y + 0.5 * dt * k2)
            k4 = rhs(y + dt * k3)
            y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            out[i + 1] = y
        t = dt * np.arange(steps + 1)
        return t, out[:, :n], out[:, n:]


class SpatialBody:
    """Spatial rigid body with one input and one output end (Fig. 5.2).

    :ivar mass: mass
    :vartype mass: float
    :ivar j_i: 3x3 inertia tensor about I, body frame
    :vartype j_i: numpy.ndarray
    :ivar l_io: output end, body frame
    :vartype l_io: numpy.ndarray
    :ivar l_ic: mass center, body frame
    :vartype l_ic: numpy.ndarray
    """

    def __init__(self, mass, j_i, l_io, l_ic):
        """Initialize the body.

        :param mass: mass
        :type mass: float
        :param j_i: inertia about I, body frame (3x3 or 3 diagonal values)
        :type j_i: array_like
        :param l_io: output end, body frame
        :type l_io: array_like
        :param l_ic: mass center, body frame
        :type l_ic: array_like
        """
        self.mass = float(mass)
        j_i = np.asarray(j_i, dtype=float)
        self.j_i = np.diag(j_i) if j_i.ndim == 1 else j_i
        self.l_io = np.asarray(l_io, dtype=float)
        self.l_ic = np.asarray(l_ic, dtype=float)

    def inertia_about_c(self):
        """Return the inertia tensor about the mass center, body frame.

        :rtype: numpy.ndarray
        """
        c = self.l_ic
        return (self.j_i
                - self.mass * (np.dot(c, c) * np.eye(3) - np.outer(c, c)))


class BallHinge:
    """Smooth ball-and-socket hinge (Sec. 5.7.1)."""

    kind = "ball"


class PinHinge:
    """Smooth pin hinge moving in space (Sec. 5.7.2).

    :ivar axis: unit axis in the inboard body frame (inertial frame for the
        base hinge)
    :vartype axis: numpy.ndarray
    """

    kind = "pin"

    def __init__(self, axis):
        """Initialize the hinge.

        :param axis: axis in the inboard body frame
        :type axis: array_like
        """
        a = np.asarray(axis, dtype=float)
        self.axis = a / np.linalg.norm(a)


class BushingHinge:
    """Spring-damper hinge treated as an acceleration hinge (Sec. 5.8).

    The translational spring acts on the gap between the two hinge points.
    The rotary spring acts on the misalignment ``n_in x n_out`` of the
    hinge axis seen from both bodies, so free rotation about the axis
    remains (Sec. 5.10.3).

    :ivar axis: axis in the inboard body frame (inertial for the base)
    :vartype axis: numpy.ndarray
    :ivar k_s: translational stiffness
    :vartype k_s: float
    :ivar c_s: translational damping
    :vartype c_s: float
    :ivar k_r: rotary stiffness
    :vartype k_r: float
    :ivar c_r: rotary damping
    :vartype c_r: float
    """

    kind = "bushing"

    def __init__(self, axis, k_s, k_r, c_s=0.0, c_r=0.0):
        """Initialize the hinge.

        :param axis: axis in the inboard body frame
        :type axis: array_like
        :param k_s: translational stiffness
        :type k_s: float
        :param k_r: rotary stiffness
        :type k_r: float
        :param c_s: translational damping
        :type c_s: float
        :param c_r: rotary damping
        :type c_r: float
        """
        a = np.asarray(axis, dtype=float)
        self.axis = a / np.linalg.norm(a)
        self.k_s, self.k_r, self.c_s, self.c_r = k_s, k_r, c_s, c_r


class SpatialChain:
    """Spatial chain of rigid bodies with smooth or elastic hinges (Ch. 5).

    Hinge 0 joins the ground point ``base`` to the input end of body 0.
    Hinge k joins the output end of body k-1 to the input end of body k.
    The output end of the last body is free. The state of body k is its
    Euler parameters and its angular velocity in the inertial frame.
    For smooth hinges the input end of body k coincides with the output
    end of body k-1. For bushing hinges every body also carries the
    position and velocity of its input end.

    :ivar bodies: bodies from the base to the tip
    :vartype bodies: list
    :ivar hinges: hinges, one per body
    :vartype hinges: list
    :ivar gravity: gravity acceleration vector
    :vartype gravity: numpy.ndarray
    :ivar base: ground point of hinge 0
    :vartype base: numpy.ndarray
    :ivar pin_mode: 'auto', 'eliminate' or 'augment'. A pin hinge is
        eliminated with the next zero axial moment (Eq. 5.61) when that
        moment depends on the hinge rate ('auto'); for orthogonal axes the
        matrix E_0 of Eq. (5.59) is singular and the relative angular
        acceleration becomes an unknown of the overall equation with the
        hinge condition ``n . m = 0`` (Eq. 5.54).
    :vartype pin_mode: str
    """

    def __init__(self, bodies, hinges, gravity=(0.0, -9.81, 0.0),
                 base=(0.0, 0.0, 0.0)):
        """Initialize the chain.

        :param bodies: list of SpatialBody
        :type bodies: list
        :param hinges: list of BallHinge, PinHinge or BushingHinge
        :type hinges: list
        :param gravity: gravity acceleration (inertial frame)
        :type gravity: array_like
        :param base: ground point of the base hinge
        :type base: array_like
        :raises ValueError: if the hinge list is not valid
        """
        if len(bodies) != len(hinges):
            raise ValueError("one hinge per body is required")
        kinds = {h.kind for h in hinges}
        if "bushing" in kinds and kinds != {"bushing"}:
            raise ValueError("bushing hinges cannot be mixed with smooth "
                             "hinges")
        for k in range(len(hinges) - 1):
            if hinges[k].kind == "ball" and hinges[k + 1].kind == "pin":
                raise ValueError("a ball hinge needs three zero moments at "
                                 "the outboard end of its outboard body (Eq. "
                                 "5.43)")
        self.bodies = list(bodies)
        self.hinges = list(hinges)
        self.gravity = np.asarray(gravity, dtype=float)
        self.base = np.asarray(base, dtype=float)
        self.elastic = kinds == {"bushing"}
        self.pin_mode = "auto"

    @property
    def n(self):
        """Return the number of bodies.

        :rtype: int
        """
        return len(self.bodies)

    def _axis_global(self, k, mats):
        """Return the axis of hinge k in the inertial frame.

        :param k: hinge index
        :type k: int
        :param mats: rotation matrices of the bodies
        :type mats: list
        :rtype: numpy.ndarray
        """
        ax = self.hinges[k].axis
        return ax if k == 0 else mats[k - 1] @ ax

    def positions(self, quats, r_in=None):
        """Return the input-end, mass-center and output-end positions.

        :param quats: Euler parameters of the bodies, shape (n, 4)
        :type quats: array_like
        :param r_in: input-end positions for an elastic chain
        :type r_in: array_like or None
        :return: (r_in, r_c, r_out) arrays of shape (n, 3)
        :rtype: tuple
        """
        ri, rc, ro = [], [], []
        p = self.base.copy()
        for k, (b, e) in enumerate(zip(self.bodies, quats)):
            a = quat_to_matrix(e)
            if r_in is not None:
                p = np.asarray(r_in[k], dtype=float)
            ri.append(p.copy())
            rc.append(p + a @ b.l_ic)
            p = p + a @ b.l_io
            ro.append(p.copy())
        return np.array(ri), np.array(rc), np.array(ro)

    def _velocities(self, mats, omegas):
        """Return the input-end velocities of a smooth chain.

        :param mats: rotation matrices
        :type mats: list
        :param omegas: angular velocities, inertial frame
        :type omegas: list
        :rtype: list
        """
        v = np.zeros(3)
        out = []
        for b, a, w in zip(self.bodies, mats, omegas):
            out.append(v.copy())
            v = v + np.cross(w, a @ b.l_io)
        return out

    def _bushing_loads(self, mats, omegas, r_in, v_in):
        """Return external forces and moments of the bushing hinges.

        :param mats: rotation matrices
        :type mats: list
        :param omegas: angular velocities, inertial frame
        :type omegas: list
        :param r_in: input-end positions
        :type r_in: numpy.ndarray
        :param v_in: input-end velocities
        :type v_in: numpy.ndarray
        :return: (forces at C, moments) per body
        :rtype: tuple
        """
        n = self.n
        f_c = [b.mass * self.gravity for b in self.bodies]
        m_c = [np.zeros(3) for _ in range(n)]
        for k, h in enumerate(self.hinges):
            b = self.bodies[k]
            a = mats[k]
            if k == 0:
                p_in, v_pin, w_in = self.base, np.zeros(3), np.zeros(3)
                n_in = h.axis
            else:
                bp = self.bodies[k - 1]
                p_in = r_in[k - 1] + mats[k - 1] @ bp.l_io
                v_pin = (v_in[k - 1]
                         + np.cross(omegas[k - 1], mats[k - 1] @ bp.l_io))
                w_in = omegas[k - 1]
                n_in = mats[k - 1] @ h.axis
            # The axis is also fixed in the outboard body; its body-frame copy
            # is set at the start of the simulation.
            n_out = a @ self._axis_out[k]
            gap = r_in[k] - p_in
            gap_rate = v_in[k] - v_pin
            # on the outboard body at I
            force = -h.k_s * gap - h.c_s * gap_rate
            mis = np.cross(n_in, n_out)
            w_rel = omegas[k] - w_in
            w_rel_perp = w_rel - np.dot(w_rel, n_in) * n_in
            torque = -h.k_r * mis - h.c_r * w_rel_perp  # on the outboard body
            # Outboard body: force at I, moment about C.
            r_ci = -(a @ b.l_ic)
            f_c[k] = f_c[k] + force
            m_c[k] = m_c[k] + torque + np.cross(r_ci, force)
            if k > 0:
                bp = self.bodies[k - 1]
                ap = mats[k - 1]
                r_co = ap @ (bp.l_io - bp.l_ic)
                f_c[k - 1] = f_c[k - 1] - force
                m_c[k - 1] = m_c[k - 1] - torque - np.cross(r_co, force)
        return f_c, m_c

    def transfer(self, quats, omegas, r_in=None, v_in=None):
        """Solve the overall transfer equation at one time instant.

        :param quats: Euler parameters, shape (n, 4)
        :type quats: array_like
        :param omegas: angular velocities (inertial frame), shape (n, 3)
        :type omegas: array_like
        :param r_in: input-end positions (elastic chain only)
        :type r_in: array_like or None
        :param v_in: input-end velocities (elastic chain only)
        :type v_in: array_like or None
        :return: (states, alpha, accel) with states[k] = (z_in, z_out),
            angular accelerations alpha (n, 3) and input-end accelerations
            accel (n, 3)
        :rtype: tuple
        """
        n = self.n
        mats = [quat_to_matrix(e) for e in quats]
        omegas = [np.asarray(w, dtype=float) for w in omegas]
        if self.elastic:
            f_c, m_c = self._bushing_loads(mats, omegas, np.asarray(r_in),
                                           np.asarray(v_in))
        else:
            f_c = [b.mass * self.gravity for b in self.bodies]
            m_c = [np.zeros(3) for _ in range(n)]
        umats = [spatial_body_matrix(b.mass, b.j_i, a, b.l_io, b.l_ic, w, f, m)
                 for b, a, w, f, m in zip(self.bodies, mats, omegas, f_c, m_c)]
        # Ground state: r = 0, Omega = 0, m = M mu, q unknown. The state is
        # carried as Z with z = Z [u; 1] for the unknown vector u.
        cols = []
        h0 = self.hinges[0]
        if h0.kind == "pin":
            ax = h0.axis
            e1 = np.cross(ax, [1.0, 0.0, 0.0])
            if np.linalg.norm(e1) < 0.5:
                e1 = np.cross(ax, [0.0, 1.0, 0.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(ax, e1)
            for e in (e1, e2):
                c = np.zeros(13)
                c[6:9] = e
                cols.append(c)
        if h0.kind != "bushing":
            for i in range(3):
                c = np.zeros(13)
                c[9 + i] = 1.0
                cols.append(c)
        z0 = np.zeros(13)
        z0[12] = 1.0
        zmat = np.column_stack(cols + [z0])
        eqs = []
        stages = []
        covered = True  # own zero-moment condition of the hinge already holds
        for k in range(n):
            h = self.hinges[k]
            u = umats[k]
            w_in = np.zeros(3) if k == 0 else omegas[k - 1]
            if h.kind in ("pin", "ball") and not covered:
                # The previous hinge did not use this hinge's zero moments:
                # add them as equations (Eqs. 5.42, 5.54).
                rows_m = (self._axis_global(k, mats).reshape(1, 3) if h.kind
                          == "pin"
                          else np.eye(3))
                for r_ in rows_m:
                    eqs.append(np.concatenate([r_ @ zmat[6:9, :-1],
                                               [r_ @ zmat[6:9, -1]]]))
            if h.kind == "bushing":
                z_in = acceleration_hinge_matrix(u) @ zmat
                covered = True
            elif h.kind == "ball":
                z_in = ball_hinge_matrix(u) @ zmat
                covered = True
            else:
                axis = self._axis_global(k, mats)
                next_pin = k + 1 < n and self.hinges[k + 1].kind == "pin"
                axis_next = (self._axis_global(k + 1, mats) if next_pin
                             else axis)
                u32 = u[6:9, 3:6]
                gain = abs(axis_next @ u32 @ axis)
                if self.pin_mode == "eliminate" or (
                        self.pin_mode == "auto"
                        and gain > 1e-3 * np.linalg.norm(u32)):
                    z_in = spatial_pin_hinge_matrix(u, axis, axis_next, w_in,
                                                    omegas[k]) @ zmat
                    covered = next_pin or k + 1 == n
                else:
                    # Augmented form: the relative angular acceleration is a
                    # new unknown of the overall equation.
                    zmat = np.column_stack([zmat[:, :-1], np.zeros(13),
                                            zmat[:, -1]])
                    z_in = zmat.copy()
                    z_in[3:6, -2] = axis
                    z_in[3:6, -1] += np.cross(w_in, omegas[k] - w_in)
                    covered = False
            zmat = u @ z_in
            stages.append((z_in, zmat))
        nu = zmat.shape[1] - 1
        rows = [np.concatenate([e[:-1], np.zeros(nu + 1 - e.size), e[-1:]])
                for e in eqs]
        rows.extend(zmat[6:12])
        if nu:
            a_mat = np.array(rows)
            x, *_ = np.linalg.lstsq(a_mat[:, :-1], -a_mat[:, -1], rcond=None)
        else:
            x = np.zeros(0)
        states = []
        alpha = np.zeros((n, 3))
        accel = np.zeros((n, 3))
        for k, (z_in, z_out) in enumerate(stages):
            zi = z_in[:, :-1] @ x[:z_in.shape[1] - 1] + z_in[:, -1]
            zo = z_out[:, :-1] @ x[:z_out.shape[1] - 1] + z_out[:, -1]
            states.append((zi, zo))
            accel[k] = zi[0:3]
            alpha[k] = zi[3:6]
        return states, alpha, accel

    def simulate(self, quats0, omegas0, t_end, dt, r_in0=None, v_in0=None):
        """Integrate the motion with the RK4 method (Sec. 5.9, step 6).

        :param quats0: initial Euler parameters, shape (n, 4)
        :type quats0: array_like
        :param omegas0: initial angular velocities, inertial frame
        :type omegas0: array_like
        :param t_end: final time
        :type t_end: float
        :param dt: time step
        :type dt: float
        :param r_in0: initial input-end positions (elastic chain)
        :type r_in0: array_like or None
        :param v_in0: initial input-end velocities (elastic chain)
        :type v_in0: array_like or None
        :return: dict with keys t, quat (N, n, 4), omega (N, n, 3) and, for
            an elastic chain, r_in and v_in (N, n, 3)
        :rtype: dict
        """
        n = self.n
        q0 = np.asarray(quats0, dtype=float).reshape(n, 4)
        w0 = np.asarray(omegas0, dtype=float).reshape(n, 3)
        if self.elastic:
            mats0 = [quat_to_matrix(e) for e in q0]
            self._axis_out = []
            for k, h in enumerate(self.hinges):
                n_in = h.axis if k == 0 else mats0[k - 1] @ h.axis
                self._axis_out.append(mats0[k].T @ n_in)
            if r_in0 is None:
                r_in0 = self.positions(q0)[0]
            if v_in0 is None:
                v_in0 = np.array(self._velocities(mats0, list(w0)))
            parts = [q0.ravel(), w0.ravel(), np.asarray(r_in0, float).ravel(),
                     np.asarray(v_in0, float).ravel()]
        else:
            parts = [q0.ravel(), w0.ravel()]
        y = np.concatenate(parts)

        def unpack(y):
            q = y[:4 * n].reshape(n, 4)
            w = y[4 * n:7 * n].reshape(n, 3)
            if self.elastic:
                return (q, w, y[7 * n:10 * n].reshape(n, 3),
                        y[10 * n:13 * n].reshape(n, 3))
            return q, w, None, None

        def rhs(y):
            q, w, r, v = unpack(y)
            _, alpha, accel = self.transfer(q, w, r, v)
            dq = np.concatenate([quat_rate(q[k], w[k]) for k in range(n)])
            if self.elastic:
                return np.concatenate([dq, alpha.ravel(), v.ravel(),
                                       accel.ravel()])
            return np.concatenate([dq, alpha.ravel()])

        steps = int(round(t_end / dt))
        hist = np.empty((steps + 1, y.size))
        hist[0] = y
        for i in range(steps):
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * dt * k1)
            k3 = rhs(y + 0.5 * dt * k2)
            k4 = rhs(y + dt * k3)
            y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            q = y[:4 * n].reshape(n, 4)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            hist[i + 1] = y
        out = {"t": dt * np.arange(steps + 1),
               "quat": hist[:, :4 * n].reshape(-1, n, 4),
               "omega": hist[:, 4 * n:7 * n].reshape(-1, n, 3)}
        if self.elastic:
            out["r_in"] = hist[:, 7 * n:10 * n].reshape(-1, n, 3)
            out["v_in"] = hist[:, 10 * n:13 * n].reshape(-1, n, 3)
        return out


def pin_relative_angles(chain, quats, quats_ref):
    """Return the relative rotation angles of the pin hinges of a chain.

    The angle of hinge k is the rotation of body k relative to body k-1
    (ground for k = 0) about the hinge axis, measured from the reference
    configuration ``quats_ref``.

    :param chain: spatial chain with pin or bushing hinges
    :type chain: SpatialChain
    :param quats: Euler parameters, shape (n, 4)
    :type quats: array_like
    :param quats_ref: reference Euler parameters, shape (n, 4)
    :type quats_ref: array_like
    :return: relative angles (rad), shape (n,)
    :rtype: numpy.ndarray
    """
    out = np.zeros(chain.n)
    for k in range(chain.n):
        a = quat_to_matrix(quats[k])
        a0 = quat_to_matrix(quats_ref[k])
        if k == 0:
            ap, ap0 = np.eye(3), np.eye(3)
        else:
            ap = quat_to_matrix(quats[k - 1])
            ap0 = quat_to_matrix(quats_ref[k - 1])
        # Relative rotation of body k w.r.t. body k-1 since the reference,
        # expressed in the inboard body frame.
        rel = (ap.T @ a) @ (ap0.T @ a0).T
        ax = chain.hinges[k].axis
        e = matrix_to_quat(rel)
        out[k] = 2.0 * np.arctan2(np.dot(e[1:], ax), e[0])
    return out

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
