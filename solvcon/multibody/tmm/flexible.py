# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Multi-flexible-body transfer matrix method (book Ch. 6) and linear library
additions.

Contents:

* Linear library additions (Ch. 15, used by tree systems of Ch. 11):

  - ``RigidBodyMultiEnd``: spatial rigid body with N input ends and L output
    ends (Eq. 15.16-15.19);
  - ``StarTree``: tree system with one rigid body, one root chain and L
    branch chains; the overall equation is formed with any of the three
    rigid body forms (Eq. 15.17, 15.18 or 15.19);
  - ``axial_load_beam_matrix``: closed form of the massless beam with axial
    load and shear (Eq. 15.24, corrected, see below);
  - ``TaperedAxialBeam`` and ``TaperedAxialBeamODE``: nonuniform massless beam
    with an axial compressive force, closed form (Eq. 15.29, corrected) and
    the numerical solution of the same first-order equations;
  - ``riccati_mode``: mode shape by the Riccati back substitution
    (Eq. 10.130, 10.135).

* Multi-flexible-body TMM (Sec. 6.2-6.6). The state vectors contain the
  absolute accelerations, angular accelerations, internal moments and
  internal forces (Eq. 6.1-6.6); every element transfer matrix is a known
  function of the positions and velocities at the time t_i (Eq. 6.7-6.10):

  - ``PlanarRigidElement``: planar rigid body (Ch. 5 state, Eq. 6.4);
  - ``PlanarFlexibleBeam``: planar beam in the floating frame of reference
    with modal coordinates (Sec. 6.4, Eq. 6.11-6.60);
  - ``SpatialRigidElement`` and ``SpatialFlexibleBeam``: spatial versions
    (Sec. 6.5, Eq. 6.61-6.113);
  - ``FlexibleChain``: chain with a pin to the ground at the input end and a
    free output end, overall transfer equation (Eq. 6.9-6.10, 6.114-6.121)
    and the fourth-order Runge-Kutta integration (Sec. 6.6).

Sign convention of the Ch. 6 state vectors (Sec. 6.4, after Fig. 6.1): q at
a connection point is the force that the inboard element applies to the
outboard element, m is the moment that the outboard element applies to the
inboard element. On an element, the input end receives the force +q_I and
the moment -m_I; the output end receives -q_O and +m_O.

Book corrections found by the tests (``tests/test_tmm_flexible.py``):

* Eq. (15.17): the block U_I(3, 1) is printed -m omega^2 l_I1C^T. Consistency
  with Eq. (15.14), (15.18) and (15.19) needs -m omega^2 l_I1C.
* Eq. (15.24): for a compressive force the printed lambda = sqrt(P/EI),
  d1 = -lambda^2 do not satisfy the beam equations. The tensile form
  lambda = sqrt(-P/EI), d1 = lambda^2 holds for both signs (lambda is
  imaginary for compression), with the entry u_14 =
  -(lambda x - sinh(lambda x)) / (d1 lambda EI) - x/GA_s.
* Eq. (15.29): u_12 and u_13 have the wrong sign, u_22 and u_23 are wrong,
  and the printed form divides by sin(beta ln c1) and tan(beta ln c1), which
  are zero for c1 = 1 (the case of Fig. 15.6b). ``TaperedAxialBeam`` uses the
  closed form derived from the same equations.
* Eq. (6.40), (6.94): the first block columns of a_2,3 and a_2,4 are zero,
  so the boundary moments do no work on the bending modes. The generalized
  forces -phi'(0) m_I and phi'(l) m_O are added here. They vanish for the
  book examples (phi'(0) = 0, m_O = 0).
"""

import numpy as np

from . import library as _lib
from . import riccati as _ric
from .linear import Mode, _normalize, _pivoted_null_vector

# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
_I3 = np.eye(3)


def _cross(v):
    """Cross-product matrix of a 3-vector (Eq. 10.13)."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def _inertia_about(j_c, m, r):
    """Inertia matrix about a point P; r is the vector from P to the mass
    center."""
    r = np.asarray(r, dtype=float)
    return np.asarray(j_c, dtype=float) + m * (r @ r * _I3 - np.outer(r, r))


def point_displacement_matrix(p_from, p_to):
    """Map [X, Theta] of a rigid body point to [X, Theta] of another point (Eq.
    15.14, row 1).

    X_to = X_from - l~ Theta with l = p_to - p_from.

    :param p_from: Coordinates of the known point.
    :type p_from: array_like
    :param p_to: Coordinates of the other point.
    :type p_to: array_like
    :return: 6 x 6 matrix.
    :rtype: numpy.ndarray
    """
    t = np.eye(6)
    t[0:3,
      3:6] = (-_cross(np.asarray(p_to, dtype=float) - np.asarray(p_from,
                                                                 dtype=float)))
    return t


# ----------------------------------------------------------------------
# Spatial rigid body with several input and output ends (Eq. 15.16-15.19)
# ----------------------------------------------------------------------
class RigidBodyMultiEnd:
    """Spatial rigid body with N input ends and L output ends (Sec.
    15.6.3-15.6.5).

    State vectors (Eq. 15.16): Z_I = [X, Y, Z, Theta_x, Theta_y, Theta_z of
    I_1, (M, Q) of I_1, ..., (M, Q) of I_N] and Z_O likewise for the outputs.
    The displacements of the other ends follow from rigid motion
    (``point_displacement_matrix``). All coordinates are in one body frame
    with any origin; every form shifts them to its reference point (I_1 for
    Eq. 15.17-15.18, I for Eq. 15.19). omega^2 is written -s^2.

    :param m: Mass.
    :type m: float
    :param j_c: Inertia matrix about the mass center (3 x 3).
    :type j_c: array_like
    :param center: Mass center coordinates.
    :type center: array_like
    :param inputs: Coordinates of the N input ends.
    :type inputs: list
    :param outputs: Coordinates of the L output ends.
    :type outputs: list
    """

    def __init__(self, m, j_c, center, inputs, outputs):
        self.m = float(m)
        self.j_c = np.asarray(j_c, dtype=float)
        self.center = np.asarray(center, dtype=float)
        self.inputs = [np.asarray(p, dtype=float) for p in inputs]
        self.outputs = [np.asarray(p, dtype=float) for p in outputs]

    def _inertia_rows(self, s, ref):
        """Inertia blocks of the moment and force rows about the point
        ``ref``."""
        w2 = -s * s
        l_c = _cross(self.center - ref)
        j_ref = _inertia_about(self.j_c, self.m, self.center - ref)
        return w2, l_c, j_ref

    def general_matrices(self, s):
        """Matrices U_O and U_I of U_O Z_O = U_I Z_I (Eq. 15.16-15.17).

        U_I(3, 1) is -m omega^2 l_I1C (printed with the opposite sign, see
        the module docstring).

        :param s: Laplace variable (s = i omega).
        :type s: complex
        :return: (U_O, U_I), shapes (12, 6 + 6 L) and (12, 6 + 6 N).
        :rtype: tuple
        """
        ref = self.inputs[0]
        w2, l_c, j_ref = self._inertia_rows(s, ref)
        n_out, n_in = len(self.outputs), len(self.inputs)
        u_o = np.zeros((12, 6 + 6 * n_out), dtype=complex)
        u_i = np.zeros((12, 6 + 6 * n_in), dtype=complex)
        u_o[0:3, 0:3] = _I3
        u_o[3:6, 3:6] = _I3
        for k, p in enumerate(self.outputs):
            c0 = 6 + 6 * k
            u_o[6:9, c0:c0 + 3] = _I3
            u_o[6:9, c0 + 3:c0 + 6] = -_cross(p - ref)
            u_o[9:12, c0 + 3:c0 + 6] = _I3
        u_i[0:3, 0:3] = _I3
        u_i[0:3, 3:6] = -_cross(self.outputs[0] - ref)
        u_i[3:6, 3:6] = _I3
        u_i[6:9, 0:3] = -self.m * w2 * l_c
        u_i[6:9, 3:6] = -w2 * j_ref
        u_i[9:12, 0:3] = self.m * w2 * _I3
        u_i[9:12, 3:6] = -self.m * w2 * l_c
        for k, p in enumerate(self.inputs):
            c0 = 6 + 6 * k
            u_i[6:9, c0:c0 + 3] = _I3
            u_i[6:9, c0 + 3:c0 + 6] = -_cross(p - ref)
            u_i[9:12, c0 + 3:c0 + 6] = _I3
        return u_o, u_i

    def inputs_matrix(self, s):
        """Transfer matrix Z_O = U Z_I for N inputs and one output (Eq. 15.18).

        :param s: Laplace variable.
        :type s: complex
        :return: U of shape (12, 6 + 6 N).
        :rtype: numpy.ndarray
        :raises ValueError: If the body has more than one output.
        """
        if len(self.outputs) != 1:
            raise ValueError("Eq. 15.18 needs exactly one output end")
        ref = self.inputs[0]
        w2, l_c, j_ref = self._inertia_rows(s, ref)
        b = self.outputs[0] - ref
        l_io = _cross(b)
        l_co = l_io - l_c
        u = np.zeros((12, 6 + 6 * len(self.inputs)), dtype=complex)
        u[0:3, 0:3] = _I3
        u[0:3, 3:6] = -l_io
        u[3:6, 3:6] = _I3
        u[6:9, 0:3] = self.m * w2 * l_co
        u[6:9, 3:6] = -w2 * (self.m * l_io @ l_c + j_ref)
        u[9:12, 0:3] = self.m * w2 * _I3
        u[9:12, 3:6] = -self.m * w2 * l_c
        for k, p in enumerate(self.inputs):
            c0 = 6 + 6 * k
            u[6:9, c0:c0 + 3] = _I3
            u[6:9, c0 + 3:c0 + 6] = _cross(self.outputs[0] - p)
            u[9:12, c0 + 3:c0 + 6] = _I3
        return u

    def outputs_matrix(self, s):
        """Transfer matrix Z_I = U Z_O for one input and L outputs (Eq. 15.19).

        :param s: Laplace variable.
        :type s: complex
        :return: U of shape (12, 6 + 6 L).
        :rtype: numpy.ndarray
        :raises ValueError: If the body has more than one input.
        """
        if len(self.inputs) != 1:
            raise ValueError("Eq. 15.19 needs exactly one input end")
        ref = self.inputs[0]
        w2, l_c, j_ref = self._inertia_rows(s, ref)
        l_io1 = _cross(self.outputs[0] - ref)
        l_o1c = _cross(self.center - self.outputs[0])
        u = np.zeros((12, 6 + 6 * len(self.outputs)), dtype=complex)
        u[0:3, 0:3] = _I3
        u[0:3, 3:6] = l_io1
        u[3:6, 3:6] = _I3
        u[6:9, 0:3] = self.m * w2 * l_c
        u[6:9, 3:6] = w2 * (self.m * l_c @ l_io1 + j_ref)
        u[9:12, 0:3] = -self.m * w2 * _I3
        u[9:12, 3:6] = self.m * w2 * l_o1c
        for k, p in enumerate(self.outputs):
            c0 = 6 + 6 * k
            u[6:9, c0:c0 + 3] = _I3
            u[6:9, c0 + 3:c0 + 6] = -_cross(p - ref)
            u[9:12, c0 + 3:c0 + 6] = _I3
        return u


_DISP = list(range(6))
_FORCE = list(range(6, 12))
_FLIP = np.diag([1.0] * 6 + [-1.0] * 6)


def _chain_matrix(elements, s):
    u = np.eye(12, dtype=complex)
    for el in elements:
        u = el.transfer(s, 0.0) @ u
    return u


class StarTree:
    """Tree system: rigid body, one root chain from the ground and L branch
    chains.

    Topology (Sec. 11, Fig. 11.x type tree): ground -> ``root`` elements ->
    input I of ``body`` -> output O_l -> ``branches[l]`` elements -> free
    tip. All elements use the 12-state of Eq. (7.20). The ground end has
    zero displacements, every tip has zero moments and forces.

    :param root: Elements from the ground to the body input, in transfer order.
    :type root: list
    :param body: Rigid body with one input and L outputs.
    :type body: RigidBodyMultiEnd
    :param branches: L lists of elements from the outputs to the tips.
    :type branches: list
    """

    def __init__(self, root, body, branches):
        if len(body.inputs) != 1 or len(body.outputs) != len(branches):
            raise ValueError("body needs one input and one output per branch")
        self.root = list(root)
        self.body = body
        self.branches = [list(b) for b in branches]

    def characteristic_matrix(self, s, form="outputs"):
        """Square matrix of the homogeneous overall equations at s.

        Forms: ``"outputs"`` uses Eq. (15.19); ``"general"`` uses
        Eq. (15.16)-(15.17); ``"inputs"`` reverses the branches (they become
        inputs of the body, Z_I = D U_b^-1 Z_tip with D = diag(I6, -I6)) and
        uses Eq. (15.18) with the root as the single output. The three forms
        give different matrices with the same determinant roots.

        :param s: Laplace variable.
        :type s: complex
        :param form: ``"outputs"``, ``"general"`` or ``"inputs"``.
        :type form: str
        :return: Square matrix.
        :rtype: numpy.ndarray
        """
        n_b = len(self.branches)
        u_root = _chain_matrix(self.root, s)
        u_br = [_chain_matrix(b, s) for b in self.branches]
        outs = self.body.outputs
        if form == "outputs":
            # unknowns: ground forces (6), X/Theta of O_1 (6), (M, Q) of every
            # output (6 L)
            n = 12 + 6 * n_b
            u_body = self.body.outputs_matrix(s)
            mat = np.zeros((n, n), dtype=complex)
            mat[0:12, 0:6] = u_root[:, _FORCE]
            mat[0:12, 6:] = -u_body
            for k in range(n_b):
                z_ol = np.zeros((12, n), dtype=complex)
                z_ol[0:6, 6:12] = point_displacement_matrix(outs[0], outs[k])
                z_ol[6:12, 12 + 6 * k:18 + 6 * k] = np.eye(6)
                mat[12 + 6 * k:18 + 6 * k] = (u_br[k] @ z_ol)[_FORCE]
            return mat
        if form == "general":
            # unknowns: ground forces (6), (M, Q) of outputs 2..L (6 (L - 1))
            n = 6 * n_b
            u_o, u_i = self.body.general_matrices(s)
            z_i = np.zeros((12, n), dtype=complex)
            z_i[:, 0:6] = u_root[:, _FORCE]
            rhs = u_i @ z_i
            # U_O Z_O = U_I Z_I: rows 0-5 give X/Theta of O_1, rows 6-11 give
            # (M, Q) of O_1
            z_o = np.zeros((6 + 6 * n_b, n), dtype=complex)
            for k in range(1, n_b):
                z_o[6 + 6 * k:12 + 6 * k, 6 * k:6 * k + 6] = np.eye(6)
            known = u_o[:, 12:] @ z_o[12:]
            sol = np.linalg.solve(u_o[:, 0:12], rhs - known)
            z_o[0:12] = sol
            mat = np.zeros((n, n), dtype=complex)
            for k in range(n_b):
                z_ol = np.zeros((12, n), dtype=complex)
                z_ol[0:6] = (point_displacement_matrix(outs[0], outs[k])
                             @ z_o[0:6])
                z_ol[6:12] = z_o[6 + 6 * k:12 + 6 * k]
                mat[6 * k:6 * k + 6] = (u_br[k] @ z_ol)[_FORCE]
            return mat
        if form == "inputs":
            # unknowns: tip displacements of every branch (6 L)
            n = 6 * n_b
            rev = RigidBodyMultiEnd(self.body.m, self.body.j_c,
                                    self.body.center, outs, self.body.inputs)
            z_in = []
            for k in range(n_b):
                z_tip = np.zeros((12, n), dtype=complex)
                z_tip[0:6, 6 * k:6 * k + 6] = np.eye(6)
                z_in.append(_FLIP @ np.linalg.solve(u_br[k], z_tip))
            rows = []
            for k in range(1, n_b):
                rows.append(z_in[k][0:6]
                            - point_displacement_matrix(outs[0], outs[k])
                            @ z_in[0][0:6])
            z_i = np.vstack([z_in[0]] + [z[6:12] for z in z_in[1:]])
            z_o = rev.inputs_matrix(s) @ z_i
            z_g = np.linalg.solve(u_root, _FLIP @ z_o)
            rows.append(z_g[_DISP])
            return np.vstack(rows)
        raise ValueError(f"unknown form {form!r}")

    def characteristic(self, omega, form="outputs"):
        """(sign, log|det|) of the characteristic matrix at s = i omega
        (undamped).

        :param omega: Circular frequency (rad/s).
        :type omega: float
        :param form: See ``characteristic_matrix``.
        :type form: str
        :return: (sign, logabs).
        :rtype: tuple
        """
        return _ric.log_determinant(self.characteristic_matrix(1j * omega,
                                                               form).real)

    def natural_frequencies(self, omega_max, form="outputs", n_grid=2000):
        """Eigenfrequencies in (0, omega_max] by the root search of
        ``linear.find_real_roots``.

        :param omega_max: Upper bound (rad/s).
        :type omega_max: float
        :param form: See ``characteristic_matrix``.
        :type form: str
        :param n_grid: Grid size of the first scan.
        :type n_grid: int
        :return: Sorted eigenfrequencies (rad/s).
        :rtype: numpy.ndarray
        """
        from .linear import find_real_roots

        return np.array(sorted(find_real_roots(lambda w: self.characteristic(
                                                   w, form),
                                               omega_max * 1e-6, omega_max,
                                               n_grid=n_grid)))


# ----------------------------------------------------------------------
# Closed-form massless beams (Eq. 15.24, 15.29)
# ----------------------------------------------------------------------
def axial_load_beam_matrix(x, ei, axial, gas=np.inf):
    """Massless beam with axial load and shear deformation, closed form (Eq.
    15.24).

    State [Y, Theta_z, M_z, Q_y]. ``axial`` is the compressive force P
    (negative for tension). The corrected parameters are lambda = sqrt(-P/EI)
    (complex for compression) and d1 = lambda^2 = -P/EI for both signs; see the
    module docstring. Same equations as ``library.AxialLoadBeam``.

    :param x: Coordinate (0 < x <= l).
    :type x: float
    :param ei: Bending stiffness.
    :type ei: float
    :param axial: Compressive force P (nonzero).
    :type axial: float
    :param gas: Shear stiffness (numpy.inf neglects shear).
    :type gas: float
    :return: U(x), real.
    :rtype: numpy.ndarray
    """
    lam = np.sqrt(complex(-axial / ei))
    d1 = lam * lam
    sh, ch = np.sinh(lam * x), np.cosh(lam * x)
    shear = 0.0 if gas == np.inf else x / gas
    u = np.array([
        [1.0, sh / lam, -(1.0 - ch) / (d1 * ei),
         -(lam * x - sh) / (d1 * lam * ei) - shear],
        [0.0, ch, sh / (lam * ei), -(1.0 - ch) / (d1 * ei)],
        [0.0, d1 * ei * sh / lam, ch, sh / lam],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return u.real


class AxialLoadBeamClosedForm(_lib.ContinuousElement):
    """Element wrapper of ``axial_load_beam_matrix`` (Eq. 15.24), state [Y,
    Theta_z, M_z, Q_y].

    :param length: Length.
    :type length: float
    :param ei: Bending stiffness.
    :type ei: float
    :param axial: Compressive force P (negative for tension, nonzero).
    :type axial: float
    :param gas: Shear stiffness.
    :type gas: float
    """

    size = 4

    def __init__(self, length, ei, axial, gas=np.inf):
        self.length = float(length)
        self.ei, self.axial, self.gas = float(ei), float(axial), gas

    def transfer_at(self, x, s, eta=1.0):
        """Transfer matrix U(x) (Eq. 15.24).

        :param x: Coordinate.
        :type x: float
        :param s: Laplace variable (unused, massless).
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U(x).
        :rtype: numpy.ndarray
        """
        return axial_load_beam_matrix(x, self.ei, self.axial,
                                      self.gas).astype(complex)


class TaperedAxialBeam(_lib.ContinuousElement):
    """Massless nonuniform beam with an axial compressive force, closed form
    (Eq. 15.29).

    State [Y, Theta_z, M_z, Q_y]. EI(x) = EI_I (c1 - c2 x)^2 (Fig. 15.6). The
    equations are those of ``library.AxialLoadBeam`` without shear:
    Y' = Theta, Theta' = M/EI(x), M' = Q - P Theta, Q' = 0. With
    xi = c1 - c2 x, M satisfies the Euler equation xi^2 M'' + k M = 0,
    k = P/(c2^2 EI_I), so M = sqrt(xi) (A cos(beta ln xi) + B sin(beta ln xi))
    with beta^2 = k - 1/4 (book definition). The particular solution is
    Theta = Q/P, and Y = Y_0 + Q x/P - (M - M_0)/P. This form replaces the
    printed Eq. (15.29) (errors listed in the module docstring). It
    requires beta^2 > 0.

    :param length: Length l (c1 - c2 l > 0).
    :type length: float
    :param ei_input: EI_I.
    :type ei_input: float
    :param axial: Compressive force P > 0.
    :type axial: float
    :param c1: Taper constant c1.
    :type c1: float
    :param c2: Taper constant c2 (nonzero).
    :type c2: float
    """

    size = 4

    def __init__(self, length, ei_input, axial, c1=1.0, c2=0.0):
        self.length = float(length)
        self.ei_input, self.axial = float(ei_input), float(axial)
        self.c1, self.c2 = float(c1), float(c2)
        b2 = self.axial / (self.c2**2 * self.ei_input) - 0.25
        if b2 <= 0.0:
            raise ValueError("Eq. 15.29 needs P/(c2^2 EI_I) > 1/4")
        self.beta = np.sqrt(b2)

    def _fundamental(self, xi):
        ph = self.beta * np.log(xi)
        co, si = np.cos(ph), np.sin(ph)
        m = np.sqrt(xi) * np.array([co, si])
        dm = np.array([0.5 * co - self.beta * si,
                       0.5 * si + self.beta * co]) / np.sqrt(xi)
        return np.array([self.c2 / self.axial * dm, m])

    def transfer_at(self, x, s, eta=1.0):
        """Transfer matrix U(x).

        :param x: Coordinate.
        :type x: float
        :param s: Laplace variable (unused).
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U(x).
        :rtype: numpy.ndarray
        """
        p = self.axial
        t = np.linalg.solve(self._fundamental(self.c1).T,
                            self._fundamental(self.c1 - self.c2 * x).T).T
        u = np.eye(4)
        u[0, 1] = -t[1, 0] / p
        u[0, 2] = -(t[1, 1] - 1.0) / p
        u[0, 3] = x / p + t[1, 0] / p**2
        u[1, 1] = t[0, 0]
        u[1, 2] = t[0, 1]
        u[1, 3] = (1.0 - t[0, 0]) / p
        u[2, 1] = t[1, 0]
        u[2, 2] = t[1, 1]
        u[2, 3] = -t[1, 0] / p
        return u.astype(complex)


class TaperedAxialBeamODE(TaperedAxialBeam):
    """Same element as ``TaperedAxialBeam`` by the fourth-order Magnus
    integrator.

    The variable-coefficient system Z' = A(x) Z (Sec. 10.4 with A depending on
    x) is integrated with ``n_steps`` steps of the two-point Gauss Magnus
    method; each step is exp(h (A1 + A2)/2 + sqrt(3) h^2 [A2, A1]/12)
    (``riccati.expm``). The error is O(h^4).

    :param n_steps: Number of steps over the element.
    :type n_steps: int
    """

    def __init__(self, length, ei_input, axial, c1=1.0, c2=0.0, n_steps=200):
        super().__init__(length, ei_input, axial, c1, c2)
        self.n_steps = int(n_steps)

    def system_matrix(self, x):
        """A(x) of Y' = Theta, Theta' = M/EI(x), M' = Q - P Theta, Q' = 0."""
        ei = self.ei_input * (self.c1 - self.c2 * x) ** 2
        return np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0 / ei, 0.0],
                         [0.0, -self.axial, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])

    def transfer_at(self, x, s, eta=1.0):
        """Transfer matrix U(x) by the Magnus integrator.

        :param x: Coordinate.
        :type x: float
        :param s: Laplace variable (unused).
        :type s: complex
        :param eta: Damping scale factor (unused).
        :type eta: float
        :return: U(x).
        :rtype: numpy.ndarray
        """
        h = x / self.n_steps
        g1, g2 = 0.5 - np.sqrt(3.0) / 6.0, 0.5 + np.sqrt(3.0) / 6.0
        u = np.eye(4)
        for k in range(self.n_steps):
            a1 = self.system_matrix((k + g1) * h)
            a2 = self.system_matrix((k + g2) * h)
            om = (0.5 * h * (a1 + a2)
                  + np.sqrt(3.0) / 12.0 * h * h * (a2 @ a1 - a1 @ a2))
            u = _ric.expm(om) @ u
        return u.astype(complex)


# ----------------------------------------------------------------------
# Riccati mode shapes (Eq. 10.130, 10.135)
# ----------------------------------------------------------------------
def _sv_ratio(mat):
    sv = np.linalg.svd(mat, compute_uv=False)
    return sv[-1] / sv[0] if sv[0] > 0 else 0.0


def riccati_mode(chain, omega):
    """Mode shape at an eigenfrequency by the Riccati transfer and back
    substitution.

    The forward sweep (Eq. 10.131) gives S_i and G_i = T21 S_i + T22. The
    Riccati characteristic determinant is det(W_c) prod det(G_i) with
    W = [S_{n+1}; I] (Eq. 10.134), so at an eigenfrequency one factor is
    singular; the factor with the smallest singular value ratio is taken.

    * W_c singular: Z_b at the output end is the null vector of W_c and
      Z_a = S_{n+1} Z_b (Eq. 10.130, e = 0).
    * G_j singular (for example when the output zeros are all in Z_b): Z_b
      at the input of element j is the null vector of G_j, so Z_b at its
      output is zero; the states after element j follow from the element
      transfer matrices.

    Up to three factors with the smallest singular value ratios are tried;
    the mode with the smallest residual of the element equations is kept
    (this rejects a zero-pole pair of G_i and S_{i+1}). The states before
    the singular factor are recovered backwards with
    Z_b,in = G_i^{-1} Z_b,out and Z_a,in = S_i Z_b,in (Eq. 10.135, 10.128
    with e = 0). The backward recursion does not form products of transfer
    matrices, so the mode keeps its precision where the plain propagation of
    ``Chain.modes`` (Eq. 2.46) amplifies rounding.

    :param chain: Chain with homogeneous boundary conditions.
    :type chain: solvcon.multibody.tmm.linear.Chain
    :param omega: Eigenfrequency (rad/s), simple root.
    :type omega: float
    :return: Mode with the state at the input of every element and at the
        output boundary; the input boundary vector is normalized as in
        ``Chain.boundary_vectors``.
    :rtype: solvcon.multibody.tmm.linear.Mode
    """
    a_idx = list(chain.input_zero)
    b_idx = [i for i in range(chain.size) if i not in a_idx]
    zeros = [np.zeros(chain.size)] * len(chain.elements)
    for attempt in range(4):
        # an exactly singular G_i stops the forward sweep; the mode is
        # continuous in omega, so it is evaluated 1e-13 away (as
        # riccati_characteristic does)
        mats = [np.real(el.transfer(1j * omega, 0.0)) for el in chain.elements]
        try:
            sweep = _ric.riccati_sweep(mats, zeros, a_idx, b_idx)
            break
        except np.linalg.LinAlgError:
            omega = omega * (1.0 + 1e-13)
    else:
        raise np.linalg.LinAlgError("Riccati recursion singular")
    s_list = [np.real(m) for m in sweep["S"]]
    g_list = [np.real(g) for g in sweep["G"]]
    nb = len(b_idx)
    n = len(mats)
    w = np.zeros((chain.size, nb))
    w[a_idx, :] = s_list[-1]
    w[b_idx, :] = np.eye(nb)
    w_c = w[list(chain.output_zero), :]

    def full(zb, i):
        z = np.zeros(chain.size)
        z[b_idx] = zb
        z[a_idx] = s_list[i] @ zb
        return z

    def build(j):
        states = np.zeros((n + 1, chain.size))
        if j == n:
            zb = np.real(_pivoted_null_vector(w_c))
            states[n] = w @ zb
            first = n - 1
        else:
            states[j] = full(np.real(_pivoted_null_vector(g_list[j])), j)
            for i in range(j, n):
                states[i + 1] = mats[i] @ states[i]
            first = j - 1
        zb = states[first + 1][b_idx]
        for i in range(first, -1, -1):
            zb = np.linalg.solve(g_list[i], zb)
            states[i] = full(zb, i)
        return states

    def residual(states):
        # element equations and output zeros, each component scaled by its
        # maximum
        scale = np.max(np.abs(states), axis=0)
        scale[scale == 0] = 1.0
        res = [np.max(np.abs(states[i + 1] - mats[i] @ states[i]) / scale)
               for i in range(n)]
        res.append(np.max(np.abs(states[n][list(chain.output_zero)]) / scale[
            list(chain.output_zero)]))
        return max(res)

    # the singular factor of det(W_c) prod det(G_i): the candidates with the
    # smallest singular value ratios; a zero-pole pair of G_i and S_{i+1} is
    # rejected by the residual of the element equations
    ratios = [_sv_ratio(g) for g in g_list] + [_sv_ratio(w_c)]
    best = None
    for j in np.argsort(ratios)[:3]:
        try:
            cand = build(int(j))
        except np.linalg.LinAlgError:
            continue
        r = residual(cand)
        if best is None or r < best[0]:
            best = (r, cand)
    states = best[1]
    iu = chain.input_unknown
    k = int(np.argmax(np.abs(states[0][iu])))
    scale = np.real(_normalize(states[0][iu])[k]) / states[0][iu][k]
    return Mode(chain, omega, states * scale)


# ----------------------------------------------------------------------
# Mode shapes of the floating frame of reference (Sec. 6.4.1, 6.5.1)
# ----------------------------------------------------------------------
def cantilever_beam_roots(n):
    """First n roots of cos(b) cosh(b) = -1 (clamped-free Euler-Bernoulli
    beam).

    :param n: Number of roots.
    :type n: int
    :return: beta_k l, k = 1..n.
    :rtype: numpy.ndarray
    """
    out = []
    for k in range(1, n + 1):
        a, b = max(0.5, (k - 0.5) * np.pi - 1.0), (k - 0.5) * np.pi + 1.0
        fa = np.cos(a) + 1.0 / np.cosh(a)
        for _ in range(200):
            c = 0.5 * (a + b)
            fc = np.cos(c) + 1.0 / np.cosh(c)
            if np.sign(fc) == np.sign(fa):
                a, fa = c, fc
            else:
                b = c
        out.append(0.5 * (a + b))
    return np.array(out)


def cantilever_bending_modes(length, n):
    """Clamped-free bending modes phi_k and derivatives (free vibration of Eq.
    6.11, 6.62).

    phi = cosh(bx) - cos(bx) - sigma (sinh(bx) - sin(bx)), written as
    ((1 - sigma) e^{bx} + (1 + sigma) e^{-bx})/2 - cos(bx) + sigma sin(bx) to
    avoid the cancellation of cosh and sinh for large b x. phi(l) = +-2.

    :param length: Beam length l.
    :type length: float
    :param n: Number of modes.
    :type n: int
    :return: Callable x -> (phi, phi', phi'') arrays of shape (n,) + x.shape.
    :rtype: callable
    """
    betas = cantilever_beam_roots(n) / length
    bl = betas * length
    den = np.sinh(bl) + np.sin(bl)
    sigma = (np.cosh(bl) + np.cos(bl)) / den
    one_m = (np.sin(bl) - np.cos(bl) - np.exp(-bl)) / den
    one_p = 1.0 + sigma

    def modes(x):
        x = np.asarray(x, dtype=float)
        b = betas.reshape((-1,) + (1,) * x.ndim)
        om, op, sg = (v.reshape(b.shape) for v in (one_m, one_p, sigma))
        ep, em = np.exp(b * (x - length)), np.exp(-b * x)
        # (1 - sigma) e^{bx} / 2 written with e^{b(x - l)} and e^{bl}(1 -
        # sigma)
        big = om * np.exp(b * length)
        c, s = np.cos(b * x), np.sin(b * x)
        phi = 0.5 * (big * ep + op * em) - c + sg * s
        d1 = b * (0.5 * (big * ep - op * em) + s + sg * c)
        d2 = b * b * (0.5 * (big * ep + op * em) + c - sg * s)
        return phi, d1, d2

    return modes


def cantilever_axial_modes(length, n):
    """Clamped-free rod modes sin((2k - 1) pi x / (2 l)) and derivatives (Eq.
    6.11, 6.61).

    :param length: Rod length l.
    :type length: float
    :param n: Number of modes.
    :type n: int
    :return: Callable x -> (phi, phi', phi'').
    :rtype: callable
    """
    ks = (2.0 * np.arange(1, n + 1) - 1.0) * np.pi / (2.0 * length)

    def modes(x):
        x = np.asarray(x, dtype=float)
        k = ks.reshape((-1,) + (1,) * x.ndim)
        return np.sin(k * x), k * np.cos(k * x), -k * k * np.sin(k * x)

    return modes


class _BeamModes:
    """Mode tables of a beam at the Gauss points and at both ends.

    Modal coordinate order: q = [q_axial, q_y, q_z]; q_z is absent in the
    planar case. ``phi`` (3 x n per point) maps q to the deformation
    u = [u, v, w] in the floating frame; ``slope`` (3 x n per point) maps q
    to the small rotation theta = [0, -w', v'] (Eq. 6.18, 6.70).
    """

    def __init__(self, length, ea, ei_z, ei_y, rho_a, n_axial, n_y, n_z,
                 n_quad):
        self.length = float(length)
        self.ea, self.ei_z = float(ea), float(ei_z)
        self.ei_y, self.rho_a = float(ei_y), float(rho_a)
        self.n_a, self.n_y, self.n_z = int(n_axial), int(n_y), int(n_z)
        self.n = self.n_a + self.n_y + self.n_z
        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        self.xg = 0.5 * length * (nodes + 1.0)
        self.wg = 0.5 * length * weights
        self.ax = cantilever_axial_modes(length, max(self.n_a, 1))
        self.bd = cantilever_bending_modes(length, max(self.n_y, self.n_z, 1))
        self.ia = slice(0, self.n_a)
        self.iy = slice(self.n_a, self.n_a + self.n_y)
        self.iz = slice(self.n_a + self.n_y, self.n)
        self.tab = [self.table(x) for x in self.xg]
        self.tab0 = self.table(0.0)
        self.tabl = self.table(self.length)
        # stacked tables (n_quad, 3, n) and (n_quad, n) for vectorized sums
        self.phi_g = np.array([t[0] for t in self.tab])
        self.du_g = np.array([t[2] for t in self.tab])
        self.dv_g = np.array([t[3] for t in self.tab])
        self.dw_g = np.array([t[4] for t in self.tab])
        # second derivatives at the Gauss points for the stiffness (Eq. 6.13,
        # 6.64)
        a2 = self.ax(self.xg)
        b2 = self.bd(self.xg)
        ka = self.ea * (a2[1][:self.n_a] * self.wg) @ a2[1][:self.n_a].T
        ky = self.ei_z * (b2[2][:self.n_y] * self.wg) @ b2[2][:self.n_y].T
        kz = self.ei_y * (b2[2][:self.n_z] * self.wg) @ b2[2][:self.n_z].T
        self.k = np.zeros((self.n, self.n))
        self.k[self.ia, self.ia] = ka
        self.k[self.iy, self.iy] = ky
        self.k[self.iz, self.iz] = kz

    def table(self, x):
        """(phi 3 x n, slope 3 x n, axial derivative row, y and z slope rows)
        at x."""
        a = self.ax(np.array(x))
        b = self.bd(np.array(x))
        phi = np.zeros((3, self.n))
        slope = np.zeros((3, self.n))
        phi[0, self.ia] = a[0][:self.n_a]
        phi[1, self.iy] = b[0][:self.n_y]
        phi[2, self.iz] = b[0][:self.n_z]
        slope[1, self.iz] = -b[1][:self.n_z]
        slope[2, self.iy] = b[1][:self.n_y]
        du = np.zeros(self.n)
        du[self.ia] = a[1][:self.n_a]
        dv = np.zeros(self.n)
        dv[self.iy] = b[1][:self.n_y]
        dw = np.zeros(self.n)
        dw[self.iz] = b[1][:self.n_z]
        return phi, slope, du, dv, dw

    def axial_coupling(self, q):
        """Generalized force f of the axial-transverse coupling (Eq. 6.43,
        6.97).

        f_k = int phi_k' EA u' v' dxi for a y bending mode k and
        int psi_k' EA u' w' dxi for a z bending mode, from the term
        d/dxi (EA du/dxi dv/dxi) of Eq. (6.31), (6.84)-(6.85).
        """
        c = self.wg * self.ea * (self.du_g @ q)
        return ((c * (self.dv_g @ q)) @ self.dv_g + (c * (self.dw_g @ q))
                @ self.dw_g)


def _transfer_from_coefficients(a1, a2, b_in, b_out):
    """Element matrices U and Ubar of Eq. (6.51)-(6.60) (planar) or
    (6.104)-(6.113) (spatial).

    :param a1: (a11, a12, a13, a14, a15) of the large-motion equation (6.33).
    :param a2: (a21, a22, a23, a24, a25) of the vibration equation (6.34).
    :param b_in: (B1, B2, B3) of the input point (Eq. 6.47).
    :param b_out: (B1, B2, B3) of the output point (Eq. 6.48).
    :return: (U, Ubar) with U of order 2 k + 1 and Ubar of shape (n, 2 k + 1).
    """
    a11, a12, a13, a14, a15 = a1
    a21, a22, a23, a24, a25 = a2
    b1i, b2i, b3i = b_in
    b1o, b2o, b3o = b_out
    b1i_inv = np.linalg.inv(b1i)
    e1 = a12 - a11 @ b1i_inv @ b2i
    e2 = a15 - a11 @ b1i_inv @ b3i
    e3 = a22 - a21 @ b1i_inv @ b2i
    e4 = a25 - a21 @ b1i_inv @ b3i
    e1e3 = e1 @ np.linalg.inv(e3)
    e5 = a11 @ b1i_inv - e1e3 @ a21 @ b1i_inv
    e6 = a13 - e1e3 @ a23
    e7 = a14 - e1e3 @ a24
    e8 = e2 - e1e3 @ e4
    e7_inv = np.linalg.inv(e7)
    e3_inv = np.linalg.inv(e3)
    e9 = e3_inv @ a24 @ e7_inv @ e5 - e3_inv @ a21 @ b1i_inv
    e10 = e3_inv @ a24 @ e7_inv @ e6 - e3_inv @ a23
    e11 = e3_inv @ a24 @ e7_inv @ e8 - e3_inv @ e4
    e12 = b2o - b1o @ b1i_inv @ b2i
    e13 = b3o - b1o @ b1i_inv @ b3i
    e14 = b1o @ b1i_inv + e12 @ e9
    e15 = e12 @ e11 + e13
    k = b1i.shape[0]
    u = np.zeros((2 * k + 1, 2 * k + 1))
    u[0:k, 0:k] = e14
    u[0:k, k:2 * k] = e12 @ e10
    u[0:k, 2 * k] = e15
    u[k:2 * k, 0:k] = -e7_inv @ e5
    u[k:2 * k, k:2 * k] = -e7_inv @ e6
    u[k:2 * k, 2 * k] = -e7_inv @ e8
    u[2 * k, 2 * k] = 1.0
    ubar = np.hstack([e9, e10, e11.reshape(-1, 1)])
    return u, ubar


def _rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


_D2 = np.array([[0.0, -1.0], [1.0, 0.0]])


def _as_func(value, size):
    """Constant or callable load -> callable of t."""
    if callable(value):
        return value
    arr = np.zeros(size) if value is None else np.asarray(value, dtype=float)
    return lambda t: arr


# ----------------------------------------------------------------------
# Planar elements (Sec. 6.4, Ch. 5 state of Eq. 6.4)
# ----------------------------------------------------------------------
class PlanarRigidElement:
    """Planar rigid body, state [xdd, ydd, Omega_z, m_z, q_x, q_y, 1] (Eq. 6.4,
    Ch. 5).

    Newton-Euler: m r_C'' = q_I - q_O + F and
    J_C Omega = -m_I + m_O + (r_I - r_C) x q_I - (r_O - r_C) x q_O + M, with
    r_C'' = r_I'' + Omega D A c - omega^2 A c. The input end is the body
    frame origin.

    :param m: Mass.
    :type m: float
    :param j_c: Moment of inertia about the mass center.
    :type j_c: float
    :param center: Mass center (body frame).
    :type center: array_like
    :param output: Output end (body frame).
    :type output: array_like
    :param force: Global force at the mass center, constant or callable of t.
    :type force: array_like or callable
    :param moment: Moment, constant or callable of t.
    :type moment: float or callable
    """

    size = 7
    n_modes = 0

    def __init__(self, m, j_c, center=(0.0, 0.0), output=(0.0, 0.0),
                 force=None, moment=None):
        self.m, self.j_c = float(m), float(j_c)
        self.c = np.asarray(center, dtype=float)
        self.b = np.asarray(output, dtype=float)
        self.force = _as_func(force, 2)
        self.moment = _as_func(moment, 1) if not callable(moment) else moment

    def frame_out(self, frame, q, qd):
        """Frame (theta, omega) of the output end."""
        return frame

    def matrices(self, frame, q, qd, t):
        """Transfer matrix U (7 x 7) at time t and an empty Ubar.

        :param frame: (theta, omega) of the body.
        :type frame: tuple
        :param q: Modal coordinates (empty).
        :param qd: Modal velocities (empty).
        :param t: Time.
        :type t: float
        :return: (U, Ubar).
        :rtype: tuple
        """
        th, om = frame
        a = _rot2(th)
        ac, ab = a @ self.c, a @ self.b
        f = np.asarray(self.force(t), dtype=float)
        mz = float(np.asarray(self.moment(t)).ravel()[0])
        u = np.zeros((7, 7))
        # accelerations of O: r_O'' = r_I'' + Omega D A b - omega^2 A b
        u[0:2, 0:2] = _I3[:2, :2]
        u[0:2, 2] = _D2 @ ab
        u[0:2, 6] = -om * om * ab
        u[2, 2] = 1.0
        # q_O = q_I + F - m r_C''
        u[4:6, 0:2] = -self.m * np.eye(2)
        u[4:6, 2] = -self.m * _D2 @ ac
        u[4:6, 4:6] = np.eye(2)
        u[4:6, 6] = f + self.m * om * om * ac
        # m_O = J_C Omega + m_I - (r_I - r_C) x q_I + (r_O - r_C) x q_O - M
        cr_i = -ac
        cr_o = ab - ac
        u[3, 2] = self.j_c
        u[3, 3] = 1.0
        u[3, 4:6] = -np.array([-cr_i[1], cr_i[0]])
        # (r_O - r_C) x q_O with q_O = row block 4:6 of u
        cross_o = np.array([-cr_o[1], cr_o[0]])
        u[3, :] += cross_o @ u[4:6, :]
        u[3, 6] -= mz
        u[6, 6] = 1.0
        return u, np.zeros((0, 7))


class PlanarFlexibleBeam:
    """Planar beam with large motion in the floating frame of reference (Sec.
    6.4).

    State [xdd, ydd, Omega_z, m_z, q_x, q_y, 1] at both ends (Eq. 6.49). The
    floating frame o2 is at the input end with the axis along the
    undeformed beam; the modes are the clamped-free axial modes phi and
    bending modes varphi (Sec. 6.4.1), so u(0) = v(0) = v'(0) = 0 and o2 = I.
    The coefficient matrices a_1,1 ... a_2,5 are those of Eq. (6.35)-(6.43)
    with the mass integrals of Eq. (6.12)-(6.17) by Gauss quadrature; the
    moment work terms of the bending modes are added (module docstring).
    U and Ubar follow Eq. (6.52)-(6.60).

    :param length: Length l.
    :type length: float
    :param ea: Tensile stiffness EA.
    :type ea: float
    :param ei: Bending stiffness EI_z.
    :type ei: float
    :param rho_a: Mass per unit length rho A.
    :type rho_a: float
    :param n_axial: Number of axial modes n_phi.
    :type n_axial: int
    :param n_bend: Number of bending modes n_varphi.
    :type n_bend: int
    :param axial_coupling: Include the term f of Eq. (6.43) (book default).
    :type axial_coupling: bool
    :param n_quad: Gauss points for the mass and stiffness integrals.
    :type n_quad: int
    """

    size = 7

    def __init__(self, length, ea, ei, rho_a, n_axial=1, n_bend=3,
                 axial_coupling=True, n_quad=48):
        self.md = _BeamModes(length, ea, ei, ei, rho_a, n_axial, n_bend, 0,
                             n_quad)
        self.n_modes = self.md.n
        self.length = float(length)
        self.axial_coupling = bool(axial_coupling)
        md = self.md
        na, nb = md.n_a, md.n_y
        a_g = md.ax(md.xg)
        b_g = md.bd(md.xg)
        ph = a_g[0][:na]
        vp = b_g[0][:nb]
        w = md.wg * md.rho_a
        xi = md.xg
        # Eq. (6.12)-(6.17), (6.42)
        self.m_aa = (ph * w) @ ph.T
        self.m_bb = (vp * w) @ vp.T
        self.m_ab = (ph * w) @ vp.T
        self.m_a1 = ph @ w
        self.m_b1 = vp @ w
        self.m_ax = ph @ (w * xi)
        self.m_bx = vp @ (w * xi)
        self.mass = float(np.sum(w))
        self.m_xic = float(np.sum(w * xi))
        self.j_o2 = float(np.sum(w * xi * xi))
        ends = [md.ax(np.array(x)) for x in (0.0, length)], \
            [md.bd(np.array(x)) for x in (0.0, length)]
        self.ph_end = [e[0][:na] for e in ends[0]]
        self.vp_end = [e[0][:nb] for e in ends[1]]
        self.dvp_end = [e[1][:nb] for e in ends[1]]

    def frame_out(self, frame, q, qd):
        """Frame of the output end: theta + v'(l), omega + v'(l) dot (Eq.
        6.18)."""
        nb = self.md.n_a
        th, om = frame
        return th + self.dvp_end[1] @ q[nb:], om + self.dvp_end[1] @ qd[nb:]

    def coefficients(self, frame, q, qd, t=0.0):
        """Coefficients of Eq. (6.33)-(6.34) and (6.44)-(6.48).

        :param frame: (theta_o2, omega_o2).
        :type frame: tuple
        :param q: Modal coordinates [q_phi, q_varphi].
        :type q: numpy.ndarray
        :param qd: Modal velocities.
        :type qd: numpy.ndarray
        :param t: Time (unused, no distributed loads).
        :type t: float
        :return: (a1, a2, b_in, b_out) tuples.
        :rtype: tuple
        """
        na = self.md.n_a
        th, om = frame
        a = _rot2(th)
        at = a.T
        qa, qb = q[:na], q[na:]
        qda, qdb = qd[:na], qd[na:]
        h1, h2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        s_vec = h1 * self.m_xic + h1 * (self.m_a1 @ qa) + h2 * (self.m_b1 @ qb)
        j_q = (self.j_o2 + 2.0 * self.m_ax @ qa + qa @ self.m_aa @ qa + qb
               @ self.m_bb @ qb)
        n = self.n_modes
        a11 = np.zeros((3, 3))
        a11[0:2, 0:2] = self.mass * at
        a11[0:2, 2] = _D2 @ s_vec
        a11[2, 0:2] = s_vec @ _D2.T @ at
        a11[2, 2] = j_q
        a12 = np.zeros((3, n))
        a12[0, :na] = self.m_a1
        a12[1, na:] = self.m_b1
        a12[2, :na] = -(qb @ self.m_ab.T)
        a12[2, na:] = self.m_bx + qa @ self.m_ab
        s_i = np.zeros(2)
        s_o = (h1 * self.length + h1 * (self.ph_end[1] @ qa)
               + h2 * (self.vp_end[1] @ qb))
        a13 = np.zeros((3, 3))
        a13[0:2, 1:3] = -at
        a13[2, 0] = 1.0
        a13[2, 1:3] = -(s_i @ _D2.T @ at)
        a14 = np.zeros((3, 3))
        a14[0:2, 1:3] = at
        a14[2, 0] = -1.0
        a14[2, 1:3] = s_o @ _D2.T @ at
        sq = h1 * (self.m_a1 @ qda) + h2 * (self.m_b1 @ qdb)
        a15 = np.zeros(3)
        a15[0:2] = 2.0 * _D2 @ sq * om - s_vec * om * om
        a15[2] = 2.0 * om * (self.m_ax @ qda + qa @ self.m_aa @ qda + qb
                             @ self.m_bb @ qdb)
        a21 = np.zeros((n, 3))
        a21[:na, 0:2] = np.outer(self.m_a1, h1 @ at)
        a21[:na, 2] = -(self.m_ab @ qb)
        a21[na:, 0:2] = np.outer(self.m_b1, h2 @ at)
        a21[na:, 2] = self.m_bx + self.m_ab.T @ qa
        a22 = np.zeros((n, n))
        a22[:na, :na] = self.m_aa
        a22[na:, na:] = self.m_bb
        a23 = np.zeros((n, 3))
        a23[:na, 1:3] = -np.outer(self.ph_end[0], h1 @ at)
        a23[na:, 1:3] = -np.outer(self.vp_end[0], h2 @ at)
        a23[na:, 0] = self.dvp_end[0]
        a24 = np.zeros((n, 3))
        a24[:na, 1:3] = np.outer(self.ph_end[1], h1 @ at)
        a24[na:, 1:3] = np.outer(self.vp_end[1], h2 @ at)
        a24[na:, 0] = -self.dvp_end[1]
        a25 = self.md.k @ q
        a25[:na] += (-2.0 * self.m_ab @ qdb * om
                     - (self.m_ax + self.m_aa @ qa) * om * om)
        a25[na:] += 2.0 * self.m_ab.T @ qda * om - self.m_bb @ qb * om * om
        if self.axial_coupling:
            a25 += self.md.axial_coupling(q)
        # kinematics Eq. (6.45)-(6.46) at I (xi = 0) and O (xi = l)
        b_pts = []
        for s_p, phe, vpe, dvpe in ((s_i, self.ph_end[0], self.vp_end[0],
                                     self.dvp_end[0]),
                                    (s_o, self.ph_end[1], self.vp_end[1],
                                     self.dvp_end[1])):
            b1 = np.eye(3)
            b1[0:2, 2] = a @ _D2 @ s_p
            b2 = np.zeros((3, n))
            b2[0:2, :na] = np.outer(a @ h1, phe)
            b2[0:2, na:] = np.outer(a @ h2, vpe)
            b2[2, na:] = dvpe
            ud = h1 * (phe @ qda) + h2 * (vpe @ qdb)
            b3 = np.zeros(3)
            b3[0:2] = 2.0 * a @ _D2 @ ud * om - a @ s_p * om * om
            b_pts.append((b1, b2, b3))
        return (a11, a12, a13, a14, a15), (a21, a22, a23, a24,
                                           a25), b_pts[0], b_pts[1]

    def matrices(self, frame, q, qd, t):
        """Transfer matrices U (7 x 7) and Ubar (n x 7) of Eq. (6.51)-(6.52).

        :param frame: (theta_o2, omega_o2).
        :type frame: tuple
        :param q: Modal coordinates.
        :type q: numpy.ndarray
        :param qd: Modal velocities.
        :type qd: numpy.ndarray
        :param t: Time.
        :type t: float
        :return: (U, Ubar).
        :rtype: tuple
        """
        a1, a2, b_in, b_out = self.coefficients(frame, q, qd, t)
        return _transfer_from_coefficients(a1, a2, b_in, b_out)


# ----------------------------------------------------------------------
# Spatial elements (Sec. 6.5, state of Eq. 6.1)
# ----------------------------------------------------------------------
def rotation_about(axis, angle):
    """Rotation matrix about a unit axis (Rodrigues formula).

    :param axis: Axis (normalized here).
    :type axis: array_like
    :param angle: Angle (rad).
    :type angle: float
    :return: 3 x 3 rotation matrix.
    :rtype: numpy.ndarray
    """
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    k = _cross(n)
    return _I3 + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


class SpatialRigidElement:
    """Spatial rigid body, state [r'' (3), Omega (3), m (3), q (3), 1] (Eq.
    6.1, Ch. 5).

    m r_C'' = q_I - q_O + F and
    J_C Omega + omega x J_C omega = -m_I + m_O + (r_I - r_C) x q_I - (r_O -
    r_C) x q_O + M, r_C'' = r_I'' + Omega x A c + omega x (omega x A c), all in
    the inertial frame; J_C = A J_c A^T. The input end is the body frame
    origin.

    :param m: Mass.
    :type m: float
    :param j_c: Inertia matrix about the mass center (body frame).
    :type j_c: array_like
    :param center: Mass center (body frame).
    :type center: array_like
    :param output: Output end (body frame).
    :type output: array_like
    :param force: Global force at the mass center, constant or callable of t.
    :type force: array_like or callable
    :param moment: Global moment, constant or callable of t.
    :type moment: array_like or callable
    """

    size = 13
    n_modes = 0

    def __init__(self, m, j_c, center=(0, 0, 0), output=(0, 0, 0), force=None,
                 moment=None):
        self.m = float(m)
        self.j_c = np.asarray(j_c, dtype=float)
        self.c = np.asarray(center, dtype=float)
        self.b = np.asarray(output, dtype=float)
        self.force = _as_func(force, 3)
        self.moment = _as_func(moment, 3)

    def frame_out(self, frame, q, qd):
        """Frame (A, omega) of the output end."""
        return frame

    def matrices(self, frame, q, qd, t):
        """Transfer matrix U (13 x 13) and an empty Ubar.

        :param frame: (A, omega) with omega in the inertial frame.
        :type frame: tuple
        :param q: Modal coordinates (empty).
        :param qd: Modal velocities (empty).
        :param t: Time.
        :type t: float
        :return: (U, Ubar).
        :rtype: tuple
        """
        a, om = frame
        ac, ab = a @ self.c, a @ self.b
        jg = a @ self.j_c @ a.T
        w = _cross(om)
        u = np.zeros((13, 13))
        u[0:3, 0:3] = _I3
        u[0:3, 3:6] = -_cross(ab)
        u[0:3, 12] = w @ w @ ab
        u[3:6, 3:6] = _I3
        # q_O = q_I + F - m r_C''
        u[9:12, 0:3] = -self.m * _I3
        u[9:12, 3:6] = self.m * _cross(ac)
        u[9:12, 9:12] = _I3
        u[9:12,
          12] = (np.asarray(self.force(t), dtype=float) - self.m * w @ w @ ac)
        # m_O = J Omega + omega x J omega + m_I - (r_I - r_C) x q_I + (r_O -
        # r_C) x q_O - M
        u[6:9, 3:6] = jg
        u[6:9, 6:9] = _I3
        u[6:9, 9:12] = -_cross(-ac)
        u[6:9, :] += _cross(ab - ac) @ u[9:12, :]
        u[6:9, 12] += w @ jg @ om - np.asarray(self.moment(t), dtype=float)
        u[12, 12] = 1.0
        return u, np.zeros((0, 13))


class SpatialFlexibleBeam:
    """Spatial beam with large motion in the floating frame of reference (Sec.
    6.5).

    State [r'', Omega, m, q, 1] at both ends (Eq. 6.103). Modal coordinates
    q = [q_phi (axial), q_varphi (bending in y), q_psi (bending in z)],
    clamped-free modes, o2 = I. The coefficients of Eq. (6.87)-(6.88) and the
    kinematics of Eq. (6.98)-(6.102) are formed from their integral definitions
    (Eq. 6.79-6.85, 6.63-6.69) by Gauss quadrature over the mass points
    s = L + u of the beam axis; the polar mass moment J_xi of the cross
    sections gives J_xi (e1 e1^T omega_dot + omega~ e1 e1^T omega) (Eq. 6.80).
    U and Ubar follow Eq. (6.105)-(6.113). In the plane this element equals
    ``PlanarFlexibleBeam``, whose coefficients are the closed forms of
    Eq. (6.35)-(6.41) (test ``tests/test_tmm_flexible.py``).

    :param length: Length l.
    :type length: float
    :param ea: Tensile stiffness.
    :type ea: float
    :param ei_z: Bending stiffness for deflection v (in y).
    :type ei_z: float
    :param ei_y: Bending stiffness for deflection w (in z).
    :type ei_y: float
    :param rho_a: Mass per unit length.
    :type rho_a: float
    :param rho_ip: Polar mass moment per unit length rho (I_y + I_z).
    :type rho_ip: float
    :param n_axial: Number of axial modes.
    :type n_axial: int
    :param n_y: Number of bending modes in y.
    :type n_y: int
    :param n_z: Number of bending modes in z.
    :type n_z: int
    :param axial_coupling: Include f of Eq. (6.97).
    :type axial_coupling: bool
    :param n_quad: Gauss points.
    :type n_quad: int
    """

    size = 13

    def __init__(self, length, ea, ei_z, ei_y, rho_a, rho_ip=0.0, n_axial=1,
                 n_y=3, n_z=3, axial_coupling=True, n_quad=48):
        self.md = _BeamModes(length, ea, ei_z, ei_y, rho_a, n_axial, n_y, n_z,
                             n_quad)
        self.n_modes = self.md.n
        self.length = float(length)
        self.j_xi = float(rho_ip) * float(length)
        self.axial_coupling = bool(axial_coupling)

    def frame_out(self, frame, q, qd):
        """Frame of the output end: A (I + theta~(l)), omega + A theta_dot(l)
        (Eq. 6.71-6.72).

        The small rotation is applied by the Rodrigues formula so that the
        output frame stays orthonormal.
        """
        a, om = frame
        slope = self.md.tabl[1]
        th = slope @ q
        ang = np.linalg.norm(th)
        rot = rotation_about(th, ang) if ang > 0.0 else _I3
        return a @ rot, om + a @ slope @ qd

    def coefficients(self, frame, q, qd, t=0.0):
        """Coefficients of Eq. (6.87)-(6.88) and (6.98)-(6.102).

        :param frame: (A_o2, omega_o2), omega in the inertial frame.
        :type frame: tuple
        :param q: Modal coordinates.
        :type q: numpy.ndarray
        :param qd: Modal velocities.
        :type qd: numpy.ndarray
        :param t: Time (unused).
        :type t: float
        :return: (a1, a2, b_in, b_out).
        :rtype: tuple
        """
        md = self.md
        a, om = frame
        at = a.T
        wb = at @ om
        wt = _cross(wb)
        n = self.n_modes
        e1 = np.array([1.0, 0.0, 0.0])
        dm = md.wg * md.rho_a
        phi_g = md.phi_g
        s_g = phi_g @ q
        s_g[:, 0] += md.xg
        st_g = np.zeros((len(dm), 3, 3))
        st_g[:, 0, 1], st_g[:, 0, 2] = -s_g[:, 2], s_g[:, 1]
        st_g[:, 1, 0], st_g[:, 1, 2] = s_g[:, 2], -s_g[:, 0]
        st_g[:, 2, 0], st_g[:, 2, 1] = -s_g[:, 1], s_g[:, 0]
        acc_g = s_g @ (wt @ wt).T + 2.0 * (phi_g @ qd) @ wt.T
        mass = float(np.sum(dm))
        s_sum = dm @ s_g
        ss = np.einsum("g,gij,gjk->ik", dm, st_g, st_g)
        sphi = np.einsum("g,gij,gjk->ik", dm, st_g, phi_g)
        phi_sum = np.einsum("g,gij->ij", dm, phi_g)
        mqq = np.einsum("g,gji,gjk->ik", dm, phi_g, phi_g)
        phis = np.einsum("g,gji,gjk->ik", dm, phi_g, st_g)
        c_t = dm @ acc_g
        c_r = np.einsum("g,gij,gj->i", dm, st_g, acc_g)
        c_q = np.einsum("g,gji,gj->i", dm, phi_g, acc_g)
        s_t = _cross(s_sum)
        jx = self.j_xi * np.outer(e1, e1)
        phi0, sl0 = md.tab0[0], md.tab0[1]
        phil, sll = md.tabl[0], md.tabl[1]
        s_i = phi0 @ q
        s_o = np.array([self.length, 0.0, 0.0]) + phil @ q
        a11 = np.zeros((6, 6))
        a11[0:3, 0:3] = mass * at
        a11[0:3, 3:6] = -s_t @ at
        a11[3:6, 0:3] = s_t @ at
        a11[3:6, 3:6] = (-ss + jx) @ at
        a12 = np.vstack([phi_sum, sphi])
        a13 = np.zeros((6, 6))
        a13[0:3, 3:6] = -at
        a13[3:6, 0:3] = at
        a13[3:6, 3:6] = -_cross(s_i) @ at
        a14 = np.zeros((6, 6))
        a14[0:3, 3:6] = at
        a14[3:6, 0:3] = -at
        a14[3:6, 3:6] = _cross(s_o) @ at
        a15 = np.concatenate([c_t, c_r + self.j_xi * wt @ e1 * (e1 @ wb)])
        a21 = np.zeros((n, 6))
        a21[:, 0:3] = phi_sum.T @ at
        a21[:, 3:6] = -phis @ at
        a22 = mqq
        a23 = np.zeros((n, 6))
        a23[:, 0:3] = sl0.T @ at
        a23[:, 3:6] = -phi0.T @ at
        a24 = np.zeros((n, 6))
        a24[:, 0:3] = -sll.T @ at
        a24[:, 3:6] = phil.T @ at
        a25 = c_q + md.k @ q
        if self.axial_coupling:
            a25 = a25 + md.axial_coupling(q)
        b_pts = []
        for s_p, phi_p, sl_p in ((s_i, phi0, sl0), (s_o, phil, sll)):
            b1 = np.eye(6)
            b1[0:3, 3:6] = -_cross(a @ s_p)
            b2 = np.vstack([a @ phi_p, a @ sl_p])
            b3 = np.concatenate([a @ (wt @ wt @ s_p + 2.0 * wt @ (phi_p @ qd)),
                                 a @ wt @ (sl_p @ qd)])
            b_pts.append((b1, b2, b3))
        return (a11, a12, a13, a14, a15), (a21, a22, a23, a24,
                                           a25), b_pts[0], b_pts[1]

    def matrices(self, frame, q, qd, t):
        """Transfer matrices U (13 x 13) and Ubar (n x 13) of Eq.
        (6.104)-(6.105).

        :param frame: (A_o2, omega_o2).
        :type frame: tuple
        :param q: Modal coordinates.
        :type q: numpy.ndarray
        :param qd: Modal velocities.
        :type qd: numpy.ndarray
        :param t: Time.
        :type t: float
        :return: (U, Ubar).
        :rtype: tuple
        """
        a1, a2, b_in, b_out = self.coefficients(frame, q, qd, t)
        return _transfer_from_coefficients(a1, a2, b_in, b_out)


# ----------------------------------------------------------------------
# Chain with a base pin (Sec. 6.3, 6.6)
# ----------------------------------------------------------------------
class FlexibleChain:
    """Chain of rigid bodies and flexible beams fixed to each other, pinned to
    the ground.

    The first element is pinned at its input end to the ground (Example
    6.6.1, Eq. 6.116): r'' = 0, Omega = n theta'', and the pin moment about
    the axis n is zero; the last output end is free (m = 0, q = 0,
    Eq. 6.117). The overall transfer equation z_n,O = U_n ... U_1 z_1,I
    (Eq. 6.9-6.10, 6.114) gives the unknown boundary state, the element
    equations give every state vector and the modal accelerations
    q'' = Ubar z_I (Eq. 6.115). The generalized coordinates are the pin
    angle theta and the modal coordinates of every beam; time integration
    is the fourth-order Runge-Kutta method (Sec. 6.6).

    :param elements: Elements in transfer order (all planar or all spatial).
    :type elements: list
    :param axis: Pin axis (spatial chains only).
    :type axis: array_like or None
    """

    def __init__(self, elements, axis=None):
        self.elements = list(elements)
        sizes = {el.size for el in self.elements}
        if len(sizes) != 1:
            raise ValueError("all elements must be planar or all spatial")
        self.size = sizes.pop()
        self.spatial = self.size == 13
        self.axis = (None if axis is None
                     else np.asarray(axis, float) / np.linalg.norm(axis))
        if self.spatial and self.axis is None:
            raise ValueError("a spatial chain needs the pin axis")
        self.offsets = np.cumsum([0] + [el.n_modes for el in self.elements])
        self.n_q = int(self.offsets[-1])

    def _base_frame(self, theta, theta_d):
        if self.spatial:
            return rotation_about(self.axis, theta), self.axis * theta_d
        return theta, theta_d

    def frames(self, y):
        """Frames of the input end of every element for the state y."""
        th, thd, q, qd = self.unpack(y)
        frame = self._base_frame(th, thd)
        out = []
        for i, el in enumerate(self.elements):
            sl = slice(self.offsets[i], self.offsets[i + 1])
            out.append(frame)
            frame = el.frame_out(frame, q[sl], qd[sl])
        return out

    def unpack(self, y):
        """Split y = [theta, theta_dot, q, q_dot]."""
        n = self.n_q
        return y[0], y[1], y[2:2 + n], y[2 + n:2 + 2 * n]

    def solve(self, y, t):
        """Boundary states and accelerations at time t (Eq. 6.114-6.117).

        :param y: State [theta, theta_dot, q, q_dot].
        :type y: numpy.ndarray
        :param t: Time.
        :type t: float
        :return: dict with ``theta_dd``, ``q_dd`` and ``states`` (input state
            of every element and the output state of the last).
        :rtype: dict
        """
        th, thd, q, qd = self.unpack(y)
        frames = self.frames(y)
        mats = []
        u_all = np.eye(self.size)
        for i, el in enumerate(self.elements):
            sl = slice(self.offsets[i], self.offsets[i + 1])
            u, ubar = el.matrices(frames[i], q[sl], qd[sl], t)
            mats.append((u, ubar))
            u_all = u @ u_all
        k = (self.size - 1) // 2
        if self.spatial:
            # unknowns: theta'', two moment components normal to the axis, q
            # (3)
            n = self.axis
            perp = np.linalg.svd(n.reshape(1, 3))[2][1:].T
            basis = np.zeros((13, 6))
            basis[3:6, 0] = n
            basis[6:9, 1:3] = perp
            basis[9:12, 3:6] = _I3
        else:
            basis = np.zeros((7, 3))
            basis[2, 0] = 1.0
            basis[4:6, 1:3] = np.eye(2)
        z0_const = np.zeros(self.size)
        z0_const[-1] = 1.0
        rows = list(range(k, 2 * k))
        x = np.linalg.solve(u_all[rows] @ basis, -u_all[rows] @ z0_const)
        z = basis @ x + z0_const
        states = [z]
        qdd = np.zeros(self.n_q)
        for i, (u, ubar) in enumerate(mats):
            if ubar.shape[0]:
                qdd[self.offsets[i]:self.offsets[i + 1]] = ubar @ z
            z = u @ z
            states.append(z)
        return {"theta_dd": x[0], "q_dd": qdd, "states": states}

    def rhs(self, t, y):
        """Time derivative of y."""
        res = self.solve(y, t)
        n = self.n_q
        return np.concatenate([[y[1], res["theta_dd"]], y[2 + n:],
                               res["q_dd"]])

    def simulate(self, t_end, dt, y0=None, t0=0.0, callback=None):
        """Fourth-order Runge-Kutta integration (Sec. 6.6).

        :param t_end: End time.
        :type t_end: float
        :param dt: Time step.
        :type dt: float
        :param y0: Initial state (zero if None).
        :type y0: numpy.ndarray or None
        :param t0: Start time.
        :type t0: float
        :param callback: Optional callable (t, y) called after every step.
        :type callback: callable or None
        :return: (times, states) arrays.
        :rtype: tuple
        """
        y = (np.zeros(2 + 2 * self.n_q) if y0 is None
             else np.array(y0, dtype=float))
        n_steps = int(round((t_end - t0) / dt))
        ts = [t0]
        ys = [y.copy()]
        t = t0
        for _ in range(n_steps):
            k1 = self.rhs(t, y)
            k2 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = self.rhs(t + dt, y + dt * k3)
            y = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += dt
            ts.append(t)
            ys.append(y.copy())
            if callback is not None:
                callback(t, y)
        return np.array(ts), np.array(ys)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
