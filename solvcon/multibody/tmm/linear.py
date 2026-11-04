# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Linear multibody system transfer matrix method (book Ch. 2-3).

Main objects:

* ``Element`` - base class of all element transfer matrices (library in
  ``solvcon.multibody.tmm.library``). Every element maps the input state to the
  output state, Z_O = U(s) Z_I + f(s) (Eq. 2.12, 2.115, 10.123).
* ``Chain`` - chain system with homogeneous boundary conditions: overall
  transfer matrix (Eq. 2.32-2.34, 2.116-2.118), characteristic equation
  (Eq. 2.39-2.40), eigenfrequencies by scanning and refinement (Sec. 2.5 and
  10.7), state vectors and eigenvectors (Eq. 2.46-2.50), damped complex
  eigenvalues (Sec. 2.7), steady-state forced response (Sec. 2.8-2.9,
  Eq. 2.119-2.120, 2.145-2.151) and static response (Sec. 3.9).
* ``Mode`` - eigenfrequency with the state vector at the input end of every
  element; the augmented eigenvector of Eq. (3.39) is formed from it.
* ``ModalModel`` - augmented eigenvectors, orthogonality (Eq. 3.48),
  modal superposition of the transient response (Eq. 3.90-3.100,
  3.135-3.157), modal damping matrix (Eq. 3.137-3.140) and modal
  steady-state response (Eq. 3.176-3.181).

Laplace variable convention: element matrices are functions of the complex
variable s. Undamped free vibration uses s = i*omega (Eq. 2.7, 2.10), damped
free vibration uses s = lambda = -lambda_r + i lambda_i (Eq. 2.85) and
harmonic excitation uses s = i*Omega (Eq. 2.130). The term m omega^2 of the
book becomes -m s^2 (Table 10.1).

Root search rule (Sec. 2.5 and 10.7). For the undamped characteristic
determinant D(omega) on [omega_min, omega_max]:

1. Rigid-body modes at omega = 0 are counted by the nullity of the
   characteristic matrix at omega = 0 (singular value ratio below
   ``null_tol``).
2. D is sampled on a uniform grid of n points. The grid is refined by
   halving the step until two successive grids give the same number of
   roots (at most ``max_halvings`` times).
3. Each sign change is refined by the improved secant step (Eq. 10.98)
   with the Illinois modification and bisection safeguard.
4. Every interior local minimum of |D| on the grid without a sign change
   is searched by golden section on log|D|. If a sign change appears, the
   two roots are refined separately (close root pair). Otherwise the
   minimum is accepted as a multiple root if at least two singular values
   of the characteristic matrix are below ``null_tol`` times the largest
   singular value at the ends of the searched interval, after a row/column
   equilibration with the scale factors of the matrix at those ends; the
   multiplicity is the number of such singular values.
5. Roots closer than 1e-6 (relative) whose mean has a nullity of at least 2
   are one multiple root found twice (a rounding sign change next to a
   double root); they are replaced by the mean repeated nullity times.

D is handled as (sign, log|D|) (Eq. 10.93-10.97), so overflow is not
possible.

Damped eigenvalue rule (Sec. 2.7.3, 10.7). The complex roots of
D(lambda) = 0 are found by continuation in a damping scale factor eta:
eta = 0 gives the undamped roots lambda = i omega_k (found by the scan),
then eta is increased to 1 in steps; at each step the complex secant
method of Eq. (10.104) starts from the root of the previous step. A step is
halved when the secant iteration fails or moves the root by more than half
of its modulus. If two starts end at the same root, the continuation is
repeated with four times more steps; a remaining duplicate is an error.
"""

import numpy as np

from . import riccati as _num


class Element:
    """Base class of an element transfer matrix (Eq. 2.12, 2.115).

    Subclasses define ``size`` (order of the state vector) and
    ``transfer(s, eta)``. Optional methods describe the element for the
    augmented eigenvector theory of Ch. 3:

    * discrete elements: ``mass_weight()`` and ``stiffness_weight()`` return
      symmetric matrices W such that the inner products of Eq. (3.48) are
      z_k^T W z_p with z the input state vector;
    * continuous elements (``continuous = True``): ``length``,
      ``transfer_at(x, s, eta)``, ``mass_density()``, ``stiffness_density()``
      and ``wavenumber(s)``; the inner products are int_0^l z_k(x)^T W z_p(x)
      dx.

    :ivar size: Order of the state vector.
    :vartype size: int
    :ivar continuous: True for distributed elements (beams, rods, shafts).
    :vartype continuous: bool
    """

    size = None
    continuous = False
    length = 0.0

    def transfer(self, s, eta=1.0):
        """Transfer matrix U(s).

        :param s: Laplace variable.
        :type s: complex
        :param eta: Scale factor of the damping parameters (1 = nominal).
        :type eta: float
        :return: Square matrix of order ``size``.
        :rtype: numpy.ndarray
        """
        raise NotImplementedError

    def forcing(self, s):
        """Load column f of the extended transfer matrix (Eq. 2.115).

        :param s: Laplace variable (i Omega for harmonic loads).
        :type s: complex
        :return: Complex vector of order ``size``.
        :rtype: numpy.ndarray
        """
        return np.zeros(self.size, dtype=complex)

    def extended(self, s, eta=1.0):
        """Extended transfer matrix [[U, f], [0, 1]] (Eq. 2.111, 2.115).

        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor.
        :type eta: float
        :return: Matrix of order ``size + 1``.
        :rtype: numpy.ndarray
        """
        n = self.size
        ext = np.zeros((n + 1, n + 1), dtype=complex)
        ext[:n, :n] = self.transfer(s, eta)
        ext[:n, n] = self.forcing(s)
        ext[n, n] = 1.0
        return ext

    def mass_weight(self):
        """Mass weight on the input state vector (Eq. 3.36-3.38), or None."""
        return None

    def stiffness_weight(self):
        """Stiffness (strain energy) weight on the input state vector, or
        None."""
        return None

    def damping_weight(self):
        """Damping weight on the input state vector (Eq. 3.138, 3.155), or
        None."""
        return None

    def displacement_index(self):
        """Indices of the displacement components of the state vector (Eq.
        2.24)."""
        return list(range(self.size // 2))


def _quadrature(n_points, length):
    """Gauss-Legendre nodes and weights on [0, length]."""
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    return 0.5 * length * (nodes + 1.0), 0.5 * length * weights


def _null_space(mat, count):
    """Right singular vectors of the ``count`` smallest singular values."""
    _, sv, vh = np.linalg.svd(mat)
    return vh[-count:, :].conj().T, sv


def _pivoted_null_vector(mat):
    """Null vector of a rank n - 1 matrix with one component fixed to 1
    (Example 2.4 Eq. b-c).

    The book fixes one unknown (for example Theta_z = 1) and solves the other
    equations. The pivot is the largest component of the right singular
    vector of the smallest singular value; the other components follow from
    the least-squares solution of C[:, others] x = -C[:, pivot], whose
    accuracy does not depend on the units (column scales) of the other
    unknowns.
    """
    n = mat.shape[1]
    if n == 1:
        return np.ones(1)
    _, _, vh = np.linalg.svd(mat)
    j = int(np.argmax(np.abs(vh[-1])))
    others = [i for i in range(n) if i != j]
    sol, *_ = np.linalg.lstsq(mat[:, others], -mat[:, j], rcond=None)
    vec = np.zeros(n, dtype=mat.dtype)
    vec[j] = 1.0
    vec[others] = sol
    return vec


def _equilibration(mat, sweeps=4):
    """Row and column scale factors r, c that balance the norms of r A c to 1.

    Applied with the factors of the matrix at the ends of the searched
    interval, the scaling makes the nullity test at a root independent of
    the units of the state components.
    """
    a = np.array(mat, dtype=float)
    r_tot = np.ones(a.shape[0])
    c_tot = np.ones(a.shape[1])
    for _ in range(sweeps):
        r = np.linalg.norm(a, axis=1)
        r[r == 0] = 1.0
        a = a / r[:, None]
        r_tot /= r
        c = np.linalg.norm(a, axis=0)
        c[c == 0] = 1.0
        a = a / c[None, :]
        c_tot /= c
    return r_tot, c_tot


def _normalize(vec):
    """Scale a vector to unit 2-norm with its largest component real
    positive."""
    vec = np.asarray(vec, dtype=complex)
    k = int(np.argmax(np.abs(vec)))
    if abs(vec[k]) == 0:
        return vec
    vec = vec / vec[k] * abs(vec[k])
    return vec / np.linalg.norm(vec)


class Chain:
    """Chain multibody system with homogeneous boundary conditions (Sec. 2.3).

    The elements are listed in transfer order from the input (root)
    boundary P_{1,0} to the output (tip) boundary P_{n,0} (Eq. 2.32).

    :param elements: Element objects in transfer order. All elements have the
        same state vector order.
    :type elements: list
    :param input_zero: Indices of the input boundary state components that
        are zero, for example [0, 1] for a fixed beam end with state
        [Y, Theta_z, M_z, Q_y] (Eq. 2.36).
    :type input_zero: list
    :param output_zero: Indices of the output boundary state components that
        are zero.
    :type output_zero: list
    :raises ValueError: If the state orders differ or the boundary
        conditions do not give a square characteristic matrix (Eq. 2.39).
    """

    def __init__(self, elements, input_zero, output_zero):
        self.elements = list(elements)
        sizes = {el.size for el in self.elements}
        if len(sizes) != 1:
            raise ValueError("all elements of a chain must have the same "
                             "state order")
        self.size = sizes.pop()
        self.input_zero = sorted(int(i) for i in input_zero)
        self.output_zero = sorted(int(i) for i in output_zero)
        self.input_unknown = [i for i in range(self.size) if i
                              not in self.input_zero]
        if len(self.output_zero) != len(self.input_unknown):
            raise ValueError("boundary conditions must give a square "
                             "characteristic matrix")

    # ------------------------------------------------------------------
    # Overall transfer equation (Sec. 2.3)
    # ------------------------------------------------------------------
    def overall_matrix(self, s, eta=1.0):
        """Overall transfer matrix U_{1-n} = U_n ... U_2 U_1 (Eq. 2.34).

        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor.
        :type eta: float
        :return: Overall matrix.
        :rtype: numpy.ndarray
        """
        u = np.eye(self.size, dtype=complex)
        for el in self.elements:
            u = el.transfer(s, eta) @ u
        return u

    def overall_extended(self, s, eta=1.0):
        """Overall extended transfer matrix [[U, f], [0, 1]] (Eq. 2.116-2.118).

        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor.
        :type eta: float
        :return: Matrix of order size + 1.
        :rtype: numpy.ndarray
        """
        u = np.eye(self.size + 1, dtype=complex)
        for el in self.elements:
            u = el.extended(s, eta) @ u
        return u

    def characteristic_matrix(self, s, eta=1.0):
        """Matrix of the homogeneous equations U Z_{1,0} = 0 (Eq. 2.39).

        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor.
        :type eta: float
        :return: Square matrix U[output_zero][:, input_unknown].
        :rtype: numpy.ndarray
        """
        u = self.overall_matrix(s, eta)
        return u[np.ix_(self.output_zero, self.input_unknown)]

    def characteristic(self, omega):
        """Real frequency determinant Delta(omega) = det U (Eq. 2.40) of an
        undamped chain.

        :param omega: Circular frequency (rad/s).
        :type omega: float
        :return: Delta(omega).
        :rtype: float
        """
        sign, logabs = self._log_char_real(omega)
        if sign == 0:
            return 0.0
        return float(sign * np.exp(logabs))

    def _log_char_real(self, omega):
        mat = self.characteristic_matrix(1j * omega, 0.0).real
        return _num.log_determinant(mat)

    def _log_char_complex(self, s, eta):
        return _num.log_determinant(self.characteristic_matrix(s, eta))

    # ------------------------------------------------------------------
    # Eigenfrequencies (Sec. 2.5, 10.7)
    # ------------------------------------------------------------------
    def natural_frequencies(self, omega_max, omega_min=0.0, n_grid=2000,
                            max_halvings=4, null_tol=1e-8, char=None):
        """Undamped eigenfrequencies in [omega_min, omega_max] (Sec. 2.5, Eq.
        2.40).

        The search rule is stated in the module docstring. Damping
        parameters are switched off (eta = 0).

        :param omega_max: Upper bound of the search interval (rad/s).
        :type omega_max: float
        :param omega_min: Lower bound (rad/s). If it is 0, rigid-body modes
            are reported as zero frequencies with their multiplicity.
        :type omega_min: float
        :param n_grid: Number of grid points of the first scan.
        :type n_grid: int
        :param max_halvings: Maximum number of grid halvings.
        :type max_halvings: int
        :param null_tol: Singular value ratio for zero and double roots.
        :type null_tol: float
        :param char: Optional callable omega -> (sign, logabs) that replaces
            the plain characteristic determinant (for example the Riccati
            form of ``solvcon.multibody.tmm.riccati.riccati_characteristic``).
        :type char: callable or None
        :return: Sorted eigenfrequencies (rad/s), repeated by multiplicity.
        :rtype: numpy.ndarray
        """
        fun = char if char is not None else self._log_char_real
        roots = []
        lo = omega_min
        if omega_min <= 0.0:
            ratio_sv = np.linalg.svd(self.characteristic_matrix(0.0, 0.0).real,
                                     compute_uv=False)
            scale = ratio_sv[0] if ratio_sv[0] > 0 else 1.0
            n_zero = int(np.sum(ratio_sv <= null_tol * scale))
            roots.extend([0.0] * n_zero)
            lo = omega_max * 1e-9

        def multiplicity(w, w_ref):
            # singular values of the row/column equilibrated matrix: the state
            # components have different units, so raw singular value ratios
            # of a simple root can fall below null_tol (grid plate, Sec. 4.7)
            refs = [self.characteristic_matrix(1j * x, 0.0).real for x
                    in w_ref]
            r, c = _equilibration(sum(np.abs(a) for a in refs))
            ref = max(np.linalg.svd(r[:, None] * a * c[None, :],
                                    compute_uv=False)[0]
                      for a in refs)
            mat = self.characteristic_matrix(1j * w, 0.0).real
            sv = np.linalg.svd(r[:, None] * mat * c[None, :], compute_uv=False)
            return int(np.sum(sv <= null_tol * ref))

        found = find_real_roots(fun, lo, omega_max, n_grid=n_grid,
                                max_halvings=max_halvings,
                                double_check=multiplicity)
        roots.extend(found)
        return np.array(sorted(roots))

    def natural_frequencies_riccati(self, omega_max, omega_min=0.0,
                                    n_grid=2000, max_halvings=4,
                                    null_tol=1e-8):
        """Undamped eigenfrequencies with the Riccati characteristic (Sec.
        10.9).

        Same search rule as ``natural_frequencies``; the characteristic
        determinant is formed by
        ``solvcon.multibody.tmm.riccati.riccati_characteristic``.

        :param omega_max: Upper bound (rad/s).
        :type omega_max: float
        :param omega_min: Lower bound (rad/s).
        :type omega_min: float
        :param n_grid: Number of grid points of the first scan.
        :type n_grid: int
        :param max_halvings: Maximum number of grid halvings.
        :type max_halvings: int
        :param null_tol: Singular value ratio for zero roots.
        :type null_tol: float
        :return: Sorted eigenfrequencies (rad/s).
        :rtype: numpy.ndarray
        """
        def fun(omega):
            sign, logabs = _num.riccati_characteristic(self, 1j * omega, 0.0)
            return (float(np.real(sign)), logabs)

        return self.natural_frequencies(omega_max, omega_min, n_grid,
                                        max_halvings, null_tol, char=fun)

    # ------------------------------------------------------------------
    # State vectors and eigenvectors (Sec. 2.5)
    # ------------------------------------------------------------------
    def boundary_vectors(self, omega, count=1):
        """Nontrivial solutions Z_{1,0} of U Z_{1,0} = 0 at an eigenfrequency
        (Eq. 2.39, 2.47).

        The book takes algebraic cofactors (Eq. h-i of Example 2.6); this
        method takes the right singular vectors of the smallest singular
        values, which gives the same direction for a rank n - 1 matrix and
        also covers multiple roots.

        :param omega: Eigenfrequency (rad/s).
        :type omega: float
        :param count: Number of independent solutions (multiplicity).
        :type count: int
        :return: Array (count, size) of full input boundary state vectors.
        :rtype: numpy.ndarray
        """
        mat = self.characteristic_matrix(1j * omega, 0.0).real
        out = np.zeros((count, self.size))
        if count == 1:
            vec = _pivoted_null_vector(mat)
            out[0, self.input_unknown] = np.real(_normalize(vec))
            return out
        vecs, _ = _null_space(mat, count)
        for j in range(count):
            out[j, self.input_unknown] = np.real(_normalize(vecs[:, j]))
        return out

    def propagate(self, z_in, s, eta=1.0, extended=False):
        """State vectors at the input end of every element (Eq. 2.46,
        2.48-2.49).

        :param z_in: Input boundary state vector Z_{1,0}.
        :type z_in: numpy.ndarray
        :param s: Laplace variable.
        :type s: complex
        :param eta: Damping scale factor.
        :type eta: float
        :param extended: If True, apply the load columns f_i (Eq. 2.114).
        :type extended: bool
        :return: Array (n_elements + 1, size): input of every element, then the
            output boundary state.
        :rtype: numpy.ndarray
        """
        z = np.asarray(z_in, dtype=complex)
        states = [z]
        for el in self.elements:
            z = el.transfer(s, eta) @ z
            if extended:
                z = z + el.forcing(s)
            states.append(z)
        return np.array(states)

    def modes(self, omegas):
        """Mode objects for a list of eigenfrequencies (Sec. 2.5, Eq. 2.50).

        Repeated values in ``omegas`` (multiple roots) give independent
        boundary vectors.

        :param omegas: Eigenfrequencies (rad/s), for example the output of
            ``natural_frequencies``.
        :type omegas: list
        :return: List of Mode.
        :rtype: list
        """
        result = []
        omegas = list(omegas)
        i = 0
        while i < len(omegas):
            w = omegas[i]
            j = i
            while j + 1 < len(omegas) and abs(omegas[j + 1]
                                              - w) <= 1e-9 * max(1.0, abs(w)):
                j += 1
            count = j - i + 1
            for z1 in self.boundary_vectors(w, count):
                states = self.propagate(z1, 1j * w, 0.0).real
                result.append(Mode(self, w, states))
            i = j + 1
        return result

    # ------------------------------------------------------------------
    # Damped eigenvalues (Sec. 2.7)
    # ------------------------------------------------------------------
    def damped_eigenvalues(self, omega_max, n_steps=8, n_grid=2000, tol=1e-12,
                           undamped=None):
        """Complex eigenvalues lambda = -lambda_r + i lambda_i (Eq. 2.40, 2.87,
        2.102).

        The continuation method is stated in the module docstring.

        :param omega_max: Upper bound of the undamped frequencies that start
            the continuation (rad/s).
        :type omega_max: float
        :param n_steps: Number of equal continuation steps in eta.
        :type n_steps: int
        :param n_grid: Grid points of the undamped scan.
        :type n_grid: int
        :param tol: Relative tolerance of the complex secant iteration.
        :type tol: float
        :param undamped: Optional undamped eigenfrequencies (rad/s).
        :type undamped: list or None
        :return: Eigenvalues with positive imaginary part, sorted by it.
            Modes with omega_k = 0 are skipped.
        :rtype: numpy.ndarray
        """
        if undamped is None:
            undamped = self.natural_frequencies(omega_max, n_grid=n_grid)
        starts = [w for w in undamped if w > 0]
        roots = []
        for w in starts:
            lam = None
            for steps in (n_steps, 4 * n_steps):
                lam = self._continue_root(w, steps, tol)
                duplicate = any(abs(lam - r) <= 1e-8 * abs(r) for r in roots)
                if not duplicate:
                    break
            if any(abs(lam - r) <= 1e-8 * abs(r) for r in roots):
                raise RuntimeError("damped eigenvalue continuation from %g "
                                   "reached a root that was already found" % w)
            roots.append(lam)
        return np.array(sorted(roots, key=lambda z: z.imag))

    def _continue_root(self, omega, n_steps, tol):
        """Continuation of the root i omega from eta = 0 to eta = 1 (module
        docstring)."""
        lam = 1j * omega
        eta = 0.0
        step = 1.0 / n_steps
        while eta < 1.0:
            eta_new = min(1.0, eta + step)

            def fun(x, e=eta_new):
                return self._log_char_complex(x, e)

            guess2 = lam * (1.0 + 1e-6) + 1e-6 * max(1.0, abs(lam))
            new, ok = _num.complex_secant_root(fun, lam, guess2, tol=tol)
            # a step that moves the root by more than half its modulus is a
            # jump
            if ok and abs(new - lam) < 0.5 * abs(lam):
                lam = new
                eta = eta_new
            else:
                step *= 0.5
                if step < 1e-6:
                    raise RuntimeError("damped eigenvalue continuation failed "
                                       "near %g" % omega)
        return lam.conjugate() if lam.imag < 0 else lam

    # ------------------------------------------------------------------
    # Steady-state and static response (Sec. 2.8-2.9, 3.8-3.9)
    # ------------------------------------------------------------------
    def steady_state(self, omega, eta=1.0):
        """Steady-state response to harmonic loads of frequency Omega (Eq.
        2.119-2.120, 2.151).

        The element load columns f_i hold complex amplitudes; a real load
        F^c cos(Omega t) + F^s sin(Omega t) has amplitude F^c - i F^s
        (Example 2.14 Eq. a-b). The physical response is
        z(t) = Re(Z e^{i Omega t}) = Z^r cos(Omega t) - Z^i sin(Omega t) (Eq.
        3.201). Omega = 0 gives the static response (Sec. 3.9).

        :param omega: Excitation frequency Omega (rad/s).
        :type omega: float
        :param eta: Damping scale factor.
        :type eta: float
        :return: Array (n_elements + 1, size) of complex state vectors at the
            input end of every element and at the output boundary.
        :rtype: numpy.ndarray
        """
        s = 1j * omega
        u = self.overall_extended(s, eta)
        n = self.size
        rows = self.output_zero
        cols = self.input_unknown
        a = u[np.ix_(rows, cols)]
        rhs = -u[rows, n]
        z1 = np.zeros(n, dtype=complex)
        z1[cols] = np.linalg.solve(a, rhs)
        return self.propagate(z1, s, eta, extended=True)

    def static_response(self):
        """Static response under the element loads (Sec. 3.9, Omega = 0).

        :return: Real array (n_elements + 1, size) of state vectors.
        :rtype: numpy.ndarray
        """
        return self.steady_state(0.0).real


def find_real_roots(fun, lo, hi, n_grid=2000, max_halvings=4,
                    double_check=None, xtol=1e-14, max_iter=200):
    """Real roots of a function given as (sign, logabs) on [lo, hi] (Sec. 2.5,
    10.7).

    Implements steps 2-4 of the search rule in the module docstring.

    :param fun: Callable x -> (sign, logabs).
    :type fun: callable
    :param lo: Lower bound.
    :type lo: float
    :param hi: Upper bound.
    :type hi: float
    :param n_grid: Number of points of the first grid.
    :type n_grid: int
    :param max_halvings: Maximum number of grid halvings.
    :type max_halvings: int
    :param double_check: Optional callable (x, (a, b)) -> int that returns
        the multiplicity of a local minimum x of |D| without a sign change
        (0 rejects it); a and b are the ends of the searched interval.
    :type double_check: callable or None
    :param xtol: Relative tolerance of the refined roots.
    :type xtol: float
    :param max_iter: Maximum iterations per root.
    :type max_iter: int
    :return: Sorted roots (double roots appear twice).
    :rtype: list
    """
    def scan(n):
        xs = np.linspace(lo, hi, n)
        vals = [fun(x) for x in xs]
        return xs, vals

    def brackets(xs, vals):
        out = []
        for i in range(len(xs) - 1):
            sa, sb = np.sign(vals[i][0]), np.sign(vals[i + 1][0])
            if sa == 0:
                out.append((xs[i], xs[i]))
            elif sa * sb < 0:
                out.append((xs[i], xs[i + 1]))
        return out

    n = n_grid
    xs, vals = scan(n)
    count = len(brackets(xs, vals))
    for _ in range(max_halvings):
        n2 = 2 * n - 1
        xs2, vals2 = scan(n2)
        count2 = len(brackets(xs2, vals2))
        xs, vals, n = xs2, vals2, n2
        if count2 == count:
            break
        count = count2

    roots = []
    for a, b in brackets(xs, vals):
        if a == b:
            roots.append(a)
        else:
            roots.append(_refine(fun, a, b, xtol, max_iter))
    # Local minima of |D| without sign change (close pairs or double roots).
    logs = np.array([v[1] if v[0] != 0 else -np.inf for v in vals])
    signs = np.array([np.sign(v[0]) for v in vals])
    for i in range(1, len(xs) - 1):
        if not (logs[i] < logs[i - 1] and logs[i] < logs[i + 1]):
            continue
        if signs[i - 1] != signs[i] or signs[i] != signs[i + 1]:
            continue
        extra = _search_minimum(fun, xs[i - 1], xs[i + 1], signs[i], xtol,
                                max_iter, double_check)
        roots.extend(extra)
    roots = sorted(roots)
    if double_check is not None:
        roots = _merge_multiple(roots, double_check, xs[1] - xs[0])
    return roots


def _merge_multiple(roots, double_check, step, rel=1e-6):
    """Merge roots of one multiple root found twice (step 5 of the search
    rule).

    Rounding can make D change sign next to a double root; the bracket root
    and the local-minimum root (counted with its multiplicity) then describe
    the same root. Neighbouring roots closer than ``rel`` (relative) are one
    multiple root if the nullity at their mean is at least 2; the group is
    replaced by the mean repeated nullity times. A close pair of simple roots
    keeps nullity 1 and is not merged.
    """
    out = []
    i = 0
    while i < len(roots):
        j = i
        while (j + 1 < len(roots)
               and roots[j + 1] - roots[i] <= rel * max(1.0, abs(roots[i]))):
            j += 1
        group = roots[i:j + 1]
        if len(group) > 1:
            xc = float(np.mean(group))
            count = double_check(xc, (xc - step, xc + step))
            if count >= 2:
                group = [xc] * count
        out.extend(group)
        i = j + 1
    return out


def _refine(fun, a, b, xtol, max_iter):
    """Bracketed root by the improved secant step (Eq. 10.98) with Illinois
    modification."""
    fa, fb = fun(a), fun(b)
    side = 0
    c = 0.5 * (a + b)
    for _ in range(max_iter):
        c = _num.improved_secant_step(a, b, fa, fb)
        if b - a <= xtol * max(1.0, abs(c)):
            break
        fc = fun(c)
        if fc[0] == 0:
            return c
        if np.sign(fc[0]) == np.sign(fa[0]):
            a, fa = c, fc
            if side == -1:
                fb = (fb[0], fb[1] - np.log(2.0))
            side = -1
        else:
            b, fb = c, fc
            if side == 1:
                fa = (fa[0], fa[1] - np.log(2.0))
            side = 1
        if b - a <= xtol * max(1.0, abs(c)):
            break
    return 0.5 * (a + b) if b - a > 0 else c


def _search_minimum(fun, a, b, sign_ref, xtol, max_iter, double_check):
    """Golden-section search of log|D| on [a, b] for hidden root pairs."""
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a0, b0 = a, b
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = fun(c), fun(d)
    for _ in range(max_iter):
        for x, fx in ((c, fc), (d, fd)):
            if fx[0] == 0:
                return [x, x]
            if np.sign(fx[0]) != sign_ref:
                r1 = _refine(fun, a, x, xtol, max_iter)
                r2 = _refine(fun, x, b, xtol, max_iter)
                return [r1, r2]
        if fc[1] < fd[1]:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = fun(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = fun(d)
        if b - a <= 1e3 * xtol * max(1.0, abs(c)):
            break
    x = 0.5 * (a + b)
    if double_check is not None:
        count = double_check(x, (a0, b0))
        return [x] * count if count >= 2 else []
    return []


class Mode:
    """Eigenfrequency and state vectors of one mode (Sec. 2.5, Eq. 2.46-2.50).

    :ivar omega: Eigenfrequency (rad/s).
    :vartype omega: float
    :ivar states: Array (n_elements + 1, size): state vector at the input end
        of every element and at the output boundary.
    :vartype states: numpy.ndarray
    """

    def __init__(self, chain, omega, states):
        """Store the mode.

        :param chain: Owner chain.
        :type chain: Chain
        :param omega: Eigenfrequency (rad/s).
        :type omega: float
        :param states: State vectors, see the class docstring.
        :type states: numpy.ndarray
        """
        self.chain = chain
        self.omega = float(omega)
        self.states = np.asarray(states)

    def state_at(self, index, x):
        """State vector inside element ``index`` at distance x from its input
        (Eq. 2.e-f, Ex. 2.4).

        :param index: Element index (0-based, transfer order).
        :type index: int
        :param x: Distance from the element input end (0 <= x <= l).
        :type x: float or numpy.ndarray
        :return: State vector(s).
        :rtype: numpy.ndarray
        """
        el = self.chain.elements[index]
        z_in = self.states[index]
        if not el.continuous:
            return z_in
        s = 1j * self.omega
        xs = np.atleast_1d(x)
        out = np.array([np.real(el.transfer_at(xi, s,
                                               0.0) @ z_in) for xi in xs])
        return out[0] if np.ndim(x) == 0 else out


def _element_inner(el, z_a, z_b, s_a, s_b, kind):
    """Inner product of Eq. (3.48) for one element and two modes."""
    if el.continuous:
        if kind == "mass":
            w = el.mass_density()
        elif kind == "stiffness":
            w = el.stiffness_density()
        else:
            w = el.damping_density()
        if w is None:
            return 0.0
        n_q = max(el.quadrature_points(s_a), el.quadrature_points(s_b))
        xs, ws = _quadrature(n_q, el.length)
        total = 0.0
        for x, wq in zip(xs, ws):
            za = np.real(el.transfer_at(x, s_a, 0.0) @ z_a)
            zb = np.real(el.transfer_at(x, s_b, 0.0) @ z_b)
            total += wq * (za @ w @ zb)
        return total
    if kind == "mass":
        w = el.mass_weight()
    elif kind == "stiffness":
        w = el.stiffness_weight()
    else:
        w = el.damping_weight()
    if w is None:
        return 0.0
    return float(np.real(z_a) @ w @ np.real(z_b))


class ModalModel:
    """Augmented eigenvectors and modal superposition (Ch. 3).

    The augmented eigenvector V^k (Eq. 3.39) collects the displacement
    components of every body element of mode k. Here it is represented by
    the state vectors of the ``Mode`` objects; the inner products
    <M V^k, V^p>, <K V^k, V^p> and <C V^k, V^p> (Eq. 3.48, 3.137) are sums of
    element contributions defined by the element weights (Eq. 3.54).

    :param chain: Chain.
    :type chain: Chain
    :param modes: Modes with positive eigenfrequency.
    :type modes: list
    :ivar modal_mass: M_p of Eq. (3.48).
    :vartype modal_mass: numpy.ndarray
    :ivar omegas: Eigenfrequencies omega_p.
    :vartype omegas: numpy.ndarray
    """

    def __init__(self, chain, modes):
        self.chain = chain
        self.modes = list(modes)
        self.omegas = np.array([m.omega for m in self.modes])
        self.modal_mass = np.array([self.inner(m, m,
                                               "mass") for m in self.modes])

    def inner(self, mode_a, mode_b, kind="mass"):
        """Inner product <M V^a, V^b>, <K V^a, V^b> or <C V^a, V^b> (Eq. 3.48,
        3.54, 3.137).

        :param mode_a: First mode.
        :type mode_a: Mode
        :param mode_b: Second mode.
        :type mode_b: Mode
        :param kind: "mass", "stiffness" or "damping".
        :type kind: str
        :return: Inner product.
        :rtype: float
        """
        s_a = 1j * mode_a.omega
        s_b = 1j * mode_b.omega
        total = 0.0
        for i, el in enumerate(self.chain.elements):
            total += _element_inner(el, mode_a.states[i], mode_b.states[i],
                                    s_a, s_b, kind)
        return total

    def matrix(self, kind="mass"):
        """Matrix of inner products [<X V^k, V^p>] for X = M, K or C (Eq. 3.48,
        3.140).

        :param kind: "mass", "stiffness" or "damping".
        :type kind: str
        :return: Symmetric matrix of order n_modes.
        :rtype: numpy.ndarray
        """
        n = len(self.modes)
        out = np.zeros((n, n))
        for k in range(n):
            for p in range(k, n):
                out[k, p] = out[p, k] = self.inner(self.modes[k],
                                                   self.modes[p], kind)
        return out

    def project(self, field, kind="mass"):
        """Projections <M v, V^p> (kind "mass") or <f, V^p> (kind "load") (Eq.
        3.94, 3.97).

        :param field: Mapping element index -> vector in state layout (only
            the displacement components are used) or callable x -> vector
            for continuous elements. For "mass" the vector is a displacement
            field; for "load" it is a force (per unit length on continuous
            elements).
        :type field: dict
        :param kind: "mass" or "load".
        :type kind: str
        :return: Array of projections, one per mode.
        :rtype: numpy.ndarray
        """
        out = np.zeros(len(self.modes))
        for p, mode in enumerate(self.modes):
            s = 1j * mode.omega
            total = 0.0
            for idx, value in field.items():
                el = self.chain.elements[idx]
                z_in = mode.states[idx]
                if el.continuous:
                    w = (el.mass_density() if kind == "mass"
                         else _load_weight(el))
                    xs, ws = _quadrature(el.quadrature_points(s), el.length)
                    for x, wq in zip(xs, ws):
                        zp = np.real(el.transfer_at(x, s, 0.0) @ z_in)
                        v = (value(x) if callable(value)
                             else np.asarray(value, dtype=float))
                        total += wq * (v @ w @ zp)
                else:
                    w = (el.mass_weight() if kind == "mass"
                         else _load_weight(el))
                    total += np.asarray(value, dtype=float) @ w @ np.real(z_in)
            out[p] = total
        return out

    def initial_conditions(self, v0=None, vdot0=None):
        """Modal initial conditions q_0^p and qdot_0^p (Eq. 3.95-3.98).

        :param v0: Initial displacement field (see ``project``).
        :type v0: dict or None
        :param vdot0: Initial velocity field.
        :type vdot0: dict or None
        :return: (q0, qdot0).
        :rtype: tuple
        """
        q0 = (self.project(v0, "mass") / self.modal_mass if v0
              else np.zeros(len(self.modes)))
        qd0 = (self.project(vdot0, "mass") / self.modal_mass if vdot0
               else np.zeros(len(self.modes)))
        return q0, qd0

    def response(self, t, v0=None, vdot0=None, loads=None, zeta=None):
        """Modal coordinates q_p(t) (Eq. 3.94, 3.99, 3.145, 3.156).

        The free part is the closed form of Eq. (3.156) (Eq. 3.99 for
        zeta = 0). The Duhamel integral of Eq. (3.99) / (3.156) is evaluated
        exactly for a load that is piecewise linear between the points of
        ``t`` (exact recurrence for the linear interpolant).

        :param t: Increasing time points starting at 0.
        :type t: numpy.ndarray
        :param v0: Initial displacement field.
        :type v0: dict or None
        :param vdot0: Initial velocity field.
        :type vdot0: dict or None
        :param loads: List of (field, g) with a load field (see ``project``,
            kind "load") and a callable g(t); f = sum field * g(t).
        :type loads: list or None
        :param zeta: Modal damping ratios zeta_p (Eq. 3.144); None means 0.
        :type zeta: numpy.ndarray or None
        :return: Array (n_modes, len(t)).
        :rtype: numpy.ndarray
        """
        t = np.asarray(t, dtype=float)
        n = len(self.modes)
        z = np.zeros(n) if zeta is None else np.asarray(zeta, dtype=float)
        q0, qd0 = self.initial_conditions(v0, vdot0)
        w = self.omegas
        wd = w * np.sqrt(1.0 - z**2)
        q = np.zeros((n, len(t)))
        for p in range(n):
            e = np.exp(-z[p] * w[p] * t)
            q[p] = e * (q0[p] * np.cos(wd[p] * t)
                        + (qd0[p] + z[p] * w[p] * q0[p]) / wd[p]
                        * np.sin(wd[p] * t))
        if loads:
            fp = np.zeros((n, len(t)))
            for field, g in loads:
                proj = self.project(field, "load")
                gt = np.array([g(ti) for ti in t])
                fp += np.outer(proj, gt)
            for p in range(n):
                q[p] += _duhamel_linear(fp[p] / self.modal_mass[p], t, w[p],
                                        z[p])
        return q

    def displacement(self, q, index, component, x=None):
        """Physical displacement from modal coordinates, v = sum V^p q_p (Eq.
        3.91, 3.100).

        :param q: Modal coordinates (n_modes, n_t).
        :type q: numpy.ndarray
        :param index: Element index.
        :type index: int
        :param component: State vector component index.
        :type component: int
        :param x: Position inside a continuous element (None: input end).
        :type x: float or None
        :return: Time history.
        :rtype: numpy.ndarray
        """
        shape = np.zeros(len(self.modes))
        for p, mode in enumerate(self.modes):
            if x is None:
                shape[p] = mode.states[index][component]
            else:
                shape[p] = mode.state_at(index, x)[component]
        return shape @ q

    def steady_state(self, omega, loads, zeta=None):
        """Modal steady-state response amplitudes (Eq. 3.176-3.181).

        :param omega: Excitation frequency Omega.
        :type omega: float
        :param loads: Load fields with complex amplitudes (see ``project``).
        :type loads: dict
        :param zeta: Modal damping ratios.
        :type zeta: numpy.ndarray or None
        :return: Complex modal amplitudes Q_p with q_p = Re(Q_p e^{i Omega t}).
        :rtype: numpy.ndarray
        """
        n = len(self.modes)
        z = np.zeros(n) if zeta is None else np.asarray(zeta, dtype=float)
        fre = {k: np.real(v) for k, v in loads.items()}
        fim = {k: np.imag(v) for k, v in loads.items()}
        proj = self.project(fre, "load") + 1j * self.project(fim, "load")
        w = self.omegas
        return proj / (self.modal_mass
                       * (w**2 - omega**2 + 2j * z * w * omega))


def _load_weight(el):
    """Identity weight on the displacement components (inner product <f, V>,
    Eq. 3.94)."""
    w = np.zeros((el.size, el.size))
    for i in el.displacement_index():
        w[i, i] = 1.0
    return w


def _duhamel_linear(f, t, w, z):
    """Response of q'' + 2 z w q' + w^2 q = f(t), zero initial state, f
    piecewise linear.

    Exact recurrence for a linearly interpolated load (Duhamel integral of
    Eq. 3.99 / 3.156 evaluated in closed form on every interval). The
    coefficients follow from the exact solution of the ODE with a linear
    right-hand side on each step.
    """
    n = len(t)
    q = np.zeros(n)
    v = np.zeros(n)
    k = w * w
    wd = w * np.sqrt(1.0 - z * z)
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        f0 = f[i]
        df = (f[i + 1] - f[i]) / h
        # particular solution q_p(tau) = a0 + a1 tau for f0 + df tau
        a1 = df / k
        a0 = (f0 - 2.0 * z * w * a1) / k
        c1 = q[i] - a0
        c2 = (v[i] - a1 + z * w * c1) / wd
        e = np.exp(-z * w * h)
        cs, sn = np.cos(wd * h), np.sin(wd * h)
        q[i + 1] = e * (c1 * cs + c2 * sn) + a0 + a1 * h
        v[i + 1] = (e * ((-z * w * c1 + wd * c2) * cs
                         + (-z * w * c2 - wd * c1) * sn) + a1)
    return q

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
