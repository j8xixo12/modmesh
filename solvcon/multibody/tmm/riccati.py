# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Transfer matrix derivation utilities, numerics and the Riccati TMM (book Ch.
10).

Contents:

* Derivation of transfer matrices (Sec. 10.3-10.5): from a fundamental
  solution matrix (Eq. 10.65), from n first-order ODEs by the matrix
  exponential (Eq. 10.69-10.70) or by the Cayley-Hamilton theorem
  (Eq. 10.73-10.74), extended matrices for constant loads (Eq. 10.80-10.81),
  and from a stiffness matrix (Eq. 10.84).
* Computational methods (Sec. 10.6-10.8): real form of a complex transfer
  matrix (Eq. 10.86-10.91), scaled determinant and improved secant step
  against overflow (Eq. 10.96-10.104), chessboard inverse (Eq. 10.111) and
  the symmetry criterion (Eq. 10.114-10.116).
* Riccati TMM (Sec. 10.9): Riccati recursion (Eq. 10.124-10.133), the
  characteristic equation (Eq. 10.134), and the back substitution
  (Eq. 10.135).

This module depends on numpy only. It does not import
``solvcon.multibody.tmm.linear``; the Riccati functions take duck-typed chain
objects (``elements``, ``size``, ``input_zero``, ``output_zero``) so that
``linear`` can import this module.
"""

import numpy as np

# Pade(13) coefficients and theta_13 of the scaling and squaring algorithm
# (Higham 2005). Used by expm.
_PADE13 = (
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
)
_THETA13 = 5.371920351148152


def expm(a):
    """Matrix exponential by scaling and squaring with a Pade(13) approximant.

    Evaluates the series of Eq. (10.70), U(s) = e^{As}, in closed form. The
    caller passes the product A*s.

    :param a: Square matrix (real or complex).
    :type a: numpy.ndarray
    :return: e^a.
    :rtype: numpy.ndarray
    """
    a = np.asarray(a)
    if a.dtype.kind not in "fc":
        a = a.astype(float)
    n = a.shape[0]
    norm1 = np.linalg.norm(a, 1)
    squarings = 0
    if norm1 > _THETA13:
        squarings = int(np.ceil(np.log2(norm1 / _THETA13)))
    a = a / (2.0**squarings)
    b = _PADE13
    ident = np.eye(n, dtype=a.dtype)
    a2 = a @ a
    a4 = a2 @ a2
    a6 = a4 @ a2
    u = a @ (a6 @ (b[13] * a6 + b[11] * a4 + b[9] * a2) + b[7] * a6
             + b[5] * a4 + b[3] * a2
             + b[1] * ident)
    v = a6 @ (b[12] * a6 + b[10] * a4
              + b[8] * a2) + b[6] * a6 + b[4] * a4 + b[2] * a2 \
        + b[0] * ident
    r = np.linalg.solve(v - u, v + u)
    for _ in range(squarings):
        r = r @ r
    return r


def transfer_from_first_order(a, x):
    """Transfer matrix of dZ/dx = A Z over length x (Eq. 10.68-10.69).

    :param a: System matrix A.
    :type a: numpy.ndarray
    :param x: Length (transfer coordinate).
    :type x: float
    :return: U(x) = e^{A x}.
    :rtype: numpy.ndarray
    """
    return expm(np.asarray(a) * x)


def cayley_hamilton_transfer(a, x):
    """Transfer matrix e^{A x} by the Cayley-Hamilton theorem (Eq.
    10.71-10.74).

    The coefficients C_0 ... C_{n-1} solve the Vandermonde system
    e^{a_i x} = sum_k C_k a_i^k for the eigenvalues a_i of A. The method
    requires distinct eigenvalues (stated in the book before Eq. 10.74).

    :param a: System matrix A with distinct eigenvalues.
    :type a: numpy.ndarray
    :param x: Length.
    :type x: float
    :return: U(x) (complex; take the real part for a real A).
    :rtype: numpy.ndarray
    :raises ValueError: If two eigenvalues coincide to working precision.
    """
    a = np.asarray(a, dtype=complex)
    n = a.shape[0]
    eig = np.linalg.eigvals(a)
    scale = max(1.0, np.max(np.abs(eig)))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(eig[i] - eig[j]) < 1e-10 * scale:
                raise ValueError("Cayley-Hamilton form needs distinct "
                                 "eigenvalues")
    vander = np.vander(eig, n, increasing=True)
    coef = np.linalg.solve(vander, np.exp(eig * x))
    result = np.zeros((n, n), dtype=complex)
    power = np.eye(n, dtype=complex)
    for k in range(n):
        result += coef[k] * power
        power = power @ a
    return result


def transfer_from_fundamental(b_of_x, x):
    """Transfer matrix U(x) = B(x) B(0)^{-1} from a fundamental matrix (Eq.
    10.63-10.65).

    :param b_of_x: Callable that returns the fundamental solution matrix B(x).
    :type b_of_x: callable
    :param x: Coordinate of the output end.
    :type x: float
    :return: U(x).
    :rtype: numpy.ndarray
    """
    return np.linalg.solve(b_of_x(0.0).T, b_of_x(x).T).T


def extended_from_first_order(a, f, x):
    """Extended transfer matrix of dZ/dx = A Z + F for a constant F (Eq.
    10.78-10.81).

    The load column f = int_0^x U(x - s) F ds (Eq. 10.80) is evaluated exactly
    with the exponential of the augmented matrix [[A, F], [0, 0]].

    :param a: System matrix A (n x n).
    :type a: numpy.ndarray
    :param f: Constant load column F (n).
    :type f: numpy.ndarray
    :param x: Length.
    :type x: float
    :return: Extended matrix [[U, f], [0, 1]] of order n + 1.
    :rtype: numpy.ndarray
    """
    a = np.asarray(a)
    f = np.asarray(f)
    n = a.shape[0]
    dtype = np.result_type(a, f, float)
    aug = np.zeros((n + 1, n + 1), dtype=dtype)
    aug[:n, :n] = a
    aug[:n, n] = f
    return expm(aug * x)


def transfer_from_stiffness(k, n_disp):
    """Transfer matrix from a stiffness matrix (Eq. 10.82-10.84).

    The stiffness relation is [f_I; f_O] = [[A, B], [C, D]] [d_I; d_O] and the
    state vector is [d; f]. Then U = [[-B^{-1} A, B^{-1}], [C - D B^{-1} A, D
    B^{-1}]].

    :param k: Stiffness matrix of order 2 * n_disp.
    :type k: numpy.ndarray
    :param n_disp: Number of displacement variables at one end.
    :type n_disp: int
    :return: Transfer matrix of order 2 * n_disp.
    :rtype: numpy.ndarray
    """
    k = np.asarray(k)
    n = n_disp
    blk_a = k[:n, :n]
    blk_b = k[:n, n:]
    blk_c = k[n:, :n]
    blk_d = k[n:, n:]
    b_inv = np.linalg.inv(blk_b)
    top = np.hstack([-b_inv @ blk_a, b_inv])
    bottom = np.hstack([blk_c - blk_d @ b_inv @ blk_a, blk_d @ b_inv])
    return np.vstack([top, bottom])


def real_form(u, f=None):
    """Real form of a complex (extended) transfer matrix (Eq. 2.95,
    10.86-10.88).

    :param u: Complex transfer matrix U = U^r + i U^i (n x n).
    :type u: numpy.ndarray
    :param f: Optional complex load column (n). If given, the extended real
        form of order 2n + 1 is returned (Eq. 2.145).
    :type f: numpy.ndarray or None
    :return: [[U^r, -U^i], [U^i, U^r]] or its extended form.
    :rtype: numpy.ndarray
    """
    u = np.asarray(u, dtype=complex)
    n = u.shape[0]
    top = np.hstack([u.real, -u.imag])
    bottom = np.hstack([u.imag, u.real])
    real = np.vstack([top, bottom])
    if f is None:
        return real
    f = np.asarray(f, dtype=complex)
    ext = np.zeros((2 * n + 1, 2 * n + 1))
    ext[:2 * n, :2 * n] = real
    ext[:n, 2 * n] = f.real
    ext[n:2 * n, 2 * n] = f.imag
    ext[2 * n, 2 * n] = 1.0
    return ext


def chessboard_inverse(u):
    """Inverse by the chessboard rule (Eq. 10.105, 10.111).

    The rule changes the sign of every element on a black square (i + j
    odd). It is exact for e^{A s} when A is a black chessboard matrix (for
    example Eq. 10.75), because e^{A(-s)} = (e^{As})^{-1}.

    :param u: Transfer matrix.
    :type u: numpy.ndarray
    :return: Matrix with the signs of the black squares changed.
    :rtype: numpy.ndarray
    """
    u = np.array(u, copy=True)
    n = u.shape[0]
    idx = np.add.outer(np.arange(n), np.arange(n))
    u[idx % 2 == 1] *= -1
    return u


def is_symmetric_transfer(u, rtol=1e-10):
    """General symmetry criterion of a transfer matrix, U U = I (Eq. 10.114).

    :param u: Transfer matrix defined with symmetric state directions.
    :type u: numpy.ndarray
    :param rtol: Relative tolerance on ||U U - I||.
    :type rtol: float
    :return: True if U is an involution.
    :rtype: bool
    """
    u = np.asarray(u)
    err = np.linalg.norm(u @ u - np.eye(u.shape[0]))
    return bool(err <= rtol * max(1.0, np.linalg.norm(u) ** 2))


def log_determinant(mat):
    """Sign (or phase) and log-magnitude of a determinant by LU
    triangularization (Eq. 10.93).

    :param mat: Square matrix (real or complex). An empty matrix has
        determinant 1.
    :type mat: numpy.ndarray
    :return: (sign, logabs). sign is +-1 or 0 (real) or a unit complex number.
    :rtype: tuple
    """
    mat = np.asarray(mat)
    if mat.size == 0:
        return (1.0, 0.0)
    sign, logabs = np.linalg.slogdet(mat)
    return (sign, float(logabs))


def scaled_determinant(mat, k=10.0):
    """Scaled characteristic determinant A = prod sign(d_jj) |d_jj|^{1/k} (Eq.
    10.96-10.97, 10.103).

    The book pairs large and small |d_jj| to avoid overflow. Here the
    product is formed as exp(sum log|d_jj| / k), which has the same value and
    cannot overflow in the intermediate steps.

    :param mat: Square characteristic matrix.
    :type mat: numpy.ndarray
    :param k: Root order k > 1.
    :type k: float
    :return: A (real for a real matrix, complex otherwise). Its sign (phase)
        is the sign (phase) of the determinant.
    :rtype: float or complex
    """
    sign, logabs = log_determinant(mat)
    if sign == 0:
        return 0.0 * sign
    return sign * np.exp(logabs / k)


def determinant_ratio(d_b, d_a):
    """Ratio D(b)/D(a) of two determinants given as (sign, logabs) pairs.

    This is R of Eq. (10.99) with A(b)/A(a) raised to the power k, evaluated
    in log form.

    :param d_b: (sign, logabs) of D(b).
    :type d_b: tuple
    :param d_a: (sign, logabs) of D(a); its sign must be nonzero.
    :type d_a: tuple
    :return: D(b)/D(a).
    :rtype: float or complex
    """
    diff = d_b[1] - d_a[1]
    diff = min(diff, 700.0)
    return (d_b[0] / d_a[0]) * np.exp(diff)


def improved_secant_step(wa, wb, d_a, d_b):
    """Improved secant step w_c = w_a + (w_b - w_a) / (1 - R) (Eq.
    10.98-10.99).

    :param wa: Left abscissa.
    :type wa: float
    :param wb: Right abscissa.
    :type wb: float
    :param d_a: (sign, logabs) of D(w_a).
    :type d_a: tuple
    :param d_b: (sign, logabs) of D(w_b).
    :type d_b: tuple
    :return: w_c. If the step is not finite or leaves (w_a, w_b), the
        midpoint is returned (rule stated after Eq. 10.99).
    :rtype: float
    """
    r = determinant_ratio(d_b, d_a)
    denom = 1.0 - r
    if denom == 0 or not np.isfinite(r):
        return 0.5 * (wa + wb)
    wc = wa + (wb - wa) / denom
    lo, hi = min(wa, wb), max(wa, wb)
    if not np.isfinite(wc) or wc <= lo or wc >= hi:
        return 0.5 * (wa + wb)
    return wc


def complex_secant_root(fun, lam0, lam1, tol=1e-13, max_iter=100):
    """Complex root of D(lambda) = 0 by the secant (tangent) method (Eq.
    10.102-10.104).

    The step lambda* = lambda_1 + h / (1 - D(lambda_1 + h) / D(lambda_1)) uses
    the ratio of determinants in log form, so D can not overflow (Eq. 10.104
    with k > 1 has the same purpose).

    :param fun: Callable returning (phase, logabs) of D(lambda).
    :type fun: callable
    :param lam0: First starting point.
    :type lam0: complex
    :param lam1: Second starting point.
    :type lam1: complex
    :param tol: Relative step tolerance.
    :type tol: float
    :param max_iter: Maximum number of iterations.
    :type max_iter: int
    :return: (root, converged).
    :rtype: tuple
    """
    x0, x1 = complex(lam0), complex(lam1)
    f0, f1 = fun(x0), fun(x1)
    for _ in range(max_iter):
        if f1[0] == 0:
            return x1, True
        if f0[0] == 0:
            return x0, True
        h = x0 - x1
        r = determinant_ratio(f0, f1)
        denom = 1.0 - r
        if denom == 0 or not np.isfinite(r):
            return x1, False
        x2 = x1 + h / denom
        if not np.isfinite(x2):
            return x1, False
        x0, f0 = x1, f1
        x1, f1 = x2, fun(x2)
        if abs(x1 - x0) <= tol * max(1.0, abs(x1)):
            return x1, True
    return x1, False


def _split(u, f, a_idx, b_idx):
    """Partition U and f into T11, T12, T21, T22, f_a, f_b (Eq. 10.124)."""
    t11 = u[np.ix_(a_idx, a_idx)]
    t12 = u[np.ix_(a_idx, b_idx)]
    t21 = u[np.ix_(b_idx, a_idx)]
    t22 = u[np.ix_(b_idx, b_idx)]
    return t11, t12, t21, t22, f[a_idx], f[b_idx]


def riccati_sweep(matrices, loads, a_idx, b_idx):
    """Forward Riccati recursion of S_i and e_i (Eq. 10.125-10.133).

    The recursion starts from S_1 = 0, e_1 = 0 (Eq. 10.133) and applies
    S_{i+1} = (T11 S_i + T12)(T21 S_i + T22)^{-1} (Eq. 10.131) and
    e_{i+1} = T11 e_i + f_a - S_{i+1}(T21 e_i + f_b) (Eq. 10.132).

    :param matrices: Element transfer matrices U_i in transfer order.
    :type matrices: list
    :param loads: Element load columns f_i (Eq. 10.123), same order.
    :type loads: list
    :param a_idx: Indices of the state components that are zero at the input
        end.
    :type a_idx: list
    :param b_idx: Indices of the other components.
    :type b_idx: list
    :return: dict with lists ``S``, ``e`` (length n + 1) and ``G`` (the
        matrices T21 S_i + T22, length n), and ``log_g`` (sum of log|det G_i|),
        ``sign_g`` (product of det signs or phases).
    :rtype: dict
    """
    na = len(a_idx)
    nb = len(b_idx)
    dtype = np.result_type(*[np.asarray(m) for m in matrices], complex)
    s_mat = np.zeros((na, nb), dtype=dtype)
    e_vec = np.zeros(na, dtype=dtype)
    s_list = [s_mat]
    e_list = [e_vec]
    g_list = []
    log_g = 0.0
    sign_g = 1.0
    for u, f in zip(matrices, loads):
        t11, t12, t21, t22, fa, fb = _split(np.asarray(u), np.asarray(f),
                                            a_idx, b_idx)
        g = t21 @ s_mat + t22
        sg, lg = log_determinant(g)
        log_g += lg
        sign_g = sign_g * sg
        # S_{i+1} G = T11 S + T12, solved as G^T S_{i+1}^T = (T11 S + T12)^T
        s_new = np.linalg.solve(g.T, (t11 @ s_mat + t12).T).T
        e_new = t11 @ e_vec + fa - s_new @ (t21 @ e_vec + fb)
        s_mat, e_vec = s_new, e_new
        s_list.append(s_mat)
        e_list.append(e_vec)
        g_list.append(g)
    return {"S": s_list, "e": e_list, "G": g_list, "log_g": log_g,
            "sign_g": sign_g}


def _riccati_partition(chain):
    a_idx = list(chain.input_zero)
    b_idx = [i for i in range(chain.size) if i not in a_idx]
    return a_idx, b_idx


def _output_rows(s_mat, e_vec, a_idx, b_idx, rows, size):
    """Rows of Z_O = W Z_b + E with W = [S; I] in the original state order (Eq.
    10.130)."""
    nb = len(b_idx)
    w = np.zeros((size, nb), dtype=s_mat.dtype)
    e_full = np.zeros(size, dtype=s_mat.dtype)
    w[a_idx, :] = s_mat
    e_full[a_idx] = e_vec
    w[b_idx, :] = np.eye(nb)
    return w[rows, :], e_full[rows], w, e_full


def riccati_characteristic(chain, s, eta=1.0):
    """Riccati characteristic determinant at the Laplace variable s (Eq.
    10.134).

    The Riccati characteristic matrix det(W_c), with W = [S_{n+1}; I] and c
    the output components that are zero, has poles where a matrix
    G_i = T21 S_i + T22 is singular. The function returns
    D = det(W_c) * prod det(G_i), which is free of poles and equals the
    plain TMM characteristic determinant det(U[c][:, b]) in exact
    arithmetic. Each factor is formed without products of large transfer
    matrices, which is the source of the improved stability (Sec. 10.9.3).
    If a G_i is exactly singular at s, D is evaluated at s (1 + 1e-12).

    :param chain: Object with ``elements``, ``size``, ``input_zero``,
        ``output_zero``; each element has ``transfer(s, eta)``.
    :type chain: object
    :param s: Laplace variable (s = i omega for undamped vibration).
    :type s: complex
    :param eta: Damping scale factor passed to the elements.
    :type eta: float
    :return: (sign or phase, logabs) of D.
    :rtype: tuple
    """
    a_idx, b_idx = _riccati_partition(chain)
    loads = [np.zeros(chain.size) for _ in chain.elements]
    for attempt in range(4):
        mats = [el.transfer(s, eta) for el in chain.elements]
        try:
            sweep = riccati_sweep(mats, loads, a_idx, b_idx)
            break
        except np.linalg.LinAlgError:
            # G_i exactly singular (pole of S_{i+1} on the evaluation point):
            # D is continuous, so evaluate it at a point 1e-12 away.
            s = s * (1.0 + 1e-12)
    else:
        raise np.linalg.LinAlgError("Riccati recursion singular")
    wc, _, _, _ = _output_rows(sweep["S"][-1], sweep["e"][-1], a_idx, b_idx,
                               list(chain.output_zero), chain.size)
    sw, lw = log_determinant(wc)
    return (sw * sweep["sign_g"], lw + sweep["log_g"])


def riccati_steady_state(chain, s, loads=None, eta=1.0):
    """Steady-state (or static) response by the Riccati TMM (Eq.
    10.123-10.135).

    The output equations Z_{a,O} = S_{n+1} Z_{b,O} + e_{n+1} (Eq. 10.134) are
    combined with the output boundary condition; then the state vectors are
    recovered backwards with Eq. (10.135).

    :param chain: Chain-like object (see riccati_characteristic). Element
        load columns come from ``forcing(s)`` unless ``loads`` is given.
    :type chain: object
    :param s: Laplace variable (s = i Omega for harmonic excitation).
    :type s: complex
    :param loads: Optional list of load columns f_i.
    :type loads: list or None
    :param eta: Damping scale factor.
    :type eta: float
    :return: dict with ``inputs`` and ``outputs`` (state vectors at the input
        and output end of every element), ``S`` and ``e`` lists.
    :rtype: dict
    """
    a_idx, b_idx = _riccati_partition(chain)
    mats = [np.asarray(el.transfer(s, eta)) for el in chain.elements]
    if loads is None:
        loads = [np.asarray(el.forcing(s)) for el in chain.elements]
    sweep = riccati_sweep(mats, loads, a_idx, b_idx)
    rows = list(chain.output_zero)
    wc, ec, w, e_full = _output_rows(sweep["S"][-1], sweep["e"][-1], a_idx,
                                     b_idx, rows, chain.size)
    zb_out = np.linalg.solve(wc, -ec)
    n = len(mats)
    outputs = [None] * n
    inputs = [None] * n
    z_out = w @ zb_out + e_full
    for i in range(n - 1, -1, -1):
        outputs[i] = z_out
        t11, t12, t21, t22, fa, fb = _split(mats[i], np.asarray(loads[i]),
                                            a_idx, b_idx)
        s_i = sweep["S"][i]
        e_i = sweep["e"][i]
        g = sweep["G"][i]
        zb_in = np.linalg.solve(g, z_out[b_idx] - t21 @ e_i - fb)
        za_in = s_i @ zb_in + e_i
        z_in = np.zeros(chain.size, dtype=z_out.dtype)
        z_in[a_idx] = za_in
        z_in[b_idx] = zb_in
        inputs[i] = z_in
        z_out = z_in
    return {"inputs": inputs, "outputs": outputs, "S": sweep["S"],
            "e": sweep["e"]}

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
