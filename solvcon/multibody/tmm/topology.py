# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Automatic deduction of the overall transfer equation (book Ch. 11).

Two levels are provided.

1. Matrix level (Secs. 11.3-11.6). Given the topology figure of a system
   (Sec. 11.2) and the element matrices, :func:`deduce_tree` builds the
   overall transfer matrix ``U_all`` of a tree system from the main
   transfer equation (coefficient matrix ``T``) and the geometric equations
   (coefficient matrix ``G``), Eqs. (11.19)-(11.27). :func:`apply_cuts`
   adds closed loops with the cutting point sign convention matrix ``C``
   (Eqs. 11.28-11.37). :func:`closed_loop_matrix` is Eq. (11.5).

2. Planar multi-rigid-body systems (Secs. 11.7.2, 11.8.2.3, 11.10.2,
   11.10.3). :class:`PlanarMechanism` deduces the overall transfer equation
   of a planar system of rigid bodies joined by smooth pin hinges, with any
   topology (chain, tree, closed loop, network), automatically from a
   traversal of the topology figure. The element relations are those of
   the DT-TMM (Eqs. 7.101, 7.119-7.137; mode ``"dt"``) or of the
   acceleration form of Ch. 5 (Eqs. 5.14-5.18; mode ``"acc"``).

The traversal starts at the ground hinges. A body is entered at one end;
its angle is removed with the zero moment at its other ends (Eq. 7.218,
Eq. 5.36). A hinge reached a second time gives closure equations
(Eq. 11.59), every free end gives ``q = 0`` and every inner hinge gives the
force balance of Eq. (11.59) or (11.119). The unknowns are the internal
forces where bodies are entered; they form the boundary state vector
``z_all`` of Eq. (11.32).
"""

import numpy as np

from . import dttmm, rigid

# ---------------------------------------------------------------------------
# Matrix level: Secs. 11.3-11.6
# ---------------------------------------------------------------------------


def closed_loop_matrix(mats):
    """Return ``U_all = U_n ... U_1 - I`` of a closed loop (Eqs. 11.1-11.5,
    11.40).

    :param mats: element matrices [U_1, ..., U_n] along the loop
    :type mats: list
    :return: overall transfer matrix of the cut loop
    :rtype: numpy.ndarray
    """
    total = np.eye(mats[0].shape[0])
    for u in mats:
        total = u @ total
    return total - np.eye(total.shape[0])


def cut_sign_matrix(n, extended=True):
    """Return the cutting point sign convention matrix C (Eq. 11.29).

    :param n: 3 for planar motion, 6 for spatial motion
    :type n: int
    :param extended: append the constant 1 of an extended state vector
    :type extended: bool
    :rtype: numpy.ndarray
    """
    size = 2 * n + (1 if extended else 0)
    c = np.eye(size)
    c[n:2 * n, n:2 * n] = -np.eye(n)
    return c


class TopologyFigure:
    """Topology figure of a tree system (Sec. 11.2).

    Every element has an ordered list of inputs and one output. An input
    is either another element id or a boundary label (str). The transfer
    direction goes from the tips to the root.

    :ivar inputs: element id -> list of inputs
    :vartype inputs: dict
    """

    def __init__(self):
        """Initialize an empty figure."""
        self.inputs = {}

    def add(self, element, inputs):
        """Add an element.

        :param element: element id
        :type element: int or str
        :param inputs: ordered inputs (element ids or boundary labels)
        :type inputs: list
        """
        self.inputs[element] = list(inputs)

    def downstream(self):
        """Return element -> (downstream element, input index).

        :rtype: dict
        """
        out = {}
        for e, ins in self.inputs.items():
            for k, src in enumerate(ins):
                if src in self.inputs:
                    out[src] = (e, k)
        return out

    def root(self):
        """Return the root element (the element without downstream element).

        :rtype: int or str
        :raises ValueError: if the figure has not exactly one root
        """
        down = self.downstream()
        roots = [e for e in self.inputs if e not in down]
        if len(roots) != 1:
            raise ValueError("a tree figure has exactly one root element")
        return roots[0]

    def tips(self):
        """Return the boundary labels of the tips in input order.

        :rtype: list
        """
        out = []
        for ins in self.inputs.values():
            for src in ins:
                if src not in self.inputs:
                    out.append(src)
        return out

    def path(self, tip):
        """Return the path [(element, input index), ...] from a tip to the
        root.

        :param tip: boundary label
        :type tip: str
        :rtype: list
        """
        down = self.downstream()
        for e, ins in self.inputs.items():
            if tip in ins:
                path = [(e, ins.index(tip))]
                break
        else:
            raise KeyError(tip)
        while path[-1][0] in down:
            path.append(down[path[-1][0]])
        return path


def deduce_tree(figure, u, u_in=None, h=None, h_in=None):
    """Build the overall transfer equation of a tree system (Secs. 11.4, 11.6).

    The main transfer equation ``-z_root + sum_t T_t z_t = 0`` uses the
    product of the element matrices along the path from tip t to the root;
    a path through a non-first input k of element j uses the force
    extraction matrix ``U_{j,Ik}`` (Eq. 11.6, theorem b). Every non-first
    input k of a multiple-input element j adds the geometric equation
    ``-H_j z_{j,I1} + H_{j,Ik} z_{j,Ik} = 0`` (Eq. 11.7), expressed in the
    tip states (Eqs. 11.13-11.18).

    :param figure: topology figure
    :type figure: TopologyFigure
    :param u: element id -> transfer matrix (first input to output)
    :type u: dict
    :param u_in: (element id, k) -> force extraction matrix, k >= 1
    :type u_in: dict or None
    :param h: element id -> position extraction matrix
    :type h: dict or None
    :param h_in: (element id, k) -> position incidence matrix, k >= 1
    :type h_in: dict or None
    :return: (U_all, labels, T, G) where labels = [root label, tips...],
        T maps tip -> matrix and G is the list of geometric rows (dict
        tip -> matrix per equation)
    :rtype: tuple
    """
    u_in = u_in or {}
    h = h or {}
    h_in = h_in or {}
    tips = figure.tips()
    n = next(iter(u.values())).shape[0]
    paths = {t: figure.path(t) for t in tips}

    def factor(e, k):
        return u[e] if k == 0 else u_in[(e, k)]

    t_mats = {}
    for t, p in paths.items():
        m = np.eye(n)
        for e, k in p:
            m = factor(e, k) @ m
        t_mats[t] = m
    g_rows = []
    for e, ins in figure.inputs.items():
        for k in range(1, len(ins)):
            row = {}
            for t, p in paths.items():
                prefix = np.eye(n)
                for e2, k2 in p:
                    if e2 == e:
                        if k2 == 0:
                            row[t] = -h[e] @ prefix
                        elif k2 == k:
                            row[t] = h_in[(e, k)] @ prefix
                        break
                    prefix = factor(e2, k2) @ prefix
            g_rows.append(row)
    nrow_g = sum(next(iter(r.values())).shape[0] for r in g_rows if r)
    nt = len(tips)
    u_all = np.zeros((n + nrow_g, n * (nt + 1)))
    u_all[0:n, 0:n] = -np.eye(n)
    for i, t in enumerate(tips):
        u_all[0:n, n * (i + 1):n * (i + 2)] = t_mats[t]
    r0 = n
    for row in g_rows:
        if not row:
            continue
        nr = next(iter(row.values())).shape[0]
        for i, t in enumerate(tips):
            if t in row:
                u_all[r0:r0 + nr, n * (i + 1):n * (i + 2)] = row[t]
        r0 += nr
    return u_all, ["root"] + tips, t_mats, g_rows


def apply_cuts(u_all, labels, cuts, c):
    """Merge the boundary states of cutting points (Eqs. 11.28-11.34).

    For a cut pair (a, b) with ``z_b = C z_a`` the columns of ``z_b`` are
    multiplied by C and added to the columns of ``z_a``; ``z_b`` is removed.

    :param u_all: overall transfer matrix of the tree system
    :type u_all: numpy.ndarray
    :param labels: block labels of the columns
    :type labels: list
    :param cuts: list of (a, b) label pairs
    :type cuts: list
    :param c: cutting point sign convention matrix
    :type c: numpy.ndarray
    :return: (U_all, labels) of the general system
    :rtype: tuple
    """
    n = c.shape[0]
    blocks = {lab: u_all[:, n * i:n * (i + 1)].copy() for i, lab
              in enumerate(labels)}
    labels = list(labels)
    for a, b in cuts:
        blocks[a] = blocks[a] + blocks[b] @ c
        labels.remove(b)
    return np.hstack([blocks[lab] for lab in labels]), labels


def solve_boundary(u_all, known):
    """Solve ``U_all z_all = 0`` with known entries of ``z_all`` (Eqs.
    11.113-11.116).

    :param u_all: overall transfer matrix
    :type u_all: numpy.ndarray
    :param known: values of z_all with numpy.nan at the unknown entries
    :type known: array_like
    :return: (z_all, residual norm)
    :rtype: tuple
    """
    known = np.asarray(known, dtype=float)
    unk = np.isnan(known)
    rhs = -u_all[:, ~unk] @ known[~unk]
    x, *_ = np.linalg.lstsq(u_all[:, unk], rhs, rcond=None)
    z = known.copy()
    z[unk] = x
    return z, float(np.linalg.norm(u_all @ z))


def reduced_matrix(u_all, zero_mask):
    """Delete the columns of known zero boundary entries (Eqs. 11.70-11.72).

    :param u_all: overall transfer matrix
    :type u_all: numpy.ndarray
    :param zero_mask: True where the entry of z_all is zero
    :type zero_mask: array_like
    :return: reduced matrix; its determinant is the characteristic function
    :rtype: numpy.ndarray
    """
    return u_all[:, ~np.asarray(zero_mask, dtype=bool)]


# ---------------------------------------------------------------------------
# Affine expressions of the boundary unknowns
# ---------------------------------------------------------------------------


class Affine:
    """Affine expression ``M u + c`` of the unknown vector u.

    :ivar m: coefficient matrix (rows x unknowns known at creation)
    :vartype m: numpy.ndarray
    :ivar c: constant vector
    :vartype c: numpy.ndarray
    """

    def __init__(self, m, c):
        """Initialize the expression.

        :param m: coefficient matrix
        :type m: numpy.ndarray
        :param c: constant vector
        :type c: numpy.ndarray
        """
        self.m = np.atleast_2d(np.asarray(m, dtype=float))
        self.c = np.atleast_1d(np.asarray(c, dtype=float))

    def width(self, nu):
        """Return the coefficient matrix padded to nu columns.

        :param nu: number of unknowns
        :type nu: int
        :rtype: numpy.ndarray
        """
        m = self.m
        if m.shape[1] < nu:
            m = np.hstack([m, np.zeros((m.shape[0], nu - m.shape[1]))])
        return m

    def __add__(self, other):
        """Return the sum.

        :param other: Affine or constant vector
        :rtype: Affine
        """
        if not isinstance(other, Affine):
            return Affine(self.m, self.c + other)
        nu = max(self.m.shape[1], other.m.shape[1])
        return Affine(self.width(nu) + other.width(nu), self.c + other.c)

    def __sub__(self, other):
        """Return the difference.

        :param other: Affine or constant vector
        :rtype: Affine
        """
        return (self + (-1.0) * other if isinstance(other, Affine) else self
                + (-other))

    def __rmul__(self, s):
        """Return the product with a scalar.

        :param s: scalar
        :rtype: Affine
        """
        return Affine(s * self.m, s * self.c)

    def lmul(self, mat):
        """Return ``mat @ self``.

        :param mat: matrix
        :type mat: numpy.ndarray
        :rtype: Affine
        """
        mat = np.atleast_2d(mat)
        return Affine(mat @ self.m, mat @ self.c)

    def outer(self, vec):
        """Return the vector expression ``vec * self`` for a scalar expression.

        :param vec: vector
        :type vec: numpy.ndarray
        :rtype: Affine
        """
        vec = np.asarray(vec, dtype=float).reshape(-1, 1)
        return Affine(vec @ self.m[0:1], vec[:, 0] * self.c[0])

    def value(self, u):
        """Evaluate the expression.

        :param u: unknown vector
        :type u: numpy.ndarray
        :rtype: numpy.ndarray
        """
        return self.width(len(u)) @ u + self.c


class AffineSystem:
    """Unknowns and equations of an overall transfer equation.

    :ivar nu: number of unknowns
    :vartype nu: int
    :ivar rows: equations, list of Affine (each = 0)
    :vartype rows: list
    """

    def __init__(self):
        """Initialize an empty system."""
        self.nu = 0
        self.rows = []

    def unknown(self, k):
        """Return an expression of k new unknowns.

        :param k: number of new unknowns
        :type k: int
        :rtype: Affine
        """
        m = np.zeros((k, self.nu + k))
        m[:, self.nu:] = np.eye(k)
        self.nu += k
        return Affine(m, np.zeros(k))

    @staticmethod
    def const(c):
        """Return a constant expression.

        :param c: constant vector
        :type c: array_like
        :rtype: Affine
        """
        c = np.atleast_1d(np.asarray(c, dtype=float))
        return Affine(np.zeros((c.size, 0)), c)

    def equation(self, expr):
        """Add the equations ``expr = 0``.

        :param expr: expression
        :type expr: Affine
        """
        self.rows.append(expr)

    def matrix(self):
        """Return (U_all, f) of the equations U_all u + f = 0.

        :rtype: tuple
        """
        if not self.rows:
            return np.zeros((0, self.nu)), np.zeros(0)
        return (np.vstack([r.width(self.nu) for r in self.rows]),
                np.concatenate([r.c for r in self.rows]))

    def solve(self, strict=True):
        """Solve the equations (least squares, exact for a consistent system).

        :param strict: raise for a rank deficient system; with False the
            minimum-norm solution is returned (redundant constraint forces)
        :type strict: bool
        :return: (u, residual norm)
        :rtype: tuple
        :raises ValueError: if the system is rank deficient and strict
        """
        a, f = self.matrix()
        if self.nu == 0:
            return np.zeros(0), float(np.linalg.norm(f))
        # Row and column equilibration: unknowns (forces, angles) and
        # equations (moments, positions) have very different scales.
        rs = np.max(np.abs(a), axis=1)
        rs[rs == 0.0] = 1.0
        a_s = a / rs[:, None]
        cs = np.max(np.abs(a_s), axis=0)
        cs[cs == 0.0] = 1.0
        a_s = a_s / cs[None, :]
        if strict and np.linalg.matrix_rank(a_s) < self.nu:
            raise ValueError("overall transfer equation is rank deficient")
        y, *_ = np.linalg.lstsq(a_s, -f / rs, rcond=None)
        u = y / cs
        return u, float(np.linalg.norm((a @ u + f) / rs))


def _cross(a, b):
    """Return the z component of the planar cross product.

    :rtype: float
    """
    return a[0] * b[1] - a[1] * b[0]


def _cross_aff(p, q):
    """Return ``p x q`` for a constant 2-vector p and an Affine q.

    :rtype: Affine
    """
    return q.lmul(np.array([[-p[1], p[0]]]))


# ---------------------------------------------------------------------------
# Planar multi-rigid-body systems with any topology
# ---------------------------------------------------------------------------


class MechanismBody:
    """Planar rigid body with any number of connection ends.

    :ivar mass: mass
    :vartype mass: float
    :ivar j_c: moment of inertia about the mass center
    :vartype j_c: float
    :ivar center: mass center, body frame
    :vartype center: numpy.ndarray
    :ivar ends: connection ends, body frame
    :vartype ends: list
    """

    def __init__(self, mass, j_c, center, ends):
        """Initialize the body.

        :param mass: mass
        :type mass: float
        :param j_c: moment of inertia about C
        :type j_c: float
        :param center: mass center, body frame
        :type center: array_like
        :param ends: end points, body frame
        :type ends: list
        """
        self.mass = float(mass)
        self.j_c = float(j_c)
        self.center = np.asarray(center, dtype=float)
        self.ends = [np.asarray(e, dtype=float) for e in ends]


class PlanarMechanism:
    """Planar system of rigid bodies and smooth pin hinges (Ch. 11).

    Topology elements: bodies (any number of ends), hinges that join body
    ends and optionally the ground, sliders (a body end pinned to a block
    that slides on a fixed ground line), and free ends (ends without a
    hinge). The overall transfer equation is deduced automatically at each
    time instant, see the module docstring.

    :ivar bodies: bodies
    :vartype bodies: list
    :ivar hinges: hinges, dict with keys ends and ground
    :vartype hinges: list
    :ivar sliders: sliders, dict with keys end, point, direction, mass
    :vartype sliders: list
    :ivar gravity: gravity acceleration
    :vartype gravity: numpy.ndarray
    """

    def __init__(self, gravity=(0.0, -9.81)):
        """Initialize an empty system.

        :param gravity: gravity acceleration
        :type gravity: array_like
        """
        self.bodies = []
        self.hinges = []
        self.sliders = []
        self.gravity = np.asarray(gravity, dtype=float)

    def add_body(self, mass, j_c, center, ends):
        """Add a body and return its index.

        :param mass: mass
        :type mass: float
        :param j_c: moment of inertia about C
        :type j_c: float
        :param center: mass center, body frame
        :type center: array_like
        :param ends: end points, body frame
        :type ends: list
        :rtype: int
        """
        self.bodies.append(MechanismBody(mass, j_c, center, ends))
        return len(self.bodies) - 1

    def add_bar(self, mass, length, j_c=None):
        """Add a bar from (0, 0) to (length, 0) with C at mid length.

        :param mass: mass
        :type mass: float
        :param length: length
        :type length: float
        :param j_c: moment of inertia about C (default m l^2 / 12)
        :type j_c: float or None
        :rtype: int
        """
        if j_c is None:
            j_c = mass * length ** 2 / 12.0
        return self.add_body(mass, j_c, [0.5 * length, 0.0],
                             [[0.0, 0.0], [length, 0.0]])

    def add_hinge(self, ends, ground=None):
        """Add a smooth pin hinge and return its index.

        :param ends: list of (body index, end index)
        :type ends: list
        :param ground: ground point if the hinge is fixed to the ground
        :type ground: array_like or None
        :rtype: int
        """
        g = None if ground is None else np.asarray(ground, dtype=float)
        self.hinges.append({"ends": [tuple(e) for e in ends], "ground": g})
        return len(self.hinges) - 1

    def add_slider(self, end, point, direction, mass=0.0):
        """Add a slider block on a ground line, pinned to a body end.

        :param end: (body index, end index)
        :type end: tuple
        :param point: point of the ground line
        :type point: array_like
        :param direction: direction of the ground line
        :type direction: array_like
        :param mass: mass of the block
        :type mass: float
        :rtype: int
        """
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        self.sliders.append({"end": tuple(end),
                             "point": np.asarray(point, dtype=float),
                             "direction": d, "mass": float(mass)})
        return len(self.sliders) - 1

    # -- traversal of the topology figure ---------------------------------

    def _connections(self):
        """Return (body, end) -> ('hinge', i) or ('slider', i).

        :rtype: dict
        """
        conn = {}
        for i, h in enumerate(self.hinges):
            for e in h["ends"]:
                conn[e] = ("hinge", i)
        for i, s in enumerate(self.sliders):
            conn[s["end"]] = ("slider", i)
        return conn

    def plan(self):
        """Return the traversal of the topology figure.

        :return: list of (body, entry end, hinge index) in entry order
        :rtype: list
        :raises ValueError: if a body cannot be reached from the ground
        """
        conn = self._connections()
        order, seen, reached = [], set(), set()
        queue = [i for i, h in enumerate(self.hinges)
                 if h["ground"] is not None]
        reached.update(queue)
        while queue:
            hi = queue.pop(0)
            for b, e in self.hinges[hi]["ends"]:
                if b in seen:
                    continue
                seen.add(b)
                order.append((b, e, hi))
                for j in range(len(self.bodies[b].ends)):
                    c = conn.get((b, j))
                    if (j != e and c is not None and c[0] == "hinge"
                            and c[1] not in reached):
                        reached.add(c[1])
                        queue.append(c[1])
        if len(seen) != len(self.bodies):
            raise ValueError(
                "every body must be reachable from a ground hinge")
        return order

    # -- element relations -------------------------------------------------

    def _coefficients(self, b, e, mode, ctx):
        """Return the linear relations of body b entered at end e.

        :param b: body index
        :type b: int
        :param e: entry end
        :type e: int
        :param mode: 'dt' or 'acc'
        :type mode: str
        :param ctx: per-step context
        :type ctx: dict
        :return: dict with the coefficients used by :meth:`_body`
        :rtype: dict
        """
        body = self.bodies[b]
        origin = body.ends[e]
        pts = {"C": body.center - origin}
        for j, p in enumerate(body.ends):
            pts[j] = p - origin
        j_i = body.j_c + body.mass * float(pts["C"] @ pts["C"])
        out = {"j_i": j_i, "geo": {}, "bar": {}}
        if mode == "dt":
            st = ctx["states"][b]
            lin = ctx["lin"]
            a_c, b_th, _, _ = lin.coefficients(st.theta)
            _, b_ri, _, _ = lin.coefficients(st.r_i)
            _, b_rc, _, _ = lin.coefficients(st.r_c)
            for key, p in pts.items():
                g_th, g_0, p_bar, _ = dttmm._planar_geometry(p, st.theta,
                                                             lin.dt)
                out["geo"][key] = (g_th, g_0)
                out["bar"][key] = p_bar
            g_th, g_0 = out["geo"]["C"]
            out["acc_c"] = (a_c, a_c * g_th, a_c * g_0 + b_rc)
            out["acc_i"] = (a_c, b_ri)
            out["alpha"] = (a_c, float(b_th))
        else:
            th, w = ctx["theta"][b], ctx["omega"][b]
            r = rigid.rot2(th)
            for key, p in pts.items():
                rp = r @ p
                out["geo"][key] = (np.array([-rp[1], rp[0]]), -w * w * rp)
                out["bar"][key] = rp
            g_th, g_0 = out["geo"]["C"]
            out["acc_c"] = (1.0, g_th, g_0)
            out["acc_i"] = (1.0, np.zeros(2))
            out["alpha"] = (1.0, 0.0)
        return out

    def _body(self, sysm, b, e, p_in, q_in, outs, mode, ctx):
        """Add the relations of body b and return its unknown-free results.

        Generalizes Eqs. (7.94)-(7.98) and (7.119)-(7.121) (mode 'dt') and
        Eqs. (5.14)-(5.16) (mode 'acc') to one input end and any number of
        output ends with zero moments. The first output takes the force
        balance; every further output force is a new unknown. The angle is
        removed with the moment equation (Eq. 7.218).

        :param sysm: overall equation under construction
        :type sysm: AffineSystem
        :param b: body index
        :type b: int
        :param e: entry end
        :type e: int
        :param p_in: position (or acceleration) of the entry end
        :type p_in: Affine
        :param q_in: internal force at the entry end
        :type q_in: Affine
        :param outs: output end indices
        :type outs: list
        :param mode: 'dt' or 'acc'
        :type mode: str
        :param ctx: per-step context
        :type ctx: dict
        :return: (theta, {end: (p, q)}, coefficients)
        :rtype: tuple
        """
        body = self.bodies[b]
        cf = self._coefficients(b, e, mode, ctx)
        m = body.mass
        f = m * self.gravity
        ap, ath, a0 = cf["acc_c"]
        bp, b0 = cf["acc_i"]
        gam, gam0 = cf["alpha"]
        pc = cf["bar"]["C"]
        extra = {j: sysm.unknown(2) for j in outs[1:]}
        # Force balance: q_O1 = q_I + f - m rdd_C - sum_extra (theta part kept
        # apart).
        q1_rest = q_in + (f - m * a0) - (m * ap) * p_in
        for j in outs[1:]:
            q1_rest = q1_rest - extra[j]
        q1_th = -m * ath
        p1 = cf["bar"][outs[0]] if outs else np.zeros(2)
        # Moment about I (zero end moments):
        # 0 = J (gam th + gam0) + m pc x (bp p_I + b0) - pc x f + sum_j p_j x
        # q_j.
        coef = cf["j_i"] * gam + _cross(p1, q1_th)
        rest = (_cross_aff(m * pc, bp * p_in)
                + (cf["j_i"] * gam0 + _cross(m * pc, b0) - _cross(pc, f)))
        if outs:
            rest = rest + _cross_aff(p1, q1_rest)
        for j in outs[1:]:
            rest = rest + _cross_aff(cf["bar"][j], extra[j])
        theta = (-1.0 / coef) * rest
        if not outs:
            # A body with one end: the force balance closes on the entry end.
            sysm.equation(q1_rest + theta.outer(q1_th))
        results = {}
        for idx, j in enumerate(outs):
            g_th, g_0 = cf["geo"][j]
            p_j = p_in + theta.outer(g_th) + g_0
            q_j = q1_rest + theta.outer(q1_th) if idx == 0 else extra[j]
            results[j] = (p_j, q_j)
        return theta, results, cf

    def deduce(self, mode, ctx):
        """Deduce and solve the overall transfer equation at one instant.

        :param mode: 'dt' (positions at t_i) or 'acc' (accelerations at t)
        :type mode: str
        :param ctx: per-step context (states and linearizer, or angles)
        :type ctx: dict
        :return: dict with u, residual, theta (per body), p_entry (per body),
            ends ((body, end) -> (p, q)), sliders (position values) and the
            system (AffineSystem)
        :rtype: dict
        """
        conn = self._connections()
        sysm = AffineSystem()
        node_p = {}
        node_in = {}
        node_out = {}
        for i, h in enumerate(self.hinges):
            if h["ground"] is not None:
                g = h["ground"] if mode == "dt" else np.zeros(2)
                node_p[i] = AffineSystem.const(g)
        entered = {}
        order = self.plan()
        pending = {}
        res_theta, res_entry, res_ends, slider_p = {}, {}, {}, {}
        for b, e, hi in order:
            h = self.hinges[hi]
            others = [x for x in h["ends"] if x[0] not in entered and x[0]
                      != b]
            n_in = len(node_in.get(hi, []))
            simple = (h["ground"] is None and len(h["ends"]) == 2 and n_in
                      == 1 and not others)
            q_in = node_in[hi][0] if simple else sysm.unknown(2)
            if simple:
                pending[hi] = "chain"
            else:
                node_out.setdefault(hi, []).append(q_in)
            entered[b] = e
            outs = [j for j in range(len(self.bodies[b].ends)) if j != e]
            theta, results, _ = self._body(sysm, b, e, node_p[hi], q_in, outs,
                                           mode, ctx)
            res_theta[b] = theta
            res_entry[b] = node_p[hi]
            res_ends[(b, e)] = (node_p[hi], q_in)
            for j, (p_j, q_j) in results.items():
                res_ends[(b, j)] = (p_j, q_j)
                c = conn.get((b, j))
                if c is None:
                    sysm.equation(q_j)
                elif c[0] == "slider":
                    self._slider(sysm, c[1], p_j, q_j, mode, ctx)
                    slider_p[c[1]] = p_j
                else:
                    h2 = c[1]
                    if h2 in node_p:
                        if self.hinges[h2]["ground"] is not None:
                            sysm.equation(p_j - node_p[h2])
                        else:
                            sysm.equation(p_j - node_p[h2])
                            node_in.setdefault(h2, []).append(q_j)
                    else:
                        node_p[h2] = p_j
                        node_in.setdefault(h2, []).append(q_j)
        for hi, h in enumerate(self.hinges):
            if h["ground"] is not None or pending.get(hi) == "chain":
                continue
            bal = None
            for q in node_in.get(hi, []):
                bal = q if bal is None else bal + q
            for q in node_out.get(hi, []):
                bal = (-1.0) * q if bal is None else bal - q
            if bal is not None:
                sysm.equation(bal)
        u, resid = sysm.solve()
        return {"u": u, "residual": resid, "system": sysm,
                "theta": {b: float(t.value(u)[0])
                          for b, t in res_theta.items()},
                "p_entry": {b: p.value(u) for b, p in res_entry.items()},
                "ends": {k: (p.value(u), q.value(u))
                         for k, (p, q) in res_ends.items()},
                "sliders": {k: p.value(u) for k, p in slider_p.items()},
                "entry": entered}

    def _slider(self, sysm, si, p, q, mode, ctx):
        """Add the relations of a slider block.

        Constraint ``n . (p - p_0) = 0`` (mode 'dt') or ``n . pdd = 0``
        (mode 'acc'); block dynamics along the line
        ``d . (q + m_s g - m_s pdd) = 0``.

        :param sysm: overall equation under construction
        :type sysm: AffineSystem
        :param si: slider index
        :type si: int
        :param p: position (or acceleration) of the pinned end
        :type p: Affine
        :param q: force of the body on the block
        :type q: Affine
        :param mode: 'dt' or 'acc'
        :type mode: str
        :param ctx: per-step context
        :type ctx: dict
        """
        s = self.sliders[si]
        d = s["direction"]
        n = np.array([-d[1], d[0]])
        if mode == "dt":
            sysm.equation(p.lmul(n.reshape(1, 2)) - float(n @ s["point"]))
            lin = ctx["lin"]
            a_c, b_s, _, _ = lin.coefficients(ctx["slider_states"][si])
            acc = a_c * p + b_s
        else:
            sysm.equation(p.lmul(n.reshape(1, 2)))
            acc = p
        dyn = q + s["mass"] * self.gravity - s["mass"] * acc
        sysm.equation(dyn.lmul(d.reshape(1, 2)))

    # -- kinematics and time integration ----------------------------------

    def kinematics(self, theta, omega):
        """Return entry-end positions and velocities from body angles.

        Positions follow the traversal from the ground hinges; closure is
        not checked.

        :param theta: body angles
        :type theta: array_like
        :param omega: body angular velocities
        :type omega: array_like
        :return: (p_entry, v_entry) dicts body -> 2-vector
        :rtype: tuple
        """
        node_p, node_v = {}, {}
        for i, h in enumerate(self.hinges):
            if h["ground"] is not None:
                node_p[i] = h["ground"]
                node_v[i] = np.zeros(2)
        conn = self._connections()
        p_entry, v_entry = {}, {}
        for b, e, hi in self.plan():
            body = self.bodies[b]
            p_entry[b], v_entry[b] = node_p[hi], node_v[hi]
            r = rigid.rot2(theta[b])
            for j, pt in enumerate(body.ends):
                c = conn.get((b, j))
                if j == e or c is None or c[0] != "hinge" or c[1] in node_p:
                    continue
                rp = r @ (pt - body.ends[e])
                node_p[c[1]] = node_p[hi] + rp
                node_v[c[1]] = (node_v[hi]
                                + omega[b] * np.array([-rp[1], rp[0]]))
        return p_entry, v_entry

    def end_positions(self, theta):
        """Return the positions of every body end from the body angles.

        :param theta: body angles
        :type theta: array_like
        :return: dict (body, end) -> position
        :rtype: dict
        """
        p_entry, _ = self.kinematics(theta, np.zeros(len(self.bodies)))
        entry = {b: e for b, e, _ in self.plan()}
        out = {}
        for b, body in enumerate(self.bodies):
            r = rigid.rot2(theta[b])
            for j, pt in enumerate(body.ends):
                out[(b, j)] = p_entry[b] + r @ (pt - body.ends[entry[b]])
        return out

    def accelerations(self, theta, omega):
        """Return body angular accelerations (acceleration form, Ch. 5).

        :param theta: body angles
        :type theta: array_like
        :param omega: body angular velocities
        :type omega: array_like
        :return: result dict of :meth:`deduce` in mode 'acc'
        :rtype: dict
        """
        return self.deduce("acc", {"theta": np.asarray(theta, float),
                                   "omega": np.asarray(omega, float)})

    def simulate_rk4(self, theta0, omega0, t_end, dt):
        """Integrate with the acceleration form and RK4 (Sec. 5.9).

        :param theta0: initial body angles
        :type theta0: array_like
        :param omega0: initial angular velocities
        :type omega0: array_like
        :param t_end: final time
        :type t_end: float
        :param dt: time step
        :type dt: float
        :return: (t, theta, omega)
        :rtype: tuple
        """
        nb = len(self.bodies)

        def rhs(y):
            res = self.accelerations(y[:nb], y[nb:])
            alpha = np.array([res["theta"][b] for b in range(nb)])
            return np.concatenate([y[nb:], alpha])

        y = np.concatenate([np.asarray(theta0, float),
                            np.asarray(omega0, float)])
        steps = int(round(t_end / dt))
        out = np.empty((steps + 1, 2 * nb))
        out[0] = y
        for i in range(steps):
            k1 = rhs(y)
            k2 = rhs(y + 0.5 * dt * k1)
            k3 = rhs(y + 0.5 * dt * k2)
            k4 = rhs(y + dt * k3)
            y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            out[i + 1] = y
        return dt * np.arange(steps + 1), out[:, :nb], out[:, nb:]

    def simulate_dt(self, theta0, omega0, t_end, dt, method="newmark",
                    **kwargs):
        """Integrate with the DT-TMM (algorithm of Sec. 7.8).

        :param theta0: initial body angles (closed loops must close)
        :type theta0: array_like
        :param omega0: initial angular velocities (consistent)
        :type omega0: array_like
        :param t_end: final time
        :type t_end: float
        :param dt: time step
        :type dt: float
        :param method: linearizer name
        :type method: str
        :return: dict with t, theta (N, nb), residual (N,) and sliders
            (N, ns, 2)
        :rtype: dict
        """
        lin = dttmm.make_linearizer(method, dt, **kwargs)
        nb = len(self.bodies)
        theta0 = np.asarray(theta0, dtype=float)
        omega0 = np.asarray(omega0, dtype=float)
        acc = self.accelerations(theta0, omega0)
        p_entry, v_entry = self.kinematics(theta0, omega0)
        states = {}
        for b, e in acc["entry"].items():
            body = self.bodies[b]
            alpha = acc["theta"][b]
            a_i = acc["p_entry"][b]
            states[b] = dttmm.PlanarBodyState.from_motion(
                [body.center - body.ends[e]], theta0[b], omega0[b], alpha,
                p_entry[b], v_entry[b], a_i)
        slider_states = {}
        for si, s in enumerate(self.sliders):
            b, j = s["end"]
            e = acc["entry"][b]
            r = rigid.rot2(theta0[b])
            rp = r @ (self.bodies[b].ends[j] - self.bodies[b].ends[e])
            drp = np.array([-rp[1], rp[0]])
            slider_states[si] = dttmm.Variable(
                p_entry[b] + rp, v_entry[b] + omega0[b] * drp,
                acc["sliders"][si])
        ctx = {"states": states, "lin": lin, "slider_states": slider_states}
        steps = int(round(t_end / dt))
        th = np.empty((steps + 1, nb))
        resid = np.zeros(steps + 1)
        sl = np.empty((steps + 1, len(self.sliders), 2))
        th[0] = theta0
        for si in range(len(self.sliders)):
            sl[0, si] = slider_states[si].x[-1]
        for i in range(steps):
            res = self.deduce("dt", ctx)
            for b, e in res["entry"].items():
                st = states[b]
                body = self.bodies[b]
                g_th, g_0, _, _ = dttmm._planar_geometry(body.center
                                                         - body.ends[e],
                                                         st.theta, dt)
                p_i = res["p_entry"][b]
                th_b = res["theta"][b]
                lin.update(st.theta, th_b)
                lin.update(st.r_i, p_i)
                lin.update(st.r_c, p_i + g_th * th_b + g_0)
                th[i + 1, b] = th_b
            for si, p in res["sliders"].items():
                lin.update(slider_states[si], p)
                sl[i + 1, si] = p
            resid[i + 1] = res["residual"]
        return {"t": dt * np.arange(steps + 1), "theta": th, "residual": resid,
                "sliders": sl}

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
