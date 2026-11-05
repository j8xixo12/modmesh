# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Geometry of a suspension corner for the 3D view, free of Qt.

The bodies are drawn as small meshes in their own frames, which the view
moves rigidly with the pose the engine solved: the upright with its wheel,
the rocker, and the rack.  The chassis pickups stay at design in a K&C
rig and ride on the sprung mass in the quarter car.  The links, the
pushrod, and the spring are line segments between the points they join,
rebuilt for every pose.

The engine works in m and the view in mm, so every coordinate here is in
mm.  A body frame is parallel to the global frame at design, with its origin
at the wheel center (upright), the rocker pivot (rocker), or the inner tie
rod point (rack), as :mod:`suspension.corner` builds them.
"""

import math

import numpy as np

from ... import core

__all__ = [
    'BODY_ORIGINS',
    'arb_segments',
    'CornerGeometry',
    'MeshBuilder',
    'arrow_segments',
    'road_segments',
    'rotation',
]

MM = 1.e3

#: The hardpoint each moving body frame has its origin at (the rack: the
#: inner point of its first link; an arm: the midpoint of its pivots).
BODY_ORIGINS = {'upright': 'wheel_center', 'rocker': 'rocker_pivot',
                'rack': 'tie_rod_inner'}
#: The design pose of the chassis: at the global origin, not rotated.
ZERO = np.zeros(3)
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def rotation(quat):
    """Return the body-to-global rotation matrix of Euler parameters.

    :param quat: ``(e0, e1, e2, e3)``, the scalar first, as the engine gives
        them (Haug, Computer-Aided Kinematics and Dynamics of Mechanical
        Systems, Eq. 6.1.13).
    :rtype: numpy.ndarray
    """
    e0, e1, e2, e3 = quat
    evec = np.array([e1, e2, e3], dtype='float64')
    skew = np.array([[0.0, -e3, e2], [e3, 0.0, -e1], [-e2, e1, 0.0]],
                    dtype='float64')
    return ((2.0 * e0 * e0 - 1.0) * np.eye(3, dtype='float64')
            + 2.0 * (np.outer(evec, evec) + e0 * skew))


def _frame(axis):
    """Unit ``axis`` and two unit vectors perpendicular to it."""
    axis = np.asarray(axis, dtype='float64')
    axis = axis / np.linalg.norm(axis)
    other = np.array([0.0, 0.0, 1.0], dtype='float64')
    if abs(axis @ other) > 0.9:
        other = np.array([1.0, 0.0, 0.0], dtype='float64')
    unit1 = np.cross(axis, other)
    unit1 /= np.linalg.norm(unit1)
    return axis, unit1, np.cross(axis, unit1)


def tire_columns(radius, rim_radius, width, grooves=(-0.32, 0.0, 0.32),
                 groove_width=0.08, groove_depth=5.0, ncol=28):
    """Cross section columns ``(y, r_lo, r_hi)`` (mm) of a tire.

    The outer contour is a superellipse of exponent 4 over the section: a
    flat tread, rounded shoulders, and sidewalls that fall to the bead on
    the rim.  Circumferential grooves of ``groove_depth`` sit at the
    section fractions ``grooves`` (-1 to 1 across the width).
    """
    half = 0.5 * width
    bead = rim_radius + 5.0
    height = radius - bead
    # Columns cluster at the sidewalls, where the contour turns.
    fractions = {math.sin(0.5 * math.pi * (-1.0 + 2.0 * it / (ncol - 1)))
                 for it in range(ncol)}
    for center in grooves:
        for edge in (-1.0, 1.0):
            for off in (0.0, 0.004):
                fractions.add(center + edge * (0.5 * groove_width + off))
    columns = []
    for frac in sorted(f for f in fractions if -1.0 <= f <= 1.0):
        top = bead + height * (1.0 - abs(frac) ** 4) ** 0.25
        if any(abs(frac - c) < 0.5 * groove_width + 1e-9 for c in grooves):
            top -= groove_depth
        columns.append((frac * half, rim_radius - 10.0, top))
    return columns


class MeshBuilder(object):
    """Collect hexahedra and build one :class:`~solvcon.core.StaticMesh`.

    The cells need not touch, so a body is drawn as a few separate bricks.
    """

    def __init__(self):
        self.nodes = []
        self.cells = []

    def hexahedron(self, corners):
        """Add a hexahedron from its 8 corners, the bottom face first."""
        base = len(self.nodes)
        self.nodes.extend(np.asarray(corners, dtype='float64'))
        self.cells.append(list(range(base, base + 8)))

    def box(self, center, half):
        """Add an axis-aligned cube of half width ``half`` at ``center``."""
        c = np.asarray(center, dtype='float64')
        signs = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                 (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        self.hexahedron([c + half * np.array(sgn, dtype='float64')
                         for sgn in signs])

    def beam(self, start, end, half):
        """Add a square bar of half width ``half`` from ``start`` to
        ``end``."""
        start = np.asarray(start, dtype='float64')
        end = np.asarray(end, dtype='float64')
        axis = end - start
        length = np.linalg.norm(axis)
        if length <= 0.0:
            return
        axis = axis / length
        other = np.array([0.0, 0.0, 1.0], dtype='float64')
        if abs(axis @ other) > 0.9:
            other = np.array([1.0, 0.0, 0.0], dtype='float64')
        side1 = np.cross(axis, other)
        side1 *= half / np.linalg.norm(side1)
        side2 = np.cross(axis, side1)
        side2 *= half / np.linalg.norm(side2)
        ring = [-side1 - side2, side1 - side2, side1 + side2, -side1 + side2]
        self.hexahedron([start + off for off in ring]
                        + [end + off for off in ring])

    def revolve(self, center, axis, columns, nseg=72):
        """Add a solid of revolution about ``axis`` through ``center``.

        The cross section is a strip of ``columns``: ``(y, r_lo, r_hi)``
        with y along the axis (increasing) and the radii r_lo < r_hi.
        Neighbor cells share their nodes, so only the outer shell is a
        boundary and a transparent solid shows no inner faces.
        """
        center = np.asarray(center, dtype='float64')
        unit0, unit1, unit2 = _frame(axis)
        ncol = len(columns)
        if ncol < 2:
            raise ValueError("revolve needs two columns or more")
        base = len(self.nodes)
        for iseg in range(nseg):
            ang = 2.0 * math.pi * iseg / nseg
            radial = math.cos(ang) * unit1 + math.sin(ang) * unit2
            for y, r_lo, r_hi in columns:
                if not r_lo < r_hi:
                    raise ValueError("revolve needs r_lo < r_hi")
                self.nodes.append(center + y * unit0 + r_lo * radial)
                self.nodes.append(center + y * unit0 + r_hi * radial)

        def node(iseg, icol, outer):
            return base + 2 * (ncol * (iseg % nseg) + icol) + outer

        for iseg in range(nseg):
            for icol in range(ncol - 1):
                bottom = [node(iseg, icol, 0), node(iseg, icol, 1),
                          node(iseg + 1, icol, 1), node(iseg + 1, icol, 0)]
                top = [node(iseg, icol + 1, 0), node(iseg, icol + 1, 1),
                       node(iseg + 1, icol + 1, 1),
                       node(iseg + 1, icol + 1, 0)]
                self.cells.append(bottom + top)

    def build(self):
        """Return the mesh of the collected cells."""
        ncell = len(self.cells)
        mesh = core.StaticMesh(ndim=3, nnode=len(self.nodes), nface=0,
                               ncell=ncell)
        mesh.ndcrd.ndarray[:, :] = np.array(self.nodes, dtype='float64')
        mesh.cltpn.ndarray[:] = core.StaticMesh.HEXAHEDRON
        clnds = np.full((ncell, 9), -1, dtype='int32')
        clnds[:, 0] = 8
        clnds[:, 1:] = np.array(self.cells, dtype='int32')
        mesh.clnds.ndarray[:, :9] = clnds
        mesh.build_interior()
        mesh.build_boundary()
        mesh.build_ghost()
        return mesh


class CornerGeometry(object):
    """The meshes and the segments of one corner model, in mm.

    :ivar model: The :class:`suspension.model.CornerModel` with its
        adjusters applied.
    :ivar suffix: Appended to every object, body, and segment name, so the
        two corners of an axle share one scene; empty for the left corner.
    """

    #: Half width of a pickup marker and of a body bar, in mm.
    PICKUP_HALF = 9.0
    BAR_HALF = 7.0

    def __init__(self, model, suffix=''):
        self.model = model
        self.suffix = suffix
        self.hp = {name: np.asarray(value, dtype='float64') * MM
                   for name, value in model.hp.items()}
        self.upright_points = {
            name: np.asarray(value, dtype='float64') * MM
            for name, value in model.upright_points().items()}

    def origin(self, body):
        """The design position of the frame origin of ``body``, in mm."""
        model = self.model
        if body.startswith('arm_'):
            names = model.arms[body[4:]]['links']
            inner = [model.chassis_point(model.links[n][1]) for n in names]
            return 0.5 * (inner[0] + inner[1]) * MM
        if 'rack' == body:
            for kind, inner, _ in model.links.values():
                if 'rack' == kind:
                    return model.chassis_point(inner) * MM
        return self.hp[BODY_ORIGINS[body]]

    def bodies(self):
        """The moving bodies of the corner model, without the suffix."""
        out = ['upright']
        if self.model.has_rocker:
            out.append('rocker')
        if self.model.has_rack:
            out.append('rack')
        out += ['arm_' + arm for arm in sorted(self.model.arms)]
        return out

    def body_of(self, name):
        """The moving body a hardpoint rides on, or None for a chassis
        point."""
        model = self.model
        if name in self.upright_points:
            return 'upright'
        arm = model.arm_of(name)
        if arm is not None:
            return 'arm_' + arm
        if name in self.ROCKER_POINTS and model.has_rocker:
            return 'rocker'
        for kind, inner, _ in model.links.values():
            if 'rack' == kind and inner == name:
                return 'rack'
        return None

    def upright_local(self, name):
        return self.upright_points[name] - self.upright_points['wheel_center']

    def chassis_pickups(self):
        """The global points fixed to the chassis, in mm."""
        names = [inner for kind, inner, _ in self.model.links.values()
                 if 'rack' != kind]
        names += [n for n in ('rocker_pivot', 'spring_chassis', 'strut_inner',
                              'arm_inner', 'arm_outer') if n in self.hp]
        return [self.model.chassis_point(name) * MM for name in names]

    def body_meshes(self):
        """Return ``{name: mesh}`` with each mesh in its body frame (mm);
        the wheel is in the upright frame and the chassis in the global
        frame."""
        meshes = {}

        model = self.model
        upright = MeshBuilder()

        def knuckle(local, half):
            path = self.knuckle_path(local)
            for start, end in zip(path[:-1], path[1:]):
                upright.beam(start, end, half)

        for name in sorted(self.upright_points):
            if 'wheel_center' == name:
                continue
            local = self.upright_local(name)
            knuckle(local, self.BAR_HALF)
            upright.box(local, self.PICKUP_HALF)
        if model.has_strut:
            # The strut tube, clamped to the upright, along the strut axis.
            low = self.upright_local('strut_outer')
            top = model.chassis_point('strut_inner') * MM \
                - self.upright_points['wheel_center']
            upright.beam(low, low + 0.5 * (top - low), 2.2 * self.BAR_HALF)
        if model.has_arm:
            # The trailing arm is the upright: bars to both pivots.
            for name in ('arm_inner', 'arm_outer'):
                local = model.chassis_point(name) * MM \
                    - self.upright_points['wheel_center']
                knuckle(local, 1.6 * self.BAR_HALF)
                upright.box(local, self.PICKUP_HALF)
        meshes['upright' + self.suffix] = upright.build()

        # The wheel turns with the upright.  The tire and the rim are drawn
        # on their own, each with a color and an opacity, so the view can
        # see through them to the pickups inside the wheel.
        meshes['wheel' + self.suffix] = self.tire_mesh()
        meshes['rim' + self.suffix] = self.rim_mesh()

        if model.has_rocker:
            rocker = MeshBuilder()
            pivot = self.hp['rocker_pivot']
            arms = [self.hp['pushrod_inner'] - pivot,
                    self.hp['spring_rocker'] - pivot]
            for arm in arms:
                rocker.beam((0.0, 0.0, 0.0), arm, self.BAR_HALF)
                rocker.box(arm, self.PICKUP_HALF)
            rocker.beam(arms[0], arms[1], self.BAR_HALF)
            meshes['rocker' + self.suffix] = rocker.build()

        if model.has_rack:
            rack = MeshBuilder()
            rack.box((0.0, 0.0, 0.0), self.PICKUP_HALF * 1.5)
            meshes['rack' + self.suffix] = rack.build()

        for arm in sorted(model.arms):
            # A rigid wishbone: bars from both pivots to the ball joint,
            # and a box at a pushrod or spring point on it.
            body = 'arm_' + arm
            origin = self.origin(body)
            names = model.arms[arm]['links']
            ball = self.hp[model.links[names[0]][2]] - origin
            mesh = MeshBuilder()
            for name in names:
                inner = model.chassis_point(model.links[name][1]) * MM \
                    - origin
                mesh.beam(inner, ball, self.BAR_HALF)
            for point in ('pushrod_outer', 'spring_upright'):
                if model.arm_of(point) == arm:
                    mesh.box(self.hp[point] - origin, self.PICKUP_HALF)
            meshes[body + self.suffix] = mesh.build()

        chassis = MeshBuilder()
        for point in self.chassis_pickups():
            chassis.box(point, self.PICKUP_HALF)
        meshes['chassis' + self.suffix] = chassis.build()
        return meshes

    #: Tire section width and rim radius, as fractions of the wheel radius;
    #: the rim grows up to :attr:`RIM_RADIUS_MAX` to hold the pickups.
    TIRE_WIDTH = 0.6
    RIM_RADIUS = 0.62
    RIM_RADIUS_MAX = 0.75
    #: Gap (mm) between a pickup marker inside the wheel and the barrel.
    BARREL_CLEARANCE = 8.0

    def wheel_sizes(self):
        """``(radius, rim radius, half width)`` of the wheel in mm.

        A real rim is picked large enough to hold the ball joints and the
        brake inside its barrel; the drawn rim does the same: it grows
        from :attr:`RIM_RADIUS` until the upright pickups inside the
        wheel width clear the barrel.
        """
        radius = self.model.wheel_radius * MM
        half = 0.5 * self.TIRE_WIDTH * radius
        axis = _frame(self.model.spin_axis)[0]
        need = 0.0
        for point in self.upright_points.values():
            local = point - self.upright_points['wheel_center']
            along = local @ axis
            radial = np.linalg.norm(local - along * axis)
            if abs(along) < half + self.FLANGE_WIDTH and radial < radius:
                need = max(need, radial + self.PICKUP_HALF
                           + self.BARREL_CLEARANCE + self.BARREL_DEPTH)
        rim = min(max(self.RIM_RADIUS * radius, need),
                  self.RIM_RADIUS_MAX * radius)
        return radius, rim, half

    #: Gap (mm) between a knuckle bar and the inboard flange of the rim.
    KNUCKLE_CLEARANCE = 15.0

    def in_wheel(self, local, margin=0.0):
        """Whether ``local`` (upright frame, mm) is inside the tire or the
        rim (barrel and flanges), grown by ``margin`` mm."""
        radius, rim, half = self.wheel_sizes()
        axis = _frame(self.model.spin_axis)[0]
        local = np.asarray(local, dtype='float64')
        along = local @ axis
        radial = np.linalg.norm(local - along * axis)
        if rim - self.BARREL_DEPTH - margin >= radial:
            return False
        if abs(along) < half + margin:
            # The tire and the barrel under it.
            return radial < radius + margin
        # The flanges outside the tire section.
        return (abs(along) < half + self.FLANGE_WIDTH + margin
                and radial < rim + self.FLANGE_HEIGHT + margin)

    def knuckle_path(self, local):
        """Points (upright frame, mm) of the knuckle bar from the hub to
        the pickup at ``local``.

        The bar is straight when it stays out of the tire and the rim.
        Else it leaves the hub inboard, rises past the inboard rim flange,
        and turns out to the pickup: the swan neck of a real knuckle that
        reaches a ball joint above the tire, or a strut clamp or an arm
        pivot inboard of it.
        """
        local = np.asarray(local, dtype='float64')
        zero = np.zeros(3, dtype='float64')
        margin = self.BAR_HALF
        straight = [zero + s * local for s in np.linspace(0.0, 1.0, 65)]
        if not any(self.in_wheel(p, margin) for p in straight):
            return [zero, local]
        _, _, half = self.wheel_sizes()
        axis = _frame(self.model.spin_axis)[0]
        inboard = -self.outboard() * axis
        depth = max(half + self.FLANGE_WIDTH + self.KNUCKLE_CLEARANCE
                    + margin, local @ inboard)
        knee = depth * inboard
        radial = local - (local @ axis) * axis
        return [zero, knee, knee + radial, local]

    #: Width (mm) of a rim flange outside the tire section, and depth (mm)
    #: of the rim barrel below the bead seat.
    FLANGE_WIDTH = 7.0
    BARREL_DEPTH = 22.0
    #: Height (mm) of a rim flange above the bead seat.
    FLANGE_HEIGHT = 14.0

    def outboard(self):
        """+1 when the spin axis points outboard (away from the chassis),
        else -1."""
        spin = np.asarray(self.model.spin_axis, dtype='float64')
        # The vehicle centerline is y = 0.
        out = np.array([0.0, self.hp['wheel_center'][1], 0.0])
        return 1.0 if out @ spin >= 0.0 else -1.0

    def tire_mesh(self):
        """The tire in the upright frame (mm)."""
        radius, rim, half = self.wheel_sizes()
        tire = MeshBuilder()
        tire.revolve((0.0, 0.0, 0.0), self.model.spin_axis,
                     tire_columns(radius, rim, 2.0 * half), nseg=96)
        return tire.build()

    def rim_mesh(self):
        """The rim in the upright frame (mm): the barrel under the tire,
        a flange at each bead, a hub, and five spokes in the outboard
        face."""
        radius, rim, half = self.wheel_sizes()
        axis, unit1, unit2 = _frame(self.model.spin_axis)
        side = self.outboard()
        out = MeshBuilder()
        zero = (0.0, 0.0, 0.0)
        barrel_in = rim - self.BARREL_DEPTH
        barrel_out = rim - 12.0
        out.revolve(zero, axis, [(-half, barrel_in, barrel_out),
                                 (half, barrel_in, barrel_out)], nseg=72)
        for sign in (-1.0, 1.0):
            ys = sorted([sign * half,
                         sign * (half + self.FLANGE_WIDTH)])
            top = rim + self.FLANGE_HEIGHT
            out.revolve(zero, axis, [(ys[0], barrel_in, top),
                                     (ys[1], barrel_in, top)],
                        nseg=72)
        face = side * 0.45 * half
        hub_in, hub_out = 0.08 * radius, 0.2 * radius
        out.revolve(zero, axis, [(face - 22.0, hub_in, hub_out),
                                 (face + 18.0, hub_in, hub_out)], nseg=36)
        for ispoke in range(5):
            ang = 2.0 * math.pi * ispoke / 5
            radial = math.cos(ang) * unit1 + math.sin(ang) * unit2
            start = face * axis + 0.9 * hub_out * radial
            # The spokes lean inboard to reach the barrel near its middle.
            end = (face - side * 0.35 * half) * axis + \
                (barrel_in + 4.0) * radial
            out.beam(start, end, 11.0)
        return out.build()

    def frames(self, pose):
        """Return ``{body: (translation_mm, rotation)}`` of a pose.

        :param pose: ``{body: (position_m, quaternion)}`` from the engine.
        """
        return {body: (np.asarray(pos, dtype='float64') * MM,
                       rotation(quat))
                for body, (pos, quat) in pose.items()}

    def chassis_pose(self, pose):
        """The ``(position_m, quaternion)`` of the chassis in ``pose``.

        A K&C rig holds the chassis at the design pose, so a pose without a
        ``chassis`` entry is that pose; the quarter car moves the chassis.
        """
        return pose.get('chassis' + self.suffix, (ZERO, IDENTITY))

    def object_poses(self, pose):
        """Return ``{object: (position_m, quaternion)}`` of the moving mesh
        objects of this corner; the wheel moves with the upright, and the
        chassis pickups (drawn in the global frame) with the chassis."""
        out = {body + self.suffix: pose[body + self.suffix]
               for body in self.bodies()}
        out['wheel' + self.suffix] = pose['upright' + self.suffix]
        out['rim' + self.suffix] = pose['upright' + self.suffix]
        out['chassis' + self.suffix] = self.chassis_pose(pose)
        return out

    #: The hardpoints carried by each moving body; the ``arb_tip`` rides on
    #: the anti-roll bar of an axle, the rest of the points on the chassis.
    ROCKER_POINTS = ('rocker_pivot', 'pushrod_inner', 'spring_rocker',
                     'arb_rocker')
    RACK_POINTS = ('tie_rod_inner',)

    def hardpoints(self, pose):
        """Return ``{name: (design_mm, current_mm)}`` of every hardpoint at
        ``pose``, the names with :attr:`suffix`.  The design position of an
        upright point includes the adjusters (it is where the upright puts
        the point at design)."""
        frames = self.frames(pose)
        pos, quat = self.chassis_pose(pose)
        chassis = (np.asarray(pos, dtype='float64') * MM, rotation(quat))

        def place(frame, local):
            return frame[0] + frame[1] @ local

        out = {}
        for name, point in self.hp.items():
            design = point
            body = self.body_of(name)
            if 'upright' == body:
                design = self.upright_points[name]
                now = place(frames['upright' + self.suffix],
                            self.upright_local(name))
            elif body is not None:
                now = place(frames[body + self.suffix],
                            point - self.origin(body))
            elif 'arb_tip' == name and 'arb' + self.suffix in frames:
                now = place(frames['arb' + self.suffix],
                            point - self.hp['arb_pivot'])
            else:
                now = place(chassis, point)
            out[name + self.suffix] = (np.array(design), now)
        return out

    def link_segments(self, pose):
        """Return ``{name: (start, end)}`` in mm of every link, the pushrod,
        and the spring at ``pose``, the names with :attr:`suffix`."""
        frames = self.frames(pose)
        pos, quat = self.chassis_pose(pose)
        chassis = (np.asarray(pos, dtype='float64') * MM, rotation(quat))

        def place(body, local):
            trans, rot = frames[body + self.suffix]
            return trans + rot @ local

        def fixed(point):
            # A chassis point: its design position is in the chassis frame.
            return chassis[0] + chassis[1] @ point

        model = self.model

        def moving(name):
            body = self.body_of(name)
            if 'upright' == body:
                return place('upright', self.upright_local(name))
            return place(body, self.hp[name] - self.origin(body))

        out = {}
        for name, (kind, inner, outer) in model.links.items():
            if 'rack' == kind:
                start = place('rack', model.chassis_point(inner) * MM
                              - self.origin('rack'))
            else:
                start = fixed(model.chassis_point(inner) * MM)
            out[name] = (start, place('upright', self.upright_local(outer)))
        if model.has_rocker:
            out[model.rod_kind] = (moving('pushrod_outer'),
                                   moving('pushrod_inner'))
        if model.has_strut:
            out['strut'] = (moving('strut_outer'),
                            fixed(model.chassis_point('strut_inner') * MM))
        out['spring'] = (moving(model.spring_point_name()),
                         fixed(model.chassis_point('spring_chassis') * MM))
        return {name + self.suffix: ends for name, ends in out.items()}


def arb_segments(left, right, pose):
    """Return the segment ends (mm) of the anti-roll bar of an axle: the
    bar between the pivots, each arm, and each drop link to its rocker.

    :param left: The :class:`CornerGeometry` of the left corner.
    :param right: The one of the right corner.
    :param pose: The axle pose, with the bar halves as ``arb`` and
        ``arb_right``.
    """
    ends = [left.hp['arb_pivot'], right.hp['arb_pivot']]
    for geom in (left, right):
        # The drop link ends on the rocker, or on the upright of a
        # direct-acting corner.
        link = 'rocker' if 'arb_rocker' in geom.hp else 'upright'
        frames = geom.frames({key: pose[key] for key in
                              ('arb' + geom.suffix, link + geom.suffix)})
        trans, rot = frames['arb' + geom.suffix]
        tip = trans + rot @ (geom.hp['arb_tip'] - geom.hp['arb_pivot'])
        trans, rot = frames[link + geom.suffix]
        if 'rocker' == link:
            drop = trans + rot @ (geom.hp['arb_rocker']
                                  - geom.hp['rocker_pivot'])
        else:
            drop = trans + rot @ geom.upright_local('arb_upright')
        ends += [geom.hp['arb_pivot'], tip, tip, drop]
    return ends


def road_segments(geometry, pose):
    """Return the outline (mm) of the effective road plane under the tire of
    ``geometry`` in a quarter car ``pose`` (tilted by its quaternion, the
    effective slope), or ``[]`` without a road."""
    if 'road' + geometry.suffix not in pose:
        return []
    position, quat = pose['road' + geometry.suffix]
    center = np.asarray(position, dtype='float64') * MM
    rot = rotation(quat)
    half_x = 0.8 * geometry.model.wheel_radius * MM
    half_y = 0.3 * geometry.model.wheel_radius * MM
    corners = [center + rot @ np.array([sx * half_x, sy * half_y, 0.0])
               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    out = []
    for it in range(4):
        out += [corners[it], corners[(it + 1) % 4]]
    return out


def arrow_segments(start, vector, head=0.18):
    """Return the 3 segments (6 ends) of an arrow from ``start`` along
    ``vector``: the shaft and two barbs ``head`` of its length."""
    start = np.asarray(start, dtype='float64')
    vector = np.asarray(vector, dtype='float64')
    length = np.linalg.norm(vector)
    if length <= 0.0:
        return []
    tip = start + vector
    axis = vector / length
    other = np.array([0.0, 0.0, 1.0], dtype='float64')
    if abs(axis @ other) > 0.9:
        other = np.array([1.0, 0.0, 0.0], dtype='float64')
    side = np.cross(axis, other)
    side /= np.linalg.norm(side)
    back = tip - head * length * axis
    return [start, tip, tip, back + 0.5 * head * length * side,
            tip, back - 0.5 * head * length * side]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
