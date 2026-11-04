# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
3D navigation math of the suspension app, free of Qt.

The vehicle axes follow ISO 8855: x forward, y left, z up.  A view is named
for the side of the car the eye looks at: the front view looks at the front
of the car (the eye is ahead, on +x), the left view at its left side (the
eye on +y), and the top view down at it with x up the screen.  Engineering
CAD tools name their views for the part, not for a vehicle, which is why
the app maps its own names onto the camera instead of using the widget
presets.

The functions here work on a :class:`CameraState` (eye position, target,
and screen up, in mm) so the view commands, the animated transitions, the
view history, and the orientation cube can be tested without a window.
"""

import collections
import math

import numpy as np

__all__ = [  # noqa: F822
    'CameraState',
    'FACES',
    'VIEWS',
    'ViewHistory',
    'aligned_view',
    'cube_faces',
    'cube_hit',
    'ease',
    'grid_segments',
    'grid_spacing',
    'interpolate',
    'nearest',
    'same_state',
    'state_for_direction',
    'view_state',
]

Z_UP = np.array([0.0, 0.0, 1.0])
X_FORWARD = np.array([1.0, 0.0, 0.0])

#: The outward normal of each face of the orientation cube, named for the
#: side of the car it shows.
FACES = {
    'front': np.array([1.0, 0.0, 0.0]),
    'rear': np.array([-1.0, 0.0, 0.0]),
    'left': np.array([0.0, 1.0, 0.0]),
    'right': np.array([0.0, -1.0, 0.0]),
    'top': np.array([0.0, 0.0, 1.0]),
    'bottom': np.array([0.0, 0.0, -1.0]),
}

#: The named views: the unit direction from the target toward the eye, and
#: the screen up axis.  Iso looks from the front left and above.
VIEWS = dict({name: (normal, X_FORWARD if abs(normal[2]) > 0.5 else Z_UP)
              for name, normal in FACES.items()},
             iso=(np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0), Z_UP))

#: A view is aligned to a named face view within this angle.
ALIGN_DEGREES = 0.5

CameraState = collections.namedtuple('CameraState',
                                     ['position', 'target', 'up'])
CameraState.__doc__ = """The eye position, the target, and the screen up
axis (unit, orthogonal to the view), as numpy arrays in mm."""


def _unit(vector):
    vector = np.asarray(vector, dtype='float64')
    return vector / np.linalg.norm(vector)


def _level_up(eye, fallback=X_FORWARD):
    """The screen up axis that keeps the horizon level for an eye direction
    ``eye`` (target toward eye), or ``fallback`` looking straight up or
    down."""
    up = Z_UP - (Z_UP @ eye) * eye
    if np.linalg.norm(up) < 1.e-9:
        up = fallback - (fallback @ eye) * eye
    return _unit(up)


def make_state(position, target, up):
    """Build a :class:`CameraState` with ``up`` made orthogonal to the
    view."""
    position = np.asarray(position, dtype='float64')
    target = np.asarray(target, dtype='float64')
    eye = _unit(position - target)
    up = np.asarray(up, dtype='float64')
    up = up - (up @ eye) * eye
    return CameraState(position, target, _unit(up))


def view_state(name, target, distance):
    """The :class:`CameraState` of the named view of :data:`VIEWS` around
    ``target`` at ``distance``.

    :raises ValueError: For a name not in :data:`VIEWS`.
    """
    if name not in VIEWS:
        raise ValueError(f"no view '{name}'; the views are "
                         f"{', '.join(VIEWS)}")
    eye, up = VIEWS[name]
    target = np.asarray(target, dtype='float64')
    position = target + eye * float(distance)
    if 'iso' == name:
        up = _level_up(eye)
    return CameraState(position, target, np.array(up, dtype='float64'))


def state_for_direction(direction, target, distance):
    """The view from ``direction`` (target toward eye, any length) with a
    level horizon, as an orientation-cube click gives it."""
    eye = _unit(direction)
    target = np.asarray(target, dtype='float64')
    return CameraState(target + eye * float(distance), target,
                       _level_up(eye))


def aligned_view(state):
    """The name of the face view ``state`` looks along, within
    :data:`ALIGN_DEGREES`, or None."""
    eye = _unit(state.position - state.target)
    limit = math.cos(math.radians(ALIGN_DEGREES))
    for name, normal in FACES.items():
        if eye @ normal >= limit:
            return name
    return None


def same_state(a, b, tol=1.e-9):
    """Whether two camera states are the same view."""
    scale = max(1.0, float(np.linalg.norm(a.position - a.target)))
    return (np.allclose(a.position, b.position, atol=tol * scale)
            and np.allclose(a.target, b.target, atol=tol * scale)
            and np.allclose(a.up, b.up, atol=tol))


def _rotate(vector, axis, angle):
    """Rotate ``vector`` about the unit ``axis`` by ``angle`` (Rodrigues)."""
    c, s = math.cos(angle), math.sin(angle)
    return (vector * c + np.cross(axis, vector) * s
            + axis * (axis @ vector) * (1.0 - c))


def ease(t):
    """Smoothstep easing of a transition fraction, clamped to [0, 1]."""
    t = min(1.0, max(0.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def interpolate(a, b, t):
    """The camera state a fraction ``t`` of the way from ``a`` to ``b``.

    The eye direction turns on the great circle between the two (about the
    up axis of ``a`` for opposite views, so a half turn swings around the
    car instead of through it); the distance and the target move linearly.
    """
    t = float(t)
    eye_a = _unit(a.position - a.target)
    eye_b = _unit(b.position - b.target)
    dist_a = float(np.linalg.norm(a.position - a.target))
    dist_b = float(np.linalg.norm(b.position - b.target))
    axis = np.cross(eye_a, eye_b)
    angle = math.atan2(float(np.linalg.norm(axis)), float(eye_a @ eye_b))
    if np.linalg.norm(axis) < 1.e-9:
        axis = a.up if eye_a @ eye_b < 0.0 else Z_UP
    axis = _unit(axis)
    eye = _unit(_rotate(eye_a, axis, t * angle))
    up = _rotate(np.asarray(a.up, dtype='float64'), axis, t * angle)
    up = (1.0 - t) * up + t * np.asarray(b.up, dtype='float64')
    up = up - (up @ eye) * eye
    up = _level_up(eye) if np.linalg.norm(up) < 1.e-9 else _unit(up)
    target = (1.0 - t) * a.target + t * b.target
    distance = (1.0 - t) * dist_a + t * dist_b
    return CameraState(target + eye * distance, target, up)


class ViewHistory(object):
    """The views the user went through, for back and forward, and the
    views saved by name.

    :meth:`push` records a view the user left; :meth:`back` and
    :meth:`forward` take the view shown now, so the first :meth:`back`
    keeps it as the way forward again.
    """

    def __init__(self):
        self._entries = []
        self._index = None  # None: the live view is past the last entry.
        self._named = collections.OrderedDict()

    def push(self, state):
        if self._index is not None:
            del self._entries[self._index + 1:]
            self._index = None
        if self._entries and same_state(self._entries[-1], state):
            return
        self._entries.append(state)

    def back(self, current):
        """The view before ``current``, or None at the oldest."""
        if self._index is None:
            if not self._entries or not same_state(self._entries[-1],
                                                   current):
                self._entries.append(current)
            self._index = len(self._entries) - 1
        if self._index <= 0:
            return None
        self._index -= 1
        return self._entries[self._index]

    def forward(self, current):
        """The view after ``current`` that :meth:`back` left, or None."""
        if self._index is None or self._index >= len(self._entries) - 1:
            return None
        self._index += 1
        return self._entries[self._index]

    def save(self, name, state):
        self._named[str(name)] = state

    def names(self):
        return list(self._named)

    def named(self, name):
        if name not in self._named:
            raise KeyError(f"no saved view '{name}'")
        return self._named[name]


def _screen_basis(state):
    """(right, screen up, toward the eye) of a view, unit vectors."""
    toward = _unit(state.position - state.target)
    right = _unit(np.cross(-toward, state.up))
    return right, np.cross(right, -toward), toward


def _project(basis, center, size, point):
    """Pixel of a cube point (cube half edge 1) drawn at ``center`` with a
    half edge of ``size`` pixels; the pixel y axis points down."""
    right, up, _ = basis
    return np.array([center[0] + size * float(point @ right),
                     center[1] - size * float(point @ up)])


def _face_axes(normal):
    """Two unit axes spanning the face with outward ``normal``."""
    helper = Z_UP if abs(normal[2]) < 0.5 else X_FORWARD
    u = _unit(np.cross(helper, normal))
    return u, np.cross(normal, u)


def cube_faces(state, center, size):
    """The faces of the orientation cube in the view ``state``, back to
    front: ``(name, polygon, facing)`` with the (4, 2) pixel polygon and
    ``facing`` the cosine between the face normal and the eye direction
    (positive for a face turned toward the eye)."""
    basis = _screen_basis(state)
    center = np.asarray(center, dtype='float64')
    out = []
    for name, normal in FACES.items():
        u, v = _face_axes(normal)
        corners = [normal + su * u + sv * v
                   for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        polygon = np.array([_project(basis, center, size, c)
                            for c in corners])
        out.append((name, polygon, float(normal @ basis[2])))
    out.sort(key=lambda item: item[2])
    return out


#: A click within this fraction of a face edge picks the edge view, and in
#: both directions the corner view, as the ViewCube does.
EDGE_BAND = 0.6


def cube_hit(state, center, size, point):
    """The eye direction (unit) a click at pixel ``point`` on the
    orientation cube asks for: the face, edge, or corner view under it; or
    None for a click off the cube."""
    basis = _screen_basis(state)
    center = np.asarray(center, dtype='float64')
    point = np.asarray(point, dtype='float64')
    for name, _, facing in reversed(cube_faces(state, center, size)):
        if facing <= 1.e-6:
            continue
        normal = FACES[name]
        u, v = _face_axes(normal)
        origin = _project(basis, center, size, normal)
        # The face projects to a parallelogram: solve for the face
        # coordinates (in [-1, 1] on the face) of the click.
        du = _project(basis, center, size, normal + u) - origin
        dv = _project(basis, center, size, normal + v) - origin
        matrix = np.column_stack([du, dv])
        if abs(np.linalg.det(matrix)) < 1.e-9:
            continue
        cu, cv = np.linalg.solve(matrix, point - origin)
        if abs(cu) > 1.0 or abs(cv) > 1.0:
            continue
        direction = normal.copy()
        if abs(cu) > EDGE_BAND:
            direction = direction + math.copysign(1.0, cu) * u
        if abs(cv) > EDGE_BAND:
            direction = direction + math.copysign(1.0, cv) * v
        return _unit(direction)
    return None


def nearest(pixels, cursor, radius):
    """The name of the point nearest to ``cursor`` on the screen within
    ``radius`` pixels, the one nearer the eye on a tie, or None.

    :param pixels: ``{name: (x, y, depth)}``, an empty tuple for a point
        that does not project (behind the eye).
    """
    best = None
    for name, pix in pixels.items():
        if len(pix) < 3:
            continue
        dist = math.hypot(pix[0] - cursor[0], pix[1] - cursor[1])
        if dist > radius:
            continue
        key = (round(dist, 6), pix[2])
        if best is None or key < best[0]:
            best = (key, name)
    return None if best is None else best[1]


def grid_spacing(extent):
    """A round grid spacing (1, 2, or 5 times a power of ten) giving about
    14 lines over ``extent``."""
    raw = max(float(extent), 1.e-9) / 14.0
    power = 10.0 ** math.floor(math.log10(raw))
    candidates = [f * power for f in (1.0, 2.0, 5.0, 10.0)]
    return min(candidates, key=lambda c: abs(math.log(c / raw)))


def grid_segments(lo, hi, z, spacing=None):
    """The lines of a ground grid at height ``z`` covering the (x, y) box
    from ``lo`` to ``hi``, on multiples of ``spacing``.

    :return: ``(ends, spacing)``: the segment ends in consecutive pairs, and
        the spacing used.
    """
    lo = np.asarray(lo, dtype='float64')
    hi = np.asarray(hi, dtype='float64')
    if spacing is None:
        spacing = grid_spacing(float(np.max(hi - lo)))
    x0 = math.floor(lo[0] / spacing) * spacing
    x1 = math.ceil(hi[0] / spacing) * spacing
    y0 = math.floor(lo[1] / spacing) * spacing
    y1 = math.ceil(hi[1] / spacing) * spacing
    ends = []
    for x in np.arange(x0, x1 + 0.5 * spacing, spacing):
        ends += [(x, y0, z), (x, y1, z)]
    for y in np.arange(y0, y1 + 0.5 * spacing, spacing):
        ends += [(x0, y, z), (x1, y, z)]
    return ends, spacing

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
