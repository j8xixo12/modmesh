# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The 3D view of the corner: bodies moved rigidly, links as lines, and the
navigation aids of an engineering viewer.

The bodies are named mesh objects of the pilot 3D widget, placed by the
translation and the Euler parameters the engine solved.  The links, the
pushrod, the spring, the selected item, the force arrows, the ground grid,
the geometry overlays, and the pick markers are named segment objects,
rebuilt for every pose.

The navigation follows the CAD viewers surveyed in ``docs/roadmap.md``
(P12): vehicle views on a toolbar and on keys, an orientation cube, a
turntable orbit about the z axis, orthographic standard views, zoom to the
cursor, animated view changes with a back and forward history, named
views, a four-view layout, a choice of mouse mapping, a coordinate readout
of the hardpoint under the cursor, and a distance and angle measure.
"""

import numpy as np

from PySide6.QtCore import QEvent, QObject, QPointF, Qt, QTimer
from PySide6.QtGui import (QAction, QActionGroup, QColor, QFont,
                           QKeySequence, QPainter, QPen, QPolygonF)
from PySide6.QtWidgets import (QComboBox, QGridLayout, QInputDialog,
                               QLabel, QMenu, QToolBar, QToolButton,
                               QWidget)

from ... import core
from ...pilot import base

from . import _navigation as nav
from . import _scene

__all__ = [
    'CornerView',
]

LINK_COLOR = (0.12, 0.12, 0.12)
SPRING_COLOR = (0.10, 0.50, 0.20)
SELECT_COLOR = (0.95, 0.55, 0.05)
FORCE_COLOR = (0.85, 0.10, 0.10)
ROAD_COLOR = (0.45, 0.45, 0.45)
GRID_COLOR = (0.80, 0.80, 0.80)
HOVER_COLOR = (0.10, 0.55, 0.95)
PICK_COLOR = (0.90, 0.10, 0.60)

#: The geometry overlays the app can draw (P11.4): segment object name and
#: color.
OVERLAYS = {
    'kingpin': (0.55, 0.20, 0.75),
    'instant_centers': (0.10, 0.45, 0.85),
    'roll_center': (0.85, 0.35, 0.10),
    'patch_path': (0.20, 0.60, 0.60),
    'road_profile': (0.35, 0.25, 0.15),
}

#: The mouse mappings of the widget, with the help text of each.
MOUSE_MAPPINGS = {
    'blender': ("Blender", "left or middle drag: orbit; Shift+middle: "
                           "pan; Ctrl+middle: zoom; right drag: pan"),
    'cad': ("CAD (SolidWorks)", "middle drag: orbit; Ctrl+middle: pan; "
                                "Shift+middle: zoom; right drag: pan; "
                                "left click: pick"),
    'default': ("Simple", "left drag: orbit; right or middle drag: pan"),
}

#: The keys of the view commands: (action id, text, keys).
VIEW_KEYS = (
    ('front', "Front", ("1",)),
    ('rear', "Rear", ("Ctrl+1",)),
    ('left', "Left", ("3",)),
    ('right', "Right", ("Ctrl+3",)),
    ('top', "Top", ("7",)),
    ('bottom', "Bottom", ("Ctrl+7",)),
    ('iso', "Iso", ("0",)),
    ('home', "Home", ("Home",)),
)


def _points(ends):
    array = np.array(ends, dtype='float32').reshape(-1, 3)
    return core.SimpleArrayFloat32(array=array)


def _cross(point, half):
    """Segment ends of a 3D cross of half size ``half`` at ``point``."""
    point = np.asarray(point, dtype='float64')
    ends = []
    for axis in np.eye(3):
        ends += [point - half * axis, point + half * axis]
    return ends


def _fmt(vector, sign=False):
    spec = "+7.1f" if sign else "7.1f"
    return "(" + ", ".join(format(float(v), spec) for v in vector) + ")"


def _domain_handle(container):
    """The Qt handle of the RDomainWidget inside a pilot 3D container."""
    for child in container.findChildren(QWidget):
        if 'solvcon::RDomainWidget' == child.metaObject().className():
            return child
    return None


class Pane(object):
    """One 3D widget of the view and its Qt handle.

    :ivar widget: The pilot ``RDomainWidget`` (its Python wrapper).
    :ivar handle: The ``QWidget`` handle of the same widget, for events,
        overlays, and layouts.
    :ivar fixed: The view name a pane of the four-view layout keeps, or
        None for the main pane.
    """

    def __init__(self, widget, handle, fixed=None):
        self.widget = widget
        self.handle = handle
        self.fixed = fixed
        self.label = None

    def state(self):
        return nav.make_state(self.widget.cameraPosition,
                              self.widget.cameraTarget,
                              self.widget.cameraUp)

    def set_state(self, state):
        self.widget.cameraTarget = tuple(float(v) for v in state.target)
        self.widget.cameraPosition = tuple(float(v) for v in state.position)
        self.widget.cameraUp = tuple(float(v) for v in state.up)

    def aspect(self):
        height = self.handle.height()
        return self.handle.width() / height if height > 0 else 1.0


class CubeOverlay(QWidget):
    """The orientation cube in the top right corner of the main pane: the
    faces turn with the camera, and a click on a face, an edge, or a corner
    turns the view there."""

    SIZE = 34.0
    EXTENT = 120

    FACE_COLOR = QColor(235, 238, 245, 225)
    EDGE_COLOR = QColor(90, 90, 110)

    def __init__(self, view, parent):
        super().__init__(parent)
        self._view = view
        self.setFixedSize(self.EXTENT, self.EXTENT)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click a face, an edge, or a corner to view the "
                        "car from there")

    def center(self):
        return np.array([self.EXTENT / 2.0, self.EXTENT / 2.0])

    def paintEvent(self, event):
        state = self._view.main_state()
        if state is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont(painter.font())
        font.setPointSizeF(7.5)
        font.setBold(True)
        painter.setFont(font)
        for name, polygon, facing in nav.cube_faces(state, self.center(),
                                                    self.SIZE):
            if facing <= 1.e-6:
                continue
            shade = int(200 + 55 * facing)
            painter.setBrush(QColor(shade - 10, shade - 6, shade, 230))
            painter.setPen(QPen(self.EDGE_COLOR, 1.2))
            painter.drawPolygon(QPolygonF([QPointF(*p) for p in polygon]))
            # A label is drawn only where it fits its face.
            width = float(np.ptp(polygon[:, 0]))
            text = name.upper()
            if facing > 0.3 and painter.fontMetrics().horizontalAdvance(
                    text) < 0.9 * width:
                mid = polygon.mean(axis=0)
                painter.setPen(QColor(30, 30, 60))
                painter.drawText(int(mid[0]) - 30, int(mid[1]) - 8, 60, 16,
                                 Qt.AlignCenter, text)
        painter.end()

    def mousePressEvent(self, event):
        state = self._view.main_state()
        if state is None:
            return
        pos = event.position()
        direction = nav.cube_hit(state, self.center(), self.SIZE,
                                 (pos.x(), pos.y()))
        if direction is not None:
            self._view.view_from(direction)


class _PaneFilter(QObject):
    """Mouse events of a pane: hover readout, click pick, and the overlay
    positions on resize.  It never consumes an event, so the widget keeps
    navigating."""

    CLICK_PIXELS = 4.0

    def __init__(self, view, pane):
        super().__init__(pane.handle)
        self._view = view
        self._pane = pane
        self._press = None

    def eventFilter(self, obj, event):
        kind = event.type()
        if QEvent.MouseMove == kind and Qt.NoButton == event.buttons():
            pos = event.position()
            self._view.hover(self._pane, (pos.x(), pos.y()))
        elif (QEvent.MouseButtonPress == kind
              and Qt.LeftButton == event.button()):
            pos = event.position()
            self._press = (pos.x(), pos.y())
        elif (QEvent.MouseButtonRelease == kind
              and Qt.LeftButton == event.button()
              and self._press is not None):
            pos = event.position()
            moved = np.hypot(pos.x() - self._press[0],
                             pos.y() - self._press[1])
            self._press = None
            if moved <= self.CLICK_PIXELS:
                self._view.click(self._pane, (pos.x(), pos.y()))
        elif QEvent.Resize == kind:
            self._view.place_overlays()
        return False


class CornerView(object):
    """Manage the 3D sub-window of the suspension app.

    :ivar widget: The main pilot ``RDomainWidget``, or None while closed.
    :ivar closed: Called after the sub-window closes, when set.
    :ivar picked: Called with the name of a hardpoint clicked in 3D (or
        None for a click on nothing), when set.
    :ivar transition_ms: The duration of an animated view change; 0 snaps.
    :ivar projection_mode: ``auto`` (orthographic in a face view,
        perspective otherwise), ``perspective``, or ``parallel``.
    """

    #: The home view direction from the target: from ahead, a little
    #: inboard, and above, so the wheel stands behind the links.
    VIEW_DIRECTION = (1.0, -0.45, 0.55)
    #: Color and opacity of the named body meshes that are not the default
    #: surface color: a dark rubber tire and a light alloy rim, both see
    #: through to the upright inside the wheel.
    BODY_LOOKS = {'wheel': ((0.16, 0.16, 0.18), 0.55),
                  'rim': ((0.80, 0.82, 0.86), 0.6)}
    #: Length in mm of the longest force arrow.
    ARROW_LENGTH = 150.0
    #: A hardpoint is picked within this many pixels of the cursor.
    PICK_PIXELS = 10.0
    #: Half size in mm of the hover and the selection markers.
    MARKER_HALF = 25.0
    #: The camera watch period, in ms.
    WATCH_MS = 40
    #: The views of the four-view layout, around the main pane.
    QUAD_VIEWS = (('front', 0, 0), ('left', 0, 1), ('top', 1, 0))

    def __init__(self, mgr):
        self._mgr = mgr
        self._mdi = None
        self.widget = None
        self.subwin = None
        self.closed = None
        self.picked = None
        self._close_filter = None
        self._camera = None
        self._meshes = set()
        self.panes = []
        self._filters = []
        self._grid_host = None
        self.toolbar = None
        self.cube = None
        self.hud = None
        self.help = None
        self.actions = {}
        self.history = nav.ViewHistory()
        self.transition_ms = 250
        self.projection_mode = 'auto'
        self.mouse_mapping = 'blender'
        self.grid_shown = True
        self.quad = False
        self.measuring = False
        self.measure_points = []
        self.measure_text = ""
        self.points = {}
        self.hovered = None
        self.selected_point = None
        self._bbox = None
        self._grid = None
        self._grid_ends = []
        self._geometries = []
        self._last_pose = None
        self._last_states = {}
        self._anim = None
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(15)
        self._anim_timer.timeout.connect(self._animate_step)
        self._watch = QTimer()
        self._watch.setInterval(self.WATCH_MS)
        self._watch.timeout.connect(self._watch_camera)

    @property
    def is_open(self):
        return self.widget is not None

    # Opening and closing.

    def open(self, title):
        if self.widget is not None:
            return
        self.widget = self._mgr.add3DWidget()
        if self._mdi is None:
            self._mdi = self._mgr.mdiArea
        self.subwin = self._mdi.activeSubWindow()
        if self.subwin is None:
            return
        self.subwin.setWindowTitle(title)
        self.subwin.setAttribute(Qt.WA_DeleteOnClose, True)
        self._close_filter = base.SubWindowCloseFilter(self._on_closed,
                                                       self.subwin)
        self.subwin.installEventFilter(self._close_filter)
        container = self.subwin.widget()
        handle = _domain_handle(container)
        if handle is None:
            return
        layout = container.layout()
        layout.removeWidget(handle)
        self._grid_host = QWidget()
        grid = QGridLayout(self._grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        grid.addWidget(handle, 1, 1)
        self.toolbar = self._build_toolbar()
        layout.addWidget(self.toolbar)
        layout.addWidget(self._grid_host)
        main = Pane(self.widget, handle)
        self._add_pane(main)
        self.cube = CubeOverlay(self, handle)
        self.hud = QLabel(handle)
        self.hud.setStyleSheet(
            "QLabel { background: rgba(255, 255, 255, 215); color: #202030;"
            " border: 1px solid #b0b0c0; padding: 3px; "
            "font-family: monospace; font-size: 8pt; }")
        self.hud.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.help = QLabel(handle)
        self.help.setStyleSheet(
            "QLabel { background: rgba(250, 250, 255, 240); color: #101020;"
            " border: 1px solid #8080a0; padding: 10px; font-size: 10pt; }")
        self.help.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.help.hide()
        self.cube.show()
        self.hud.show()
        self.place_overlays()
        self._watch.start()

    def _add_pane(self, pane):
        widget = pane.widget
        widget.cameraMode = "orbit"
        widget.worldUp = (0.0, 0.0, 1.0)
        widget.zoomToCursor = True
        widget.showAxis(True)
        widget.navigationMapping = self.mouse_mapping
        pane.handle.setMouseTracking(True)
        pane.handle.setFocusPolicy(Qt.StrongFocus)
        filt = _PaneFilter(self, pane)
        pane.handle.installEventFilter(filt)
        self._filters.append(filt)
        self.panes.append(pane)

    def close(self):
        if self.subwin is not None:
            self.subwin.close()
        else:
            self._on_closed()

    def _on_closed(self):
        self._watch.stop()
        self._anim_timer.stop()
        self._anim = None
        self._meshes = set()
        self.widget = None
        self.subwin = None
        self._close_filter = None
        self.panes = []
        self._filters = []
        self._grid_host = None
        self.toolbar = self.cube = self.hud = self.help = None
        self.actions = {}
        self.points = {}
        self.hovered = self.selected_point = None
        self.measure_points = []
        if self.closed is not None:
            self.closed()

    # The toolbar and the keys.

    def _action(self, key, text, func, keys=(), tip=None, checkable=False):
        action = QAction(text, self.subwin)
        if keys:
            action.setShortcuts([QKeySequence(k) for k in keys])
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        hint = tip or text
        if keys:
            hint += f" ({', '.join(keys)})"
        action.setToolTip(hint)
        action.setCheckable(checkable)
        if checkable:
            action.toggled.connect(func)
        else:
            action.triggered.connect(lambda _=False: func())
        self.subwin.addAction(action)
        self.actions[key] = action
        return action

    def _build_toolbar(self):
        bar = QToolBar("3D view")
        bar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        for key, text, keys in VIEW_KEYS:
            bar.addAction(self._action(
                key, text, lambda name=key: self.set_view(name), keys,
                tip=f"{text} view"))
        bar.addSeparator()
        bar.addAction(self._action('fit', "Fit all", self.fit_all, ("F",),
                                   tip="Frame the whole corner"))
        bar.addAction(self._action(
            'fit_selection', "Fit pick", self.fit_selection, ("Shift+F",),
            tip="Center and frame the picked hardpoint"))
        bar.addAction(self._action('back', "Back", self.back,
                                   ("Alt+Left",), tip="Previous view"))
        bar.addAction(self._action('forward', "Fwd", self.forward,
                                   ("Alt+Right",), tip="Next view"))
        bar.addSeparator()
        self.projection_box = QComboBox()
        for mode, text in (('auto', "Auto ortho"),
                           ('perspective', "Perspective"),
                           ('parallel', "Orthographic")):
            self.projection_box.addItem(text, mode)
        self.projection_box.setToolTip(
            "Auto ortho: orthographic in the front, side, and top views, "
            "perspective while orbiting (5 toggles)")
        self.projection_box.currentIndexChanged.connect(
            lambda _: self.set_projection_mode(
                self.projection_box.currentData()))
        bar.addWidget(self.projection_box)
        self._action('toggle_projection', "Ortho/persp",
                     self.toggle_projection, ("5",))
        bar.addSeparator()
        grid = self._action('grid', "Grid", self.set_grid, ("G",),
                            tip="Ground grid at the road plane",
                            checkable=True)
        grid.setChecked(True)
        bar.addAction(grid)
        bar.addAction(self._action('measure', "Measure", self.set_measuring,
                                   ("M",), checkable=True,
                                   tip="Click 2 hardpoints for a distance, "
                                       "3 for an angle"))
        bar.addAction(self._action('quad', "4 views", self.set_quad,
                                   ("Q",), checkable=True,
                                   tip="Front, side, top, and 3D views"))
        bar.addSeparator()
        # Parented, so the menu and its actions live as long as the view.
        menu = QMenu(self.subwin)
        self._options_menu = menu
        menu.addAction(self._action('save_view', "Save view...",
                                    self._ask_save_view))
        self._views_menu = menu.addMenu("Saved views")
        menu.addSeparator()
        mouse = menu.addMenu("Mouse")
        group = QActionGroup(mouse)
        for key, (text, _) in MOUSE_MAPPINGS.items():
            action = mouse.addAction(text)
            action.setCheckable(True)
            action.setChecked(key == self.mouse_mapping)
            action.triggered.connect(
                lambda _=False, name=key: self.set_mouse_mapping(name))
            group.addAction(action)
            self.actions['mouse_' + key] = action
        orbit = menu.addMenu("Orbit")
        group = QActionGroup(orbit)
        for key, text in (('turntable', "Turntable (z stays up)"),
                          ('trackball', "Trackball (free tumble)")):
            action = orbit.addAction(text)
            action.setCheckable(True)
            action.setChecked('turntable' == key)
            action.triggered.connect(
                lambda _=False, name=key: self.set_orbit_style(name))
            group.addAction(action)
            self.actions['orbit_' + key] = action
        transitions = menu.addMenu("View change")
        group = QActionGroup(transitions)
        for ms, text in ((0, "Snap"), (250, "Animate 0.25 s"),
                         (500, "Animate 0.5 s")):
            action = transitions.addAction(text)
            action.setCheckable(True)
            action.setChecked(ms == self.transition_ms)
            action.triggered.connect(
                lambda _=False, value=ms: setattr(self, 'transition_ms',
                                                  value))
            group.addAction(action)
        zoom = menu.addAction("Zoom toward the cursor")
        zoom.setCheckable(True)
        zoom.setChecked(True)
        zoom.toggled.connect(self.set_zoom_to_cursor)
        self.actions['zoom_to_cursor'] = zoom
        button = QToolButton()
        button.setText("View options")
        button.setMenu(menu)
        button.setPopupMode(QToolButton.InstantPopup)
        bar.addWidget(button)
        bar.addAction(self._action('help', "?", self.toggle_help,
                                   ("F1", "?"), tip="Navigation help"))
        for key, yaw, pitch, keys in (
                ('orbit_left', -15.0, 0.0, ("Left",)),
                ('orbit_right', 15.0, 0.0, ("Right",)),
                ('orbit_up', 0.0, 15.0, ("Up",)),
                ('orbit_down', 0.0, -15.0, ("Down",)),
                ('orbit_left90', -90.0, 0.0, ("Shift+Left",)),
                ('orbit_right90', 90.0, 0.0, ("Shift+Right",)),
                ('orbit_up90', 0.0, 90.0, ("Shift+Up",)),
                ('orbit_down90', 0.0, -90.0, ("Shift+Down",))):
            self._action(key, key, lambda y=yaw, p=pitch: self.orbit(y, p),
                         keys)
        self._action('escape', "Clear pick", self.clear_pick, ("Escape",))
        return bar

    # Camera state and the view commands.

    def main_state(self):
        return self.panes[0].state() if self.panes else None

    def _fit_distance(self, pane, radius):
        half_v = np.radians(22.5)
        half_h = np.arctan(np.tan(half_v) * pane.aspect())
        return radius / np.tan(min(half_v, half_h)) * 1.05

    def _fit_state(self, pane, eye, up, lo=None, hi=None):
        """The view from the unit direction ``eye`` that frames the box
        from ``lo`` to ``hi`` (the scene by default): the box corners
        projected on the screen axes set the distance, so a face view of a
        flat corner fills the pane instead of fitting its bounding sphere."""
        if lo is None:
            if self._bbox is None:
                return None
            lo, hi = self._bbox
        lo = np.asarray(lo, dtype='float64')
        hi = np.asarray(hi, dtype='float64')
        target = 0.5 * (lo + hi)
        eye = eye / np.linalg.norm(eye)
        up = np.asarray(up, dtype='float64')
        up = up - (up @ eye) * eye
        up /= np.linalg.norm(up)
        right = np.cross(-eye, up)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])]) - target
        half_w = float(np.max(np.abs(corners @ right)))
        half_h = float(np.max(np.abs(corners @ up)))
        depth = float(np.max(corners @ eye))
        need = max(half_h, half_w / pane.aspect(), 1.0)
        distance = need / np.tan(np.radians(22.5)) * 1.1 + depth
        return nav.CameraState(target + eye * distance, target, up)

    def _scene_target(self):
        if self._bbox is None:
            return np.zeros(3), 1000.0
        lo, hi = self._bbox
        return 0.5 * (lo + hi), 0.5 * float(np.linalg.norm(hi - lo))

    def home_state(self):
        if self._camera is None:
            return None
        position, target, up = self._camera
        return nav.make_state(position, target, up)

    def set_view(self, name, pane=None):
        """Turn the view to a named vehicle view (or ``home``), keeping the
        target and the distance."""
        pane = pane or self.panes[0]
        if 'home' == name:
            state = self.home_state()
        else:
            now = pane.state()
            distance = float(np.linalg.norm(now.position - now.target))
            state = nav.view_state(name, now.target, distance)
        if state is not None:
            self.animate_to(state, pane)

    def view_from(self, direction):
        """Turn the main view to look from ``direction`` (a cube click)."""
        now = self.main_state()
        distance = float(np.linalg.norm(now.position - now.target))
        self.animate_to(nav.state_for_direction(direction, now.target,
                                                distance))

    def fit_all(self, pane=None):
        """Frame the whole corner, keeping the view direction."""
        pane = pane or self.panes[0]
        now = pane.state()
        state = self._fit_state(pane, now.position - now.target, now.up)
        if state is not None:
            self.animate_to(state, pane)

    def fit_selection(self):
        """Center the picked hardpoint and frame its neighborhood."""
        if self.selected_point not in self.points:
            self.fit_all()
            return
        pane = self.panes[0]
        target = np.asarray(self.points[self.selected_point][1])
        now = pane.state()
        eye = now.position - now.target
        eye /= np.linalg.norm(eye)
        distance = self._fit_distance(pane, 150.0)
        self.animate_to(nav.CameraState(target + eye * distance, target,
                                        now.up), pane)

    def orbit(self, yaw, pitch):
        pane = self.panes[0]
        self.history.push(pane.state())
        pane.widget.orbitStep(yaw, pitch)

    def back(self):
        state = self.history.back(self.main_state())
        if state is not None:
            self.animate_to(state, record=False)

    def forward(self):
        state = self.history.forward(self.main_state())
        if state is not None:
            self.animate_to(state, record=False)

    def save_view(self, name):
        self.history.save(name, self.main_state())
        self._views_menu.clear()
        for saved in self.history.names():
            action = self._views_menu.addAction(saved)
            action.triggered.connect(
                lambda _=False, n=saved: self.restore_view(n))

    def restore_view(self, name):
        self.animate_to(self.history.named(name))

    def _ask_save_view(self):
        name, ok = QInputDialog.getText(None, "Save view", "View name:")
        if ok and name:
            self.save_view(name)

    def animate_to(self, state, pane=None, record=True):
        """Move a pane camera to ``state``, animated over
        :attr:`transition_ms`."""
        pane = pane or self.panes[0]
        start = pane.state()
        if record and pane is self.panes[0]:
            self.history.push(start)
        self._anim_timer.stop()
        if self.transition_ms <= 0:
            self._anim = None
            pane.set_state(state)
            self._watch_camera()
            return
        self._anim = (pane, start, state, 0.0)
        self._anim_timer.start()

    def _animate_step(self):
        if self._anim is None:
            self._anim_timer.stop()
            return
        pane, start, end, elapsed = self._anim
        elapsed += self._anim_timer.interval()
        frac = min(1.0, elapsed / float(self.transition_ms))
        pane.set_state(nav.interpolate(start, end, nav.ease(frac)))
        if frac >= 1.0:
            pane.set_state(end)
            self._anim = None
            self._anim_timer.stop()
        else:
            self._anim = (pane, start, end, elapsed)
        self._watch_camera()

    def finish_animation(self):
        """Jump to the end of a running view change (for tests and before
        a new command)."""
        if self._anim is not None:
            pane, _, end, _ = self._anim
            pane.set_state(end)
            self._anim = None
            self._anim_timer.stop()
            self._watch_camera()

    # Settings.

    def set_projection_mode(self, mode):
        self.projection_mode = mode
        index = self.projection_box.findData(mode)
        if index >= 0 and index != self.projection_box.currentIndex():
            self.projection_box.setCurrentIndex(index)
        self._last_states = {}
        self._watch_camera()

    def toggle_projection(self):
        pane = self.panes[0]
        current = pane.widget.projection
        self.set_projection_mode('perspective' if 'parallel' == current
                                 else 'parallel')

    def projection_for(self, pane):
        if pane.fixed is not None:
            return 'parallel'
        if 'auto' == self.projection_mode:
            return ('parallel' if nav.aligned_view(pane.state()) is not None
                    else 'perspective')
        return self.projection_mode

    def set_mouse_mapping(self, name):
        self.mouse_mapping = name
        for pane in self.panes:
            pane.widget.navigationMapping = name
        action = self.actions.get('mouse_' + name)
        if action is not None and not action.isChecked():
            action.setChecked(True)
        self._update_help()

    def set_orbit_style(self, name):
        for pane in self.panes:
            pane.widget.orbitStyle = name

    def set_zoom_to_cursor(self, on):
        for pane in self.panes:
            pane.widget.zoomToCursor = bool(on)

    def set_grid(self, shown):
        self.grid_shown = bool(shown)
        self._draw_grid()

    def set_measuring(self, on):
        self.measuring = bool(on)
        self.measure_points = []
        self.measure_text = ""
        for pane in self.panes:
            pane.widget.clearMeasurements()
        self._update_hud()

    def clear_pick(self):
        self.selected_point = None
        self.measure_points = []
        self.measure_text = ""
        for pane in self.panes:
            pane.widget.clearMeasurements()
        self._draw_markers()
        self._update_hud()

    def toggle_help(self):
        if self.help is None:
            return
        self._update_help()
        self.help.setVisible(not self.help.isVisible())
        self.place_overlays()

    def _update_help(self):
        if self.help is None:
            return
        rows = [("Views", "1 front, 3 left, 7 top, Ctrl+1/3/7 the "
                          "opposite, 0 iso, Home"),
                ("Cube", "click a face, an edge, or a corner"),
                ("Orbit step", "arrows 15 deg, Shift+arrows 90 deg"),
                ("Mouse", MOUSE_MAPPINGS[self.mouse_mapping][1]),
                ("Wheel", "zoom toward the cursor"),
                ("Frame", "F all, Shift+F the picked point"),
                ("Projection", "5 toggles orthographic and perspective"),
                ("History", "Alt+Left back, Alt+Right forward"),
                ("Pick", "click a hardpoint; Esc clears"),
                ("Measure", "M, then click 2 or 3 hardpoints"),
                ("Layout", "Q four views, G ground grid")]
        body = "".join(f"<tr><td><b>{k}</b>&nbsp;&nbsp;</td><td>{v}</td>"
                       f"</tr>" for k, v in rows)
        self.help.setText(f"<b>3D navigation</b> (F1 closes)<table>{body}"
                          f"</table>")
        self.help.adjustSize()

    # The four-view layout.

    def set_quad(self, on):
        on = bool(on)
        if on and len(self.panes) == 1:
            self._create_quad_panes()
        for pane in self.panes[1:]:
            pane.handle.setVisible(on)
        self.quad = on
        action = self.actions.get('quad')
        if action is not None and action.isChecked() != on:
            action.setChecked(on)

    def _create_quad_panes(self):
        grid = self._grid_host.layout()
        for name, row, col in self.QUAD_VIEWS:
            widget = self._mgr.add3DWidget()
            subwin = self._mdi.activeSubWindow()
            handle = _domain_handle(subwin.widget())
            subwin.widget().layout().removeWidget(handle)
            grid.addWidget(handle, row, col)
            subwin.close()
            pane = Pane(widget, handle, fixed=name)
            self._add_pane(pane)
            pane.label = QLabel(name.capitalize(), handle)
            pane.label.setStyleSheet(
                "QLabel { background: rgba(255, 255, 255, 200); "
                "padding: 2px; font-weight: bold; }")
            pane.label.move(6, 6)
            pane.label.show()
        self._mdi.setActiveSubWindow(self.subwin)
        self._build_meshes(self.panes[1:])
        for pane in self.panes[1:]:
            self._aim_fixed(pane)
        # The panes take their size when the grid lays them out; frame them
        # again for that size.
        QTimer.singleShot(0, self._refit_fixed)
        if self._last_pose is not None:
            self.show_pose(*self._last_pose)

    def _aim_fixed(self, pane):
        eye, up = nav.VIEWS[pane.fixed]
        state = self._fit_state(pane, eye, up)
        if state is not None:
            pane.set_state(state)
        pane.widget.projection = 'parallel'

    def _refit_fixed(self):
        for pane in self.panes[1:]:
            self._aim_fixed(pane)

    # Overlays.

    def place_overlays(self):
        if not self.panes:
            return
        handle = self.panes[0].handle
        if self.cube is not None:
            self.cube.move(handle.width() - self.cube.width() - 4, 4)
            self.cube.raise_()
        if self.hud is not None:
            # Bottom right: the cube holds the top right corner and the axis
            # triad the bottom left one.
            self.hud.adjustSize()
            self.hud.move(max(0, handle.width() - self.hud.width() - 6),
                          max(0, handle.height() - self.hud.height() - 6))
            self.hud.raise_()
        if self.help is not None and self.help.isVisible():
            self.help.adjustSize()
            self.help.move(max(0, (handle.width() - self.help.width()) // 2),
                           max(0, (handle.height() - self.help.height())
                               // 2))
            self.help.raise_()

    def _view_name(self, pane):
        name = nav.aligned_view(pane.state())
        return f"{name} view" if name is not None else "free view"

    def hud_text(self):
        """The readout of the main pane: the view, the projection, the grid,
        the axes, and the hovered and picked hardpoints."""
        if not self.panes:
            return ""
        pane = self.panes[0]
        projection = ('orthographic' if 'parallel' == pane.widget.projection
                      else 'perspective')
        line = f"{self._view_name(pane)}, {projection}"
        if self.grid_shown and self._grid is not None:
            line += f", grid {self._grid:g} mm"
        lines = [line, "axes: x forward, y left, z up (mm)"]
        for title, name in (("hover", self.hovered),
                            ("pick", self.selected_point)):
            if name in self.points and not (title == 'pick'
                                            and name == self.hovered):
                design, now = self.points[name]
                lines.append(f"{title} {name}: design {_fmt(design)}")
                lines.append(f"  now {_fmt(now)}  delta "
                             f"{_fmt(np.asarray(now) - design, sign=True)}")
        if self.measuring:
            picked = ", ".join(self.measure_points) or "click hardpoints"
            lines.append(f"measure: {picked}")
            if self.measure_text:
                lines.append(f"  {self.measure_text}")
        return "\n".join(lines)

    def _update_hud(self):
        if self.hud is None:
            return
        self.hud.setText(self.hud_text())
        self.place_overlays()

    def _watch_camera(self):
        """Follow the camera of every pane: the projection of the auto mode,
        the cube, and the readout."""
        changed = False
        for pane in self.panes:
            state = pane.state()
            last = self._last_states.get(id(pane))
            if last is not None and nav.same_state(last, state):
                continue
            self._last_states[id(pane)] = state
            projection = self.projection_for(pane)
            if pane.widget.projection != projection:
                pane.widget.projection = projection
            changed = True
        if changed:
            if self.cube is not None:
                self.cube.update()
            self._update_hud()

    # Picking.

    def _pixels(self, pane):
        return {name: tuple(pane.widget.projectPoint(*[float(v) for v in
                                                       now]))
                for name, (_, now) in self.points.items()}

    def pick(self, pane, cursor):
        """The hardpoint under ``cursor`` in ``pane``, or None."""
        return nav.nearest(self._pixels(pane), cursor, self.PICK_PIXELS)

    def hover(self, pane, cursor):
        name = self.pick(pane, cursor)
        if name != self.hovered:
            self.hovered = name
            self._draw_markers()
            self._update_hud()

    def click(self, pane, cursor):
        name = self.pick(pane, cursor)
        if self.measuring:
            if name is not None:
                if len(self.measure_points) >= 3:
                    self.measure_points = []
                self.measure_points.append(name)
                self._draw_measure()
        else:
            self.selected_point = name
            self._draw_markers()
            if self.picked is not None:
                self.picked(name)
        self._update_hud()

    def select_point(self, name):
        """Select a hardpoint from the app (the table), or None."""
        self.selected_point = name if name in self.points else None
        self._draw_markers()
        self._update_hud()

    def _draw_markers(self):
        for pane in self.panes:
            for group, name, color in (("hover", self.hovered, HOVER_COLOR),
                                       ("pick", self.selected_point,
                                        PICK_COLOR)):
                ends = []
                if name in self.points:
                    ends = _cross(self.points[name][1], self.MARKER_HALF)
                pane.widget.setObjectSegments(group, _points(ends), *color)

    def _draw_measure(self):
        names = [n for n in self.measure_points if n in self.points]
        coords = [tuple(float(v) for v in self.points[n][1]) for n in names]
        self.measure_text = ""
        for pane in self.panes:
            pane.widget.clearMeasurements()
            if 2 == len(coords):
                value = pane.widget.measureDistance(coords[0], coords[1])
                self.measure_text = f"distance {value:.2f} mm"
            elif 3 == len(coords):
                value = pane.widget.measureAngle(*coords)
                self.measure_text = f"angle at {names[1]} {value:.2f} deg"

    def measure_value(self):
        """The distance (mm) or the angle (deg) of the picked hardpoints,
        computed from their current positions, or None."""
        coords = [np.asarray(self.points[n][1]) for n in self.measure_points
                  if n in self.points]
        if 2 == len(coords):
            return float(np.linalg.norm(coords[1] - coords[0]))
        if 3 == len(coords):
            a, b = coords[0] - coords[1], coords[2] - coords[1]
            cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
            return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        return None

    # The scene.

    def _build_meshes(self, panes):
        shown = set()
        for pane in panes:
            for geometry in self._geometries:
                for name, mesh in geometry.body_meshes().items():
                    pane.widget.addObject(name, mesh)
                    shown.add(name)
                for body, (color, opacity) in self.BODY_LOOKS.items():
                    pane.widget.setObjectColor(body + geometry.suffix,
                                               *color)
                    pane.widget.setObjectOpacity(body + geometry.suffix,
                                                 opacity)
            for name in self._meshes - shown:
                pane.widget.setObjectVisible(name, False)
        return shown

    def build(self, geometries, aim=True):
        """Register the body meshes of every corner in ``geometries`` and
        hide those of a corner no longer posed; aim the camera at them when
        ``aim``."""
        if self.widget is None:
            return
        self._geometries = list(geometries)
        # Adding a mesh reframes the widget camera to its default; keep the
        # camera the user had.
        before = [pane.state() for pane in self.panes]
        self._meshes |= self._build_meshes(self.panes)
        for pane, state in zip(self.panes, before):
            pane.set_state(state)
        points = []
        for geometry in geometries:
            radius = geometry.model.wheel_radius * _scene.MM
            wheel = geometry.origin('upright')
            points += list(geometry.chassis_pickups())
            points += [wheel - radius, wheel + radius]
        lo = np.min(points, axis=0)
        hi = np.max(points, axis=0)
        self._bbox = (lo, hi)
        geometry = geometries[0]
        road = (geometry.origin('upright')[2]
                - geometry.model.wheel_radius * _scene.MM)
        margin = 0.15 * float(np.max(hi - lo))
        self._grid_ends, self._grid = nav.grid_segments(
            lo[:2] - margin, hi[:2] + margin, road)
        self._draw_grid()
        if not aim:
            return
        target = 0.5 * (lo + hi)
        direction = np.array(self.VIEW_DIRECTION, dtype='float64')
        direction /= np.linalg.norm(direction)
        distance = 1.05 * np.linalg.norm(hi - lo)
        self._camera = (tuple(target + distance * direction), tuple(target),
                        (0.0, 0.0, 1.0))
        self.aim()
        for pane in self.panes[1:]:
            self._aim_fixed(pane)

    def _draw_grid(self):
        ends = self._grid_ends if (self.grid_shown
                                   and self._grid is not None) else []
        for pane in self.panes:
            pane.widget.setObjectSegments("grid", _points(ends), *GRID_COLOR)
        self._update_hud()

    def aim(self):
        """Put the camera at the app's home view of the corner."""
        if self.widget is None or self._camera is None:
            return
        position, target, up = self._camera
        self.widget.cameraTarget = target
        self.widget.cameraPosition = position
        self.widget.cameraUp = up
        self._watch_camera()

    def show_pose(self, geometries, pose, selected=None, forces=None,
                  overlays=None):
        """Move the bodies of every corner to ``pose`` and redraw the
        segments in every pane.

        :param selected: The name of a link to highlight, or None.
        :param forces: ``[(name, point_mm, force_N)]`` to draw as arrows,
            or None.
        :param overlays: ``{name: segment ends (mm)}`` of the geometry
            overlays of :data:`OVERLAYS` to draw; the others are cleared.
        """
        if self.widget is None or pose is None:
            return
        self._last_pose = (geometries, pose, selected, forces, overlays)
        segments = {}
        points = {}
        for geometry in geometries:
            segments.update(geometry.link_segments(pose))
            points.update(geometry.hardpoints(pose))
        self.points = points
        springs = [name for name in segments
                   if name in {'spring' + g.suffix for g in geometries}]
        # The highlight is left out of its own group: a line drawn twice
        # at the same depth keeps the color drawn first.
        links = [end for name, ends in segments.items()
                 if name not in springs and name != selected
                 for end in ends]
        spring = [end for name in springs if name != selected
                  for end in segments[name]]
        chosen = segments.get(selected, ())
        arb = []
        if 2 == len(geometries) and 'arb' in pose:
            arb = _scene.arb_segments(geometries[0], geometries[1], pose)
        road = []
        for geometry in geometries:
            road += _scene.road_segments(geometry, pose)
        arrows = []
        if forces:
            largest = max(np.linalg.norm(f) for _, _, f in forces)
            scale = self.ARROW_LENGTH / largest if largest > 0.0 else 0.0
            for _, point, force in forces:
                arrows.extend(_scene.arrow_segments(point, force * scale))
        overlays = overlays or {}
        for pane in self.panes:
            widget = pane.widget
            before = pane.state()
            for geometry in geometries:
                for name, (pos, quat) in geometry.object_poses(pose).items():
                    trans = np.asarray(pos, dtype='float64') * _scene.MM
                    widget.setObjectTransform(
                        name, *[float(v) for v in trans], 1.0, 1.0, 1.0,
                        *[float(v) for v in quat])
            widget.setObjectSegments("links", _points(links), *LINK_COLOR)
            widget.setObjectSegments("spring", _points(spring),
                                     *SPRING_COLOR)
            widget.setObjectSegments("selected", _points(chosen),
                                     *SELECT_COLOR)
            widget.setObjectSegments("arb", _points(arb), *SPRING_COLOR)
            widget.setObjectSegments("road", _points(road), *ROAD_COLOR)
            widget.setObjectSegments("forces", _points(arrows), *FORCE_COLOR)
            for name, color in OVERLAYS.items():
                widget.setObjectSegments(name, _points(overlays.get(name, [])),
                                         *color)
            # The widget reframes when a pose grows the framing box, which
            # resets its camera; keep the camera the user had.
            if not nav.same_state(before, pane.state()):
                pane.set_state(before)
        self._draw_markers()
        if self.measuring and self.measure_points:
            self._draw_measure()
        self._update_hud()

    def model_of(self, body):
        """The 4x4 model transform of a body object, row-major, or None."""
        if self.widget is None:
            return None
        values = self.widget.objectModel(body)
        if not values:
            return None
        return np.array(values, dtype='float64').reshape(4, 4)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
