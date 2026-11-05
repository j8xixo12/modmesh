# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The suspension K&C session: what the app shows, held without Qt.

A session owns one corner model, its adjusters, and the analyses run on it.
The GUI, a script, and a test drive it the same way::

    sess = SuspensionSession(_engine.load())
    sess.run_sweep()
    sess.set_adjuster('camber_shim', 2.0)   # mm; re-runs the sweep
    sess.current.curve('camber')            # the new curve, in deg
    sess.previous.curve('camber')           # the one before, drawn dashed

Two rules of the UI live here, so that they hold whatever drives the
session:

- A failed analysis never replaces a result.  The last valid result and its
  pose stay current, and the failure goes to :attr:`warnings` with its
  reason.
- An adjuster change keeps the result it replaces as :attr:`previous`, so
  the view can draw the effect of the change.

The engine works in SI units.  The session keeps the results in SI too, and
converts to the display units (mm, deg, N, N/mm) only where a curve is read
out or exported.
"""

import copy
import json
import math
import os
import sys

import numpy as np

from . import _analysis
from . import _scene

__all__ = [
    'ADJUSTERS',
    'RIGHT',
    'Rig',
    'CURVES',
    'SWEEPS',
    'Curve',
    'QuarterCarRun',
    'ROAD_INPUTS',
    'make_road_input',
    'SuspensionSession',
    'SweepKind',
    'SweepResult',
]

DEG = 180.0 / math.pi
MM = 1.e3


def _arrays(result):
    """An engine result dict with its arrays read as numpy arrays (the
    engine returns SimpleArray, which numpy reads without a copy)."""
    return {key: (value if value is None
                  or isinstance(value, (str, bool, int, float))
                  else np.asarray(value)) for key, value in result.items()}


def _fmt_mm(xyz):
    return "(" + ", ".join(f"{v:.2f}" for v in xyz) + ") mm"


def _num(value):
    return "-" if value is None else f"{value:+.5g}"


class Curve(object):
    """One K&C quantity: its key in the result, its title, its display unit,
    and the factor from SI to that unit."""

    def __init__(self, key, title, unit, scale):
        self.key = key
        self.title = title
        self.unit = unit
        self.scale = scale

    @property
    def label(self):
        return f"{self.title} [{self.unit}]"


class SweepKind(object):
    """A sweep: the driven quantity and the curves it produces.

    :ivar xkey: The key of the swept value in the result of the suspension
        package, in m.
    :ivar default: ``(lo, hi, count)`` of the sweep, in mm.
    """

    def __init__(self, name, xkey, title, default, curves):
        self.name = name
        self.xkey = xkey
        self.x = Curve(xkey, title, 'mm', MM)
        self.default = default
        self.curves = curves


#: The curves of each sweep, in the order they are plotted.
CURVES = {
    'bump': (
        Curve('camber', "Camber", 'deg', DEG),
        Curve('toe', "Toe-in", 'deg', DEG),
        Curve('caster', "Caster", 'deg', DEG),
        Curve('kpi', "KPI", 'deg', DEG),
        Curve('scrub', "Scrub radius", 'mm', MM),
        Curve('trail', "Mech. trail", 'mm', MM),
        Curve('motion_ratio', "Motion ratio", '-', 1.0),
        Curve('roll_center', "Roll center height", 'mm', MM),
        Curve('wheel_rate', "Wheel rate", 'N/mm', 1.e-3),
    ),
    'steer': (
        Curve('toe', "Road wheel angle (toe)", 'deg', DEG),
        Curve('camber', "Camber", 'deg', DEG),
    ),
    'roll': (
        Curve('roll', "Roll angle", 'deg', DEG),
        Curve('moment', "Roll moment", 'N m', 1.0),
        Curve('roll_stiffness', "Roll stiffness", 'N m/deg', 1.0 / DEG),
    ),
}

SWEEPS = {
    'bump': SweepKind('bump', 'travel', "Wheel travel (+ bump)",
                      (-40.0, 40.0, 33), CURVES['bump']),
    'steer': SweepKind('steer', 'rack', "Rack travel",
                       (-30.0, 30.0, 25), CURVES['steer']),
    'roll': SweepKind('roll', 's', "Opposite wheel travel",
                      (-20.0, 20.0, 21), CURVES['roll']),
}

#: The adjusters of :meth:`suspension.model.CornerModel.set_adjusters`, with
#: their titles, all in mm.
ADJUSTERS = (('camber_shim', "Camber shim"), ('caster_shim', "Caster shim"),
             ('toe_link', "Toe link"))

#: The name suffix of the right corner of an axle.
RIGHT = '_right'

#: The road inputs of the quarter car: kind, title, and the parameters with
#: their defaults (lengths in mm, speed in km/h, frequency in Hz, the
#: simulated ``duration`` in s).  A road duration shorter than the time the
#: wheel needs to pass the profile ends the run on the profile.
ROAD_INPUTS = {
    'bump': ("Bump (stone, speed hump)",
             {'speed': 30.0, 'height': 30.0, 'length': 150.0,
              'duration': 1.0}),
    'pothole': ("Pothole", {'speed': 40.0, 'depth': 60.0, 'length': 300.0,
                            'duration': 1.0}),
    'step': ("Step (curb)", {'speed': 10.0, 'height': 30.0,
                             'duration': 1.5}),
    'sine': ("Sine road (waves)",
             {'speed': 60.0, 'amplitude': 10.0, 'wavelength': 5000.0,
              'duration': 2.0}),
    'csv': ("Road profile from CSV (x_mm, z_mm)",
            {'speed': 30.0, 'path': '', 'duration': 3.0}),
    'rig': ("Shaker rig (the whole road plate moves)",
            {'amplitude': 5.0, 'frequency': 3.0, 'duration': 1.0}),
}


class Rig(object):
    """A K&C rig kept to read the loads at a sweep point.

    :ivar system: The engine system.
    :ivar corners: ``(suffix, corner handles)`` of each corner.
    :ivar drivers: The driver, or drivers, the sweep moves.
    :ivar values: The driver values of each sweep point, as the engine
        sweep takes them.
    :ivar rates: The rates of a multi-driver sweep, or None.
    """

    def __init__(self, system, corners, drivers, values, rates=None):
        self.system = system
        self.corners = corners
        self.drivers = drivers
        self.values = values
        self.rates = rates

    def joint_names(self):
        """Name the joints by the labels the corners keep (with the corner
        suffix); the joints of no corner (the anti-roll bar) stay
        unnamed."""
        handles = [(label + suffix, joint) for suffix, corner in self.corners
                   for label, joint in corner.joints]
        names = {}
        for index, joint in enumerate(self.system.get_joints()):
            for name, handle in handles:
                if joint is handle:
                    names[index] = name
                    break
        return names


class SweepResult(object):
    """One completed sweep, with the poses the 3D view draws.

    :ivar kind: The :class:`SweepKind`.
    :ivar data: The dict the suspension package returned, in SI units.
    :ivar poses: ``{body: (position_m, quaternion)}`` of each sweep point.
    :ivar setup: The adjusters (mm) the sweep ran with.
    """

    def __init__(self, kind, data, poses, setup, diagnostics=()):
        self.kind = kind
        self.data = data
        self.poses = poses
        self.setup = dict(setup)
        self.diagnostics = list(diagnostics)
        self.corners = ['']
        self.rig = None

    def __len__(self):
        return len(self.poses)

    @property
    def x(self):
        """The swept value of each point, in mm."""
        return np.asarray(self.data[self.kind.xkey],
                          dtype='float64') * self.kind.x.scale

    def curve(self, key):
        """Return ``(x, y)`` of curve ``key`` in its display units."""
        spec = {c.key: c for c in self.kind.curves}[key]
        return self.x, np.asarray(self.data[key],
                                  dtype='float64') * spec.scale

    def setup_text(self):
        return ", ".join(f"{title.lower()} {self.setup[name]:+.2f} mm"
                         for name, title in ADJUSTERS)


class SuspensionSession(object):
    """The model, the adjusters, and the analyses of one corner.

    :ivar engine: The :class:`~._engine.Engine` the analyses run on.
    :ivar model_path: The file the model was read from.
    :ivar current: The last valid :class:`SweepResult`, or None.
    :ivar previous: The result :attr:`current` replaced on the last adjuster
        change, or None.
    :ivar cursor: The index of the sweep point the view shows.
    :ivar warnings: ``dict(level, source, joint, message)`` of the last
        analysis run.
    """

    DEFAULT_MODEL = 'race_front_corner.json'
    #: The engine holds the Python thread through a sweep, so the GUI
    #: cannot run it in the background; the points are capped to keep the
    #: longest sweep (bump, about 1 ms a point) well under 0.5 s.
    MAX_POINTS = 201

    def __init__(self, engine, model_path=None):
        self.engine = engine
        self.kind = SWEEPS['bump']
        self.ranges = {name: kind.default for name, kind in SWEEPS.items()}
        self.adjusters = {name: 0.0 for name, _ in ADJUSTERS}
        self.current = None
        self.previous = None
        self.cursor = 0
        self.warnings = []
        self._rig = None
        self._compliance = None
        self.base_model = None
        self.model_path = None
        self.runs = []
        self.baseline = None
        self.measured = []
        self.layout = {name: [c.key for c in kind.curves]
                       for name, kind in SWEEPS.items()}
        self.link_coincident = True
        self._run_count = 0
        self.load(model_path or os.path.join(engine.data_dir,
                                             self.DEFAULT_MODEL))

    def load(self, path):
        """Read a corner model file and drop every result of the last
        one."""
        self.base_model = self.engine.model.CornerModel.load(path)
        self.model_path = os.path.abspath(str(path))
        if self.kind.name not in self.kinds():
            self.kind = SWEEPS['bump']
        self.adjusters = {name: 0.0 for name, _ in ADJUSTERS}
        #: The model differs from its file (an edit kept in the session).
        self.model_dirty = False
        #: The setup changes from the model file (see suspension.tuning).
        self.setup = {}
        #: The setup the setup sheet compares with (the model file first).
        self.setup_reference = {}
        self._reference_readouts = None
        self.current = self.previous = None
        self.cursor = 0
        self.warnings = []
        self._rig = self._compliance = None
        self.runs = []
        self.baseline = None
        self.measured = []
        self._run_count = 0

    def outline(self):
        """Return the model tree as ``[(group, [(item, [(property,
        text)])])]``, every number with its unit."""
        model = self.model()
        raw = model.raw

        def point(name):
            x, y, z = np.asarray(model.hp[name], dtype='float64') * MM
            return f"({x:.1f}, {y:.1f}, {z:.1f}) mm"

        def inertia(values):
            return "(" + ", ".join(f"{v:g}" for v in values) + ") kg m^2"

        bodies = [
            ("chassis", [("role", "ground of the K&C rig")]),
            ("upright", [("mass", f"{model.upright_mass:g} kg"),
                         ("inertia", inertia(model.upright_inertia)),
                         ("wheel center", point('wheel_center')),
                         ("wheel radius",
                          f"{model.wheel_radius * MM:.1f} mm")])]
        if model.has_rocker:
            bodies.append(("rocker", [
                ("mass", f"{model.rocker_mass:g} kg"),
                ("inertia", inertia(model.rocker_inertia)),
                ("pivot", point('rocker_pivot')),
                ("axis", str(list(model.rocker_axis)))]))
        if model.has_rack:
            bodies.append(("rack", [("mass", f"{model.rack_mass:g} kg"),
                                    ("axis", str(list(model.rack_axis)))]))
        for arm, spec in sorted(model.arms.items()):
            bodies.append((f"{arm} arm", [
                ("mass", f"{spec['mass']:g} kg"),
                ("links", ", ".join(spec['links']))]))
        links = []
        for name, (kind, inner, outer) in model.links.items():
            links.append((name, [
                ("inner (" + kind + ")", point(inner)),
                ("outer (upright)", point(outer)),
                ("length", f"{model.link_length(name) * MM:.2f} mm")]))
        if model.has_rocker:
            length = np.linalg.norm(model.hp['pushrod_outer']
                                    - model.hp['pushrod_inner']) * MM
            links.append((model.rod_kind, [
                (model.rod_body if model.rod_body == 'upright'
                 else f"{model.rod_body} arm", point('pushrod_outer')),
                ("rocker", point('pushrod_inner')),
                ("length", f"{length:.2f} mm")]))
        if model.has_strut:
            links.append(("strut", [
                ("upright point", point('strut_outer')),
                ("top mount", point('strut_inner')),
                ("constraint", "the top mount stays on the strut axis")]))
        if model.has_arm:
            links.append(("trailing arm", [
                ("inner pivot", point('arm_inner')),
                ("outer pivot", point('arm_outer')),
                ("constraint", "revolute about the pivot axis")]))
        drivers = [
            ("wheel travel driver", [
                ("kind", "absolute z of the wheel center"),
                ("range", "{:+.1f} to {:+.1f} mm, {} points".format(
                    *self.ranges['bump']))])]
        if model.has_rack:
            drivers.append(("rack driver (steer)", [
                ("kind", "translation of the rack along its axis"),
                ("range", "{:+.1f} to {:+.1f} mm, {} points".format(
                    *self.ranges['steer']))]))
        spring_name = model.spring_point_name()
        on = model.spring_body()
        forces = [("spring", [
            ("rate", f"{model.spring_k * 1.e-3:g} N/mm"),
            ("damping", f"{model.spring_c * 1.e-3:g} N.s/mm"
             if model.damper_curve is None else _curve_text(model)),
            (on.replace('arm_', '') + " point", point(spring_name)),
            ("chassis point", point('spring_chassis')),
            ("free length offset",
             f"{model.spring_free_offset * MM:g} mm")])]
        if model.arb is not None:
            forces.append(("anti-roll bar", [
                ("torsion rate", f"{model.arb['k']:g} N m/rad "
                                 f"({model.arb['k'] / DEG:.1f} N m/deg)"),
                ("bar pivot", point('arb_pivot')),
                ("arm tip", point('arb_tip')),
                ("mass", f"{model.arb['mass']:g} kg")]))
        forces.append(("tire", [
            ("vertical rate", f"{model.tire_k * 1.e-3:g} N/mm"),
            ("damping", f"{model.tire_c * 1.e-3:g} N.s/mm")]))
        if model.stops is not None:
            st = model.stops
            forces.append(("travel stops", [
                ("bump stop at", f"{st['bump_travel'] * MM:+.1f} mm"),
                ("droop stop at", f"{st['droop_travel'] * MM:+.1f} mm"),
                ("rate", f"{st['k'] * 1.e-3:g} N/mm + "
                         f"{st['k2'] * 1.e-6:g} N/mm^2 d")]))
        notes = raw.get('adjusters', {})
        adjusters = [(title.lower(), [
            ("value", f"{self.adjusters[name]:+.2f} mm"),
            ("note", notes.get(name, {}).get('note', ''))])
            for name, title in ADJUSTERS if name in model.adjusters]
        return [("Bodies", bodies), ("Links", links), ("Drivers", drivers),
                ("Forces", forces), ("Adjusters", adjusters)]

    def model(self, adjusters=None):
        """Return a copy of the model with ``adjusters`` (mm) applied,
        :attr:`adjusters` by default."""
        values = self.adjusters if adjusters is None else adjusters
        model = copy.deepcopy(self.base_model)
        model.set_adjusters(**{name: values[name] / MM
                               for name, _ in ADJUSTERS})
        # The setup (springs, ride height, damper, bar, tires) on top.
        return self.engine.tuning.apply(model, self.setup)

    def geometry(self, result=None):
        """The :class:`~._scene.CornerGeometry` of a result, of the current
        one by default, or of the model when there is none."""
        result = self.current if result is None else result
        setup = None if result is None else result.setup
        return _scene.CornerGeometry(self.model(setup))

    def geometries(self, result=None):
        """The geometry of every corner a result poses: the left one, and
        the mirrored right one of a roll sweep."""
        left = self.geometry(result)
        result = self.current if result is None else result
        if result is None or RIGHT not in result.corners:
            return [left]
        return [left, _scene.CornerGeometry(left.model.mirrored(), RIGHT)]

    def travels(self, kind=None):
        """The swept values of ``kind`` (the current kind by default), in
        m."""
        lo, hi, count = self.ranges[(kind or self.kind).name]
        return [float(v) / MM for v in np.linspace(lo, hi, int(count),
                                                   dtype='float64')]

    def library(self):
        """``[(stem, name, type)]`` of the corner models in the engine data
        directory, the default model first."""
        out = []
        for fname in sorted(os.listdir(self.engine.data_dir)):
            if not fname.endswith('.json'):
                continue
            path = os.path.join(self.engine.data_dir, fname)
            try:
                with open(path) as fobj:
                    data = json.load(fobj)
                name, kind = data['name'], data.get('type',
                                                    'double_wishbone')
            except (OSError, ValueError, KeyError):
                continue
            out.append((fname[:-5], name, kind))
        default = self.DEFAULT_MODEL[:-5]
        out.sort(key=lambda row: (row[0] != default, row[0]))
        return out

    def load_library(self, stem):
        """Load a model of :meth:`library` by its file stem."""
        self.load(os.path.join(self.engine.data_dir, f"{stem}.json"))

    def kinds(self):
        """The sweep kinds the model can run: no steer sweep without a
        rack."""
        return [name for name in SWEEPS
                if 'steer' != name or self.base_model.has_rack]

    def has_adjuster(self, name):
        """Whether the model defines adjuster ``name``."""
        return name in self.base_model.adjusters

    def set_kind(self, name):
        """Switch the sweep and run it; the previous curve is dropped, since
        it was swept over another variable.  On failure the last valid
        result, of the other sweep, stays current.

        :raises ValueError: For a sweep the model cannot run.
        """
        if name not in self.kinds():
            raise ValueError(f"model '{self.base_model.name}' has no "
                             f"{name} sweep (no steering rack)")
        self.kind = SWEEPS[name]
        return self.run_sweep()

    def set_range(self, lo, hi, count):
        """Set the range (mm) of the current sweep and run it.

        :raises ValueError: For more than :attr:`MAX_POINTS` points.
        """
        if not 2 <= int(count) <= self.MAX_POINTS:
            raise ValueError(f"a sweep takes 2 to {self.MAX_POINTS} points, "
                             f"not {count}")
        self.ranges[self.kind.name] = (float(lo), float(hi), int(count))
        self.previous = None
        return self.run_sweep(keep_previous=False)

    def set_adjuster(self, name, value):
        """Set an adjuster (mm) and re-run the current sweep.

        :return: Whether the sweep succeeded.  On failure the adjuster still
            holds the requested value, while :attr:`current` keeps the
            result of the setup it was valid for.
        """
        if name not in self.adjusters:
            raise KeyError(f"unknown adjuster {name!r}")
        if float(value) != 0.0 and not self.has_adjuster(name):
            raise ValueError(f"model '{self.base_model.name}' has no "
                             f"{name} adjuster")
        self.adjusters[name] = float(value)
        self._compliance = None
        return self.run_sweep(keep_previous=True)

    # Editing the model file from the model tree (P22).

    #: The model file blocks behind each model tree item.
    EDIT_BLOCKS = {
        ('Bodies', 'chassis'): (('vehicle',),),
        ('Bodies', 'upright'): (('upright',), ('wheel',)),
        ('Bodies', 'rocker'): (('rocker',),),
        ('Bodies', 'rack'): (('rack',),),
        ('Links', 'pushrod'): (('pushrod',),),
        ('Links', 'pullrod'): (('pushrod',),),
        ('Links', 'strut'): (('strut',),),
        ('Links', 'trailing arm'): (('arm',),),
        ('Forces', 'spring'): (('spring',),),
        ('Forces', 'anti-roll bar'): (('arb',),),
        ('Forces', 'tire'): (('tire',),),
        ('Forces', 'travel stops'): (('stops',),),
    }
    #: The unit of a model file value, by its key (the file stores mm, kg,
    #: N/mm, N.s/mm).
    EDIT_UNITS = {
        'mass': "kg", 'inertia': "kg m^2", 'radius': "mm", 'center': "mm",
        'free_length_offset': "mm", 'bump_travel': "mm",
        'droop_travel': "mm", 'k2': "N/mm^2", 'c': "N.s/mm",
        'knee': "mm/s", 'bump': "N.s/mm", 'rebound': "N.s/mm",
        'clicks': "clicks", 'bump_clicks': "clicks",
        'rebound_clicks': "clicks", 'velocity': "mm/s", 'force': "N",
        'corner_mass': "kg", 'wheelbase': "mm", 'cg_height': "mm",
        'half_track': "mm", 'ride_height': "mm", 'axis': "unit vector",
        'spin_axis': "unit vector",
    }

    def _edit_blocks(self, group, name):
        raw = self.base_model.raw
        if 'Links' == group and name in raw.get('links', {}):
            return (('links', name),)
        if 'Bodies' == group and name.endswith(' arm') \
                and name[:-4] in raw.get('arms', {}):
            return (('arms', name[:-4]),)
        return self.EDIT_BLOCKS.get((group, name), ())

    def _unit(self, path):
        key = path[-1] if not isinstance(path[-1], int) else path[-2]
        if 'k' == key:
            return {'arb': "N m/rad", 'bushings': "N/mm"}.get(path[0],
                                                              "N/mm")
        if 'k_rot' == key:
            return "N m/rad"
        value = self._raw_value(path)
        if isinstance(value, list) and 3 == len(value) \
                and key not in self.EDIT_UNITS:
            return "mm"
        return self.EDIT_UNITS.get(key, "")

    def _raw_value(self, path):
        node = self.base_model.raw
        for key in path:
            node = node[key]
        return node

    def editable(self, group, name):
        """The model file values behind a model tree item: ``[(label,
        path, unit, text)]``, ``path`` the keys into the file; the numbers
        in the units of the file.  An adjuster item edits its value (mm)."""
        if 'Adjusters' == group:
            for key, title in ADJUSTERS:
                if title.lower() == name and self.has_adjuster(key):
                    return [("value", ('__adjuster__', key), "mm",
                             f"{self.adjusters[key]:g}")]
            return []
        rows = []

        def walk(node, path):
            for key, value in node.items():
                if 'note' == key:
                    continue
                here = path + (key,)
                if isinstance(value, dict):
                    walk(value, here)
                elif isinstance(value, (int, float, str, list)) \
                        and not isinstance(value, bool):
                    rows.append((" / ".join(str(k) for k in here[
                        len(block):]), here, self._unit(here),
                        json.dumps(value)))

        for block in self._edit_blocks(group, name):
            node = self._raw_value(block)
            if isinstance(node, dict):
                walk(node, block)
        return rows

    def edit_model(self, path, text):
        """Change a value of the model file (see :meth:`editable`) and
        re-run the sweep.  The file on disk does not change; see
        :meth:`save_model`.

        :param text: the new value as JSON (a number, a list, a string)
        :raises ValueError: for a value of another type or a model the
            change breaks (the model keeps its last valid state)
        """
        if '__adjuster__' == path[0]:
            return self.set_adjuster(path[1], float(text))
        old = self._raw_value(path)
        try:
            value = json.loads(text)
        except ValueError:
            value = json.loads(f"[{text}]") if ',' in str(text) else text
        if isinstance(old, (int, float)) and not isinstance(old, bool):
            if not isinstance(value, (int, float)):
                raise ValueError(f"{'/'.join(map(str, path))} takes a "
                                 f"number, not {text!r}")
        elif isinstance(old, list):
            if not isinstance(value, list) or len(value) != len(old):
                raise ValueError(f"{'/'.join(map(str, path))} takes a "
                                 f"list of {len(old)} values")
        raw = copy.deepcopy(self.base_model.raw)
        node = raw
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        model = self.engine.model.CornerModel(raw)
        # The setup and the K&C rig must still work on the changed model.
        trial = self.engine.tuning.apply(model, self.setup)
        trial.set_adjusters(**{name: self.adjusters[name] / MM
                               for name, _ in ADJUSTERS})
        check = self.engine.kc.bump_sweep(self.engine.mbs, trial, [0.0])
        if not check['completed']:
            raise ValueError(f"the changed model does not assemble: "
                             f"{check['message']}")
        self.base_model = model
        self.model_dirty = True
        self._reference_readouts = None
        self._compliance = None
        return self.run_sweep(keep_previous=True)

    def save_model(self, path=None):
        """Write the model file (the current one by default); the model
        is then the file."""
        path = os.path.abspath(str(path or self.model_path))
        with open(path, 'w') as fobj:
            json.dump(self.base_model.raw, fobj, indent=2)
            fobj.write("\n")
        self.model_path = path
        self.model_dirty = False
        return path

    # The setup sheet (P20).

    def setup_fields(self):
        """``[(key, label, unit, value, reference, (lo, hi), step,
        decimals)]`` of the setup fields of the model."""
        tuning = self.engine.tuning
        base = tuning.baseline(self.base_model)
        clicks = tuning.click_range(self.base_model)
        ranges = {
            'spring_rate': ((1.0, 500.0), 5.0, 1),
            'ride_height': ((-80.0, 60.0), 5.0, 1),
            'bump_stop_gap': ((5.0, 150.0), 5.0, 1),
            'bump_clicks': ((0, clicks or 0), 1, 0),
            'rebound_clicks': ((0, clicks or 0), 1, 0),
            'damper_bump': ((0.0, 50.0), 0.5, 2),
            'damper_rebound': ((0.0, 50.0), 0.5, 2),
            'arb_rate': ((0.0, 500.0), 2.0, 1),
            'tire_rate': ((50.0, 1000.0), 10.0, 0),
        }
        out = []
        for key, label, unit in tuning.FIELDS:
            if key not in base:
                continue
            value = self.setup.get(key, base[key])
            ref = self.setup_reference.get(key, base[key])
            (lo, hi), step, decimals = ranges[key]
            out.append((key, label, unit, value, ref, (lo, hi), step,
                        decimals))
        return out

    def set_setup(self, name, value):
        """Change one setup field and re-run the sweep.

        :raises ValueError: for a field the model does not have or a value
            out of range (the setup keeps its last valid state)
        """
        trial = dict(self.setup, **{name: float(value)})
        self.engine.tuning.apply(self.base_model, trial)
        self.setup = trial
        self._compliance = None
        return self.run_sweep(keep_previous=True)

    def reset_setup(self):
        """Go back to the setup of the model file."""
        self.setup = {}
        self._compliance = None
        return self.run_sweep(keep_previous=True)

    def set_setup_reference(self):
        """Make the current setup the one the sheet compares with."""
        self.setup_reference = dict(self.setup)
        self._reference_readouts = None

    def setup_readouts(self, sprung_mass):
        """The readouts of the current setup and of the reference setup.

        :return: dict with ``completed``, ``message``, ``current``,
            ``reference`` (readout dicts of :func:`tuning.readouts`), and
            ``rows`` ``[(label, unit, format, reference, current)]``
        """
        tuning = self.engine.tuning
        mbs = self.engine.mbs
        self.warnings = []
        try:
            current = tuning.readouts(mbs, self.model(), float(sprung_mass))
            key = (tuple(sorted(self.setup_reference.items())),
                   float(sprung_mass))
            if self._reference_readouts is None \
                    or self._reference_readouts[0] != key:
                base = copy.deepcopy(self.base_model)
                base.set_adjusters(**{name: self.adjusters[name] / MM
                                      for name, _ in ADJUSTERS})
                ref = tuning.readouts(mbs, tuning.apply(
                    base, self.setup_reference), float(sprung_mass))
                self._reference_readouts = (key, ref)
            reference = self._reference_readouts[1]
        except Exception as exc:
            self.warnings.append(self._entry(
                'error', 'setup', None, f"setup readouts failed: {exc}"))
            return {'completed': False, 'message': str(exc)}
        rows = [(label, unit, fmt, reference.get(key), current.get(key))
                for key, label, unit, fmt in tuning.READOUTS
                if key in current]
        return {'completed': True, 'message': '', 'current': current,
                'reference': reference, 'rows': rows}

    def save_setup(self, path, note=""):
        """Write the current setup as a setup sheet file."""
        tuning = self.engine.tuning
        full = dict(tuning.baseline(self.base_model), **self.setup)
        tuning.save(path, {'corner': full, 'model': self.model_path},
                    note=note)

    def load_setup(self, path):
        """Read a setup sheet file into the current setup.

        :raises ValueError: when the file holds no corner setup or a field
            the model does not have
        """
        setups = self.engine.tuning.load(path)
        if 'corner' not in setups:
            raise ValueError(f"{path}: no corner setup")
        trial = {k: float(v) for k, v in setups['corner'].items()}
        self.engine.tuning.apply(self.base_model, trial)
        self.setup = trial
        self._compliance = None
        return self.run_sweep(keep_previous=True)

    def run_sweep(self, keep_previous=False):
        """Run the current sweep with the current adjusters.

        :param keep_previous: Keep the replaced result as :attr:`previous`.
        :return: Whether the sweep succeeded and became :attr:`current`.
        """
        kind = self.kind
        setup = dict(self.adjusters)
        self.warnings = []
        try:
            result = self._sweep(kind, setup)
        except Exception as exc:
            return self._fail(kind, setup, str(exc))
        if not result.data['completed']:
            done = len(result)
            return self._fail(kind, setup, (
                f"{result.data['message']} ({done} of "
                f"{len(self.travels(kind))} points converged)"))
        if keep_previous and self.current is not None:
            self.previous = self.current
        if self.current is None or self.current.kind is not kind:
            self.cursor = len(result) // 2
        self.current = result
        self._rig = result.rig
        self._add_run(result)
        self.cursor = min(self.cursor, len(result) - 1)
        if not keep_previous and self.previous is not None:
            if self.previous.kind is not kind:
                self.previous = None
        self.warnings = [self._entry('info', kind.name, None,
                                     f"{kind.name} sweep completed: "
                                     f"{len(result)} points, "
                                     f"{result.setup_text()}")]
        self.warnings.extend(result.diagnostics)
        rcond = np.asarray(result.data.get('rcond', []), dtype='float64')
        if rcond.size and rcond.min() < 1.e-10:
            self.warnings.append(self._entry(
                'warning', kind.name, None,
                f"near lock-up: constraint Jacobian rcond "
                f"{rcond.min():.2e} at {result.x[int(rcond.argmin())]:+.1f} "
                f"mm"))
        return True

    # The run list (P11.5).

    #: The run list keeps at most this many runs; the oldest one that is
    #: not the baseline goes first.
    MAX_RUNS = 12

    def _add_run(self, result):
        self._run_count += 1
        color = _analysis.RUN_COLORS[(self._run_count - 1)
                                     % len(_analysis.RUN_COLORS)]
        ident = f"#{self._run_count}"
        name = f"{ident} {result.kind.name}: {result.setup_text()}"
        self.runs.append(_analysis.Run(name, result, color, ident))
        while len(self.runs) > self.MAX_RUNS:
            for index, run in enumerate(self.runs):
                if run is not self.baseline:
                    del self.runs[index]
                    break

    def current_run(self):
        """The :class:`~._analysis.Run` of :attr:`current`, or None."""
        for run in self.runs:
            if run.result is self.current:
                return run
        return None

    def visible_runs(self, kind=None):
        """The visible runs of a sweep kind (the current one by default),
        oldest first."""
        kind = kind or (self.current.kind if self.current else self.kind)
        return [run for run in self.runs
                if run.visible and run.result.kind is kind]

    def set_run_visible(self, index, visible):
        self.runs[index].visible = bool(visible)

    def rename_run(self, index, name):
        name = str(name).strip()
        if not name:
            raise ValueError("a run needs a name")
        self.runs[index].name = name

    def delete_run(self, index):
        """Drop a run from the list; the current result stays shown."""
        run = self.runs.pop(index)
        if run is self.baseline:
            self.baseline = None

    def set_baseline(self, index):
        """Mark a run as the baseline (None clears it): it is drawn solid in
        its color and kept when the list is full."""
        self.baseline = None if index is None else self.runs[index]

    # Measured curves (P11.1).

    def import_measured(self, path):
        """Read measured curves from a CSV file and keep them for the plots.

        :return: The :class:`~._analysis.MeasuredSet`, or None when the file
            cannot be read (the reason is in :attr:`warnings`).
        """
        try:
            measured, unknown = _analysis.read_measured_csv(path, SWEEPS)
        except (OSError, ValueError) as exc:
            self.warnings.append(self._entry('error', 'measured', None,
                                             f"cannot import {path}: {exc}"))
            return None
        self.measured.append(measured)
        text = (f"measured {measured.kind} curves from {measured.name}: "
                f"{', '.join(measured.curves)}")
        if unknown:
            text += f"; columns not read: {', '.join(unknown)}"
        self.warnings.append(self._entry('info', 'measured', None, text))
        return measured

    def measured_for(self, kind=None):
        kind = kind or (self.current.kind if self.current else self.kind)
        return [m for m in self.measured if m.kind == kind.name]

    def clear_measured(self):
        self.measured = []

    # Summaries and overlays (P11.3, P11.4).

    def summary(self, run, key):
        """The :func:`~._analysis.summary` of curve ``key`` of a run, in
        display units (the gradient per mm of the swept value)."""
        xs, ys = run.result.curve(key)
        return _analysis.summary(xs, ys)

    def overlays(self, index=None):
        """The geometry overlays of the current bump sweep at a point (the
        cursor by default), ``{name: segment ends in mm}``."""
        if self.current is None or 'bump' != self.current.kind.name:
            return {}
        index = self.cursor if index is None else index
        return _analysis.overlay_segments(self.current.data, index)

    # Plot layouts (P11.10).

    def set_layout(self, kind_name, keys):
        """Choose the plots a sweep kind shows, in its curve order."""
        kind = SWEEPS[kind_name]
        known = [c.key for c in kind.curves]
        unknown = [k for k in keys if k not in known]
        if unknown:
            raise ValueError(f"{kind_name} sweep has no curve "
                             f"{', '.join(unknown)}")
        self.layout[kind_name] = [k for k in known if k in keys]

    def save_layout(self, path):
        with open(path, 'w', encoding='utf-8') as fobj:
            json.dump({'suspension_plot_layout': 1, 'plots': self.layout},
                      fobj, indent=2)

    def load_layout(self, path):
        with open(path, encoding='utf-8') as fobj:
            data = json.load(fobj)
        if 'plots' not in data:
            raise ValueError(f"{path} is not a plot layout file")
        for name, keys in data['plots'].items():
            if name in SWEEPS:
                self.set_layout(name, keys)

    def _fail(self, kind, setup, reason):
        text = ", ".join(f"{title.lower()} {setup[name]:+.2f} mm"
                         for name, title in ADJUSTERS)
        self.warnings.append(self._entry(
            'error', kind.name, None,
            f"{kind.name} sweep failed ({text}): {reason}"))
        if self.current is not None:
            self.warnings.append(self._entry(
                'info', kind.name, None,
                f"showing the last valid result ({self.current.kind.name} "
                f"sweep, {self.current.setup_text()})"))
        else:
            self.warnings.append(self._entry(
                'info', kind.name, None, "no valid result to show"))
        return False

    @staticmethod
    def _entry(level, source, joint, message):
        return {'level': level, 'source': source, 'joint': joint,
                'message': message}

    def _sweep(self, kind, setup):
        eng = self.engine
        model = self.model(setup)
        travels = self.travels(kind)
        if 'bump' == kind.name:
            data = eng.kc.bump_sweep(eng.mbs, model, travels)
            sys_obj, corner = eng.corner.kc_rig(eng.mbs, model,
                                                travel=travels[0])
            raw = _arrays(sys_obj.sweep(corner.travel,
                                        [corner.z0 + t for t in travels]))
            rig = Rig(sys_obj, [('', corner)], corner.travel,
                      [[v] for v in raw['values']])
        elif 'steer' == kind.name:
            data = eng.kc.steer_sweep(eng.mbs, model, travels)
            sys_obj, corner = eng.corner.kc_rig(eng.mbs, model,
                                                steer=travels[0])
            raw = _arrays(sys_obj.sweep(corner.steer, travels))
            rig = Rig(sys_obj, [('', corner)], corner.steer,
                      [[v] for v in raw['values']])
        else:
            arb = model.arb is not None
            data = eng.axle.roll_sweep(eng.mbs, model, travels, with_arb=arb)
            raw = _arrays(data['raw'])
            # The axle of roll_sweep is not returned, so a twin is built to
            # read the loads at a sweep point.
            sys_obj, axle = eng.axle.kc_axle(eng.mbs, model, with_arb=arb)
            z0 = axle['z0']
            rig = Rig(sys_obj, [('', axle['left']), (RIGHT, axle['right'])],
                      [axle['left'].travel, axle['right'].travel],
                      [[[z0 + t, z0 - t]] for t in data['s']], [1.0, -1.0])
        count = min(len(raw['values']), len(data[kind.xkey]))
        if not raw['completed']:
            data = dict(data, completed=False, message=raw['message'])
        numbers = {}
        for suffix, corner in rig.corners:
            numbers.update({body + suffix: num
                            for body, num in corner.nums.items()})
        if 'roll' == kind.name and 'arb_nums' in data['axle']:
            numbers['arb'], numbers['arb' + RIGHT] = data['axle']['arb_nums']
        poses = [{body: (np.array(raw['positions'][it, num],
                                  dtype='float64'),
                         np.array(raw['quaternions'][it, num],
                                  dtype='float64'))
                  for body, num in numbers.items()}
                 for it in range(count)]
        diagnostics = []
        names = rig.joint_names()
        for entry in rig.system.get_diagnostics():
            joint = entry.get('joint')
            diagnostics.append(self._entry(
                entry.get('level', 'warning'), kind.name,
                names.get(joint, None if joint is None else f"joint {joint}"),
                entry.get('message', '')))
        result = SweepResult(kind, data, poses, setup, diagnostics)
        result.corners = [suffix for suffix, _ in rig.corners]
        result.rig = rig
        return result

    @staticmethod
    def _joint_names(rig):
        """Name the joints of a K&C rig ``(system, corner)``."""
        sys_obj, corner = rig
        return Rig(sys_obj, [('', corner)], None, []).joint_names()

    def pose(self, index=None):
        """``{body: (position_m, quaternion)}`` at a sweep point (the cursor
        by default), or None without a result."""
        if self.current is None:
            return None
        index = self.cursor if index is None else index
        return self.current.poses[index]

    def set_cursor(self, index):
        """Move the shared cursor to a sweep point, clamped to the result."""
        if self.current is None:
            self.cursor = 0
            return 0
        self.cursor = max(0, min(int(index), len(self.current) - 1))
        return self.cursor

    def cursor_value(self):
        """The swept value (mm) at the cursor, or None."""
        if self.current is None:
            return None
        return float(self.current.x[self.cursor])

    def forces(self, index=None):
        """Return ``[(name, point_mm, force_N)]`` on each upright and rocker
        at a sweep point.

        The rig is moved to the point and the joint reactions and the
        spring loads are read there; the wheel travel drivers, which the
        rig holds the wheels with, are left out.
        """
        rig = self._rig
        if rig is None or self.current is None:
            return []
        index = self.cursor if index is None else index
        args = [rig.drivers, rig.values[index]]
        if rig.rates is not None:
            args.append(rig.rates)
        if not rig.system.sweep(*args)['completed']:
            return []
        names = rig.joint_names()
        uprights = {corner.nums['upright'] for _, corner in rig.corners}
        out = []
        for jt, entry in enumerate(rig.system.get_joint_reactions()):
            entry = _arrays(entry)
            name = names.get(jt, f"joint {jt}")
            if name.startswith('wheel travel driver'):
                continue
            if entry['body_j'] in uprights:
                point, force = entry['point_j'], entry['force_j']
            elif entry['body_i'] in uprights:
                point, force = entry['point_i'], entry['force_i']
            else:
                continue
            out.append((name, np.asarray(point, dtype='float64') * MM,
                        np.asarray(force, dtype='float64')))
        for suffix, corner in rig.corners:
            loads = self.engine.mbs.get_force_element_loads(corner.spring)
            out.append(('spring' + suffix,
                        np.asarray(loads['point_i'], dtype='float64') * MM,
                        np.asarray(loads['force_i'], dtype='float64')))
        return out

    def export_csv(self, path):
        """Write the current sweep to ``path`` with a header of the model,
        the setup, and the units.

        :raises ValueError: Without a valid result to write.
        """
        result = self.current
        if result is None:
            raise ValueError("no valid sweep result to export")
        kind = result.kind
        columns = [(kind.x, result.x)]
        columns += [(c, result.curve(c.key)[1]) for c in kind.curves]
        with open(path, 'w', encoding='ascii', errors='replace') as fobj:
            fobj.write("# solvcon suspension K&C sweep\n")
            fobj.write(f"# model file: {self.model_path}\n")
            fobj.write(f"# model name: {self.base_model.name}\n")
            fobj.write(f"# sweep: {kind.name}, {len(result)} points\n")
            fobj.write(f"# adjusters [mm]: {result.setup_text()}\n")
            fobj.write("# units: " + ", ".join(
                f"{c.key} [{c.unit}]" for c, _ in columns) + "\n")
            fobj.write(",".join(c.label for c, _ in columns) + "\n")
            for row in range(len(result)):
                fobj.write(",".join(f"{values[row]:.10g}"
                                    for _, values in columns) + "\n")

    def run_compliance(self, force):
        """Apply ``force`` (N, global) at the contact patch of the compliant
        rig and return the alignment change.

        :return: dict with ``completed``, ``message``, ``toe`` and
            ``camber`` (deg, change from the unloaded rig), and
            ``patch`` (mm, contact patch displacement); a failure keeps the
            values of the last load case.
        """
        eng = self.engine
        self.warnings = []
        try:
            if self._compliance is None:
                model = self.model()
                sys_obj, _ = eng.compliance.compliant_rig(eng.mbs, model)
                base = eng.compliance.apply_patch_load(eng.mbs, sys_obj,
                                                       model, [0.0] * 3)
                self._compliance = (model, sys_obj, base, None)
            model, sys_obj, base, last = self._compliance
            loaded = eng.compliance.apply_patch_load(
                eng.mbs, sys_obj, model, [float(v) for v in force])
        except Exception as exc:
            last = None if self._compliance is None else self._compliance[3]
            self.warnings.append(self._entry(
                'error', 'statics', None,
                f"compliance load case {list(force)} N failed: {exc}"))
            out = dict(last or {}, completed=False, message=str(exc))
            # The rig keeps the pose of its failed solve; rebuild next time.
            self._compliance = None
            return out
        out = {'completed': True, 'message': '', 'force': list(force),
               'toe': (loaded['toe'] - base['toe']) * DEG,
               'camber': (loaded['camber'] - base['camber']) * DEG,
               'patch': (loaded['contact_patch']
                         - base['contact_patch']) * MM}
        self._compliance = (model, sys_obj, base, out)
        self.warnings.append(self._entry(
            'info', 'statics', None,
            f"compliance load case {list(force)} N converged"))
        return out

    def run_modal(self, sprung_mass):
        """Return the modes of the quarter car at its static equilibrium.

        :return: dict with ``completed``, ``message``, and ``modes``, a list
            of ``(name, frequency_hz, damping_ratio)`` in ascending
            frequency.
        """
        eng = self.engine
        self.warnings = []
        try:
            sys_obj, _ = eng.vehicle.quarter_car(eng.mbs, self.model(),
                                                 float(sprung_mass))
            eq = sys_obj.find_equilibrium(tol=1.e-9)
            if not eq['converged']:
                raise RuntimeError(f"static equilibrium failed: "
                                   f"{eq['message']}")
            modal = eng.mbs.modal_analysis(sys_obj)
        except Exception as exc:
            self.warnings.append(self._entry('error', 'modal', None,
                                             f"modal analysis failed: {exc}"))
            return {'completed': False, 'message': str(exc), 'modes': []}
        pairs = sorted(pair for pair in zip(
            np.asarray(modal.frequencies).tolist(),
            np.asarray(modal.damping_ratios).tolist())
                       if pair[0] > 1.e-3)
        names = ["ride", "wheel hop"]
        modes = [(names[it] if it < len(names) else f"mode {it + 1}", f, z)
                 for it, (f, z) in enumerate(pairs)]
        self.warnings.append(self._entry(
            'info', 'modal', None,
            f"modal analysis: {len(modes)} modes, sprung mass "
            f"{sprung_mass:g} kg"))
        return {'completed': True, 'message': '', 'modes': modes}

    def mode_frames(self, sprung_mass, index, frames=24, amplitude=0.025):
        """The frames of a mode shape of the quarter car for a 3D view.

        The linear corner gives the mode (its frequencies and damping
        equal the engine modal analysis); the K&C rig poses the corner at
        the travel of each frame and the chassis moves with the mode.

        :param index: the mode, 0 the ride mode, 1 the wheel hop
        :param amplitude: the largest motion (m)
        :return: dict with ``model``, ``poses`` (quarter car poses),
            ``labels``, ``frequency``, ``damping``, and ``shape``
        """
        eng = self.engine
        model = self.model()
        lin = eng.ride.LinearCorner(eng.mbs, model, float(sprung_mass))
        modes = lin.modes()
        if not 0 <= index < len(modes):
            raise ValueError(f"no mode {index + 1}")
        freq, zeta, shape = modes[index]
        phases = [2.0 * math.pi * k / frames for k in range(frames)]
        heave = [amplitude * float((shape[0] * np.exp(1j * p)).real)
                 for p in phases]
        travel = [model.static_travel + amplitude * float(
            (shape[1] * np.exp(1j * p)).real) for p in phases]
        s, c = eng.corner.kc_rig(eng.mbs, model)
        res = eng.arrays.result_arrays(s.sweep(c.travel,
                                               [c.z0 + t for t in travel]))
        poses, labels = [], []
        for k in range(len(res['values'])):
            lift = np.array([0.0, 0.0, heave[k]])
            pose = {'chassis': (lift, np.array([1.0, 0.0, 0.0, 0.0]))}
            for body, num in c.nums.items():
                pose[body] = (res['positions'][k, num] + lift,
                              res['quaternions'][k, num].copy())
            wheel = pose['upright'][0]
            pose['road'] = (np.array([wheel[0], wheel[1], 0.0]),
                            np.array([1.0, 0.0, 0.0, 0.0]))
            poses.append(pose)
            labels.append(f"{freq:.2f} Hz, phase "
                          f"{math.degrees(phases[k]):.0f} deg")
        return {'model': model, 'poses': poses, 'labels': labels,
                'frequency': freq, 'damping': zeta, 'shape': shape}

    # Target alignment (P11.6).

    def solve_alignment(self, targets):
        """Solve the adjusters for target static alignment and sweep.

        :param targets: ``{camber, toe, caster: deg}``, any subset.
        :return: dict with ``converged``, ``message``, ``adjusters`` (mm),
            and ``achieved`` (deg); on failure the adjusters stay.
        """
        eng = self.engine
        self.warnings = []
        rad = {key: float(value) / DEG for key, value in targets.items()}
        out = eng.alignment.solve_alignment(eng.mbs, self.model(), rad)
        result = {'converged': bool(out['converged']),
                  'message': out['message'],
                  'adjusters': {k: v * MM for k, v in
                                out['adjusters'].items()},
                  'achieved': ({k: v * DEG for k, v in
                                out['achieved'].items()}
                               if out['achieved'] else None)}
        if not out['converged']:
            self.warnings.append(self._entry(
                'error', 'alignment', None,
                f"target alignment failed: {out['message']}; the adjusters "
                f"are unchanged"))
            return result
        # The solved adjusters apply on top of the current ones.
        for name, value in result['adjusters'].items():
            self.adjusters[name] = float(value)
        self._compliance = None
        self.run_sweep(keep_previous=True)
        text = ", ".join(f"{k} {v:+.4f} deg" for k, v in
                         result['achieved'].items() if k in targets)
        self.warnings.append(self._entry(
            'info', 'alignment', None,
            f"target alignment reached in {out['iterations']} iterations: "
            f"{text}"))
        return result

    # Compliance sweeps (P11.7).

    #: The load cases of the compliance sweep: title and load unit.
    COMPLIANCE_CASES = {
        'lateral': ("Lateral force at the contact patch", 'N'),
        'braking': ("Braking force at the contact patch", 'N'),
        'traction': ("Traction force at the wheel center", 'N'),
        'aligning': ("Aligning torque about the vertical", 'N m'),
    }

    def run_compliance_sweep(self, case, fmax, count=11):
        """Ramp one load on the compliant rig.

        :return: dict with ``completed``, ``message``, ``load`` (N or N m),
            ``toe``, ``camber`` (deg), ``patch`` ((n, 3) mm), and
            ``gradient`` per kN (per kN m for the torque): ``toe``,
            ``camber`` (deg), ``patch_x``, ``patch_y``, ``patch_z`` (mm).
        """
        eng = self.engine
        self.warnings = []
        try:
            out = eng.compliance.compliance_sweep(
                eng.mbs, self.model(), case, float(fmax), int(count))
        except (ValueError, RuntimeError) as exc:
            self.warnings.append(self._entry(
                'error', 'compliance', None,
                f"{case} compliance sweep failed: {exc}"))
            return {'completed': False, 'message': str(exc)}
        grad = out['gradient']
        result = {
            'completed': bool(out['completed']), 'message': out['message'],
            'case': case, 'load': np.asarray(out['load']),
            'toe': np.asarray(out['toe']) * DEG,
            'camber': np.asarray(out['camber']) * DEG,
            'patch': np.asarray(out['patch']) * MM,
            'gradient': {'toe': grad['toe'] * DEG * 1.e3,
                         'camber': grad['camber'] * DEG * 1.e3,
                         'patch_x': grad['patch_x'] * MM * 1.e3,
                         'patch_y': grad['patch_y'] * MM * 1.e3,
                         'patch_z': grad['patch_z'] * MM * 1.e3}}
        level = 'info' if result['completed'] else 'error'
        self.warnings.append(self._entry(
            level, 'compliance', None,
            f"{case} compliance sweep: {len(result['load'])} loads"
            + ("" if result['completed'] else f", {out['message']}")))
        return result

    # Hardpoints (P11.8, P11.12).

    def hardpoint_rows(self):
        """``[(name, (x, y, z) mm, limits)]`` of the editable hardpoints,
        ``limits`` being ``(lo, hi)`` in mm or None."""
        model = self.base_model
        rows = []
        for name in model.hardpoint_names():
            box = model.limits.get(name)
            limits = None
            if box is not None:
                limits = (np.asarray(box['min']) * MM,
                          np.asarray(box['max']) * MM)
            rows.append((name, np.asarray(model.hp[name]) * MM, limits))
        return rows

    def coincident(self, name):
        """The points that move with ``name`` when :attr:`link_coincident`
        is on (the two outer points of a wishbone share the ball joint)."""
        names = [name]
        if self.link_coincident:
            names += [n for n in self.base_model.coincident_points(name)
                      if n != name]
        return names

    def set_hardpoint(self, name, xyz):
        """Move a hardpoint (mm, unadjusted design value) and re-run the
        sweep.  The coincident points move with it.

        :return: Whether the edit was taken.  An edit outside a limit box,
            or one that leaves the corner unassemblable, is rejected and
            the model is unchanged.
        """
        xyz = np.asarray(xyz, dtype='float64') / MM
        backup = copy.deepcopy(self.base_model)
        self.warnings = []
        try:
            for point in self.coincident(name):
                self.base_model.set_hardpoint(point, xyz)
        except (KeyError, ValueError) as exc:
            self.base_model = backup
            self.warnings.append(self._entry('error', 'hardpoint', name,
                                             f"edit rejected: {exc}"))
            return False
        self._compliance = None
        if not self.run_sweep(keep_previous=True):
            # A failed sweep keeps the last valid result, which is the one
            # of the restored model.
            self.base_model = backup
            self.warnings.append(self._entry(
                'error', 'hardpoint', name,
                f"edit rejected: the corner does not assemble with "
                f"{name} at {_fmt_mm(xyz * MM)}; the model is unchanged"))
            return False
        return True

    def set_limit(self, name, lo, hi):
        """Set the limit box (mm) of a hardpoint; None, None clears it."""
        if lo is None:
            self.base_model.set_limit(name, None, None)
            return
        self.base_model.set_limit(name, np.asarray(lo, dtype='float64') / MM,
                                  np.asarray(hi, dtype='float64') / MM)

    # Hardpoint batch sweep (P11.9).

    #: The summaries of a batch sweep: key, title, unit, factor from SI.
    BATCH_SUMMARIES = (
        ('camber_gain', "Camber gain", 'deg/mm', DEG / MM),
        ('bump_steer', "Bump steer", 'deg/mm', DEG / MM),
        ('roll_center', "Roll center height", 'mm', MM),
        ('caster', "Caster", 'deg', DEG),
        ('kpi', "KPI", 'deg', DEG),
        ('scrub', "Scrub radius", 'mm', MM),
        ('trail', "Mech. trail", 'mm', MM),
        ('motion_ratio', "Motion ratio", '-', 1.0),
    )

    def run_batch(self, point, axis, offsets):
        """Vary one coordinate of a hardpoint and record the K&C summary at
        design for each offset (mm).

        :return: dict with ``offset`` (mm), every key of
            :attr:`BATCH_SUMMARIES` in its unit, ``completed``, ``message``.
        """
        eng = self.engine
        self.warnings = []
        model = self.model()
        try:
            out = eng.sensitivity.hardpoint_sweep(
                eng.mbs, model, point, int(axis),
                [float(v) / MM for v in offsets])
        except (KeyError, ValueError) as exc:
            self.warnings.append(self._entry('error', 'batch', point,
                                             f"batch sweep failed: {exc}"))
            return {'completed': False, 'message': str(exc)}
        result = {'offset': np.asarray(out['offset']) * MM,
                  'completed': bool(out['completed']),
                  'message': out['message']}
        for key, _, _, factor in self.BATCH_SUMMARIES:
            result[key] = np.asarray(out[key]) * factor
        level = 'info' if result['completed'] else 'error'
        self.warnings.append(self._entry(
            level, 'batch', point,
            f"batch sweep of {point} {'xyz'[int(axis)]}: "
            f"{len(result['offset'])} of {len(offsets)} offsets"
            + ("" if result['completed'] else f", {out['message']}")))
        return result

    # Motion profile (P11.11).

    def read_profile(self, path):
        """Read a profile CSV; return rows (heave mm, roll deg, rack mm)."""
        rows = self.engine.axle.read_profile_csv(path)
        return [(h * MM, r * DEG, k * MM) for h, r, k in rows]

    def run_profile(self, rows):
        """Drive the axle through rows of (heave mm, roll deg, rack mm).

        :return: dict with ``completed``, ``message``, the per-row
            ``camber_left``, ``toe_left``, ``camber_right``, ``toe_right``
            (deg), ``rows`` (the solved rows), and ``poses`` for the 3D
            view (both corners, the bar halves).
        """
        eng = self.engine
        self.warnings = []
        model = self.model()
        si = [(h / MM, r / DEG, k / MM) for h, r, k in rows]
        try:
            arb = model.arb is not None
            out = eng.axle.profile_sweep(eng.mbs, model, si, with_arb=arb)
            _, axle = eng.axle.kc_axle(eng.mbs, model, with_arb=arb)
        except (ValueError, RuntimeError) as exc:
            self.warnings.append(self._entry('error', 'profile', None,
                                             f"profile failed: {exc}"))
            return {'completed': False, 'message': str(exc), 'poses': []}
        numbers = {}
        for suffix, side in (('', 'left'), (RIGHT, 'right')):
            numbers.update({body + suffix: num
                            for body, num in axle[side].nums.items()})
        if 'arb_nums' in axle:
            numbers['arb'], numbers['arb' + RIGHT] = axle['arb_nums']
        count = len(out['heave'])
        poses = []
        if count:
            for it in range(count):
                poses.append({body: (np.array(out['positions'][it, num]),
                                     np.array(out['quaternions'][it, num]))
                              for body, num in numbers.items()})
        result = {'completed': bool(out['completed']),
                  'message': out['message'], 'rows': list(rows[:count]),
                  'poses': poses}
        for key in ('camber_left', 'toe_left', 'camber_right', 'toe_right'):
            result[key] = np.asarray(out[key]) * DEG
        level = 'info' if result['completed'] else 'error'
        self.warnings.append(self._entry(
            level, 'profile', None,
            f"motion profile: {count} of {len(rows)} rows"
            + ("" if result['completed'] else f", {out['message']}")))
        return result

    # The report (P11.10).

    def report_content(self):
        """The setup sheet, the plot pages, and the summary table of the
        PDF report, in display units."""
        model = self.model()
        sheet = [("Model", [("file", self.model_path),
                            ("name", self.base_model.name)])]
        sheet.append(("Adjusters", [(title.lower(),
                                     f"{self.adjusters[name]:+.2f} mm")
                                    for name, title in ADJUSTERS]))
        current = self.current
        if current is not None:
            fields = []
            for curve in current.kind.curves:
                value = _analysis.summary(*current.curve(curve.key))['value']
                if value is not None:
                    fields.append((curve.title,
                                   f"{value:+.4f} {curve.unit}"))
            sheet.append((f"Static alignment ({current.kind.name} sweep at "
                          f"0)", fields))
        sheet.append(("Hardpoints (design, mm)", [
            (name, _fmt_mm(np.asarray(model.hp[name]) * MM))
            for name in model.hardpoint_names()]))
        pages = []
        table_rows = []
        kind = current.kind if current is not None else self.kind
        runs = self.visible_runs(kind)
        plots = []
        for curve in kind.curves:
            if curve.key not in self.layout[kind.name]:
                continue
            curves = []
            for run in runs:
                xs, ys = run.result.curve(curve.key)
                curves.append((run.name, xs, ys, run.color,
                               run is not self.baseline
                               and run.result is not current, False))
                stats = self.summary(run, curve.key)
                table_rows.append([
                    run.name, curve.label,
                    _num(stats['value']), _num(stats['gradient']),
                    _num(stats['min']), _num(stats['max'])])
            for measured in self.measured_for(kind):
                if curve.key in measured.curves:
                    curves.append((measured.name, measured.x,
                                   measured.curves[curve.key], (60, 60, 60),
                                   False, True))
            plots.append((curve.title, kind.x.label, curve.label, curves))
        if plots:
            pages.append((f"{kind.name} sweep", plots))
        columns = ["run", "curve", "at 0", "gradient /mm", "min", "max"]
        return sheet, pages, (columns, table_rows)

    def export_report(self, path):
        """Write the PDF report; return the number of pages."""
        sheet, pages, table = self.report_content()
        return _analysis.write_report(path, sheet, pages, table)

    def quarter_car(self, sprung_mass, amplitude=5.0, frequency=3.0,
                    duration=1.0, dt=2.e-3, road=None):
        """Return a :class:`QuarterCarRun` of the current setup.

        :param road: ``(kind, parameters)`` of :data:`ROAD_INPUTS`; None is
            the shaker rig with ``amplitude`` (mm), ``frequency`` (Hz), and
            ``duration`` (s).
        """
        if road is None:
            road = ('rig', {'amplitude': amplitude, 'frequency': frequency,
                            'duration': duration})
        return QuarterCarRun(self.engine, self.model(), sprung_mass, road,
                             dt)

    # Ride comfort and road holding (P17).

    def run_ride(self, road_class, speed_kmh, sprung_mass, count=600):
        """The linear ride of the current model on an ISO 8608 road.

        :return: dict with ``completed``, ``message``, ``stats`` (the
            random response of :class:`suspension.ride.LinearCorner`, with
            ``road_class`` and ``speed``), and the frequency response
            ``freq`` (Hz), ``body_acc`` and ``body_acc_w`` (m/s^2 per mm of
            road), ``tire_force`` (N per mm)
        """
        eng = self.engine
        self.warnings = []
        try:
            speed = float(speed_kmh) / 3.6
            lin = eng.ride.LinearCorner(eng.mbs, self.model(),
                                        float(sprung_mass))
            stats = lin.random_response(road_class, speed)
            freq = np.geomspace(0.3, 50.0, count)
            resp = lin.response(freq)
        except Exception as exc:
            self.warnings.append(self._entry('error', 'ride', None,
                                             f"ride analysis failed: {exc}"))
            return {'completed': False, 'message': str(exc)}
        stats.update({'road_class': road_class, 'speed': speed,
                      'static_load': lin.static_load})
        weight = np.abs(eng.ride.weighting(freq, 'k'))
        return {'completed': True, 'message': '', 'stats': stats,
                'freq': freq,
                'body_acc': np.abs(resp['body_acc']) / MM,
                'body_acc_w': np.abs(resp['body_acc']) * weight / MM,
                'tire_force': np.abs(resp['tire_force']) / MM}

    def run_ride_time(self, road_class, speed_kmh, sprung_mass,
                      duration=12.0, seed=1):
        """March the quarter car of the current model over a random road of
        the class (a point follower, the input of the linear theory).

        :return: dict with ``completed``, ``message``, and the result of
            :func:`suspension.ride.simulate`
        """
        eng = self.engine
        self.warnings = []
        try:
            speed = float(speed_kmh) / 3.6
            prof = eng.ride.random_profile(road_class,
                                           speed * duration + 5.0, seed=seed)
            out = eng.ride.simulate(
                eng.mbs, self.model(), float(sprung_mass),
                eng.ride.point_road_input(prof, speed, duration), settle=2.0)
        except Exception as exc:
            self.warnings.append(self._entry(
                'error', 'ride', None, f"ride time run failed: {exc}"))
            return {'completed': False, 'message': str(exc)}
        out.update({'completed': True, 'message': ''})
        return out

    # The full vehicle (P18).

    def vehicles(self):
        """``[(stem, name)]`` of the bundled vehicle files."""
        folder = self.engine.car.DATA_DIR
        out = []
        for fname in sorted(os.listdir(folder)):
            if fname.endswith('.json'):
                with open(os.path.join(folder, fname)) as fobj:
                    out.append((fname[:-5], json.load(fobj)['name']))
        return out

    #: The vehicle setup fields of the Vehicle tab (per axle).
    VEHICLE_FIELDS = ('spring_rate', 'arb_rate', 'ride_height',
                      'bump_clicks', 'rebound_clicks')

    def vehicle_setup(self, stem):
        """``{"front": setup, "rear": setup}`` of a bundled vehicle (the
        fields of :attr:`VEHICLE_FIELDS` the corners have)."""
        vm = self.engine.car.VehicleModel.load(stem)
        out = {}
        for axle_name, model in (("front", vm.front), ("rear", vm.rear)):
            base = self.engine.tuning.baseline(model)
            out[axle_name] = {k: base[k] for k in self.VEHICLE_FIELDS
                              if k in base}
        return out

    def _vehicle(self, stem, setups):
        vm = self.engine.car.VehicleModel.load(stem)
        setups = setups or {}
        return vm.with_setup(front=setups.get('front'),
                             rear=setups.get('rear'))

    def vehicle_readouts(self, stem, setups=None):
        """The ride and roll numbers of each axle of a vehicle: the corner
        readouts of :func:`tuning.readouts` at the corner sprung masses.

        :return: dict with ``completed``, ``message``, ``front``, ``rear``
        """
        self.warnings = []
        try:
            vm = self._vehicle(stem, setups)
            out = {axle_name: self.engine.tuning.readouts(
                self.engine.mbs, model,
                vm.corner_sprung_mass(axle_name == 'front'))
                for axle_name, model in (('front', vm.front),
                                         ('rear', vm.rear))}
        except Exception as exc:
            self.warnings.append(self._entry(
                'error', 'vehicle', None, f"vehicle readouts failed: {exc}"))
            return {'completed': False, 'message': str(exc)}
        out.update({'completed': True, 'message': ''})
        return out

    def run_handling(self, stem, setups=None):
        """The handling diagram of a vehicle with a setup.

        :return: dict with ``completed``, ``message``, and the result of
            :func:`suspension.car.handling_diagram`
        """
        car = self.engine.car
        self.warnings = []
        try:
            vm = self._vehicle(stem, setups)
            out = car.handling_diagram(self.engine.mbs, vm)
        except Exception as exc:
            self.warnings.append(self._entry(
                'error', 'vehicle', None, f"handling diagram failed: {exc}"))
            return {'completed': False, 'message': str(exc)}
        out.update({'completed': True, 'message': '', 'name': vm.name})
        return out

    #: Drive layouts of the Vehicle tab: (label, front share).
    DRIVES = (('rear drive', 0.0), ('front drive', 1.0),
              ('all-wheel drive 40 / 60', 0.4))

    def run_longitudinal(self, stem, brake_bias, drive_split, setups=None):
        """Braking and traction of a vehicle with a setup: the brake limit
        of the bias and the traction limit of the drive, the pitch and the
        corner travel per g, and the anti fractions (engine and the RCVD
        formula).

        :param brake_bias: front share of the braking force (0 to 1)
        :param drive_split: front share of the drive force (0 to 1)
        :return: dict with ``completed``, ``message``, ``brake``,
            ``traction``, ``braking`` and ``driving`` (dicts of ``pitch``
            deg/g, ``travel`` front and rear mm/g, ``anti`` front and rear
            fractions), and ``formula`` (RCVD anti fractions)
        """
        lo = self.engine.longitudinal
        mbs = self.engine.mbs
        self.warnings = []
        try:
            vm = self._vehicle(stem, setups)
            brake = lo.brake_balance(mbs, vm, float(brake_bias))
            traction = lo.traction_limit(mbs, vm, float(drive_split))
            geo = lo.anti_geometry(mbs, vm)
            out = {}
            for key, a, kw in (('braking', -0.1,
                                {'brake_bias': float(brake_bias)}),
                               ('driving', 0.1,
                                {'drive_split': float(drive_split)})):
                st = lo.steady_state(mbs, vm, [0.0, a], **kw)
                _, anti = lo.predicted_travel(mbs, vm, st)
                out[key] = {
                    'pitch': math.degrees(st['pitch'][1]) / abs(a),
                    'travel': [st['travel'][1][0] / abs(a) * MM,
                               st['travel'][1][2] / abs(a) * MM],
                    'anti': [anti[1][0], anti[1][2]]}
        except Exception as exc:
            self.warnings.append(self._entry(
                'error', 'vehicle', None,
                f"braking and traction failed: {exc}"))
            return {'completed': False, 'message': str(exc)}
        bias, split = float(brake_bias), float(drive_split)
        out.update({
            'completed': True, 'message': '', 'brake': brake,
            'traction': traction, 'brake_bias': bias, 'drive_split': split,
            'formula': {
                'anti_dive': bias * geo['front']['anti_brake'],
                'anti_lift': (1.0 - bias) * geo['rear']['anti_brake'],
                'anti_squat': (1.0 - split) * geo['rear']['anti_drive'],
                'anti_lift_drive': split * geo['front']['anti_drive']}})
        return out

    def run_vehicle(self, stem, accelerations_g=(0.0, 0.1, 0.2, 0.3, 0.4),
                    setups=None):
        """Steady-state cornering of a bundled vehicle, with the closed form
        and the understeer gradient.

        :return: dict with ``completed``, ``message``, ``steady``
            (:func:`suspension.car.steady_state`), ``closed``
            (:func:`suspension.car.closed_form`), ``understeer`` (rad/g),
            and ``name``
        """
        car = self.engine.car
        mbs = self.engine.mbs
        self.warnings = []
        try:
            vm = self._vehicle(stem, setups)
            steady = car.steady_state(mbs, vm, accelerations_g)
            closed = car.closed_form(mbs, vm)
            kus = car.understeer_gradient(vm, steady['roll_steer'])
        except Exception as exc:
            self.warnings.append(self._entry(
                'error', 'vehicle', None, f"vehicle analysis failed: {exc}"))
            return {'completed': False, 'message': str(exc)}
        return {'completed': True, 'message': '', 'steady': steady,
                'closed': closed, 'understeer': kus, 'name': vm.name,
                'wheelbase': vm.wheelbase}

    def run_step_steer(self, stem, speed_kmh, rack_mm, understeer=None,
                       duration=4.0, setups=None):
        """A step steer of a bundled vehicle, with the steady yaw rate of
        the linear bicycle model (understeer gradient from the steady-state
        rig).

        :return: dict with ``completed``, ``message``, the run of
            :func:`suspension.car.step_steer`, ``delta`` (rad, the front
            wheel angle of the rack at rest), and ``bicycle_yaw_rate``
            (rad/s)
        """
        car = self.engine.car
        mbs = self.engine.mbs
        self.warnings = []
        try:
            vm = self._vehicle(stem, setups)
            if understeer is None:
                steady = car.steady_state(mbs, vm, (0.0, 0.05, 0.1))
                understeer = car.understeer_gradient(vm,
                                                     steady['roll_steer'])
            rack = float(rack_mm) / MM
            run = car.step_steer(mbs, vm, float(speed_kmh) / 3.6, rack,
                                 duration=duration)
            delta = car.kinematic_steer(mbs, vm, rack)
        except Exception as exc:
            self.warnings.append(self._entry(
                'error', 'vehicle', None, f"step steer failed: {exc}"))
            return {'completed': False, 'message': str(exc)}
        v = float(run['steady']['speed'])
        run.update({'completed': True, 'message': '', 'delta': delta,
                    'understeer': understeer,
                    'bicycle_yaw_rate': v * delta / (
                        vm.wheelbase + understeer * v * v / 9.81)})
        return run

    def road_input(self, kind, params, sprung_mass=200.0):
        """Build the :class:`suspension.road.RoadInput` of a road input.

        :raises ValueError: For an unknown kind or bad parameters.
        """
        return make_road_input(self.engine, self.model(), kind, params,
                               sprung_mass)


def _curve_text(model):
    """The damper curve of a model in words: the low-speed slopes and the
    force at the fastest table point of each direction."""
    curve = model.damper_curve
    mod = sys.modules[type(model).__module__]
    bump = mod.curve_slope(curve, -1e-9) * 1.e-3
    rebound = mod.curve_slope(curve, 1e-9) * 1.e-3
    return (f"curve: bump {bump:g}, rebound {rebound:g} N.s/mm at low "
            f"speed; {curve[1][0]:.0f} N at {curve[0][0] * MM:g} mm/s to "
            f"{curve[1][-1]:.0f} N at {curve[0][-1] * MM:g} mm/s")


#: ISO 8608 road classes for the ride analysis.
ROAD_CLASSES = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')


def road_pass_time(road_input):
    """Time (s) the wheel needs to drive over the whole profile of a road
    input (its cams on the flat road again), or None for a rig."""
    if road_input.profile is None:
        return None
    lead = road_input.position[0]
    end = road_input.profile.end + road_input.cam.reach
    return (end - lead) / road_input.speed


def make_road_input(engine, model, kind, params, sprung_mass=200.0):
    """The road input of the quarter car, from :data:`ROAD_INPUTS`.

    :param sprung_mass: quarter sprung mass (kg), for the static tire
        deflection that sets the contact length of the enveloping cams
    :raises ValueError: For an unknown kind or bad parameters.
    """
    road = engine.road
    if kind not in ROAD_INPUTS:
        raise ValueError(f"unknown road input {kind!r}")
    p = dict(ROAD_INPUTS[kind][1], **(params or {}))
    if 'rig' == kind:
        return road.rig_input(float(p['amplitude']) / MM,
                              float(p['frequency']), float(p['duration']))
    if 'bump' == kind:
        profile = road.RoadProfile.bump(float(p['height']) / MM,
                                        float(p['length']) / MM)
    elif 'pothole' == kind:
        profile = road.RoadProfile.pothole(float(p['depth']) / MM,
                                           float(p['length']) / MM,
                                           edge=0.01)
    elif 'step' == kind:
        profile = road.RoadProfile.step(float(p['height']) / MM, ramp=0.002)
    elif 'sine' == kind:
        profile = road.RoadProfile.sine(float(p['amplitude']) / MM,
                                        float(p['wavelength']) / MM,
                                        cycles=4)
    else:
        if not p['path']:
            raise ValueError("pick a road profile CSV file")
        profile = road.RoadProfile.from_csv(p['path'])
    speed = float(p['speed']) / 3.6
    if speed <= 0.0:
        raise ValueError("the speed must be positive")
    # The static tire deflection sets the contact length of the cams.
    deflection = engine.vehicle.static_tire_deflection(
        model, float(sprung_mass))[0]
    cam = road.TandemCam.for_tire(model.wheel_radius, deflection)
    duration = float(p['duration'])
    if duration <= 0.0:
        raise ValueError("the duration must be positive")
    return road.road_input(profile, speed, cam, duration=duration)


class QuarterCarRun(object):
    """A quarter car on a road input, marched a chunk at a time.

    :ivar road: The :class:`suspension.road.RoadInput` (None before
        :meth:`start`).
    :ivar history: ``dict`` of lists: ``time`` (s); the ``body``, ``wheel``
        (from the static equilibrium) and ``road`` (effective road height)
        heights in mm; the suspension ``travel`` (mm, wheel center to the
        body from design, + bump); the tire ``force`` (N) and ``contact``
        (bool); the travel ``stop`` force (N) and ``stop_state`` (-1 bump
        stop, +1 droop stop, 0 free).
    :ivar finished: True when the run reached the duration, was cancelled,
        or failed; :attr:`message` says why it stopped.
    """

    #: Half length (m) of the road profile drawn around the wheel.
    ROAD_VIEW = 0.8

    def __init__(self, engine, model, sprung_mass, road, dt):
        self.engine = engine
        self.model = model
        self.sprung_mass = float(sprung_mass)
        self.road_spec = road
        self.dt = float(dt)
        self.road = None
        self.duration = 0.0
        self.history = {'time': [], 'body': [], 'wheel': [], 'road': [],
                        'travel': [], 'force': [], 'contact': [],
                        'stop': [], 'stop_state': []}
        #: The pose of every recorded step, for the replay.
        self.poses = []
        self.finished = False
        self.completed = False
        self.message = ''
        self._sys = None
        self._tire = None
        self._travel_stop = None
        #: The engine body numbers of the quarter car (after start).
        self.numbers = {}
        self._z0 = None

    @property
    def steps(self):
        return int(round(self.duration / self.dt))

    @property
    def airborne_time(self):
        """Time (s) the tire was off the road so far."""
        return self.dt * sum(1 for c in self.history['contact'] if not c)

    def stop_time(self, state):
        """Time (s) on the bump stop (``state`` -1) or the droop stop (+1)
        so far."""
        return self.dt * sum(1 for c in self.history['stop_state']
                             if c == state)

    @property
    def pass_time(self):
        """Time (s) the wheel needs to pass the profile; None for a rig."""
        return None if self.road is None else road_pass_time(self.road)

    def start(self):
        """Build the quarter car at its static equilibrium.

        :return: Whether the run can march.
        """
        eng = self.engine
        try:
            kind, params = self.road_spec
            self.road = make_road_input(eng, self.model, kind, params,
                                        self.sprung_mass)
            # The tabulated road ends at its duration; stop a step before.
            self.duration = self.road.duration - self.dt
            sys_obj, handles = eng.vehicle.quarter_car(
                eng.mbs, self.model, self.sprung_mass, road=self.road,
                dt=self.dt)
            sys_obj.set_integrator('rk45')
            sys_obj.set_tolerances(1e-7, 1e-10)
        except Exception as exc:
            return self._stop(False, str(exc))
        self._sys = sys_obj
        self._tire = handles['tire']
        self._travel_stop = handles.get('stop')
        self.numbers = {'chassis': eng.vehicle.CHASSIS}
        self.numbers.update(handles['corner'].nums)
        pos = np.asarray(sys_obj.get_body_positions())
        nums = self.numbers
        self._z0 = (pos[nums['chassis']][2], pos[nums['upright']][2])
        self._record()
        return True

    def _record(self):
        pos = np.asarray(self._sys.get_body_positions())
        hist = self.history
        t = self._sys.get_sim_time()
        hist['time'].append(t)
        self.poses.append(self.pose())
        nums = self.numbers
        hist['body'].append((pos[nums['chassis']][2] - self._z0[0]) * MM)
        hist['wheel'].append((pos[nums['upright']][2] - self._z0[1]) * MM)
        hist['road'].append(self.road.at(t)[0] * MM)
        hist['travel'].append(hist['wheel'][-1] - hist['body'][-1])
        state = self.engine.mbs.get_tire_state(self._tire)
        hist['force'].append(float(state['normal_force']))
        hist['contact'].append(bool(state['in_contact']))
        if self._travel_stop is None:
            hist['stop'].append(0.0)
            hist['stop_state'].append(0)
        else:
            stop = self.engine.mbs.get_length_stop_state(self._travel_stop)
            hist['stop'].append(float(stop['force']))
            hist['stop_state'].append(int(stop['engaged']))

    def pose(self):
        """``{body: (position_m, quaternion)}`` of the quarter car now, with
        the ``chassis`` and the effective ``road`` plane (its center under
        the wheel, tilted by the effective slope), or None before
        :meth:`start`."""
        if self._sys is None:
            return None
        pos = np.asarray(self._sys.get_body_positions())
        quat = np.asarray(self._sys.get_body_quaternions())
        out = {body: (np.array(pos[num], dtype='float64'),
                      np.array(quat[num], dtype='float64'))
               for body, num in self.numbers.items()}
        height, slope, _ = self.road.at(self._sys.get_sim_time())
        wheel = pos[self.numbers['upright']]
        # A forward slope b turns the plane about -y (its normal is
        # (-sin b, 0, cos b)).
        half = -0.5 * slope
        out['road'] = (np.array([wheel[0], wheel[1], height]),
                       np.array([math.cos(half), 0.0, math.sin(half), 0.0]))
        return out

    def road_profile(self, pose=None, t=None):
        """Segment ends (mm) of the road profile around the wheel at time
        ``t`` (now by default), in the side view plane of the wheel; empty
        for a rig."""
        road = self.road
        if road is None or road.profile is None or self._sys is None:
            return []
        t = self._sys.get_sim_time() if t is None else t
        pose = self.pose() if pose is None else pose
        wheel = pose['upright'][0]
        _, _, x_road = road.at(t)
        dx = np.linspace(-self.ROAD_VIEW, self.ROAD_VIEW, 161)
        z = road.profile.height(x_road + dx)
        pts = np.column_stack([wheel[0] + dx, np.full_like(dx, wheel[1]), z])
        pts = pts * MM
        ends = []
        for it in range(len(pts) - 1):
            ends += [pts[it], pts[it + 1]]
        return ends

    def advance(self, steps=10):
        """March up to ``steps`` time steps; return whether it can go on."""
        if self.finished or self._sys is None:
            return False
        for _ in range(steps):
            if len(self.history['time']) > self.steps:
                return self._stop(True, "completed")
            try:
                self._sys.solve()
            except Exception as exc:
                return self._stop(False, f"time step failed at t = "
                                         f"{self.history['time'][-1]:.3f} s: "
                                         f"{exc}")
            self._record()
        if len(self.history['time']) > self.steps:
            return self._stop(True, "completed")
        return True

    def cancel(self):
        if not self.finished:
            self._stop(False, f"cancelled at t = "
                              f"{self.history['time'][-1]:.3f} s"
                       if self.history['time'] else "cancelled")

    def _stop(self, completed, message):
        self.finished = True
        self.completed = completed
        self.message = message
        return False

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
