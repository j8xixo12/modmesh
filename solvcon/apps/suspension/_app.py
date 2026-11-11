# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The suspension K&C app on App > Suspension K&C.

The app lays the corner out as the pilot does a domain: the 3D view in the
MDI area, the model tree docked on the left, the properties and the
adjusters on the right, and the analysis tabs with the warnings below.  It
reads every result from a :class:`~._session.SuspensionSession` and moves
the 3D pose and the plot cursors together.
"""

import copy
import math
import os
import time

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog,
                               QLabel, QMessageBox)

from ...pilot import base

from . import _engine
from . import _glossary
from . import _jobs
from . import _overview
from . import _panel
from . import _plots
from . import _scene
from . import _session
from . import _view
from . import _window

__all__ = [  # noqa: F822
    'SuspensionApp',
    'build_app',
]

CURRENT = "current"
PREVIOUS = "previous"


def _rotation_vector(r):
    """The rotation vector of a rotation matrix (axis times angle)."""
    angle = math.acos(max(-1.0, min(1.0, 0.5 * (np.trace(r) - 1.0))))
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0],
                     r[1, 0] - r[0, 1]]) / (2.0 * math.sin(angle))
    return axis * angle


def _axis_quat(vec):
    angle = float(np.linalg.norm(vec))
    if angle < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = np.asarray(vec) / angle
    return np.array([math.cos(0.5 * angle), *(math.sin(0.5 * angle) * axis)])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


class SuspensionApp(base.PilotFeature):
    """Suspension kinematics and compliance of one corner.

    :ivar session: The :class:`~._session.SuspensionSession`, or None
        before the app opens or when the engine is missing.
    :ivar error: Why the app could not start, or None.
    """

    TITLE = "Suspension K&C"
    #: Time steps of the quarter car per tick of the timer, about 15 ms of
    #: engine time, so the window stays live and Cancel is heard while a
    #: run marches.
    STEPS_PER_TICK = 20

    def __init__(self, *args, engine=None, model_path=None, **kw):
        super().__init__(*args, **kw)
        self._engine = engine
        self._model_path = model_path
        self.session = None
        self.error = None
        self.error_window = None
        self.error_label = None
        self.view = _view.CornerView(self._mgr)
        self.view.closed = self._on_closed
        self.tree = None
        self.adjusters = None
        self.analysis = None
        self.docks = []
        self.selected = None
        self.show_forces = False
        self.run = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self._geometry = None
        self._geometries = []
        self._geometry_setup = None
        #: The geometry of the model :attr:`run` was built from.
        self._run_geometry = None
        #: The understeer gradient of each vehicle the Vehicle tab solved.
        self._understeer = {}
        #: Run the vehicle analyses in a child process (the window stays
        #: live and a run can be cancelled); False runs them in place.
        self.background = True
        self._job = None
        self._job_done = None
        self._job_text = ""
        self._job_started = 0.0
        self.job_timer = QTimer()
        self.job_timer.timeout.connect(self._job_tick)
        #: The geometries the view shows instead of the sweep, or False.
        self._showing_run = False
        self.overlays = set()
        self.profile = None
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._play_tick)
        #: The 3D frames of each analysis tab: ``{key: [(label,
        #: geometries, pose, overlays)]}``, and the one that plays.
        self.animations = {}
        self._animating = None
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self._anim_tick)

    #: The playback period at 1x, in ms per sweep point.
    PLAY_MS = 80

    def populate_menu(self):
        self.add_action(
            "App", text="Suspension K&C",
            tip="Kinematics and compliance of a race car front corner",
            func=self.open, id="app.suspension", weight=20)

    def open(self):
        """Open the app, or bring it back; a missing engine is reported in
        a window instead."""
        if self.view.is_open:
            self._mgr.mdiArea.setActiveSubWindow(self.view.subwin)
            return
        if self.session is None and not self._start_session():
            return
        self.view.open(f"{self.TITLE}: {self.session.base_model.name}")
        self._build_docks()
        if self.session.current is None:
            self.session.run_sweep()
        self._rebuild_view(aim=True)
        self.refresh()
        if self.view.subwin is not None:
            self.view.subwin.showMaximized()

    def _start_session(self):
        try:
            engine = self._engine or _engine.load()
            self.session = _session.SuspensionSession(engine,
                                                      self._model_path)
        except Exception as exc:
            self.error = str(exc)
            self._show_error(self.error)
            return False
        self.error = None
        return True

    def _show_error(self, message):
        text = (f"The suspension app cannot start.\n\n{message}\n\n"
                f"Nothing was computed.")
        if self.error_window is None:
            label = QLabel(text)
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignCenter)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setMargin(24)
            subwin = self._mgr.addSubWindow(label)
            subwin.setWindowTitle(f"{self.TITLE}: unavailable")
            subwin.setAttribute(Qt.WA_DeleteOnClose, True)
            self._error_filter = base.SubWindowCloseFilter(
                self._on_error_closed, subwin)
            subwin.installEventFilter(self._error_filter)
            subwin.show()
            self.error_window = subwin
            # Held: the sub-window does not keep the Python wrapper alive.
            self.error_label = label
        else:
            self.error_label.setText(text)
        self._pycon.writeToHistory(f"suspension: {message}\n")

    def _on_error_closed(self):
        self.error_window = None
        self.error_label = None
        self._error_filter = None

    def _build_docks(self):
        handlers = {'kind': self.on_kind, 'range': self.on_range,
                    'cursor': self.on_cursor, 'export': self.on_export,
                    'forces': self.on_forces, 'statics': self.on_statics,
                    'dynamics': self.on_dynamics, 'cancel': self.on_cancel,
                    'modal': self.on_modal, 'measured': self.on_measured,
                    'report': self.on_report, 'layout': self.on_layout,
                    'overlay': self.on_overlay, 'step': self.on_step,
                    'play': self.on_play, 'replay': self.on_replay,
                    'compliance': self.on_compliance_sweep,
                    'hardpoint': self.on_hardpoint, 'batch': self.on_batch,
                    'profile': self.on_profile,
                    'profile_import': self.on_profile_import,
                    'profile_cursor': self.on_profile_cursor,
                    'road_csv': self.on_road_csv, 'ride': self.on_ride,
                    'ride_time': self.on_ride_time,
                    'vehicle': self.on_vehicle,
                    'step_steer': self.on_step_steer,
                    'setup': self.on_setup,
                    'setup_action': self.on_setup_action,
                    'vehicle_picked': self.on_vehicle_picked,
                    'vehicle_open': self.on_vehicle_open,
                    'vehicle_import': self.on_vehicle_import,
                    'vehicle_input_changed': self.on_vehicle_input_changed,
                    'log_open': self.on_log_open,
                    'log_replay': self.on_log_replay,
                    'handling': self.on_handling,
                    'vehicle_reference': self.on_vehicle_reference,
                    'longitudinal': self.on_longitudinal,
                    'frame': self.on_frame, 'animate': self.on_animate,
                    'mode_shape': self.on_mode_shape,
                    'vehicle_setup_changed': self.on_vehicle_setup_changed,
                    'vehicle_cancel': self.cancel_vehicle_job,
                    'overview_update': self.on_overview_update,
                    'overview_targets': self.on_overview_targets}
        self.tree = _panel.ModelTree()
        self.tree.selected = self.on_selected
        self.tree.open_clicked = self.on_open_model
        self.tree.library_picked = self.on_library
        self.tree.save_clicked = self.on_save_model
        self.tree.activated = self.on_tree_activated
        self.adjusters = _panel.AdjusterPanel()
        self.adjusters.changed = self.on_adjuster
        self.adjusters.edited = self.on_property_edited
        self.adjusters.solved = self.on_solve_alignment
        self.analysis = _window.AnalysisPanel(handlers)
        self.analysis.warnings.picked = self.tree.select
        self.analysis.sweep.runs.changed = self.on_run_list
        self.analysis.vehicle.show_vehicles(self.session.vehicles())
        self.on_vehicle_picked(self.analysis.vehicle.vehicle.currentData())
        self.analysis.setup.show_fields(self.session.setup_fields())
        self.view.picked = self.on_point_picked
        main = self._mgr.mainWindow
        for title, widget, area in (
                ("Suspension model", self.tree, Qt.LeftDockWidgetArea),
                ("Suspension properties", self.adjusters,
                 Qt.RightDockWidgetArea),
                ("Suspension analysis", self.analysis,
                 Qt.BottomDockWidgetArea)):
            dock = QDockWidget(title)
            # The model tree and the analysis tabs scroll by themselves;
            # the properties stack of group boxes needs a scroll area.
            dock.setWidget(_window.scrollable(widget, 0)
                           if widget is self.adjusters else widget)
            main.addDockWidget(area, dock)
            self.docks.append(dock)

    def close(self):
        self.view.close()

    def _on_closed(self):
        self.timer.stop()
        self.play_timer.stop()
        self.cancel_vehicle_job()
        if self.run is not None:
            self.run.cancel()
        for dock in self.docks:
            self._mgr.mainWindow.removeDockWidget(dock)
            dock.deleteLater()
        self.docks = []
        self.tree = self.adjusters = self.analysis = None
        self._geometry = self._geometry_setup = None
        self._geometries = []

    # The readouts, filled from the session.

    def _rebuild_view(self, aim=False):
        """Rebuild the body meshes when the setup or the corners they were
        built for changed; an adjuster moves the upright pickups."""
        result = self.session.current
        key = (None if result is None
               else (tuple(sorted(result.setup.items())),
                     tuple(result.corners)))
        if self._geometry is not None and key == self._geometry_setup:
            return
        count = len(self._geometries)
        self._geometries = self.session.geometries()
        self._geometry = self._geometries[0]
        self._geometry_setup = key
        # A corner more or less changes what the camera has to hold.
        self.view.build(self._geometries,
                        aim=aim or count != len(self._geometries))

    def refresh(self):
        """Show the session in every panel."""
        sess = self.session
        if self.tree is None or sess is None:
            return
        picked = self.tree.current_name()
        self.tree.show_outline(sess.outline(), sess.model_path)
        if picked is not None:
            self.tree.select(picked)
        self.adjusters.show_adjusters(sess.adjusters,
                                      sess.base_model.adjusters)
        self.tree.show_library(sess.library(), os.path.splitext(
            os.path.basename(sess.model_path))[0])
        sweep = self.analysis.sweep
        sweep.show_kinds(sess.kinds())
        lo, hi, count = sess.ranges[sess.kind.name]
        sweep.show_range(sess.kind.name, lo, hi, count)
        current = sess.current
        kind = sess.kind if current is None else current.kind
        sweep.set_kind(kind, sess.layout[kind.name])
        runs = [run for run in sess.visible_runs(kind)
                if run.result is not current]
        measured = sess.measured_for(kind)
        summaries = {}
        for curve in kind.curves:
            if curve.key not in sweep.plots:
                continue
            curves = []
            lines = []
            # The other runs are dashed in their own colors, the baseline
            # solid; the current result is the solid blue curve.
            for run in runs:
                xs, ys = run.result.curve(curve.key)
                curves.append((run.label, xs, ys, run.color,
                               run is not sess.baseline))
            for data in measured:
                if curve.key in data.curves:
                    curves.append((f"measured: {data.name}", data.x,
                                   data.curves[curve.key], (60, 60, 60),
                                   False))
            if current is not None:
                xs, ys = current.curve(curve.key)
                curves.append((CURRENT, xs, ys, _plots.CURRENT_COLOR,
                               False))
                stats = sess.summary(sess.current_run(), curve.key) \
                    if sess.current_run() else None
                if stats is not None:
                    lines.append(_window._summary_text(stats, curve.unit,
                                                       kind.x.unit))
            if sess.baseline is not None and sess.baseline.result.kind \
                    is kind and sess.baseline.result is not current:
                stats = sess.summary(sess.baseline, curve.key)
                lines.append("baseline " + _window._summary_text(
                    stats, curve.unit, kind.x.unit))
            sweep.plots[curve.key].set_curves(curves)
            sweep.plots[curve.key].set_markers(
                {f"measured: {d.name}" for d in measured})
            summaries[curve.key] = lines
        sweep.show_summaries(summaries)
        sweep.runs.show_runs(sess.runs, sess.baseline)
        legend = []
        if current is not None:
            legend.append(f"solid blue: current {kind.name} sweep, "
                          f"{current.setup_text()}")
        if runs:
            legend.append(f"dashed: {len(runs)} other run(s), colors in "
                          f"the run list")
        if sess.baseline is not None:
            legend.append(f"solid color: baseline {sess.baseline.label}")
        if measured:
            legend.append("dots: measured " + ", ".join(
                d.name for d in measured))
        sweep.legend.setText(";    ".join(legend) or "no valid result")
        self.analysis.hardpoints.show_rows(sess.hardpoint_rows())
        self.analysis.batch.show_points(
            [row[0] for row in sess.hardpoint_rows()])
        self._rebuild_view()
        self.update_cursor()
        self.analysis.warnings.show_entries(sess.warnings)

    def update_cursor(self, index=None):
        """Move the shared cursor: the 3D pose and every plot marker."""
        sess = self.session
        if self.analysis is None or sess is None:
            return
        if index is not None:
            sess.set_cursor(index)
        sweep = self.analysis.sweep
        current = sess.current
        if current is None:
            sweep.show_cursor(0, 0, "no valid result")
            return
        value = sess.cursor_value()
        kind = current.kind
        sweep.show_cursor(sess.cursor, len(current),
                          f"{kind.x.title} {value:+.2f} mm "
                          f"(point {sess.cursor + 1} of {len(current)})")
        for plot in sweep.plots.values():
            plot.set_cursor(value)
        if self._showing_run:
            # The quarter car posed the view: rebuild the corners of the
            # sweep it hid, and pose the sweep again.
            self._showing_run = False
            self._geometry_setup = None
            self._rebuild_view()
        forces = sess.forces() if self.show_forces else None
        overlays = {key: ends for key, ends in sess.overlays().items()
                    if key in self.overlays}
        self.view.show_pose(self._geometries, sess.pose(),
                            selected=self.selected, forces=forces,
                            overlays=overlays)
        rows = []
        for curve in kind.curves:
            ys = current.curve(curve.key)[1]
            rows.append(f"{curve.title}: {ys[sess.cursor]:+.4f} "
                        f"{curve.unit}")
        self.adjusters.show_alignment("\n".join(rows))

    # The handlers of the controls.

    def on_adjuster(self, name, value):
        self.session.set_adjuster(name, value)
        self._drop_results(self.MODEL_CHANGED)
        self.refresh()

    def on_kind(self, name):
        self.session.set_kind(name)
        self.refresh()

    def on_range(self, lo, hi, count):
        self.session.set_range(lo, hi, count)
        self.refresh()

    def on_cursor(self, index):
        self.update_cursor(index)

    def on_forces(self, shown):
        self.show_forces = shown
        self.update_cursor()

    def on_selected(self, group, name, props):
        self._picked = (group, name)
        self.adjusters.show_properties(f"{group}: {name}", props,
                                       self.session.editable(group, name))
        links = self._geometry.model.links if self._geometry else {}
        self.selected = (name if name in links
                         or name in ('pushrod', 'pullrod', 'strut', 'spring')
                         else None)
        self.update_cursor()

    # Editing the model from the model tree (P22).

    def ask_save(self, path):
        """Ask how to keep a model change: ``overwrite`` the model file,
        keep it ``temporary`` (in this session), or ``cancel`` it.  Tests
        replace this method."""
        box = QMessageBox(self._mgr.mainWindow)
        box.setWindowTitle("Model changed")
        box.setText(f"Keep the change of the model?\n\nfile: {path}")
        over = box.addButton("Overwrite the file", QMessageBox.AcceptRole)
        temp = box.addButton("Keep it temporary", QMessageBox.ActionRole)
        box.addButton("Cancel the change", QMessageBox.RejectRole)
        box.setDefaultButton(temp)
        box.exec()
        clicked = box.clickedButton()
        return ('overwrite' if clicked is over
                else 'temporary' if clicked is temp else 'cancel')

    def _refresh_tree(self):
        if self.tree is None:
            return
        picked = self.tree.current_name()
        self.tree.show_outline(self.session.outline(),
                               self.session.model_path)
        if picked is not None:
            self.tree.select(picked)

    def ask_switch(self, path):
        """Ask what to do with the unsaved edits of the tree model before
        another corner opens: ``save`` them to the file, ``discard`` them,
        or ``cancel``.  Tests replace this method."""
        box = QMessageBox(self._mgr.mainWindow)
        box.setWindowTitle("Unsaved model edits")
        box.setText(f"The model in the tree has edits not in its file.\n\n"
                    f"file: {path}")
        save = box.addButton("Save them to the file", QMessageBox.AcceptRole)
        drop = box.addButton("Discard them", QMessageBox.DestructiveRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(save)
        box.exec()
        clicked = box.clickedButton()
        return ('save' if clicked is save
                else 'discard' if clicked is drop else 'cancel')

    def on_tree_activated(self, group, name):
        """A double-click on an axle of the Vehicle group opens its corner
        file in the tree."""
        if 'Vehicle' != group or name not in ('front axle', 'rear axle'):
            return
        sess = self.session
        path = sess.axle_corner_path(name.split()[0])
        if path is None or path == sess.model_path:
            return
        if sess.model_dirty:
            choice = self.ask_switch(sess.model_path)
            if 'cancel' == choice:
                return
            if 'save' == choice:
                sess.save_model()
        self.load_model(path)

    def _on_vehicle_edited(self, path, text):
        sess = self.session
        panel = self.adjusters
        before = (copy.deepcopy(sess.vehicle_raw), set(sess.vehicle_dirty))
        try:
            sess.edit_model(path, text)
        except Exception as exc:
            panel.edit_status.setText(f"not changed: {exc}")
            self._show_picked()
            return
        choice = self.ask_save(str(sess._vehicle_file(sess.vehicle_stem)))
        if 'cancel' == choice:
            sess.vehicle_raw, sess.vehicle_dirty = before
            panel.edit_status.setText("the change is cancelled")
        elif 'overwrite' == choice:
            panel.edit_status.setText(f"saved to {sess.save_vehicle()}")
        else:
            panel.edit_status.setText("kept in this session")
        # The vehicle results are of the old vehicle data.
        self.on_vehicle_setup_changed()
        self.analysis.vehicle.status.setText(
            "the vehicle changed: run the analysis again")
        self._refresh_tree()
        self._show_picked()

    def on_property_edited(self, path, text):
        if '__vehicle__' == path[0]:
            return self._on_vehicle_edited(path, text)
        sess = self.session
        panel = self.adjusters
        before = (sess.base_model, sess.model_dirty, sess.current,
                  sess.previous, dict(sess.adjusters))
        try:
            sess.edit_model(path, text)
        except Exception as exc:
            panel.edit_status.setText(f"not changed: {exc}")
            self._show_picked()
            return
        if '__adjuster__' != path[0]:
            choice = self.ask_save(sess.model_path)
            if 'cancel' == choice:
                (sess.base_model, sess.model_dirty, sess.current,
                 sess.previous, adjusters) = before
                sess.adjusters.update(adjusters)
                panel.edit_status.setText("the change is cancelled")
                self._drop_results(self.MODEL_CHANGED)
                self._geometry = None
                self._rebuild_view()
                self.refresh()
                self._show_picked()
                return
            if 'overwrite' == choice:
                sess.save_model()
                panel.edit_status.setText(f"saved to {sess.model_path}")
            else:
                panel.edit_status.setText(
                    "kept in this session; Save model... writes it")
        # The geometry may have moved: rebuild the meshes; the results of
        # the old model go; the setup sheet starts from the changed model.
        self._drop_results(self.MODEL_CHANGED)
        self._geometry = None
        self._rebuild_view()
        self.refresh()
        self._show_picked()
        self.analysis.setup.show_fields(sess.setup_fields())

    def _show_picked(self):
        picked = getattr(self, '_picked', None)
        if picked is None:
            return
        group, name = picked
        props = dict(self.session.outline()).get(group, [])
        rows = dict(props).get(name, [])
        self.adjusters.show_properties(f"{group}: {name}", rows,
                                       self.session.editable(group, name))
        self.tree.dirty_label.setText(
            "the model has changes not in its file"
            if self.session.model_dirty else "")

    def on_save_model(self):
        sess = self.session
        path, _ = QFileDialog.getSaveFileName(
            None, "Save the corner model", sess.model_path,
            "Corner model (*.json)")
        if path:
            sess.save_model(path)
            self.tree.dirty_label.setText("")
            self.adjusters.edit_status.setText(f"saved to {path}")
            self.refresh()

    def on_library(self, stem):
        self.load_model(os.path.join(self.session.engine.data_dir,
                                     f"{stem}.json"))

    def on_open_model(self):
        path, _ = QFileDialog.getOpenFileName(
            None, "Open a corner model", self.session.engine.data_dir,
            "Corner model (*.json)")
        if path:
            self.load_model(path)

    def load_model(self, path):
        """Read another corner model and sweep it; a file that cannot be
        read keeps the model shown."""
        try:
            self.session.load(path)
        except Exception as exc:
            self.session.warnings = [{
                'level': 'error', 'source': 'model', 'joint': None,
                'message': f"cannot read {path}: {exc}; the model shown is "
                           f"unchanged"}]
            self.analysis.warnings.show_entries(self.session.warnings)
            return False
        # The run and the results of the old model have its bodies and its
        # numbers: drop them before the new model is drawn.
        self._drop_results()
        self.session.run_sweep()
        self._geometry = None
        self._rebuild_view(aim=True)
        # The quarter car mass of the model, else the default; a mass left
        # from another model can be far from this one's springs.
        mass = self.session.base_model.vehicle.get('corner_mass') \
            or self.DEFAULT_SPRUNG_MASS
        self.analysis.dynamics.mass.setValue(float(mass))
        self.analysis.modal.mass.setValue(float(mass))
        self.analysis.ride.mass.setValue(float(mass))
        self.analysis.setup.mass.setValue(float(mass))
        self.analysis.setup.show_fields(self.session.setup_fields())
        self.refresh()
        return True

    #: The quarter car sprung mass (kg) of a model without a corner mass.
    DEFAULT_SPRUNG_MASS = 200.0

    #: The note of a tab whose results the last change dropped.
    MODEL_CHANGED = "the model changed: run the analysis again"

    def _drop_results(self, note=""):
        """Stop and drop the quarter car run, and clear every tab that shows
        results of the corner model (each was computed for the model before
        the change); ``note`` tells the user why."""
        self.timer.stop()
        self.cancel_vehicle_job(note)
        if self.run is not None and not self.run.finished:
            self.run.cancel()
        self.run = None
        self._run_geometry = None
        self._showing_run = False
        if self.analysis is not None:
            self.analysis.dynamics.clear()
            self.analysis.modal.clear()
            self.analysis.statics.clear()
            self.analysis.ride.clear()
            self.analysis.setup.clear()
            self.analysis.batch.clear(note)
            self.analysis.profile.clear(note)
            for key in ('ride', 'modal', 'statics'):
                if key in self.animations:
                    self.set_animation(key, [])
            for tab in (self.analysis.dynamics, self.analysis.modal,
                        self.analysis.statics, self.analysis.ride):
                tab.status.setText(note)
            self.analysis.overview.clear(
                _overview.CORNER_GROUPS + _overview.VEHICLE_GROUPS, note)
            # The vehicle uses the corner of the tree for its axle: its
            # results are of the old corner too.
            vehicle = self.analysis.vehicle
            vehicle.clear_results(note)
            if 'vehicle' in self.animations:
                self.set_animation('vehicle', [])
            if vehicle.vehicle.count():
                vehicle.show_vehicle_setup(self.session.vehicle_setup(
                    vehicle.vehicle.currentData()))
                self._show_vehicle_sources()

    def on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            None, "Export the sweep", "suspension_sweep.csv",
            "CSV (*.csv)")
        if path:
            self.export_csv(path)

    def export_csv(self, path):
        try:
            self.session.export_csv(path)
        except (OSError, ValueError) as exc:
            self.session.warnings.append({
                'level': 'error', 'source': 'export', 'joint': None,
                'message': f"export failed: {exc}"})
        else:
            self.session.warnings.append({
                'level': 'info', 'source': 'export', 'joint': None,
                'message': f"sweep written to {path}"})
        self.analysis.warnings.show_entries(self.session.warnings)

    def _busy(self, label, text, func):
        """Run ``func`` with a wait cursor and ``text`` in ``label``; the
        engine call holds the thread, so the label is painted first."""
        label.setText(text)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            return func()
        finally:
            QApplication.restoreOverrideCursor()

    def on_statics(self, force):
        tab = self.analysis.statics
        out = self._busy(tab.status, "solving the compliance equilibrium...",
                         lambda: self.session.run_compliance(force))
        if out['completed']:
            tab.add_row(out)
            self.set_animation('statics', self._load_frames(out))
            tab.status.setText(f"load {list(force)} N: toe change "
                               f"{out['toe']:+.4f} deg, camber change "
                               f"{out['camber']:+.4f} deg")
        else:
            tab.status.setText(f"failed, no result shown: {out['message']}")
        self.analysis.warnings.show_entries(self.session.warnings)

    def on_modal(self, mass):
        tab = self.analysis.modal
        out = self._busy(tab.status, "solving the static equilibrium and "
                                     "the modes...",
                         lambda: self.session.run_modal(mass))
        if out['completed']:
            tab.show_modes(out['modes'])
            tab.status.setText(f"quarter car, sprung mass {mass:g} kg, "
                               f"{self.session.current.setup_text()}"
                               if self.session.current else "")
        else:
            tab.status.setText(f"failed, the table keeps the last valid "
                               f"modes: {out['message']}")
        self.analysis.warnings.show_entries(self.session.warnings)

    def on_ride(self, road_class, speed_kmh, mass):
        tab = self.analysis.ride
        out = self._busy(tab.status, "solving the linear ride response...",
                         lambda: self.session.run_ride(road_class, speed_kmh,
                                                       mass))
        if out['completed']:
            tab.show_response(out)
        else:
            tab.status.setText(f"failed: {out['message']}")
        self.analysis.warnings.show_entries(self.session.warnings)

    def on_ride_time(self, road_class, speed_kmh, mass):
        tab = self.analysis.ride
        if not tab.status.text():
            self.on_ride(road_class, speed_kmh, mass)
        before = tab.status.text()
        out = self._busy(tab.status, "marching the quarter car over a "
                                     "random road (12 s)...",
                         lambda: self.session.run_ride_time(
                             road_class, speed_kmh, mass))
        tab.status.setText(before)
        if out['completed']:
            tab.show_time(out)
            self.set_animation('ride', self._quarter_frames(
                self.session.model(), out['poses'],
                [f"t = {t:.2f} s" for t in out['pose_times']]), index=0)
        else:
            tab.status.setText(f"failed: {out['message']}")
        self.analysis.warnings.show_entries(self.session.warnings)

    # The setup sheet (P20).

    def on_setup(self, name, value, mass):
        tab = self.analysis.setup
        try:
            self.session.set_setup(name, value)
        except ValueError as exc:
            tab.status.setText(f"not applied: {exc}")
            tab.show_fields(self.session.setup_fields())
            return
        self._drop_results(self.MODEL_CHANGED)
        self.refresh()
        self.on_setup_action('readouts', mass)

    def on_setup_action(self, action, mass):
        tab = self.analysis.setup
        sess = self.session
        if 'reference' == action:
            sess.set_setup_reference()
            tab.show_fields(sess.setup_fields())
        elif 'reset' == action:
            sess.reset_setup()
            self._drop_results(self.MODEL_CHANGED)
            tab.show_fields(sess.setup_fields())
            self.refresh()
        elif 'save' == action:
            path, _ = QFileDialog.getSaveFileName(
                None, "Save the setup sheet", "", "Setup sheet (*.json)")
            if not path:
                return
            sess.save_setup(path)
            tab.status.setText(f"setup saved to {path}")
            return
        elif 'load' == action:
            path, _ = QFileDialog.getOpenFileName(
                None, "Load a setup sheet", "", "Setup sheet (*.json)")
            if not path:
                return
            try:
                sess.load_setup(path)
            except (OSError, ValueError) as exc:
                tab.status.setText(f"not loaded: {exc}")
                return
            self._drop_results(self.MODEL_CHANGED)
            tab.show_fields(sess.setup_fields())
            self.refresh()
        out = self._busy(tab.status, "computing the setup numbers...",
                         lambda: sess.setup_readouts(mass))
        if out['completed']:
            tab.show_readouts(out)
            tab.status.setText(
                f"setup changes from the model file: "
                f"{sess.setup or 'none'}")
        else:
            tab.status.setText(f"failed: {out['message']}")
        self.analysis.warnings.show_entries(sess.warnings)

    # The vehicle (P18, P21).

    def on_vehicle_picked(self, stem):
        if stem is None:
            return
        self.cancel_vehicle_job()
        self.session.set_vehicle(stem)
        self._refresh_tree()
        tab = self.analysis.vehicle
        tab.show_vehicle_setup(self.session.vehicle_setup(stem))
        tab.apply_defaults(self.session.vehicle_defaults(stem))
        self._show_vehicle_sources()
        # Nothing of the last vehicle stays: its results, its reference, and
        # its 3D frames.
        tab.clear_results(keep_reference=False)
        self.set_animation('vehicle', [])
        self.analysis.overview.clear(_overview.VEHICLE_GROUPS,
                                     f"vehicle {stem}: run the analyses")

    def on_vehicle_open(self):
        path, _ = QFileDialog.getOpenFileName(
            None, "Open a vehicle file", "", "Vehicle (*.json)")
        if path:
            self.open_vehicle_file(path)

    def open_vehicle_file(self, path):
        """List a vehicle file in the Vehicle tab and pick it."""
        try:
            stem = self.session.add_vehicle_file(path)
        except (OSError, ValueError) as exc:
            self._vehicle_failed(f"the vehicle is not opened: {exc}")
            return None
        self._pick_vehicle(stem)
        return stem

    def on_vehicle_import(self):
        folder = QFileDialog.getExistingDirectory(
            None, "The data folder of an Assetto Corsa car")
        if not folder:
            return
        default = os.path.join(os.path.dirname(os.path.abspath(folder)),
                               "solvcon_vehicle.json")
        target, _ = QFileDialog.getSaveFileName(
            None, "Save the converted vehicle as", default,
            "Vehicle (*.json)")
        if target:
            self.import_ac_vehicle(folder, target)

    def import_ac_vehicle(self, folder, target):
        """Convert Assetto Corsa car data to vehicle and corner files
        (:meth:`SuspensionSession.import_ac_vehicle`) and pick the
        vehicle."""
        tab = self.analysis.vehicle
        try:
            stem = self._busy(tab.status, "converting the AC data",
                              lambda: self.session.import_ac_vehicle(
                                  folder, target))
        except Exception as exc:
            self._vehicle_failed(f"the AC data are not converted: {exc}")
            return None
        self._pick_vehicle(stem)
        tab.status.setText(
            f"converted to {stem}; double-click an axle in the model tree "
            "to open its corner")
        return stem

    def _vehicle_failed(self, message):
        sess = self.session
        sess.warnings = [sess._entry('error', 'vehicle', None, message)]
        self.analysis.vehicle.status.setText(message)
        self.analysis.warnings.show_entries(sess.warnings)

    def _pick_vehicle(self, stem):
        tab = self.analysis.vehicle
        tab.show_vehicles(self.session.vehicles())
        index = tab.vehicle.findData(stem)
        tab.vehicle.blockSignals(True)
        tab.vehicle.setCurrentIndex(index)
        tab.vehicle.blockSignals(False)
        self.on_vehicle_picked(stem)

    def on_log_open(self):
        path, _ = QFileDialog.getOpenFileName(
            None, "Open a RaceChrono CSV export", "", "CSV (*.csv)")
        if path:
            self.set_log(path)

    def set_log(self, path):
        """Take ``path`` as the log of the replay: the replay results of
        another log go."""
        self.analysis.vehicle.set_log(path)
        self.on_vehicle_input_changed()

    def on_log_replay(self, stem, path, start, length, gyro_axes,
                      gyro_signs, brake_bias, drive_split, setups=None):
        if not path:
            return

        def done(outs):
            out = outs[0]
            tab = self.analysis.vehicle
            if out['completed']:
                tab.show_replay(out)
                self.set_animation('vehicle', self._vehicle_frames(
                    [(f"log replay t = {t:.2f} s", p)
                     for t, p in zip(out['pose_times'], out['poses'])],
                    out['models']), index=0)
            else:
                tab.status.setText(f"failed: {out['message']}")

        self._vehicle_run(
            f"replaying {os.path.basename(path)} ({length:.0f} s of the "
            "log)...",
            [('run_log_replay', (stem, path),
              {'start': start, 'length': length, 'gyro_axes': gyro_axes,
               'gyro_signs': gyro_signs, 'brake_bias': brake_bias,
               'drive_split': drive_split, 'setups': setups})], done)

    def on_vehicle_input_changed(self):
        """A run input (speed, rack, brake bias, drive) changed: the
        vehicle results shown are of the old inputs."""
        self.cancel_vehicle_job()
        self.analysis.vehicle.clear_results(
            "an input changed: run the analysis again")
        self.set_animation('vehicle', [])
        self.analysis.overview.clear(
            _overview.VEHICLE_GROUPS,
            "a vehicle input changed: run the analyses again")

    def _show_vehicle_sources(self):
        tab = self.analysis.vehicle
        stem = tab.vehicle.currentData()
        if stem is None:
            return
        sess = self.session
        tab.show_sources(sess.vehicle_sources(stem),
                         os.path.basename(sess.model_path),
                         sess.model_dirty)

    def on_vehicle_setup_changed(self):
        """A vehicle setup field changed: the shown results are of the old
        setup, so they go (the reference stays: it is a chosen setup)."""
        self.cancel_vehicle_job()
        tab = self.analysis.vehicle
        tab.clear_results("the setup changed: run the analysis again")
        self.set_animation('vehicle', [])
        self.analysis.overview.clear(
            _overview.VEHICLE_GROUPS,
            "the vehicle setup changed: run the analyses again")

    def on_handling(self, stem, setups):
        def done(outs):
            out = outs[0]
            tab = self.analysis.vehicle
            if out['completed']:
                out['readouts'] = outs[1]
                ro = out['readouts']
                tab.show_handling(out, ro if ro['completed'] else None)
                self._vehicle_kpis('handling', out)
                st = out['steady']
                self.set_animation('vehicle', self._vehicle_frames(
                    [(f"steady turn {a:.2f} g", p)
                     for a, p in zip(st['a_y'], st['poses'])],
                    st['models']))
            else:
                tab.status.setText(f"failed: {out['message']}")

        self._vehicle_run("solving the handling diagram...",
                          [('run_handling', (stem, setups), {}),
                           ('vehicle_readouts', (stem, setups), {})], done)

    def on_longitudinal(self, stem, brake_bias, drive_split, setups=None):
        def done(outs):
            out = outs[0]
            tab = self.analysis.vehicle
            if out['completed']:
                tab.show_longitudinal(out)
                self._vehicle_kpis('longitudinal', out)
                brake = out['brake']['state']
                drive = out['traction']['state']
                items = [(f"braking {-a:.2f} g", p)
                         for a, p in zip(brake['a_x'], brake['poses'])]
                items += [(f"accelerating {a:.2f} g", p)
                          for a, p in zip(drive['a_x'], drive['poses'])]
                self.set_animation('vehicle', self._vehicle_frames(
                    items, brake['models']), index=0)
            else:
                tab.status.setText(f"failed: {out['message']}")

        self._vehicle_run("solving the braking and the traction states...",
                          [('run_longitudinal',
                            (stem, brake_bias, drive_split, setups), {})],
                          done)

    # The overview KPIs.

    def _vehicle_kpis(self, kind, out):
        overview = self.analysis.overview
        overview.set_values(self.session.vehicle_kpis(kind, out))
        if 'step' == kind and not out.get('plausible', True):
            overview.set_note('handling', "step steer NOT PLAUSIBLE: " + "; "
                              .join(c['message'] for c in out['checks']
                                    if 'invalid' == c['level']))
        else:
            overview.set_note('handling' if kind != 'longitudinal'
                              else 'braking', "")

    def on_overview_update(self, mass):
        overview = self.analysis.overview
        out = self._busy(overview.group_notes['ride'],
                         "computing the corner KPIs...",
                         lambda: self.session.corner_kpis(mass))
        if out['completed']:
            overview.clear(_overview.CORNER_GROUPS)
            overview.set_values(out['values'])
            name = os.path.basename(self.session.model_path)
            overview.set_note('ride', f"{name} at {mass:.0f} kg per corner")
        else:
            overview.clear(_overview.CORNER_GROUPS,
                           f"failed: {out['message']}")
        self.analysis.warnings.show_entries(self.session.warnings)

    def on_overview_targets(self):
        path, _ = QFileDialog.getOpenFileName(
            None, "Load KPI targets", "", "JSON (*.json)")
        if path:
            self.load_targets(path)

    def load_targets(self, path):
        try:
            targets = _glossary.load_targets(path)
        except (OSError, ValueError) as exc:
            self.session.warnings.append(self.session._entry(
                'error', 'overview', None, f"targets not loaded: {exc}"))
        else:
            self.analysis.overview.set_custom_targets(targets)
        self.analysis.warnings.show_entries(self.session.warnings)

    # Background runs of the vehicle analyses.

    def _vehicle_run(self, text, calls, done):
        """Run session ``calls`` (``[(method, args, kwargs)]``) and give
        their results to ``done``: in a child process when
        :attr:`background`, else in place.  A new run replaces a running
        one."""
        tab = self.analysis.vehicle
        self.cancel_vehicle_job()
        if not self.background:
            sess = self.session
            outs = self._busy(tab.status, text, lambda: [
                getattr(sess, name)(*args, **kwargs)
                for name, args, kwargs in calls])
            done(outs)
            self.analysis.warnings.show_entries(sess.warnings)
            return
        self._job = _jobs.Job(_jobs.session_state(self.session), calls)
        self._job_done = done
        self._job_text = text
        self._job_started = time.monotonic()
        tab.set_running(True)
        tab.status.setText(f"{text} (running in the background; Cancel "
                           "stops it)")
        self.job_timer.start(100)

    def vehicle_job_running(self):
        return self._job is not None

    def _job_tick(self):
        job = self._job
        if job is None:
            self.job_timer.stop()
            return
        tab = self.analysis.vehicle
        result = job.poll()
        if result is None:
            tab.status.setText(
                f"{self._job_text} {time.monotonic() - self._job_started:.0f}"
                " s (running in the background; Cancel stops it)")
            return
        self.job_timer.stop()
        self._job = None
        done, self._job_done = self._job_done, None
        tab.set_running(False)
        status, outs, warnings = result
        if 'error' == status:
            last = outs.strip().splitlines()[-1] if outs.strip() else outs
            tab.status.setText(f"failed: {last}")
            self.session.warnings = [self.session._entry(
                'error', 'vehicle', None, f"the analysis failed: {last}")]
        else:
            self.session.warnings = list(warnings)
            done(outs)
        self.analysis.warnings.show_entries(self.session.warnings)

    def cancel_vehicle_job(self, note=""):
        """Stop a background vehicle run; nothing of it is shown."""
        if self._job is None:
            return
        self._job.cancel()
        self._job = None
        self._job_done = None
        self.job_timer.stop()
        if self.analysis is not None:
            tab = self.analysis.vehicle
            tab.set_running(False)
            tab.status.setText(note or "cancelled: no result shown")

    def wait_vehicle_job(self, timeout=600.0):
        """Process events until the background run ends (for scripts and
        tests)."""
        start = time.monotonic()
        while self._job is not None:
            if time.monotonic() - start > timeout:
                raise TimeoutError("the vehicle analysis did not finish")
            QApplication.processEvents()
            time.sleep(0.02)
            self._job_tick()

    # The 3D frames of the analyses (P22).

    #: The period of a played frame, in ms.
    ANIM_MS = 60

    def _bar(self, key):
        tabs = {'vehicle': self.analysis.vehicle, 'ride': self.analysis.ride,
                'modal': self.analysis.modal,
                'statics': self.analysis.statics}
        return tabs[key].replay

    def set_animation(self, key, frames, index=None):
        """Keep the 3D frames of a tab and show one (the last by
        default)."""
        self.anim_timer.stop()
        self._animating = None
        self.animations[key] = frames
        bar = self._bar(key)
        bar.play.blockSignals(True)
        bar.play.setChecked(False)
        bar.play.blockSignals(False)
        index = len(frames) - 1 if index is None else index
        bar.set_count(len(frames), max(index, 0))
        if frames:
            self.on_frame(key, max(index, 0))

    def on_frame(self, key, index):
        frames = self.animations.get(key) or []
        if not frames:
            return
        index = max(0, min(int(index), len(frames) - 1))
        label, geometries, pose, overlays = frames[index][:4]
        forces = frames[index][4] if len(frames[index]) > 4 else None
        self._show_other(geometries, pose, overlays, forces)
        self._bar(key).label.setText(f"{label} ({index + 1} of "
                                     f"{len(frames)})")

    def on_animate(self, key, on):
        if on and self.animations.get(key):
            for other in self.animations:
                if other != key:
                    bar = self._bar(other)
                    bar.play.blockSignals(True)
                    bar.play.setChecked(False)
                    bar.play.blockSignals(False)
            self._animating = key
            self.anim_timer.start(self.ANIM_MS)
        elif key == self._animating:
            self.anim_timer.stop()
            self._animating = None

    def _anim_tick(self):
        key = self._animating
        if key is None:
            self.anim_timer.stop()
            return
        bar = self._bar(key)
        count = len(self.animations.get(key) or [])
        if count < 2:
            self.anim_timer.stop()
            return
        # Loop: the slider drives the frame.
        bar.slider.setValue((bar.slider.value() + 1) % count)

    def _vehicle_frames(self, items, models):
        """Frames of vehicle poses: ``items`` of (label, vehicle pose)."""
        geometries = _scene.vehicle_geometries(models)
        frames = []
        for label, vpose in items:
            pose = _scene.vehicle_view_pose(vpose)
            body = _scene.vehicle_body_segments(geometries, pose)
            frames.append((label, geometries, pose,
                           {'vehicle_body': body}))
        return frames

    #: The displacement scale of the compliance frames in the 3D view.
    COMPLIANCE_SCALE = 20.0

    def _compliance_frames(self, out, unit):
        """The frames of a compliance sweep: the bushing deflection of each
        load scaled by :attr:`COMPLIANCE_SCALE` about the unloaded pose,
        with the load as an arrow."""
        geometry = _scene.CornerGeometry(self.session.model())
        loads = list(out['load'])
        base = out['poses'][loads.index(0.0)] if 0.0 in loads \
            else out['poses'][0]
        scale = self.COMPLIANCE_SCALE
        point = None if out['load_point'] is None \
            else np.asarray(out['load_point']) * 1e3
        frames = []
        for load, pose in zip(loads, out['poses']):
            shown = {}
            for body, (p, q) in pose.items():
                p0, q0 = base[body]
                delta = _scene.rotation(q) @ _scene.rotation(q0).T
                vec = _rotation_vector(delta) * scale
                shown[body] = (p0 + scale * (np.asarray(p) - p0),
                               _quat_mul(_axis_quat(vec), q0))
            forces = None
            if point is not None and out['case'] in ('lateral', 'braking',
                                                     'traction'):
                axis = 1 if 'lateral' == out['case'] else 0
                vector = np.zeros(3)
                vector[axis] = load
                if abs(load) > 0.0:
                    forces = [(out['case'], point, vector)]
            frames.append((f"{out['case']} {load:+.0f} {unit} (deflection "
                           f"x{scale:g})", [geometry], shown, None, forces))
        return frames

    def _load_frames(self, out):
        """The unloaded and the loaded compliant rig of one load case, the
        deflection scaled, with the load as an arrow."""
        ramp = {'load': [0.0, 1.0], 'poses': out['poses'],
                'load_point': out['load_point'], 'case': 'load'}
        frames = self._compliance_frames(ramp, "")
        point = np.asarray(out['load_point']) * 1e3
        force = np.asarray(out['force'], dtype='float64')
        label = f"load {list(out['force'])} N (deflection " \
                f"x{self.COMPLIANCE_SCALE:g})"
        return [("no load",) + frames[0][1:4] + (None,),
                (label,) + frames[1][1:4]
                + ([('load', point, force)],)]

    def _quarter_frames(self, model, poses, labels):
        geometry = _scene.CornerGeometry(model)
        return [(label, [geometry], pose, None)
                for label, pose in zip(labels, poses)]

    def on_mode_shape(self, mass, index):
        tab = self.analysis.modal
        try:
            out = self._busy(tab.status, "posing the mode shape...",
                             lambda: self.session.mode_frames(mass, index))
        except Exception as exc:
            tab.status.setText(f"no mode shape: {exc}")
            return
        name = ("ride", "wheel hop")[index] if index < 2 else \
            f"mode {index + 1}"
        tab.status.setText(
            f"{name} mode {out['frequency']:.2f} Hz, damping ratio "
            f"{out['damping']:.3f}: chassis / travel amplitude "
            f"{abs(out['shape'][0]):.2f} / {abs(out['shape'][1]):.2f} "
            f"(25 mm at the largest)")
        self.set_animation('modal', self._quarter_frames(
            out['model'], out['poses'], out['labels']), index=0)
        tab.replay.play.setChecked(True)

    def on_vehicle_reference(self):
        tab = self.analysis.vehicle
        if tab.last_handling is not None:
            tab.reference_handling = tab.last_handling
            tab.show_handling(tab.last_handling)

    def on_vehicle(self, stem, setups=None):
        def done(outs):
            out = outs[0]
            tab = self.analysis.vehicle
            if out['completed']:
                key = (stem, repr(sorted((setups or {}).items())))
                self._understeer[key] = out['understeer']
                tab.show_steady(out)
                self._vehicle_kpis('steady', out)
                st = out['steady']
                self.set_animation('vehicle', self._vehicle_frames(
                    [(f"steady turn {a:.2f} g", p)
                     for a, p in zip(st['a_y'], st['poses'])],
                    st['models']))
            else:
                tab.status.setText(f"failed: {out['message']}")

        self._vehicle_run("solving the steady turns...",
                          [('run_vehicle', (stem,), {'setups': setups})],
                          done)

    def on_step_steer(self, stem, speed_kmh, rack_mm, setups=None):
        key = (stem, repr(sorted((setups or {}).items())))

        def done(outs):
            out = outs[0]
            tab = self.analysis.vehicle
            if out['completed']:
                self._understeer[key] = out['understeer']
                tab.show_step(out)
                self._vehicle_kpis('step', out)
                self.set_animation('vehicle', self._vehicle_frames(
                    [(f"step steer t = {t:.2f} s", p)
                     for t, p in zip(out['pose_times'], out['poses'])],
                    out['models']), index=0)
            else:
                tab.status.setText(f"failed: {out['message']}")

        self._vehicle_run("marching the step steer (4 s of driving)...",
                          [('run_step_steer',
                            (stem, speed_kmh, rack_mm,
                             self._understeer.get(key)),
                            {'setups': setups})], done)

    def on_dynamics(self):
        tab = self.analysis.dynamics
        self.on_cancel()
        self.run = self.session.quarter_car(tab.mass.value(),
                                            road=tab.road())
        # The run keeps the bodies of the model it was built from, also
        # when an adjuster changes the sweep geometry later.
        self._run_geometry = _scene.CornerGeometry(self.run.model)
        started = self._busy(tab.status, "building the road input and the "
                                         "static equilibrium...",
                             self.run.start)
        if not started:
            tab.status.setText(f"failed: {self.run.message}")
            return
        tab.run.setEnabled(False)
        tab.cancel.setEnabled(True)
        self._draw_run()
        self.timer.start(0)

    def on_cancel(self):
        if self.run is not None and not self.run.finished:
            self.run.cancel()
        self.timer.stop()
        if self.analysis is not None:
            tab = self.analysis.dynamics
            tab.run.setEnabled(True)
            tab.cancel.setEnabled(False)
            if self.run is not None:
                self._draw_run()
                count = len(self.run.poses)
                tab.replay.blockSignals(True)
                tab.replay.setRange(0, max(0, count - 1))
                tab.replay.setValue(max(0, count - 1))
                tab.replay.setEnabled(count > 1)
                tab.replay.blockSignals(False)

    def _tick(self):
        if self.run is None:
            self.timer.stop()
            return
        going = self.run.advance(self.STEPS_PER_TICK)
        self._draw_run()
        if not going:
            self.on_cancel()

    def _draw_run(self):
        run = self.run
        tab = self.analysis.dynamics
        hist = run.history
        road_label = ("effective road [mm]" if run.road is not None
                      and run.road.kind == 'road' else "rig plate [mm]")
        tab.plot.set_curves([
            (road_label, hist['time'], hist['road'], (120, 120, 120), True),
            ("wheel center [mm]", hist['time'], hist['wheel'],
             _plots.PREVIOUS_COLOR, False),
            ("body [mm]", hist['time'], hist['body'],
             _plots.CURRENT_COLOR, False),
            ("suspension travel [mm] (+ bump)", hist['time'],
             hist['travel'], (40, 150, 70), False)])
        tab.force_plot.set_curves([
            ("tire normal force [N]", hist['time'], hist['force'],
             (200, 60, 40), False),
            ("travel stop force [N]", hist['time'], hist['stop'],
             (150, 60, 170), False)])
        self._pose_run()
        now = hist['time'][-1] if hist['time'] else 0.0
        state = run.message if run.finished else "running"
        text = f"t = {now:.3f} / {run.duration:.3f} s, {state}"
        if run.road is not None:
            text = f"{run.road.name}: " + text
        airborne = run.airborne_time
        if airborne > 0.0:
            text += (f"; the tire left the road for {airborne * 1e3:.1f} "
                     f"ms (force 0)")
        for state, name in ((-1, "bump stop"), (1, "droop stop")):
            on_stop = run.stop_time(state)
            if on_stop > 0.0:
                text += f"; on the {name} for {on_stop * 1e3:.1f} ms"
        need = run.pass_time
        if need is not None and run.duration < need:
            text += (f"; the duration ends before the wheel passes the "
                     f"road (needs {need:.2f} s)")
        tab.status.setText(text)

    def _pose_run(self, pose=None, time=None):
        """Move the 3D view to the quarter car now (or to a recorded
        ``pose`` at ``time``): one corner on the sprung mass, above the
        effective road plane, with the road profile around the wheel."""
        pose = self.run.pose() if pose is None else pose
        if pose is None or self._run_geometry is None:
            return
        self._show_other([self._run_geometry], pose,
                         {'road_profile': self.run.road_profile(pose, time)})

    def _show_other(self, geometries, pose, overlays=None, forces=None):
        """Pose the view with something else than the sweep (the quarter
        car, the profile); the next sweep cursor move rebuilds the sweep
        corners."""
        # The geometry objects, not their names: two models share names.
        shown = tuple(geometries)
        if not (self._showing_run
                and len(self._showing_run) == len(shown)
                and all(a is b for a, b in zip(self._showing_run, shown))):
            # Frame the camera when the number of corners changes (a whole
            # vehicle after a corner).
            count = len(self._showing_run) if self._showing_run else \
                len(self._geometries)
            self.view.build(geometries, aim=count != len(shown))
            self._showing_run = shown
        self.view.show_pose(geometries, pose, selected=self.selected,
                            overlays=overlays, forces=forces)

    def on_road_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            None, "Open a road profile", "", "CSV (*.csv)")
        if path:
            tab = self.analysis.dynamics
            tab.csv_path = path
            tab.status.setText(f"road profile: {path}")

    # The playback of the shared cursor (P11.2).

    def on_play(self, on):
        sweep = self.analysis.sweep
        if on:
            speed = sweep.speed.currentData() or 1.0
            self.play_timer.start(int(self.PLAY_MS / speed))
            sweep.play.setText("Pause")
        else:
            self.play_timer.stop()
            sweep.play.setText("Play")

    def _play_tick(self):
        """Advance the cursor one point; wrap with Loop on, else stop."""
        sess = self.session
        sweep = self.analysis.sweep if self.analysis else None
        if sess is None or sess.current is None or sweep is None:
            self.play_timer.stop()
            return
        speed = sweep.speed.currentData() or 1.0
        self.play_timer.setInterval(int(self.PLAY_MS / speed))
        last = len(sess.current) - 1
        index = sess.cursor + 1
        if index > last:
            if not sweep.loop.isChecked():
                sweep.play.setChecked(False)
                return
            index = 0
        self.update_cursor(index)

    def on_step(self, delta):
        if self.session is not None and self.session.current is not None:
            self.update_cursor(self.session.cursor + int(delta))

    def on_replay(self, index):
        if self.run is None or not self.run.poses:
            return
        index = max(0, min(int(index), len(self.run.poses) - 1))
        time = self.run.history['time'][index]
        self._pose_run(self.run.poses[index], time)
        self.analysis.dynamics.replay_label.setText(
            f"t = {time:.3f} s (step {index} of {len(self.run.poses) - 1})")
        self.analysis.dynamics.plot.set_cursor(time)

    # The run list, measured curves, layouts, overlays, report.

    def on_run_list(self, action, index, value):
        sess = self.session
        try:
            if 'visible' == action:
                sess.set_run_visible(index, value)
            elif 'rename' == action:
                sess.rename_run(index, value)
            elif 'delete' == action:
                sess.delete_run(index)
            elif 'baseline' == action:
                same = sess.baseline is sess.runs[index]
                sess.set_baseline(None if same else index)
        except (IndexError, ValueError) as exc:
            sess.warnings.append(sess._entry('error', 'runs', None,
                                             str(exc)))
        self.refresh()

    def on_measured(self):
        path, _ = QFileDialog.getOpenFileName(
            None, "Import measured curves", "", "CSV (*.csv)")
        if path:
            self.import_measured(path)

    def import_measured(self, path):
        measured = self.session.import_measured(path)
        self.refresh()
        return measured

    def on_report(self):
        path, _ = QFileDialog.getSaveFileName(
            None, "Export the report", "suspension_report.pdf",
            "PDF (*.pdf)")
        if path:
            self.export_report(path)

    def export_report(self, path):
        sess = self.session
        try:
            pages = sess.export_report(path)
        except (OSError, ValueError, RuntimeError) as exc:
            sess.warnings.append(sess._entry('error', 'report', None,
                                             f"report failed: {exc}"))
        else:
            sess.warnings.append(sess._entry(
                'info', 'report', None, f"report written to {path}, "
                                        f"{pages} pages"))
        self.analysis.warnings.show_entries(sess.warnings)

    def on_layout(self, action, key, on):
        sess = self.session
        kind = sess.current.kind if sess.current else sess.kind
        if 'toggle' == action:
            keys = set(sess.layout[kind.name])
            if on:
                keys.add(key)
            elif len(keys) > 1:
                keys.discard(key)
            sess.set_layout(kind.name, keys)
        elif 'save' == action:
            path, _ = QFileDialog.getSaveFileName(
                None, "Save the plot layout", "suspension_layout.json",
                "Layout (*.json)")
            if path:
                sess.save_layout(path)
        elif 'load' == action:
            path, _ = QFileDialog.getOpenFileName(
                None, "Load a plot layout", "", "Layout (*.json)")
            if path:
                try:
                    sess.load_layout(path)
                except (OSError, ValueError) as exc:
                    sess.warnings.append(sess._entry(
                        'error', 'layout', None,
                        f"cannot load {path}: {exc}"))
        self.refresh()

    def on_overlay(self, key, on):
        if on:
            self.overlays.add(key)
        else:
            self.overlays.discard(key)
        self.update_cursor()

    # Target alignment, compliance sweep (P11.6, P11.7).

    def on_solve_alignment(self, targets):
        panel = self.adjusters
        if not targets:
            panel.target_status.setText("check at least one target")
            return
        out = self._busy(panel.target_status, "solving the adjusters...",
                         lambda: self.session.solve_alignment(targets))
        if out['converged']:
            panel.target_status.setText(
                "solved: " + ", ".join(f"{k} {v:+.3f} mm" for k, v in
                                       out['adjusters'].items()))
        else:
            panel.target_status.setText(f"not solved: {out['message']}")
        if out['converged']:
            self._drop_results(self.MODEL_CHANGED)
        self.refresh()

    def on_compliance_sweep(self, case, fmax, count):
        tab = self.analysis.statics
        if 0 == int(count) % 2:
            tab.ramp_status.setText("the point count must be odd, so that "
                                    "zero load is a point")
            return
        out = self._busy(tab.ramp_status, "ramping the load...",
                         lambda: self.session.run_compliance_sweep(
                             case, fmax, count))
        if 'load' in out and len(out['load']):
            unit = self.session.COMPLIANCE_CASES[case][1]
            tab.show_ramp(out, unit)
            for plot in tab.ramp_plots.values():
                plot.set_xlabel(f"Load [{unit}]")
            if out.get('poses'):
                self.set_animation('statics', self._compliance_frames(
                    out, unit), index=len(out['poses']) - 1)
        else:
            tab.ramp_status.setText(f"failed: {out['message']}")
        self.analysis.warnings.show_entries(self.session.warnings)

    # Hardpoints, batch, profile (P11.8, P11.9, P11.11).

    def on_hardpoint(self, action, name, value):
        sess = self.session
        if 'linked' == action:
            sess.link_coincident = bool(value)
            return
        if 'select' == action:
            self.view.select_point(name)
            return
        if 'invalid' == action:
            sess.warnings = [sess._entry('error', 'hardpoint', name,
                                         f"{value!r} is not a number")]
            self.refresh()
            return
        if 'limit' == action:
            if name is None:
                return
            try:
                if value is None:
                    sess.set_limit(name, None, None)
                else:
                    here = dict((r[0], r[1]) for r in
                                sess.hardpoint_rows())[name]
                    sess.set_limit(name, here - value, here + value)
            except ValueError as exc:
                sess.warnings = [sess._entry('error', 'hardpoint', name,
                                             str(exc))]
            self.refresh()
            return
        if 'set' == action:
            self._busy(self.analysis.sweep.cursor_label,
                       f"moving {name}...",
                       lambda: sess.set_hardpoint(name, value))
            self._geometry = None
            self._drop_results(self.MODEL_CHANGED)
            self.refresh()
            self.view.select_point(name)

    def on_point_picked(self, name):
        """A hardpoint clicked in 3D: select its table row and show its
        coordinates in the properties."""
        if self.analysis is None:
            return
        self.analysis.hardpoints.select(name)
        if name is not None and name in self.view.points:
            design, now = self.view.points[name]
            self.adjusters.show_properties(f"hardpoint: {name}", [
                ("design", _view._fmt(design) + " mm"),
                ("now", _view._fmt(now) + " mm"),
                ("delta", _view._fmt(now - design, sign=True) + " mm")])

    def on_batch(self, point, axis, lo, hi, count):
        tab = self.analysis.batch
        import numpy as np
        offsets = list(np.linspace(lo, hi, int(count)))
        out = self._busy(tab.status, f"sweeping {point}...",
                         lambda: self.session.run_batch(point, axis, offsets))
        if 'offset' in out and len(out['offset']):
            tab.show_result(out, f"{point} {'xyz'[int(axis)]}")
            tab.status.setText(
                f"{point} {'xyz'[int(axis)]}: {len(out['offset'])} offsets"
                + ("" if out['completed'] else f"; stopped: "
                                               f"{out['message']}"))
        else:
            tab.status.setText(f"failed: {out['message']}")
        self.analysis.warnings.show_entries(self.session.warnings)

    def on_profile_import(self):
        path, _ = QFileDialog.getOpenFileName(
            None, "Import a motion profile", "", "CSV (*.csv)")
        if path:
            self.import_profile(path)

    def import_profile(self, path):
        tab = self.analysis.profile
        try:
            rows = self.session.read_profile(path)
        except (OSError, ValueError) as exc:
            tab.status.setText(f"cannot read {path}: {exc}")
            return False
        tab.set_rows(rows)
        tab.status.setText(f"{len(rows)} rows from {path}")
        return True

    def on_profile(self, rows=None):
        tab = self.analysis.profile
        try:
            if rows is None:
                rows = tab.rows()
            else:
                tab.set_rows(rows)  # the table shows what was run
        except ValueError as exc:
            tab.status.setText(str(exc))
            return
        out = self._busy(tab.status, "driving the axle...",
                         lambda: self.session.run_profile(rows))
        self.profile = out
        if out['poses']:
            tab.show_result(out)
            tab.status.setText(
                f"{len(out['poses'])} of {len(rows)} rows solved"
                + ("" if out['completed'] else f"; stopped: "
                                               f"{out['message']}"))
            self.on_profile_cursor(0)
        else:
            tab.status.setText(f"failed: {out['message']}")
        self.analysis.warnings.show_entries(self.session.warnings)

    def on_profile_cursor(self, index):
        out = self.profile
        if not out or not out['poses']:
            return
        index = max(0, min(int(index), len(out['poses']) - 1))
        left = self.session.geometry()
        right = _scene.CornerGeometry(left.model.mirrored(),
                                      _session.RIGHT)
        self._show_other([left, right], out['poses'][index])
        heave, roll, rack = out['rows'][index]
        tab = self.analysis.profile
        tab.cursor_label.setText(
            f"row {index + 1}: heave {heave:+.1f} mm, roll {roll:+.2f} deg, "
            f"rack {rack:+.1f} mm")
        for plot in tab.plots.values():
            plot.set_cursor(index + 1)


def build_app(controller):
    """Build the app on the pilot."""
    return SuspensionApp(mgr=controller.mgr)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
