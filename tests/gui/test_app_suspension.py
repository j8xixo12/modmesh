# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The suspension app in a live pilot window: the plots draw the K&C
analysis, an adjuster change leaves the old curve dashed, a failed sweep
keeps the last valid pose, and the slider moves the pose and the plot
cursors together.

The engine lives outside solvcon, so these tests skip where it is not
installed.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from solvcon import testing

try:
    from solvcon import pilot
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication
    if pilot.enable:
        from solvcon.apps.suspension import _app, _engine, _window
except ImportError:
    pilot = None

ENGINE = None
if pilot is not None and pilot.enable:
    try:
        ENGINE = _engine.load()
    except _engine.EngineUnavailable:
        ENGINE = None


class _Fixture(object):

    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.area = self.mgr.mdiArea

    def _open(self, engine=ENGINE):
        feature = _app.SuspensionApp(mgr=self.mgr, engine=engine)
        # The vehicle runs in place: a click gives its result at once.
        feature.background = False
        self.addCleanup(self._dispose, feature)
        feature.open()
        QApplication.processEvents()
        return feature

    def _march(self, feature):
        for _ in range(10000):
            if not feature.timer.isActive():
                return
            QApplication.processEvents()
        self.fail("the quarter car did not finish")

    def _dispose(self, feature):
        feature.close()
        if feature.error_window is not None:
            feature.error_window.close()
        QApplication.processEvents()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@testing.requires_live_window
@unittest.skipUnless(ENGINE, "the multibody engine is not installed")
class SuspensionAppTC(_Fixture, unittest.TestCase):

    def test_static_quality_returns_after_pose_updates_stop(self):
        from PySide6.QtCore import QEventLoop, QTimer
        feature = self._open()
        feature.update_cursor()
        self.assertTrue(feature.view._motion_quality)
        loop = QEventLoop()
        QTimer.singleShot(600, loop.quit)
        loop.exec()
        self.assertFalse(feature.view._motion_quality)
        feature.update_cursor()
        self.assertTrue(feature.view._motion_quality)

    def test_component_search_filters_and_restores_external_selection(self):
        feature = self._open()
        tree = feature.tree
        names = tree.names()
        self.assertGreater(len(names), 1)
        tree.search.setText('no such component 987654321')
        self.assertFalse(tree.search_empty.isHidden())
        tree.search.setText(names[0])
        self.assertTrue(tree.search_empty.isHidden())
        self.assertFalse(tree._items[names[0]].isHidden())
        hidden = next(name for name in names
                      if tree._items[name].isHidden()
                      or tree._items[name].parent().isHidden())
        self.assertTrue(tree.select(hidden))
        self.assertEqual('', tree.search.text())
        self.assertEqual(hidden, tree.current_name())

    def test_tree_expansion_survives_refresh_and_search(self):
        feature = self._open()
        tree = feature.tree
        group = tree.tree.topLevelItem(0)
        name = group.text(0)
        group.setExpanded(False)
        feature.refresh()
        group = tree.tree.topLevelItem(0)
        self.assertEqual(name, group.text(0))
        self.assertFalse(group.isExpanded())
        tree.search.setText(name)
        self.assertTrue(group.isExpanded())
        tree.search.clear()
        self.assertFalse(group.isExpanded())

    def test_setup_fields_scroll_instead_of_collapsing_after_refresh(self):
        from PySide6.QtCore import QPoint
        feature = self._open()
        self.mgr.mainWindow.show()
        self.mgr.mainWindow.resize(1100, 650)
        feature.analysis.select(feature.analysis.setup)
        feature.refresh()
        QApplication.processEvents()
        for spin in feature.analysis.setup.spins.values():
            self.assertGreaterEqual(spin.height(),
                                    spin.minimumSizeHint().height())
            page = feature.analysis.setup
            bottom = spin.mapTo(page, QPoint()).y() + spin.height()
            self.assertLessEqual(bottom,
                                 page.mass.mapTo(page, QPoint()).y())

    def test_switching_analysis_restores_its_pose_and_stops_playback(self):
        feature = self._open()
        panel = feature.analysis
        panel.select(panel.vehicle)
        vehicle = panel.vehicle
        feature.on_handling(vehicle.vehicle.currentData(), vehicle.setups())
        self.assertEqual(4, len(feature.view._geometries))
        vehicle.replay.play.setChecked(True)
        self.assertTrue(feature.anim_timer.isActive())

        panel.select(panel.sweep)
        self.assertFalse(feature.anim_timer.isActive())
        self.assertFalse(vehicle.replay.play.isChecked())
        self.assertFalse(feature._showing_run)
        self.assertEqual(1, len(feature.view._geometries))
        panel.sweep.play.setChecked(True)
        feature.background = True
        feature.set_animation('vehicle', feature.animations['vehicle'])
        self.assertEqual(1, len(feature.view._geometries))
        self.assertTrue(feature.play_timer.isActive())
        panel.select(panel.vehicle)
        self.assertFalse(feature.play_timer.isActive())
        self.assertEqual(4, len(feature.view._geometries))

    def test_expanding_a_chart_keeps_its_data_and_restores_the_grid(self):
        feature = self._open()
        sweep = feature.analysis.sweep
        feature.analysis.select(sweep)
        before = sweep.plots['camber'].data(_app.CURRENT)
        sweep.cells['camber'].focus.click()
        self.assertTrue(all(cell.isHidden() for key, cell in
                            sweep.cells.items() if key != 'camber'))
        self.assertEqual(before, sweep.plots['camber'].data(_app.CURRENT))
        sweep.cells['camber'].focus.click()
        self.assertTrue(all(not cell.isHidden()
                            for cell in sweep.cells.values()))

    def test_messages_do_not_take_space_until_a_problem_occurs(self):
        feature = self._open()
        panel = feature.analysis
        self.assertFalse(panel.messages.isChecked())
        panel.warnings.show_entries([{
            'level': 'error', 'message': 'The model could not be solved.'}])
        self.assertTrue(panel.messages.isChecked())
        self.assertIn('1 need attention', panel.messages.text())

    def test_current_run_uses_the_curve_color_in_the_run_list(self):
        from PySide6.QtCore import Qt
        from solvcon.apps.suspension import _plots
        feature = self._open()
        feature.adjusters.spins['camber_shim'].setValue(0.1)
        tree = feature.analysis.sweep.runs.tree
        current = tree.topLevelItem(tree.topLevelItemCount() - 1)
        self.assertTrue(current.text(0).startswith('[current]'))
        self.assertEqual(_plots.CURRENT_COLOR,
                         tuple(current.data(0, Qt.UserRole + 1)))

    def test_changing_theme_does_not_edit_run_visibility(self):
        from PySide6.QtGui import QColor, QPalette
        feature = self._open()
        runs = feature.analysis.sweep.runs
        runs.changed = mock.Mock()
        palette = runs.palette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        runs.setPalette(palette)
        QApplication.processEvents()
        runs.changed.assert_not_called()

    def test_short_adjuster_panel_keeps_inputs_readable(self):
        feature = self._open()
        self.mgr.mainWindow.show()
        self.mgr.mainWindow.resize(1100, 650)
        QApplication.processEvents()
        panel = feature.adjusters
        for spin in (*panel.spins.values(), *panel.target_value.values()):
            self.assertGreaterEqual(spin.height(),
                                    spin.minimumSizeHint().height())
        scroll = panel.tabs.widget(0)
        self.assertGreater(scroll.verticalScrollBar().maximum(), 0)

    def test_overview_shortcuts_open_the_named_vehicle_task(self):
        feature = self._open()
        panel = feature.analysis
        for (name, task), button in panel.overview.shortcuts.items():
            panel.select(panel.overview)
            button.click()
            self.assertIs(panel.tabs.currentWidget().widget(),
                          getattr(panel, name))
            if name == 'vehicle':
                self.assertEqual(task, panel.vehicle.tasks.currentIndex())

    def test_smooth_wheel_surfaces_have_no_normal_or_uv_seam(self):
        from solvcon.apps.suspension import _appearance
        from PySide6.QtGui import QImage
        feature = self._open()
        geometry = feature.session.geometry()
        points, normals, uv, indices = _appearance.tire_surface(geometry)
        self.assertTrue(np.isfinite(points).all())
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1,
                                   atol=1e-6)
        np.testing.assert_allclose(points[uv[:, 0] == 0],
                                   points[uv[:, 0] == 1], atol=1e-5)
        self.assertLess(indices.max(), len(points))
        image = QImage.fromData(_appearance.rubber_texture())
        self.assertEqual((2048, 1024), (image.width(), image.height()))
        points, normals, indices = _appearance.rim_surface(geometry)
        self.assertTrue(np.isfinite(normals).all())
        self.assertLess(indices.max(), len(points))

    def test_spring_surface_tracks_endpoints_and_skips_zero_length(self):
        from solvcon.apps.suspension import _appearance
        self.assertIsNone(_appearance.spring_surface([(0, 0, 0)] * 2))
        points, normals, indices = _appearance.spring_surface(
            [(0, 0, 0), (0, 0, 100)])
        self.assertTrue(np.isfinite(points).all())
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1,
                                   atol=1e-6)
        self.assertLess(indices.max(), len(points))
        self.assertLess(float(points[:, 2].min()), 0)
        self.assertGreater(float(points[:, 2].max()), 100)

    def test_link_surface_handles_short_and_degenerate_links(self):
        from solvcon.apps.suspension import _appearance
        self.assertIsNone(_appearance.link_surface([(0, 0, 0)] * 2))
        points, normals, indices = _appearance.link_surface(
            [(0, 0, 0), (0, 0, 1)])
        self.assertTrue(np.isfinite(points).all())
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1,
                                   atol=1e-6)
        self.assertLess(indices.max(), len(points))

    def _reference(self, feature, **adjusters):
        sess = feature.session
        model = ENGINE.model.CornerModel.load(sess.model_path)
        model.set_adjusters(**{k: v / 1.e3 for k, v in adjusters.items()})
        return ENGINE.kc.bump_sweep(ENGINE.mbs, model, sess.travels())

    def test_the_window_lays_out_the_corner(self):
        feature = self._open()
        self.assertIn(feature.view.subwin, self.area.subWindowList())
        self.assertEqual(
            ["Suspension model", "Suspension properties",
             "Suspension analysis"],
            [dock.windowTitle() for dock in feature.docks])
        names = feature.tree.names()
        for name in ("upright", "rocker", "tie_rod", "pushrod", "spring",
                     "anti-roll bar", "wheel travel driver", "camber shim"):
            self.assertIn(name, names)
        self.assertEqual(
            {"chassis", "grid", "link_rods", "rack", "rim", "rocker",
             "spring_coils", "spring_dampers", "upright", "wheel"},
            set(feature.view.widget.objectNames()))

    def test_the_plots_draw_the_bump_sweep(self):
        feature = self._open()
        ref = self._reference(feature)
        plots = feature.analysis.sweep.plots
        self.assertEqual("deg", plots["camber"]._ylabel)
        self.assertEqual("Wheel travel (+ bump) [mm]",
                         plots['camber']._xlabel)
        for curve in feature.session.current.kind.curves:
            with self.subTest(curve=curve.key):
                xs, ys = plots[curve.key].data(_app.CURRENT)
                np.testing.assert_array_equal(ref['travel'] * 1.e3, xs)
                np.testing.assert_array_equal(ref[curve.key] * curve.scale,
                                              ys)
                self.assertEqual([_app.CURRENT], plots[curve.key].labels())

    def test_an_adjuster_leaves_the_old_curve_dashed(self):
        feature = self._open()
        base = self._reference(feature)
        spin = feature.adjusters.spins['camber_shim']
        spin.setValue(2.0)
        QApplication.processEvents()
        plot = feature.analysis.sweep.plots['camber']
        # The replaced curve is the first run of the run list, dashed in its
        # own color (roadmap P11.5).
        self.assertIn("camber shim +0.00 mm", feature.session.runs[0].name)
        first = feature.session.runs[0].label
        self.assertEqual("#1", first)
        self.assertEqual([first, _app.CURRENT], plot.labels())
        self.assertTrue(plot.dashed(first))
        self.assertFalse(plot.dashed(_app.CURRENT))
        scale = 180.0 / np.pi
        np.testing.assert_array_equal(base['camber'] * scale,
                                      plot.data(first)[1])
        np.testing.assert_array_equal(
            self._reference(feature, camber_shim=2.0)['camber'] * scale,
            plot.data(_app.CURRENT)[1])
        self.assertIn("dashed: 1 other run(s)",
                      feature.analysis.sweep.legend.text())
        # Hiding the first run in the list removes it from every plot.
        feature.on_run_list('visible', 0, False)
        for key, other in feature.analysis.sweep.plots.items():
            self.assertEqual([_app.CURRENT], other.labels(), key)

    def test_a_failed_sweep_keeps_the_last_valid_pose(self):
        feature = self._open()
        sweep = feature.analysis.sweep
        sweep.slider.setValue(30)
        upright = feature.view.model_of('upright')
        curve = sweep.plots['camber'].data(_app.CURRENT)
        sweep.hi.setValue(400.0)
        sweep.count.setValue(12)
        sweep.run.click()
        QApplication.processEvents()
        texts = feature.analysis.warnings.texts()
        self.assertTrue(texts[0].startswith("ERROR: bump sweep failed"))
        self.assertIn("did not converge", texts[0])
        self.assertIn("showing the last valid result", texts[1])
        np.testing.assert_array_equal(upright,
                                      feature.view.model_of('upright'))
        self.assertEqual(curve, sweep.plots['camber'].data(_app.CURRENT))
        self.assertEqual(30, sweep.slider.value())

    def test_the_slider_moves_the_pose_and_the_cursor(self):
        feature = self._open()
        sweep = feature.analysis.sweep
        sess = feature.session
        sweep.slider.setValue(3)
        pos, quat = sess.current.poses[3]['upright']
        model = feature.view.model_of('upright')
        np.testing.assert_allclose(pos * 1.e3, model[:3, 3], rtol=1.e-6)
        # The rotation part is the Euler parameters of the pose.
        from solvcon.apps.suspension import _scene
        np.testing.assert_allclose(_scene.rotation(quat), model[:3, :3],
                                   atol=1.e-6)
        x3 = sess.current.x[3]
        for plot in sweep.plots.values():
            self.assertEqual(x3, plot.cursor)
        self.assertIn(f"{x3:+.2f} mm", sweep.cursor_label.text())

    def test_csv_export_and_force_arrows(self):
        feature = self._open()
        feature.analysis.sweep.forces.setChecked(True)
        self.assertIn("forces", feature.view.widget.objectNames())
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "sweep.csv")
            feature.export_csv(path)
            with open(path) as fobj:
                head = fobj.readline()
        self.assertEqual("# solvcon suspension K&C sweep\n", head)
        self.assertIn(f"sweep written to {path}",
                      feature.analysis.warnings.texts()[-1])

    def test_the_selected_link_is_highlighted(self):
        feature = self._open()
        self.assertTrue(feature.tree.select('tie_rod'))
        self.assertEqual('tie_rod', feature.selected)
        self.assertIn("selected", feature.view.widget.objectNames())
        rows = dict(feature.adjusters.property_rows())
        self.assertTrue(rows['length'].endswith(" mm"))


@testing.requires_live_window
@unittest.skipUnless(ENGINE, "the multibody engine is not installed")
class AnalysisTabsTC(_Fixture, unittest.TestCase):

    DEG = 180.0 / np.pi

    def test_statics_tab_shows_the_compliance_output(self):
        feature = self._open()
        tab = feature.analysis.statics
        for spin, value in zip(tab.force, (0.0, 1000.0, 0.0)):
            spin.setValue(value)
        tab.run.click()
        model = feature.session.model()
        sys_obj, _ = ENGINE.compliance.compliant_rig(ENGINE.mbs, model)
        base = ENGINE.compliance.apply_patch_load(ENGINE.mbs, sys_obj, model,
                                                  [0.0] * 3)
        loaded = ENGINE.compliance.apply_patch_load(
            ENGINE.mbs, sys_obj, model, [0.0, 1000.0, 0.0])
        toe = (loaded['toe'] - base['toe']) * self.DEG
        camber = (loaded['camber'] - base['camber']) * self.DEG
        row = tab.table.topLevelItem(0)
        self.assertEqual("(0, 1000, 0)", row.text(0))
        self.assertEqual(f"{toe:+.4f}", row.text(1))
        self.assertEqual(f"{camber:+.4f}", row.text(2))
        self.assertIn(f"toe change {toe:+.4f} deg", tab.status.text())

    def test_modal_tab_shows_the_engine_modes(self):
        feature = self._open()
        tab = feature.analysis.modal
        tab.mass.setValue(250.0)
        tab.run.click()
        sys_obj, _ = ENGINE.vehicle.quarter_car(
            ENGINE.mbs, feature.session.model(), 250.0)
        sys_obj.find_equilibrium(tol=1.e-9)
        modal = ENGINE.mbs.modal_analysis(sys_obj)
        pairs = sorted(zip(modal.frequencies, modal.damping_ratios))
        self.assertEqual(
            [[name, f"{f:.3f}", f"{z:.4f}"]
             for name, (f, z) in zip(("ride", "wheel hop"), pairs)],
            tab.rows())

    def test_dynamics_tab_marches_with_progress(self):
        feature = self._open()
        tab = feature.analysis.dynamics
        tab.set_road('rig', amplitude=5.0, frequency=3.0, duration=0.1)
        tab.run.click()
        self.assertFalse(tab.run.isEnabled())
        self.assertTrue(tab.cancel.isEnabled())
        self._march(feature)
        self.assertIn("t = 0.098 / 0.098 s, completed", tab.status.text())
        self.assertTrue(tab.run.isEnabled())
        times, body = tab.plot.data("body [mm]")
        self.assertEqual(feature.run.history['time'], times)
        self.assertEqual(feature.run.history['body'], body)
        self.assertEqual(50, len(times))
        self.assertTrue(tab.plot.dashed("rig plate [mm]"))
        _, force = tab.force_plot.data("tire normal force [N]")
        self.assertEqual(feature.run.history['force'], force)

    def test_dynamics_animates_the_3d_view(self):
        feature = self._open()
        tab = feature.analysis.dynamics
        tab.set_road('rig', amplitude=20.0, frequency=3.0, duration=0.4)
        tab.run.click()
        self._march(feature)
        pose = feature.run.pose()
        view = feature.view
        for body in ('upright', 'rocker', 'chassis'):
            np.testing.assert_allclose(
                pose[body][0] * 1.e3, view.model_of(body)[:3, 3],
                rtol=1.e-6, atol=1.e-6)
        self.assertGreater(abs(view.model_of('chassis')[2, 3]), 0.5)
        self.assertIn('road', view.widget.objectNames())
        # Back on the sweep, the slider poses the K&C rig again.
        feature.analysis.sweep.slider.setValue(0)
        feature.update_cursor(0)
        np.testing.assert_allclose([0.0, 0.0, 0.0],
                                   view.model_of('chassis')[:3, 3],
                                   atol=1.e-9)
        self.assertNotIn('road', view.widget.objectNames())

    def test_driving_over_a_bump_draws_the_road(self):
        feature = self._open()
        tab = feature.analysis.dynamics
        tab.set_road('bump', speed=30.0, height=30.0, length=150.0)
        tab.run.click()
        self._march(feature)
        self.assertIn("bump 30 mm x 150 mm at 30 km/h", tab.status.text())
        self.assertTrue(tab.plot.dashed("effective road [mm]"))
        names = feature.view.widget.objectNames()
        self.assertIn('road_profile', names)
        self.assertIn('road', names)

    def test_pothole_reports_the_lift_off(self):
        feature = self._open()
        tab = feature.analysis.dynamics
        tab.set_road('pothole', speed=40.0, depth=60.0, length=300.0)
        tab.run.click()
        self._march(feature)
        self.assertIn("the tire left the road for", tab.status.text())

    def test_dynamics_cancel_keeps_what_was_marched(self):
        feature = self._open()
        tab = feature.analysis.dynamics
        tab.set_road('rig', duration=5.0)
        tab.run.click()
        QApplication.processEvents()
        tab.cancel.click()
        self.assertFalse(feature.timer.isActive())
        self.assertIn("cancelled at t = ", tab.status.text())
        self.assertTrue(tab.run.isEnabled())
        self.assertLess(len(feature.run.history['time']), 2501)

    def test_roll_shows_both_corners(self):
        feature = self._open()
        sweep = feature.analysis.sweep
        sweep.forces.setChecked(True)
        sweep.kind.setCurrentIndex(sweep.kind.findData('roll'))
        names = set(feature.view.widget.objectNames())
        for name in ('upright_right', 'wheel_right', 'rocker_right',
                     'chassis_right', 'arb'):
            self.assertIn(name, names)
        # The middle of the sweep is the unrolled axle, which carries no
        # load; off it the links load up and the arrows appear.
        self.assertNotIn('forces', names)
        sweep.slider.setValue(2)
        self.assertIn('forces', feature.view.widget.objectNames())
        pos, _ = feature.session.current.poses[2]['upright_right']
        np.testing.assert_allclose(
            pos * 1.e3, feature.view.model_of('upright_right')[:3, 3],
            rtol=1.e-6)
        self.assertEqual(['roll', 'moment', 'roll_stiffness'],
                         list(sweep.plots))


@testing.requires_live_window
class MissingEngineTC(_Fixture, unittest.TestCase):

    def test_a_missing_engine_is_reported_not_raised(self):
        reason = "cannot import the multibody engine solvcon.multibody"
        with mock.patch.object(_engine, 'load',
                               side_effect=_engine.EngineUnavailable(reason)):
            feature = self._open(engine=None)
        self.assertIsNone(feature.session)
        self.assertEqual(reason, feature.error)
        self.assertIsNotNone(feature.error_window)
        self.assertIn(reason, feature.error_label.text())
        self.assertFalse(feature.view.is_open)


@testing.requires_live_window
@unittest.skipUnless(ENGINE, "the multibody engine is not installed")
class ViewNavigationTC(_Fixture, unittest.TestCase):
    """The 3D view navigation (roadmap P12): vehicle views, the
    orientation cube, auto orthographic, fit, history, pick, measure, and
    the four-view layout."""

    def _view(self):
        feature = self._open()
        view = feature.view
        view.transition_ms = 0
        return feature, view

    @staticmethod
    def _forward(state):
        d = state.target - state.position
        return d / np.linalg.norm(d)

    def test_vehicle_views_and_auto_orthographic(self):
        _, view = self._view()
        for name, forward in (('front', [-1, 0, 0]), ('left', [0, -1, 0]),
                              ('top', [0, 0, -1]), ('rear', [1, 0, 0])):
            view.actions[name].trigger()
            np.testing.assert_allclose(forward,
                                       self._forward(view.main_state()),
                                       atol=1.e-6, err_msg=name)
            self.assertEqual('parallel', view.widget.projection, name)
        view.widget.rotateCamera(60.0, 20.0)
        view._watch_camera()
        self.assertEqual('perspective', view.widget.projection)
        self.assertIn("free view, perspective", view.hud.text())
        view.set_projection_mode('parallel')
        self.assertEqual('parallel', view.widget.projection)

    def test_keys_are_bound(self):
        _, view = self._view()
        keys = {name: [k.toString() for k in view.actions[name].shortcuts()]
                for name in ('front', 'left', 'top', 'iso', 'fit', 'back')}
        self.assertEqual(['1'], keys['front'])
        self.assertEqual(['3'], keys['left'])
        self.assertEqual(['7'], keys['top'])
        self.assertEqual(['0'], keys['iso'])
        self.assertEqual(['F'], keys['fit'])
        self.assertEqual(['Alt+Left'], keys['back'])

    def test_cube_click_turns_the_view(self):
        _, view = self._view()
        view.set_view('front')
        cube = view.cube
        center = cube.center()
        top_edge = center + [0.0, -0.9 * cube.SIZE]
        direction = _navigation_hit(view, top_edge)
        np.testing.assert_allclose([1.0, 0.0, 1.0] / np.sqrt(2.0), direction,
                                   atol=1.e-9)
        view.view_from(direction)
        eye = -self._forward(view.main_state())
        np.testing.assert_allclose(direction, eye, atol=1.e-6)

    def test_fit_all_shows_every_pickup(self):
        feature, view = self._view()
        view.widget.zoomCamera(-8.0)  # far away first
        view.fit_all()
        pane = view.panes[0]
        width, height = pane.handle.width(), pane.handle.height()
        for name, (_, now) in view.points.items():
            px, py, _ = view.widget.projectPoint(*[float(v) for v in now])
            self.assertTrue(0 <= px <= width and 0 <= py <= height, name)

    def test_history_back_and_forward(self):
        _, view = self._view()
        view.set_view('front')
        view.set_view('top')
        view.back()
        np.testing.assert_allclose([-1, 0, 0],
                                   self._forward(view.main_state()),
                                   atol=1.e-6)
        view.forward()
        np.testing.assert_allclose([0, 0, -1],
                                   self._forward(view.main_state()),
                                   atol=1.e-6)

    def test_animated_transition_ends_at_the_view(self):
        _, view = self._view()
        view.transition_ms = 120
        view.set_view('left')
        import time
        deadline = time.monotonic() + 5.0
        while view._anim is not None and time.monotonic() < deadline:
            QApplication.processEvents()
        self.assertIsNone(view._anim)
        np.testing.assert_allclose([0, -1, 0],
                                   self._forward(view.main_state()),
                                   atol=1.e-6)

    def _pixel(self, view, name):
        now = view.points[name][1]
        return view.widget.projectPoint(*[float(v) for v in now])[:2]

    def test_pick_reads_the_hardpoint(self):
        feature, view = self._view()
        view.set_view('front')
        picked = []
        view.picked = picked.append
        name = 'upper_front_outer'
        view.click(view.panes[0], self._pixel(view, name))
        # The front view stacks the front and rear outer points; the pick is
        # one of the points under the cursor.
        self.assertEqual(1, len(picked))
        got = picked[0]
        np.testing.assert_allclose(self._pixel(view, got),
                                   self._pixel(view, name), atol=1.0)
        text = view.hud.text()
        self.assertIn(f"pick {got}: design", text)
        self.assertIn("design", text)
        self.assertIn('pick', view.widget.objectNames())
        view.clear_pick()
        self.assertIsNone(view.selected_point)

    def test_measure_distance_and_angle(self):
        feature, view = self._view()
        view.set_view('iso')
        view.set_measuring(True)
        names = ['upper_front_inner', 'upper_front_outer', 'lower_front_outer']
        for name in names[:2]:
            view.click(view.panes[0], self._pixel(view, name))
        self.assertEqual(names[:2], view.measure_points)
        a, b = (np.asarray(view.points[n][1]) for n in names[:2])
        self.assertAlmostEqual(np.linalg.norm(b - a), view.measure_value(),
                               places=6)
        self.assertIn("distance", view.hud.text())
        view.click(view.panes[0], self._pixel(view, names[2]))
        self.assertIn("angle at upper_front_outer", view.hud.text())
        # The upper front link length is the distance its ends keep.
        model = feature.session.current and feature.session.geometry().model
        self.assertAlmostEqual(model.link_length('upper_front') * 1.e3,
                               np.linalg.norm(b - a), places=4)

    def test_four_views_show_the_same_pose(self):
        feature, view = self._view()
        view.actions['quad'].trigger()
        self.assertEqual(4, len(view.panes))
        fixed = {pane.fixed for pane in view.panes[1:]}
        self.assertEqual({'front', 'left', 'top'}, fixed)
        feature.analysis.sweep.slider.setValue(3)
        main = view.model_of('upright')
        for pane in view.panes[1:]:
            self.assertEqual('parallel', pane.widget.projection)
            other = np.array(pane.widget.objectModel('upright')).reshape(4, 4)
            np.testing.assert_allclose(main, other, atol=1.e-6)
        view.actions['quad'].trigger()
        self.assertFalse(view.panes[1].handle.isVisible())

    def test_pose_update_keeps_the_user_camera(self):
        feature, view = self._view()
        view.set_view('top')
        state = view.main_state()
        feature.analysis.sweep.slider.setValue(0)
        feature.analysis.sweep.slider.setValue(32)
        self.assertTrue(_same(state, view.main_state()))

    def test_adjuster_rebuild_keeps_the_user_camera(self):
        # An adjuster change rebuilds the meshes, and adding a mesh
        # reframes the widget; the user's view must survive it.
        feature, view = self._view()
        view.set_view('front')
        view.widget.zoomCamera(2.0)
        state = view.main_state()
        feature.adjusters.spins['camber_shim'].setValue(1.5)
        QApplication.processEvents()
        self.assertTrue(_same(state, view.main_state()))

    def test_mouse_mapping_applies_to_every_pane(self):
        _, view = self._view()
        view.set_quad(True)
        view.set_mouse_mapping('cad')
        self.assertEqual(['cad'] * 4,
                         [p.widget.navigationMapping for p in view.panes])
        view.toggle_help()
        self.assertIn("Ctrl+middle: pan", view.help.text())


@testing.requires_live_window
@unittest.skipUnless(ENGINE, "the multibody engine is not installed")
class WorkflowUITC(_Fixture, unittest.TestCase):
    """The engineering workflow controls of the app (roadmap P11) show what
    the session computes."""

    def _tmp(self, suffix):
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        self.addCleanup(os.remove, path)
        return path

    def test_playback_moves_the_cursor_and_loops(self):
        feature = self._open()
        sess = feature.session
        sweep = feature.analysis.sweep
        feature.update_cursor(len(sess.current) - 2)
        feature._play_tick()
        self.assertEqual(len(sess.current) - 1, sess.cursor)
        feature._play_tick()
        self.assertEqual(0, sess.cursor)  # loop wraps
        sweep.loop.setChecked(False)
        feature.update_cursor(len(sess.current) - 1)
        sweep.play.setChecked(True)
        feature._play_tick()
        self.assertFalse(sweep.play.isChecked())
        self.assertFalse(feature.play_timer.isActive())
        sweep.step_back.click()
        self.assertEqual(len(sess.current) - 2, sess.cursor)
        sweep.step_forward.click()
        self.assertEqual(len(sess.current) - 1, sess.cursor)

    def test_measured_curves_are_drawn_as_dots(self):
        feature = self._open()
        path = self._tmp('.csv')
        feature.session.export_csv(path)
        measured = feature.import_measured(path)
        plot = feature.analysis.sweep.plots['camber']
        label = f"measured: {measured.name}"
        self.assertIn(label, plot.labels())
        self.assertTrue(plot.markers(label))
        np.testing.assert_allclose(plot.data(_app.CURRENT)[1],
                                   plot.data(label)[1], rtol=1.e-9,
                                   atol=1.e-12)

    def test_summary_lines_under_the_plots(self):
        feature = self._open()
        sess = feature.session
        stats = sess.summary(sess.current_run(), 'camber')
        text = feature.analysis.sweep.cells['camber'].summary.text()
        self.assertIn(f"at 0: {stats['value']:+.4g} deg", text)
        self.assertIn(f"gradient {stats['gradient']:+.4g} deg/mm", text)

    def test_overlay_toggle_draws_the_kingpin(self):
        feature = self._open()
        sweep = feature.analysis.sweep
        self.assertNotIn('kingpin', feature.view.widget.objectNames())
        sweep.overlay_actions['kingpin'].setChecked(True)
        self.assertIn('kingpin', feature.view.widget.objectNames())
        sweep.overlay_actions['kingpin'].setChecked(False)
        self.assertNotIn('kingpin', feature.view.widget.objectNames())

    def test_layout_hides_a_plot(self):
        feature = self._open()
        feature.on_layout('toggle', 'toe', False)
        self.assertNotIn('toe', feature.analysis.sweep.plots)
        feature.on_layout('toggle', 'toe', True)
        self.assertIn('toe', feature.analysis.sweep.plots)

    def test_target_alignment_sets_the_adjusters(self):
        feature = self._open()
        panel = feature.adjusters
        panel.target_on['caster'].setChecked(False)
        panel.target_value['camber'].setValue(-1.0)
        panel.target_value['toe'].setValue(0.05)
        panel.solve.click()
        sess = feature.session
        for name, spin in panel.spins.items():
            self.assertAlmostEqual(sess.adjusters[name], spin.value(),
                                   places=2)
        self.assertIn("solved", panel.target_status.text())
        xs, camber = sess.current.curve('camber')
        self.assertAlmostEqual(-1.0, camber[int(np.flatnonzero(xs == 0)[0])],
                               delta=1.e-6)

    def test_compliance_sweep_plots_the_session_values(self):
        feature = self._open()
        feature.on_compliance_sweep('lateral', 1000.0, 5)
        tab = feature.analysis.statics
        out = feature.session.run_compliance_sweep('lateral', 1000.0, 5)
        np.testing.assert_allclose(out['toe'],
                                   tab.ramp_plots['toe'].data('toe')[1],
                                   rtol=1.e-12, atol=1.e-15)
        self.assertIn("deg/kN", tab.ramp_status.text())
        self.assertIn("Parallel and opposed", tab.ramp_status.text())

    def test_hardpoint_table_edit_and_3d_pick(self):
        feature = self._open()
        tab = feature.analysis.hardpoints
        sess = feature.session
        names = tab.names()
        self.assertEqual(list(sess.base_model.hardpoint_names()), names)
        row = names.index('tie_rod_inner')
        z = float(tab.table.item(row, 3).text())
        tab.table.item(row, 3).setText(f"{z + 1.0:.2f}")
        QApplication.processEvents()
        self.assertAlmostEqual((z + 1.0) / 1.e3,
                               sess.base_model.hp['tie_rod_inner'][2])
        self.assertEqual(2, len(sess.runs))
        # A click on a hardpoint in 3D selects its row.
        feature.view.transition_ms = 0
        feature.view.set_view('iso')
        name = 'upper_rear_inner'
        now = feature.view.points[name][1]
        px, py, _ = feature.view.widget.projectPoint(*[float(v)
                                                       for v in now])
        feature.view.click(feature.view.panes[0], (px, py))
        picked = feature.view.selected_point
        self.assertEqual(picked, tab.current_name())
        self.assertEqual(f"hardpoint: {picked}", feature.adjusters.title
                         .text())
        # Jog moves the selected point by the step along the axis.
        tab.select('tie_rod_inner')
        tab.table.selectRow(row)
        tab.axis.setCurrentIndex(2)
        tab.step.setValue(0.5)
        tab._jog(1.0)
        self.assertAlmostEqual((z + 1.5) / 1.e3,
                               sess.base_model.hp['tie_rod_inner'][2])

    def test_batch_plots_the_sensitivity(self):
        feature = self._open()
        feature.on_batch('tie_rod_inner', 2, -2.0, 2.0, 3)
        tab = feature.analysis.batch
        out = feature.session.run_batch('tie_rod_inner', 2, [-2.0, 0.0, 2.0])
        label = tab.plots['bump_steer'].labels()[0]
        np.testing.assert_allclose(out['bump_steer'],
                                   tab.plots['bump_steer'].data(label)[1],
                                   rtol=1.e-12)

    def test_profile_poses_the_axle(self):
        feature = self._open()
        rows = [(0.0, 0.0, 0.0), (10.0, 1.0, 5.0), (-10.0, -1.0, -5.0)]
        feature.on_profile(rows)
        tab = feature.analysis.profile
        self.assertEqual(3, len(tab.plots['camber'].data('left')[0]))
        self.assertEqual(rows, tab.rows())
        self.assertIn('upright_right', feature.view.widget.objectNames())
        tab.slider.setValue(1)
        pos, _ = feature.profile['poses'][1]['upright_right']
        np.testing.assert_allclose(
            pos * 1.e3, feature.view.model_of('upright_right')[:3, 3],
            rtol=1.e-6)
        # Back on the sweep, the slider poses the K&C rig again.
        feature.analysis.sweep.slider.setValue(3)
        pos, _ = feature.session.current.poses[3]['upright']
        np.testing.assert_allclose(
            pos * 1.e3, feature.view.model_of('upright')[:3, 3], rtol=1.e-6)

    def test_dynamics_replay(self):
        feature = self._open()
        tab = feature.analysis.dynamics
        tab.set_road('rig', amplitude=20.0, frequency=3.0, duration=0.4)
        tab.run.click()
        for _ in range(10000):
            if not feature.timer.isActive():
                break
            QApplication.processEvents()
        self.assertTrue(tab.replay.isEnabled())
        tab.replay.setValue(10)
        pose = feature.run.poses[10]
        np.testing.assert_allclose(
            pose['chassis'][0] * 1.e3,
            feature.view.model_of('chassis')[:3, 3], atol=1.e-6)
        self.assertIn("step 10 of", tab.replay_label.text())

    def test_switching_the_model_drops_the_run(self):
        # A run of one model must not be drawn with the bodies of the next
        # one; the next run starts from the next model and its mass.
        feature = self._open()
        tab = feature.analysis.dynamics
        tab.set_road('rig', amplitude=5.0, frequency=3.0, duration=0.1)
        tab.run.click()
        self._march(feature)
        self.assertTrue(feature.run.completed)
        feature.on_library('five_link_rear')
        self.assertIsNone(feature.run)
        self.assertFalse(feature.timer.isActive())
        self.assertTrue(tab.run.isEnabled())
        self.assertFalse(tab.replay.isEnabled())
        self.assertEqual("", tab.status.text())
        self.assertEqual(350.0, tab.mass.value())
        tab.set_road('rig', amplitude=5.0, frequency=3.0, duration=0.1)
        tab.run.click()
        self._march(feature)
        self.assertTrue(feature.run.completed, feature.run.message)
        self.assertNotIn('rack', feature.run.numbers)
        # Back to a model without a corner mass: the default mass, not
        # the one of the last model.
        feature.on_library('race_front_corner')
        self.assertEqual(feature.DEFAULT_SPRUNG_MASS, tab.mass.value())
        tab.run.click()
        self._march(feature)
        self.assertTrue(feature.run.completed, feature.run.message)

    def test_ride_and_vehicle_tabs(self):
        feature = self._open()
        feature.on_library('macpherson_front')
        ride = feature.analysis.ride
        self.assertEqual(400.0, ride.mass.value())
        ride.run.click()
        self.assertIn("ISO 2631-1 Wk weighted", ride.status.text())
        self.assertIn("body acceleration", ride.plot.labels())
        vehicle = feature.analysis.vehicle
        self.assertGreater(vehicle.vehicle.count(), 0)
        vehicle.run.click()
        self.assertIn("LLTD", vehicle.status.text())
        self.assertIn("understeer", vehicle.status.text())
        self.assertIn("roll [deg]", vehicle.roll_plot.labels())
        # A model switch drops the ride result of the old model.
        feature.on_library('five_link_rear')
        self.assertEqual("", ride.status.text())

    def test_setup_sheet_and_vehicle_tuning(self):
        feature = self._open()
        feature.on_library('macpherson_coilover')
        tab = feature.analysis.setup
        self.assertIn('damper_rebound', tab.spins)
        self.assertEqual(400.0, tab.mass.value())
        tab.spins['ride_height'].setValue(-20.0)
        rows = {row[0]: row for row in tab.rows()}
        self.assertEqual("+20.0", rows["static travel from design (+ bump)"]
                         [3])
        self.assertEqual(-20.0, feature.session.setup['ride_height'])
        tab.reset_button.click()
        self.assertEqual({}, feature.session.setup)
        self.assertEqual(0.0, tab.spins['ride_height'].value())
        vehicle = feature.analysis.vehicle
        vehicle.vehicle.setCurrentIndex(vehicle.vehicle.findData(
            'coilover_car'))
        self.assertIn(('rear', 'arb_rate'), vehicle.setup_spins)
        vehicle.handling.click()
        self.assertIn("Limit lateral acceleration", vehicle.status.text())
        vehicle.reference.click()
        vehicle.setup_spins[('rear', 'arb_rate')].setValue(80.0)
        vehicle.setup_spins[('front', 'arb_rate')].setValue(5.0)
        vehicle.handling.click()
        self.assertIn("rear axle saturates first", vehicle.status.text())
        self.assertIn("reference", vehicle.handling_plot.labels())

    def test_vehicle_braking_and_traction(self):
        feature = self._open()
        vehicle = feature.analysis.vehicle
        vehicle.vehicle.setCurrentIndex(vehicle.vehicle.findData(
            'coilover_car'))
        vehicle.bias.setValue(80.0)
        vehicle.drive.setCurrentIndex(vehicle.drive.findData(1.0))
        vehicle.longitudinal.click()
        text = vehicle.status.text()
        self.assertIn("the front axle locks first", text)
        self.assertIn("the front wheels spin first", text)
        self.assertIn("anti-dive", text)
        self.assertIn("front", vehicle.brake_plot.labels())

    def test_edit_the_model_from_the_tree(self):
        import shutil
        feature = self._open()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'corner.json')
            shutil.copy(os.path.join(feature.session.engine.data_dir,
                                     'macpherson_coilover.json'), path)
            feature.load_model(path)
            feature.tree.select('spring')
            asked = []
            feature.ask_save = lambda p: asked.append(p) or 'temporary'
            feature.adjusters.edit_property('k', '90')
            self.assertEqual([feature.session.model_path], asked)
            self.assertEqual(90e3, feature.session.base_model.spring_k)
            self.assertIn("not in its file", feature.tree.dirty_label.text())
            feature.ask_save = lambda p: 'overwrite'
            feature.adjusters.edit_property('free_length_offset', '2')
            with open(path) as fobj:
                data = json.load(fobj)
            self.assertEqual(90, data['spring']['k'])
            self.assertEqual(2, data['spring']['free_length_offset'])
            feature.ask_save = lambda p: 'cancel'
            feature.adjusters.edit_property('k', '120')
            self.assertEqual(90e3, feature.session.base_model.spring_k)
            feature.adjusters.edit_property('k', '"x"')
            self.assertIn("not changed", feature.adjusters.edit_status.text())

    def test_vehicle_and_corner_animations(self):
        feature = self._open()
        vehicle = feature.analysis.vehicle
        vehicle.vehicle.setCurrentIndex(vehicle.vehicle.findData(
            'coilover_car'))
        vehicle.run.click()
        frames = feature.animations['vehicle']
        self.assertEqual(5, len(frames))
        names = feature.view.widget.objectNames()
        for suffix in ('_fl', '_fr', '_rl', '_rr'):
            self.assertIn('upright' + suffix, names)
        vehicle.replay.slider.setValue(0)
        self.assertIn("0.00 g", vehicle.replay.label.text())
        # The view follows the frame: the chassis rolls with a_y.
        vehicle.replay.slider.setValue(4)
        roll = feature.view.model_of('chassis_fl')
        self.assertGreater(abs(roll[2, 1]), 1e-3)
        vehicle.replay.play.setChecked(True)
        self.assertTrue(feature.anim_timer.isActive())
        vehicle.replay.play.setChecked(False)
        self.assertFalse(feature.anim_timer.isActive())
        feature.on_library('macpherson_coilover')
        modal = feature.analysis.modal
        modal.run.click()
        modal.animate.click()
        self.assertEqual(24, len(feature.animations['modal']))
        self.assertTrue(feature.anim_timer.isActive())
        modal.replay.play.setChecked(False)
        # Back on the sweep, the corner is shown again.
        feature.update_cursor(0)
        self.assertIn('upright', feature.view.widget.objectNames())

    def test_compliance_in_3d(self):
        feature = self._open()
        statics = feature.analysis.statics
        statics.run.click()
        frames = feature.animations['statics']
        self.assertEqual(2, len(frames))
        self.assertIn("deflection x20", frames[1][0])
        self.assertTrue(frames[1][4])
        statics.ramp_run.click()
        frames = feature.animations['statics']
        self.assertGreater(len(frames), 2)
        # The scaled deflection: the upright of the largest load moves 20
        # times its engine displacement from the unloaded pose.
        statics.replay.slider.setValue(0)
        self.assertIn("deflection", statics.replay.label.text())

    def _fill_corner_results(self, feature):
        """Run every corner analysis so each result widget holds data."""
        a = feature.analysis
        a.dynamics.set_road('rig', amplitude=5.0, frequency=3.0,
                            duration=0.1)
        a.dynamics.run.click()
        self._march(feature)
        a.modal.run.click()
        a.modal.animate.click()
        a.modal.replay.play.setChecked(False)
        a.statics.run.click()
        a.ride.run.click()
        a.ride.run_time.click()
        a.batch.run.click()
        a.profile.run.click()
        self.assertIsNotNone(feature.run)
        self.assertTrue(a.modal.rows())
        self.assertTrue(a.ride.plot.labels())
        for key in ('ride', 'modal', 'statics'):
            self.assertTrue(feature.animations.get(key), key)

    def _assert_corner_results_dropped(self, feature, note):
        a = feature.analysis
        self.assertIsNone(feature.run)
        self.assertFalse(feature.timer.isActive())
        self.assertFalse(feature.anim_timer.isActive())
        self.assertEqual([], a.modal.rows())
        self.assertEqual(0, a.statics.table.topLevelItemCount())
        for plot in (a.dynamics.plot, a.dynamics.force_plot, a.ride.plot,
                     a.ride.time_plot, *a.statics.ramp_plots.values(),
                     *a.batch.plots.values(), *a.profile.plots.values()):
            self.assertEqual([], plot.labels())
        for key in ('ride', 'modal', 'statics'):
            self.assertEqual([], feature.animations.get(key, []), key)
            self.assertFalse(feature._bar(key).slider.isEnabled(), key)
        for tab in (a.dynamics, a.modal, a.statics, a.ride):
            self.assertEqual(note, tab.status.text())

    def test_every_change_of_the_corner_drops_its_results(self):
        import shutil
        feature = self._open()
        note = feature.MODEL_CHANGED
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'corner.json')
            shutil.copy(os.path.join(feature.session.engine.data_dir,
                                     'macpherson_coilover.json'), path)
            feature.load_model(path)
            feature.ask_save = lambda p: 'temporary'
            changes = {
                'adjuster': lambda: feature.adjusters.spins[
                    'camber_shim'].setValue(1.0),
                'setup': lambda: feature.analysis.setup.spins[
                    'spring_rate'].setValue(70.0),
                'setup reset': lambda: feature.analysis.setup
                .reset_button.click(),
                'tree edit': lambda: (feature.tree.select('spring'),
                                      feature.adjusters.edit_property(
                                          'k', '65')),
                'hardpoint': lambda: feature.on_hardpoint(
                    'set', 'lower_front_inner', (-17.88, 375.0, 245.7529)),
                'model load': lambda: feature.on_library(
                    'macpherson_coilover'),
            }
            for name, change in changes.items():
                with self.subTest(change=name):
                    self._fill_corner_results(feature)
                    change()
                    self._assert_corner_results_dropped(
                        feature, "" if 'model load' == name else note)

    def test_every_change_of_the_vehicle_drops_its_results(self):
        feature = self._open()
        vehicle = feature.analysis.vehicle
        vehicle.vehicle.setCurrentIndex(vehicle.vehicle.findData(
            'coilover_car'))

        def fill():
            vehicle.run.click()
            vehicle.handling.click()
            vehicle.longitudinal.click()
            vehicle.step.click()
            for plot in (vehicle.roll_plot, vehicle.handling_plot,
                         vehicle.brake_plot, vehicle.yaw_plot):
                self.assertTrue(plot.labels())
            self.assertTrue(feature.animations['vehicle'])

        fill()
        vehicle.reference.click()
        # A setup change: the results go, the chosen reference stays.
        vehicle.setup_spins[('front', 'spring_rate')].setValue(70.0)
        self.assertEqual(['reference'], vehicle.handling_plot.labels())
        for plot in (vehicle.roll_plot, vehicle.brake_plot,
                     vehicle.yaw_plot):
            self.assertEqual([], plot.labels())
        self.assertEqual([], feature.animations['vehicle'])
        self.assertIn("run the analysis again", vehicle.status.text())
        # Another vehicle: nothing of the last one stays.
        fill()
        vehicle.vehicle.setCurrentIndex(vehicle.vehicle.findData(
            'road_car'))
        self.assertIsNone(vehicle.reference_handling)
        for plot in (vehicle.roll_plot, vehicle.handling_plot,
                     vehicle.brake_plot, vehicle.yaw_plot):
            self.assertEqual([], plot.labels())
        self.assertEqual([], feature.animations['vehicle'])
        self.assertEqual("", vehicle.status.text())
        # The same inputs give the same numbers run after run.
        vehicle.run.click()
        first = vehicle.status.text()
        vehicle.handling.click()
        vehicle.run.click()
        self.assertEqual(first, vehicle.status.text())

    def test_the_vehicle_uses_the_corner_of_the_tree(self):
        feature = self._open()
        feature.ask_save = lambda p: 'temporary'
        # The coilover car's front axle is macpherson_coilover.
        feature.on_library('macpherson_coilover')
        vehicle = feature.analysis.vehicle
        vehicle.vehicle.setCurrentIndex(vehicle.vehicle.findData(
            'coilover_car'))
        self.assertIn("front axle: the model tree", vehicle.sources.text())
        self.assertIn("rear axle: its library file", vehicle.sources.text())
        before = vehicle.setup_spins[('front', 'spring_rate')].value()
        vehicle.run.click()
        vehicle.step.click()
        self.assertTrue(vehicle.roll_plot.labels())
        self.assertTrue(feature.animations['vehicle'])
        # A tree edit of the spring: the vehicle results of the old corner
        # go, and the vehicle setup shows the edited spring.
        feature.tree.select('spring')
        feature.adjusters.edit_property('k', '75')
        for plot in (vehicle.roll_plot, vehicle.yaw_plot):
            self.assertEqual([], plot.labels())
        self.assertEqual([], feature.animations['vehicle'])
        self.assertIn(feature.MODEL_CHANGED, vehicle.status.text())
        self.assertIn("unsaved edits", vehicle.sources.text())
        after = vehicle.setup_spins[('front', 'spring_rate')].value()
        self.assertGreater(after, before)

    def test_overview_kpis_targets_and_their_reset(self):
        from solvcon.apps.suspension import _overview
        feature = self._open()
        feature.ask_save = lambda p: 'temporary'
        feature.on_library('macpherson_coilover')
        overview = feature.analysis.overview
        self.assertEqual(0, feature.analysis.tabs.currentIndex())
        self.assertEqual('missing', overview.state('ride_frequency'))
        overview.update_button.click()
        ride = overview.values['ride_frequency']
        self.assertGreater(ride, 0.5)
        self.assertIn("Hz", overview.cards['ride_frequency'].text())
        self.assertIn("OptimumG", overview.cards['ride_frequency']
                      .toolTip())
        # The profile moves the band, so the verdict.
        states = set()
        for k in range(overview.profile.count()):
            overview.profile.setCurrentIndex(k)
            states.add(overview.state('ride_frequency'))
        self.assertTrue({'in', 'out'} & states)
        self.assertEqual('none', overview.state('wheel_rate'))
        # The vehicle runs fill the vehicle cards.
        vehicle = feature.analysis.vehicle
        vehicle.vehicle.setCurrentIndex(vehicle.vehicle.findData(
            'coilover_car'))
        vehicle.run.click()
        vehicle.longitudinal.click()
        for key in ('roll_gradient', 'lltd', 'understeer_gradient',
                    'brake_limit', 'anti_dive'):
            self.assertIn(key, overview.values)
        # The baseline: the change shows.
        overview.set_baseline()
        overview.update_button.click()
        self.assertIn("change from baseline",
                      overview.cards['ride_frequency'].text())
        # A vehicle setup change drops the vehicle cards only.
        vehicle.setup_spins[('front', 'spring_rate')].setValue(70.0)
        self.assertIn('ride_frequency', overview.values)
        self.assertNotIn('roll_gradient', overview.values)
        # A tree edit drops every card (the vehicle uses the tree corner).
        vehicle.run.click()
        feature.tree.select('spring')
        feature.adjusters.edit_property('k', '75')
        for key, kpi in _overview._glossary.KPIS.items():
            self.assertEqual('missing', overview.state(key), key)
        self.assertIn(feature.MODEL_CHANGED,
                      overview.group_notes['ride'].text())
        # The updated corner: a stiffer spring, a higher ride frequency.
        overview.update_button.click()
        self.assertGreater(overview.values['ride_frequency'], ride)
        # User targets win over the published band.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'targets.json')
            with open(path, 'w') as fobj:
                fobj.write('{"wheel_rate": [10, 20]}')
            feature.load_targets(path)
        self.assertEqual('out', overview.state('wheel_rate'))
        self.assertIn("your targets file",
                      overview.cards['wheel_rate'].toolTip())

    def test_vehicle_tree_edits_and_opens_the_axles(self):
        feature = self._open()
        feature.ask_save = lambda p: 'temporary'
        vehicle = feature.analysis.vehicle
        vehicle.vehicle.setCurrentIndex(vehicle.vehicle.findData(
            'coilover_car'))
        tree = feature.tree
        for name in ('vehicle data', 'front axle', 'rear axle'):
            self.assertIn(name, tree.names())
        # Edit the vehicle data from the tree: the vehicle results go.
        vehicle.run.click()
        self.assertTrue(vehicle.roll_plot.labels())
        tree.select('vehicle data')
        feature.adjusters.edit_property('sprung_mass', '1600')
        sess = feature.session
        self.assertEqual(1600, sess.vehicle_data('coilover_car')
                         ['sprung_mass'])
        self.assertIn('coilover_car', sess.vehicle_dirty)
        self.assertEqual([], vehicle.roll_plot.labels())
        self.assertIn("vehicle changed", vehicle.status.text())
        self.assertEqual(1600.0, sess._vehicle('coilover_car',
                                               None).sprung_mass)
        # A bad value keeps the vehicle.
        feature.adjusters.edit_property('front_fraction', '1.5')
        self.assertEqual(1600, sess.vehicle_data('coilover_car')
                         ['sprung_mass'])
        self.assertNotEqual(1.5, sess.vehicle_data('coilover_car')
                            ['front_fraction'])
        # Double-click an axle: its corner file opens in the tree.
        tree.activate('rear axle')
        self.assertTrue(sess.model_path.endswith('five_link_coilover.json'))
        self.assertIn("rear axle: the model tree", vehicle.sources.text())
        # Unsaved corner edits: cancel keeps the tree model.
        tree.select('spring')
        feature.adjusters.edit_property('k', '70')
        feature.ask_switch = lambda p: 'cancel'
        tree.activate('front axle')
        self.assertTrue(sess.model_path.endswith('five_link_coilover.json'))
        feature.ask_switch = lambda p: 'discard'
        tree.activate('front axle')
        self.assertTrue(sess.model_path.endswith(
            'macpherson_coilover.json'))

    def test_vehicle_runs_in_the_background(self):
        feature = self._open()
        feature.background = True
        vehicle = feature.analysis.vehicle
        vehicle.vehicle.setCurrentIndex(vehicle.vehicle.findData(
            'road_car'))
        vehicle.run.click()
        # The window is live: the run button waits, Cancel shows.
        self.assertTrue(feature.vehicle_job_running())
        self.assertFalse(vehicle.run.isEnabled())
        self.assertTrue(vehicle.cancel.isVisibleTo(vehicle))
        feature.wait_vehicle_job()
        self.assertTrue(vehicle.run.isEnabled())
        self.assertTrue(vehicle.roll_plot.labels())
        self.assertTrue(feature.animations['vehicle'])
        background = vehicle.status.text()
        # The same numbers as a run in place.
        feature.background = False
        vehicle.run.click()
        self.assertEqual(background, vehicle.status.text())
        # Cancel: nothing of the run is shown.
        feature.background = True
        vehicle.clear_results()
        vehicle.step.click()
        vehicle.cancel.click()
        self.assertFalse(feature.vehicle_job_running())
        self.assertEqual([], vehicle.yaw_plot.labels())
        self.assertIn("cancelled", vehicle.status.text())
        # An input change while it runs drops the run.
        vehicle.step.click()
        feature.on_library('five_link_rear')
        self.assertFalse(feature.vehicle_job_running())
        self.assertEqual([], vehicle.yaw_plot.labels())

    def test_imported_ac_car_runs_every_analysis(self):
        # The synthetic Assetto Corsa car of the test data: converted to a
        # vehicle file with two corner files, then every vehicle analysis,
        # its corners in the tree, and the result reset on input changes.
        fixture = os.path.join(os.path.dirname(__file__), os.pardir, 'data',
                               'multibody', 'ac_synthetic')
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        target = os.path.join(tmp.name, 'synthetic.json')
        feature = self._open()
        feature.ask_save = lambda p: 'temporary'
        stem = feature.import_ac_vehicle(fixture, target)
        self.assertEqual(os.path.abspath(target), stem)
        for name in ('synthetic.json', 'synthetic_front.json',
                     'synthetic_rear.json'):
            self.assertTrue(os.path.exists(os.path.join(tmp.name, name)))
        vehicle = feature.analysis.vehicle
        self.assertEqual(stem, vehicle.vehicle.currentData())
        # The brake share and the drive split of the AC files, and the
        # assumptions of the conversion.
        self.assertEqual(70.0, vehicle.bias.value())
        self.assertEqual(0.6, vehicle.drive.currentData())
        self.assertIn("Data assumptions", vehicle.notes.text())
        self.assertIn("tyre", vehicle.notes.toolTip())
        # Every analysis runs on it.
        vehicle.run.click()
        self.assertTrue(vehicle.roll_plot.labels())
        vehicle.handling.click()
        self.assertTrue(vehicle.handling_plot.labels())
        vehicle.longitudinal.click()
        self.assertTrue(vehicle.brake_plot.labels())
        vehicle.step.click()
        self.assertTrue(vehicle.yaw_plot.labels())
        self.assertTrue(feature.animations['vehicle'])
        self.assertTrue(feature.analysis.overview.values)
        # A run input change drops the results of the old inputs.
        vehicle.speed.setValue(vehicle.speed.value() + 10.0)
        for plot in (vehicle.roll_plot, vehicle.handling_plot,
                     vehicle.brake_plot, vehicle.yaw_plot):
            self.assertEqual([], plot.labels())
        self.assertEqual([], feature.animations['vehicle'])
        self.assertIn("input changed", vehicle.status.text())
        for key in ('roll_gradient', 'yaw_gain', 'brake_limit'):
            self.assertNotIn(key, feature.analysis.overview.values)
        # The corner of an axle opens in the tree for the K&C analyses.
        feature.tree.activate('front axle')
        sess = feature.session
        self.assertTrue(sess.model_path.endswith('synthetic_front.json'))
        self.assertIn("front axle: the model tree", vehicle.sources.text())
        # Another window opens the converted file.
        other = self._open()
        self.assertEqual(stem, other.open_vehicle_file(target))
        self.assertEqual(stem, other.analysis.vehicle.vehicle.currentData())
        self.assertIsNone(other.open_vehicle_file(
            os.path.join(fixture, 'README.md')))
        self.assertIn("not opened", other.analysis.vehicle.status.text())

    def test_log_replay_and_its_reset(self):
        log = os.path.join(os.path.dirname(__file__), os.pardir, 'data',
                           'multibody', 'racechrono_synthetic',
                           'session_v3.csv')
        feature = self._open()
        vehicle = feature.analysis.vehicle
        vehicle.vehicle.setCurrentIndex(vehicle.vehicle.findData(
            'road_car'))
        self.assertFalse(vehicle.log_run.isEnabled())
        feature.set_log(log)
        self.assertTrue(vehicle.log_run.isEnabled())
        self.assertIn("session_v3.csv", vehicle.log_label.text())
        vehicle.log_length.setValue(3.0)
        vehicle.log_run.click()
        labels = vehicle.replay_rate_plot.labels()
        self.assertIn("roll rate, model", labels)
        self.assertIn("roll rate, log", labels)
        self.assertTrue(vehicle.replay_angle_plot.labels())
        self.assertIn("roll rate: correlation", vehicle.status.text())
        self.assertTrue(feature.animations['vehicle'])
        self.assertTrue(any("flipped" in w['message']
                            for w in feature.session.warnings))
        # Every replay input clears the replay: the window, the mount.
        for change in (lambda: vehicle.log_start.setValue(1.0),
                       lambda: vehicle.gyro.setCurrentIndex(1),
                       lambda: vehicle.invert_roll.setChecked(True),
                       lambda: feature.set_log(log)):
            vehicle.log_run.click()
            self.assertTrue(vehicle.replay_rate_plot.labels())
            change()
            self.assertEqual([], vehicle.replay_rate_plot.labels())
            self.assertEqual([], vehicle.replay_angle_plot.labels())
            self.assertEqual([], feature.animations['vehicle'])
        # Another vehicle clears it too.
        vehicle.log_run.click()
        vehicle.vehicle.setCurrentIndex(vehicle.vehicle.findData(
            'coilover_car'))
        self.assertEqual([], vehicle.replay_rate_plot.labels())

    def test_background_run_sees_the_vehicle_edits(self):
        feature = self._open()
        feature.ask_save = lambda p: 'temporary'
        vehicle = feature.analysis.vehicle
        vehicle.vehicle.setCurrentIndex(vehicle.vehicle.findData(
            'road_car'))
        feature.tree.select('vehicle data')
        feature.adjusters.edit_property('sprung_mass', '1700')
        vehicle.run.click()
        in_place = vehicle.status.text()
        feature.background = True
        vehicle.clear_results()
        vehicle.run.click()
        feature.wait_vehicle_job()
        self.assertEqual(in_place, vehicle.status.text())

    def test_switching_the_model_while_a_run_marches(self):
        feature = self._open()
        tab = feature.analysis.dynamics
        tab.set_road('rig', duration=5.0)
        tab.run.click()
        QApplication.processEvents()
        self.assertTrue(feature.timer.isActive())
        feature.on_library('fsae_front_pullrod')
        self.assertIsNone(feature.run)
        self.assertFalse(feature.timer.isActive())
        tab.set_road('rig', amplitude=5.0, frequency=3.0, duration=0.1)
        tab.run.click()
        self._march(feature)
        self.assertTrue(feature.run.completed, feature.run.message)
        self.assertEqual(feature.session.base_model.name,
                         feature.run.model.name)

    def test_short_docks_scroll_instead_of_clipping(self):
        # Every analysis tab and the properties dock sit in a scroll area;
        # shrinking the dock keeps each control reachable by scrolling.
        from PySide6.QtWidgets import QScrollArea
        feature = self._open()
        tabs = feature.analysis.tabs
        for it in range(tabs.count()):
            page = tabs.widget(it)
            self.assertIsInstance(page, QScrollArea, tabs.tabText(it))
            self.assertTrue(page.widgetResizable())
        dock = [d for d in feature.docks
                if d.windowTitle() == "Suspension properties"][0]
        self.assertIsInstance(dock.widget(), QScrollArea)
        # A short main window gives a short dock: every page keeps its
        # readable height and scrolls instead of squeezing the plots.
        self.mgr.mainWindow.resize(900, 520)
        for it in range(tabs.count()):
            tabs.setCurrentIndex(it)
            QApplication.processEvents()
            page = tabs.currentWidget()
            content = page.widget()
            self.assertGreaterEqual(content.height(),
                                    _window.PAGE_MIN_HEIGHT, tabs.tabText(it))
            if page.viewport().height() < _window.PAGE_MIN_HEIGHT:
                self.assertGreater(page.verticalScrollBar().maximum(), 0,
                                   tabs.tabText(it))

    def test_report_is_written(self):
        feature = self._open()
        path = self._tmp('.pdf')
        feature.export_report(path)
        self.assertGreater(os.path.getsize(path), 1000)
        self.assertIn("report written", feature.session.warnings[-1]
                      ['message'])


def _navigation_hit(view, pixel):
    from solvcon.apps.suspension import _navigation
    return _navigation.cube_hit(view.main_state(), view.cube.center(),
                                view.cube.SIZE, pixel)


def _same(a, b):
    from solvcon.apps.suspension import _navigation
    return _navigation.same_state(a, b, tol=1.e-5)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
