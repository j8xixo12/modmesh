# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The engineering workflow of the suspension session (roadmap P11), free
of Qt: the run list, measured curves, summaries, geometry overlays, target
alignment, compliance sweeps, hardpoint edits and limit boxes, the batch
sweep, the motion profile, the plot layout, the PDF report, and the
dynamics replay.  Every number is checked against the suspension package
or its independent reference solver.

The engine lives outside solvcon, so these tests skip where it is not
installed.
"""

import copy
import os
import tempfile
import unittest

import numpy as np

from solvcon.apps.suspension import _analysis, _engine, _session

try:
    ENGINE = _engine.load()
except _engine.EngineUnavailable as exc:
    ENGINE = None
    WHY = str(exc)
else:
    WHY = ""

requires_engine = unittest.skipUnless(ENGINE, f"no engine: {WHY}")
DEG = 180.0 / np.pi


def _session_with_sweep():
    sess = _session.SuspensionSession(ENGINE)
    assert sess.run_sweep()
    return sess


def _tmp(test, suffix):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    test.addCleanup(os.remove, path)
    return path


@requires_engine
class RunListTC(unittest.TestCase):

    def test_every_valid_sweep_is_a_named_run(self):
        sess = _session_with_sweep()
        sess.set_adjuster('camber_shim', 2.0)
        self.assertEqual(2, len(sess.runs))
        first, second = sess.runs
        self.assertIs(second.result, sess.current)
        self.assertIs(second, sess.current_run())
        self.assertNotEqual(first.color, second.color)
        self.assertIn("camber shim +2.00 mm", second.name)
        self.assertEqual([first, second], sess.visible_runs())
        sess.set_run_visible(0, False)
        self.assertEqual([second], sess.visible_runs())
        sess.rename_run(1, "shim 2")
        self.assertEqual("shim 2", sess.runs[1].name)
        with self.assertRaisesRegex(ValueError, "needs a name"):
            sess.rename_run(1, "  ")
        sess.set_baseline(0)
        self.assertIs(first, sess.baseline)
        sess.delete_run(0)
        self.assertIsNone(sess.baseline)
        self.assertIs(second.result, sess.current)

    def test_a_failed_sweep_adds_no_run(self):
        sess = _session_with_sweep()
        self.assertFalse(sess.set_adjuster('toe_link', 500.0))
        self.assertEqual(1, len(sess.runs))

    def test_the_list_keeps_the_baseline_when_full(self):
        sess = _session_with_sweep()
        sess.set_range(-10.0, 10.0, 3)
        sess.set_baseline(0)
        baseline = sess.baseline
        for it in range(sess.MAX_RUNS + 2):
            sess.set_adjuster('toe_link', 0.01 * it)
        self.assertEqual(sess.MAX_RUNS, len(sess.runs))
        self.assertIn(baseline, sess.runs)


@requires_engine
class MeasuredTC(unittest.TestCase):

    def test_exported_sweep_imports_as_the_same_curves(self):
        sess = _session_with_sweep()
        path = _tmp(self, '.csv')
        sess.export_csv(path)
        measured = sess.import_measured(path)
        self.assertIsNotNone(measured)
        self.assertEqual('bump', measured.kind)
        np.testing.assert_allclose(sess.current.x, measured.x, rtol=1.e-9)
        for curve in sess.current.kind.curves:
            # The export writes 10 significant digits.
            np.testing.assert_allclose(sess.current.curve(curve.key)[1],
                                       measured.curves[curve.key],
                                       rtol=1.e-9, atol=1.e-12,
                                       err_msg=curve.key)
        self.assertEqual([measured], sess.measured_for())

    def test_short_column_names(self):
        sess = _session_with_sweep()
        path = _tmp(self, '.csv')
        with open(path, 'w') as fobj:
            fobj.write("# rig data\ntravel_mm,camber_deg,toe_deg,extra\n"
                       "-10,0.2,0.01,7\n0,0,0,7\n10,-0.2,-0.01,7\n")
        measured = sess.import_measured(path)
        self.assertEqual({'camber', 'toe'}, set(measured.curves))
        np.testing.assert_allclose([0.2, 0.0, -0.2],
                                   measured.curves['camber'])
        self.assertIn("columns not read: extra",
                      sess.warnings[-1]['message'])

    def test_a_file_without_a_sweep_column_is_rejected(self):
        sess = _session_with_sweep()
        path = _tmp(self, '.csv')
        with open(path, 'w') as fobj:
            fobj.write("time_s,camber_deg\n0,1\n")
        self.assertIsNone(sess.import_measured(path))
        self.assertEqual([], sess.measured)
        self.assertEqual('error', sess.warnings[-1]['level'])
        self.assertIn("no sweep column", sess.warnings[-1]['message'])


@requires_engine
class SummaryOverlayTC(unittest.TestCase):

    def test_summary_is_the_reference_solver_gradient(self):
        sess = _session_with_sweep()
        run = sess.current_run()
        stats = sess.summary(run, 'camber')
        xs, ys = sess.current.curve('camber')
        self.assertEqual(0.0, xs[16])
        self.assertAlmostEqual(ys[16], stats['value'], places=12)
        self.assertEqual(ys.min(), stats['min'])
        self.assertEqual(ys.max(), stats['max'])
        # The gradient is the central difference over the neighbors at
        # +/- 2.5 mm; the reference solver gives the same difference.
        # The independent solver of the package.
        ref = ENGINE.reference.ReferenceCorner(sess.model())
        z0 = sess.model().hp['wheel_center'][2]
        lo = ref.metrics(z0 - 2.5e-3)['camber'] * DEG
        hi = ref.metrics(z0 + 2.5e-3)['camber'] * DEG
        # Both solves converge to 1e-10 rad; over 5 mm that is 1e-8 deg/mm.
        self.assertAlmostEqual((hi - lo) / 5.0, stats['gradient'], delta=1e-8)

    def test_summary_without_zero_in_the_range(self):
        out = _analysis.summary([1.0, 2.0], [3.0, 4.0])
        self.assertIsNone(out['value'])
        self.assertIsNone(out['gradient'])
        self.assertEqual(3.0, out['min'])

    def test_summary_of_an_undefined_quantity(self):
        # A rear corner without a rack has no steering axis (NaN).
        out = _analysis.summary([-1.0, 0.0, 1.0], [np.nan] * 3)
        self.assertEqual({'value': None, 'gradient': None, 'min': None,
                          'max': None}, out)

    def test_overlays_are_the_sweep_geometry(self):
        sess = _session_with_sweep()
        out = sess.overlays(16)
        data = sess.current.data
        self.assertEqual({'kingpin', 'instant_centers', 'roll_center',
                          'patch_path'}, set(out))
        start, end = out['kingpin']
        axis = data['kingpin_dir'][16]
        point = data['kingpin_point'][16] * 1.e3
        # Both ends lie on the kingpin axis through the kingpin point.
        for end_point in (start, end):
            offset = end_point - point
            np.testing.assert_allclose(np.cross(offset, axis), 0.0,
                                       atol=1.e-9)
        cross = np.array(out['roll_center']).reshape(-1, 2, 3)
        np.testing.assert_allclose(cross.mean(axis=(0, 1)),
                                   data['roll_center_point'][16] * 1.e3,
                                   atol=1.e-9)
        self.assertEqual(2 * (len(sess.current) - 1), len(out['patch_path']))
        sess.set_kind('steer')
        self.assertEqual({}, sess.overlays())


@requires_engine
class AlignmentTC(unittest.TestCase):

    def test_targets_are_reached_and_swept(self):
        sess = _session_with_sweep()
        out = sess.solve_alignment({'camber': -1.5, 'toe': 0.1})
        self.assertTrue(out['converged'], out['message'])
        self.assertEqual(2, len(sess.runs))
        xs, camber = sess.current.curve('camber')
        _, toe = sess.current.curve('toe')
        middle = int(np.flatnonzero(xs == 0.0)[0])
        # The engine solves to 1e-10 rad (6e-9 deg).
        self.assertAlmostEqual(-1.5, camber[middle], delta=1.e-6)
        self.assertAlmostEqual(0.1, toe[middle], delta=1.e-6)
        for name in ('camber_shim', 'toe_link'):
            self.assertAlmostEqual(out['adjusters'][name],
                                   sess.adjusters[name], places=9)

    def test_an_unreachable_target_changes_nothing(self):
        sess = _session_with_sweep()
        before = dict(sess.adjusters)
        out = sess.solve_alignment({'camber': -30.0})
        self.assertFalse(out['converged'])
        self.assertEqual(before, sess.adjusters)
        self.assertEqual(1, len(sess.runs))
        self.assertIn("unchanged", sess.warnings[-1]['message'])


@requires_engine
class ComplianceSweepTC(unittest.TestCase):

    def test_lateral_ramp(self):
        sess = _session_with_sweep()
        out = sess.run_compliance_sweep('lateral', 1000.0, 5)
        self.assertTrue(out['completed'], out['message'])
        np.testing.assert_allclose([-1000, -500, 0, 500, 1000], out['load'])
        self.assertEqual(0.0, out['toe'][2])
        # The gradient per kN is the central difference at zero load.
        want = (out['toe'][3] - out['toe'][1]) / 1.0
        self.assertAlmostEqual(want, out['gradient']['toe'], places=12)
        engine = ENGINE.compliance.compliance_sweep(
            ENGINE.mbs, sess.model(), 'lateral', 1000.0, 5)
        np.testing.assert_allclose(np.asarray(engine['toe']) * DEG,
                                   out['toe'], rtol=1.e-12, atol=1.e-15)

    def test_a_bad_case_is_reported(self):
        sess = _session_with_sweep()
        out = sess.run_compliance_sweep('vertical', 1000.0, 5)
        self.assertFalse(out['completed'])
        self.assertEqual('error', sess.warnings[-1]['level'])


@requires_engine
class HardpointTC(unittest.TestCase):

    def test_an_edit_equals_the_edited_model(self):
        sess = _session_with_sweep()
        name = 'tie_rod_inner'
        xyz = np.asarray(sess.base_model.hp[name]) * 1.e3 + [0.0, 0.0, 1.0]
        self.assertTrue(sess.set_hardpoint(name, xyz))
        model = ENGINE.model.CornerModel.load(sess.model_path)
        model.set_hardpoint(name, xyz / 1.e3)
        ref = ENGINE.kc.bump_sweep(ENGINE.mbs, model, sess.travels())
        np.testing.assert_allclose(np.asarray(ref['toe']) * DEG,
                                   sess.current.curve('toe')[1],
                                   rtol=0.0, atol=1.e-12)
        rows = {row[0]: row[1] for row in sess.hardpoint_rows()}
        np.testing.assert_allclose(xyz, rows[name])

    def test_coincident_outer_points_move_together(self):
        sess = _session_with_sweep()
        name = 'lower_front_outer'
        partners = sess.coincident(name)
        self.assertGreater(len(partners), 1)
        xyz = np.asarray(sess.base_model.hp[name]) * 1.e3 + [2.0, 0.0, 0.0]
        self.assertTrue(sess.set_hardpoint(name, xyz))
        for partner in partners:
            np.testing.assert_allclose(
                xyz, np.asarray(sess.base_model.hp[partner]) * 1.e3)

    def test_limit_box_rejects_and_keeps_the_model(self):
        sess = _session_with_sweep()
        name = 'tie_rod_inner'
        here = np.asarray(sess.base_model.hp[name]) * 1.e3
        sess.set_limit(name, here - 5.0, here + 5.0)
        before = copy.deepcopy(sess.base_model.hp)
        current = sess.current
        self.assertFalse(sess.set_hardpoint(name, here + [0.0, 0.0, 9.0]))
        for key, value in before.items():
            np.testing.assert_array_equal(value, sess.base_model.hp[key])
        self.assertIs(current, sess.current)
        self.assertIn("edit rejected", sess.warnings[-1]['message'])
        rows = {row[0]: row[2] for row in sess.hardpoint_rows()}
        np.testing.assert_allclose(here - 5.0, rows[name][0])
        sess.set_limit(name, None, None)
        self.assertTrue(sess.set_hardpoint(name, here + [0.0, 0.0, 9.0]))

    def test_an_unassemblable_edit_is_rejected(self):
        # A 30 mm lower front link cannot follow 40 mm of wheel travel: the
        # sweep stops, and the edit is rejected.
        sess = _session_with_sweep()
        name = 'lower_front_inner'
        before = np.array(sess.base_model.hp[name])
        outer = np.asarray(sess.base_model.hp['lower_front_outer']) * 1.e3
        current = sess.current
        self.assertFalse(sess.set_hardpoint(name, outer + [0.0, -30.0, 0.0]))
        np.testing.assert_array_equal(before, sess.base_model.hp[name])
        self.assertIn("does not assemble", " ".join(
            w['message'] for w in sess.warnings))
        np.testing.assert_allclose(current.curve('camber')[1],
                                   sess.current.curve('camber')[1])


@requires_engine
class BatchProfileTC(unittest.TestCase):

    def test_batch_is_the_engine_sensitivity(self):
        sess = _session_with_sweep()
        out = sess.run_batch('tie_rod_inner', 2, [-2.0, 0.0, 2.0])
        self.assertTrue(out['completed'], out['message'])
        engine = ENGINE.sensitivity.hardpoint_sweep(
            ENGINE.mbs, sess.model(), 'tie_rod_inner', 2,
            [-2.e-3, 0.0, 2.e-3])
        np.testing.assert_allclose(np.asarray(engine['bump_steer'])
                                   * DEG / 1.e3, out['bump_steer'],
                                   rtol=1.e-12)
        # The shipped tie rod height minimizes bump steer: it changes sign
        # across the offsets.
        self.assertLess(out['bump_steer'][0] * out['bump_steer'][2], 0.0)

    def test_heave_profile_is_the_bump_sweep(self):
        sess = _session_with_sweep()
        rows = [(-20.0, 0.0, 0.0), (0.0, 0.0, 0.0), (20.0, 0.0, 0.0)]
        out = sess.run_profile(rows)
        self.assertTrue(out['completed'], out['message'])
        self.assertEqual(3, len(out['poses']))
        self.assertIn('upright_right', out['poses'][0])
        ref = ENGINE.kc.bump_sweep(ENGINE.mbs, sess.model(),
                                   [-0.02, 0.0, 0.02])
        # Both solve the same corner to 1e-10 rad.
        np.testing.assert_allclose(np.asarray(ref['camber']) * DEG,
                                   out['camber_left'], atol=1.e-7)
        np.testing.assert_allclose(out['camber_left'], out['camber_right'],
                                   atol=1.e-7)

    def test_profile_csv(self):
        sess = _session_with_sweep()
        path = _tmp(self, '.csv')
        with open(path, 'w') as fobj:
            fobj.write("heave_mm,roll_deg,rack_mm\n10,0.5,2\n")
        np.testing.assert_allclose([(10.0, 0.5, 2.0)],
                                   sess.read_profile(path))


@requires_engine
class LayoutReportTC(unittest.TestCase):

    def test_layout_round_trip(self):
        sess = _session_with_sweep()
        sess.set_layout('bump', ['toe', 'camber'])
        self.assertEqual(['camber', 'toe'], sess.layout['bump'])
        path = _tmp(self, '.json')
        sess.save_layout(path)
        other = _session.SuspensionSession(ENGINE)
        other.load_layout(path)
        self.assertEqual(sess.layout, other.layout)
        with self.assertRaisesRegex(ValueError, "no curve spin"):
            sess.set_layout('bump', ['spin'])

    def test_report_holds_the_session_values(self):
        import pypdf
        sess = _session_with_sweep()
        sess.set_adjuster('camber_shim', 1.0)
        path = _tmp(self, '.pdf')
        pages = sess.export_report(path)
        reader = pypdf.PdfReader(path)
        self.assertEqual(pages, len(reader.pages))
        self.assertGreaterEqual(pages, 3)
        text = "".join(page.extract_text() for page in reader.pages)
        self.assertIn("Setup sheet", text)
        self.assertIn(sess.base_model.name, text)
        self.assertIn("camber shim", text)
        self.assertIn("+1.00 mm", text)
        camber = _analysis.summary(*sess.current.curve('camber'))['value']
        self.assertIn(f"{camber:+.4f} deg", text)
        self.assertIn("Summary at design", text)


@requires_engine
class ReplayTC(unittest.TestCase):

    def test_every_step_pose_is_kept(self):
        sess = _session.SuspensionSession(ENGINE)
        run = sess.quarter_car(200.0, 5.0, 3.0, 0.02, dt=2.e-3)
        self.assertTrue(run.start())
        while run.advance(5):
            pass
        self.assertEqual(len(run.history['time']), len(run.poses))
        last = run.pose()
        for body, (pos, quat) in last.items():
            np.testing.assert_array_equal(pos, run.poses[-1][body][0])
            np.testing.assert_array_equal(quat, run.poses[-1][body][1])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
