# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The suspension session without the GUI: the curves it shows are the ones
the suspension package computes, an adjuster change keeps the replaced
curve, and a failed sweep never replaces a valid result.

The engine lives outside solvcon, so these tests skip where it is not
installed.
"""

import os
import tempfile
import unittest

import numpy as np

from solvcon.apps.suspension import _engine, _scene, _session

try:
    ENGINE = _engine.load()
except _engine.EngineUnavailable as exc:
    ENGINE = None
    WHY = str(exc)
else:
    WHY = ""

requires_engine = unittest.skipUnless(ENGINE, f"no engine: {WHY}")


@requires_engine
class LibraryTC(unittest.TestCase):
    """Every model of the library loads, sweeps, draws, and runs the quarter
    car; the controls follow the parts the model has."""

    def test_library_lists_the_types(self):
        sess = _session.SuspensionSession(ENGINE)
        rows = sess.library()
        self.assertEqual('race_front_corner', rows[0][0])
        kinds = {kind for _, _, kind in rows}
        self.assertTrue({'macpherson', 'double_wishbone_pullrod',
                         'five_link', 'semi_trailing_arm',
                         'four_link_virtual_kingpin'} <= kinds)

    def test_every_model_sweeps_and_draws(self):
        sess = _session.SuspensionSession(ENGINE)
        for stem, _, _ in sess.library():
            sess.load_library(stem)
            model = sess.base_model
            self.assertEqual(model.has_rack, 'steer' in sess.kinds(), stem)
            self.assertTrue(sess.set_kind('bump'), stem)
            pose = sess.pose()
            geom = sess.geometry()
            meshes = geom.body_meshes()
            self.assertEqual(1.0, geom.outboard(), stem)
            for body in geom.bodies():
                self.assertIn(body, meshes, stem)
                self.assertIn(body, pose, stem)
            segments = geom.link_segments(pose)
            for name in model.links:
                self.assertIn(name, segments)
            self.assertIn('spring', segments)
            self.assertEqual(model.has_strut, 'strut' in segments)
            # The upright points ride on the upright at the design pose.
            points = geom.hardpoints(pose)
            for name in model.upright_point_names():
                design, now = points[name]
                cursor = sess.current.x[sess.cursor]
                if abs(cursor) < 1e-9:
                    np.testing.assert_allclose(now, design, atol=1e-6)
            groups = dict(sess.outline())
            names = [name for name, _ in groups['Bodies']]
            self.assertEqual(model.has_rocker, 'rocker' in names, stem)
            self.assertEqual(model.has_rack, 'rack' in names, stem)
            links = [name for name, _ in groups['Links']]
            self.assertEqual(model.has_strut, 'strut' in links, stem)
            self.assertEqual(model.has_arm, 'trailing arm' in links, stem)
            self.assertTrue(sess.forces(), stem)
            out = sess.run_compliance([0.0, 1000.0, 0.0])
            self.assertTrue(out['completed'], (stem, out['message']))
            self.assertTrue(sess.set_kind('roll'), stem)
            if model.has_rack:
                self.assertTrue(sess.set_kind('steer'), stem)
            if not model.has_rack:
                with self.assertRaisesRegex(ValueError, "no steer sweep"):
                    sess.set_kind('steer')

    def test_missing_adjuster_is_refused(self):
        sess = _session.SuspensionSession(ENGINE)
        sess.load_library('semi_trailing_rear')
        self.assertFalse(sess.has_adjuster('caster_shim'))
        with self.assertRaisesRegex(ValueError, "no caster_shim"):
            sess.set_adjuster('caster_shim', 1.0)
        self.assertTrue(sess.set_adjuster('camber_shim', 1.0))

    def test_quarter_car_without_a_rocker(self):
        sess = _session.SuspensionSession(ENGINE)
        sess.load_library('macpherson_front')
        run = sess.quarter_car(400.0, road=('bump', {
            'speed': 30.0, 'height': 40.0, 'length': 150.0,
            'duration': 0.3}), dt=2.e-3)
        self.assertTrue(run.start(), run.message)
        self.assertNotIn('rocker', run.numbers)
        while run.advance(50):
            pass
        self.assertTrue(run.completed, run.message)
        self.assertGreater(max(run.history['travel']), 5.0)
        self.assertIn('upright', run.poses[-1])


class WheelMeshTC(unittest.TestCase):
    """The wheel of the 3D view: a tire of revolution with a rounded
    section and grooves, and a rim with a barrel, flanges, hub, and
    spokes."""

    def test_tire_section(self):
        radius, rim, width = 330.0, 0.62 * 330.0, 0.6 * 330.0
        cols = _scene.tire_columns(radius, rim, width)
        ys = [y for y, _, _ in cols]
        self.assertEqual(sorted(ys), ys)
        self.assertAlmostEqual(-0.5 * width, ys[0])
        self.assertAlmostEqual(0.5 * width, ys[-1])
        tops = np.array([top for _, _, top in cols])
        self.assertAlmostEqual(radius, tops.max(), delta=0.01)
        # The sidewalls fall to the bead, just above the rim.
        self.assertAlmostEqual(rim + 5.0, tops[0])
        self.assertAlmostEqual(rim + 5.0, tops[-1])
        # The center groove (4 percent of the width) is 5 mm deep.
        center = [top for y, _, top in cols
                  if abs(y) <= 0.02 * width + 1e-9]
        self.assertTrue(center)
        for top in center:
            self.assertAlmostEqual(radius - 5.0, top, delta=0.01)

    def test_revolve_shares_the_nodes(self):
        # Only the outer shell is a boundary: inside and outside surfaces
        # of every strip and the two end walls, nothing between the cells.
        cols = _scene.tire_columns(330.0, 204.6, 198.0)
        builder = _scene.MeshBuilder()
        builder.revolve((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), cols, nseg=96)
        mesh = builder.build()
        self.assertEqual(96 * len(cols) * 2, len(builder.nodes))
        self.assertEqual(96 * (len(cols) - 1), mesh.ncell)
        self.assertEqual(96 * (2 * (len(cols) - 1) + 2), mesh.nbound)
        nodes = np.array(builder.nodes)
        self.assertLessEqual(np.hypot(nodes[:, 0], nodes[:, 2]).max(),
                             330.0 + 1e-9)
        with self.assertRaises(ValueError):
            builder.revolve((0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                            [(0.0, 2.0, 1.0), (1.0, 2.0, 3.0)])

    def test_wheel_meshes_ride_with_the_upright(self):
        from solvcon.multibody.suspension.model import CornerModel
        geom = _scene.CornerGeometry(CornerModel.load("race_front_corner"))
        meshes = geom.body_meshes()
        self.assertIn('wheel', meshes)
        self.assertIn('rim', meshes)
        self.assertEqual(1.0, geom.outboard())
        pose = {body: (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
                for body in ('chassis', 'upright', 'rocker', 'rack')}
        poses = geom.object_poses(pose)
        self.assertIs(poses['wheel'], pose['upright'])
        self.assertIs(poses['rim'], pose['upright'])

    def test_knuckle_turns_around_the_tire(self):
        # A ball joint above the tire (the Audi upper links) is reached by
        # a swan neck past the inboard flange, not a bar through the tread.
        from solvcon.multibody.suspension.model import CornerModel
        geom = _scene.CornerGeometry(CornerModel.load(
            "virtual_kingpin_front"))
        local = geom.upright_local('upper_front_outer')
        self.assertFalse(geom.in_wheel(local))
        self.assertTrue(any(geom.in_wheel(s * local)
                            for s in np.linspace(0.0, 1.0, 65)))
        path = geom.knuckle_path(local)
        self.assertEqual(4, len(path))
        np.testing.assert_allclose(local, path[-1])
        for start, end in zip(path[:-1], path[1:]):
            for s in np.linspace(0.0, 1.0, 65):
                self.assertFalse(geom.in_wheel(start + s * (end - start)))
        # A pickup inside the barrel keeps its straight bar.
        low = geom.upright_local('lower_front_outer')
        self.assertEqual(2, len(geom.knuckle_path(low)))

    @requires_engine
    def test_no_part_enters_the_wheel(self):
        # Over the bump sweep, no knuckle bar, link, rod, or strut of a
        # library model enters the tire or the rim; the rim grows to hold
        # the ball joints inside its barrel.
        sess = _session.SuspensionSession(ENGINE)
        for stem, _, _ in sess.library():
            sess.load_library(stem)
            self.assertTrue(sess.run_sweep(), stem)
            geom = sess.geometry()
            for name in geom.upright_points:
                if 'wheel_center' == name:
                    continue
                path = geom.knuckle_path(geom.upright_local(name))
                for start, end in zip(path[:-1], path[1:]):
                    for s in np.linspace(0.0, 1.0, 41):
                        self.assertFalse(
                            geom.in_wheel(start + s * (end - start)),
                            f"{stem}: knuckle bar of {name}")
            for index in range(0, len(sess.current), 4):
                pose = sess.pose(index)
                trans, rot = geom.frames(pose)['upright']
                for seg, (p0, p1) in geom.link_segments(pose).items():
                    if 'virtual_kingpin_front' == stem and 'spring' == seg:
                        # The damper point of the source grazes the inner
                        # sidewall by up to 3 mm (a line, not a body).
                        continue
                    for s in np.linspace(0.0, 1.0, 41):
                        local = rot.T @ (p0 + s * (p1 - p0) - trans)
                        self.assertFalse(geom.in_wheel(local),
                                         f"{stem}: {seg} at point {index}")


@requires_engine
class BumpSweepTC(unittest.TestCase):

    def setUp(self):
        self.sess = _session.SuspensionSession(ENGINE)
        self.assertTrue(self.sess.run_sweep())

    def _reference(self, **adjusters):
        model = ENGINE.model.CornerModel.load(self.sess.model_path)
        model.set_adjusters(**{k: v / 1.e3 for k, v in adjusters.items()})
        # The session sweeps the mm range it shows, converted to m.
        travels = self.sess.travels()
        np.testing.assert_allclose(
            np.linspace(-0.04, 0.04, 33, dtype='float64'), travels,
            rtol=0.0, atol=1.e-15)
        return ENGINE.kc.bump_sweep(ENGINE.mbs, model, travels)

    def test_curves_are_the_package_output(self):
        # The app must draw what the K&C analysis computes, not a copy that
        # drifted: every curve equals bump_sweep bit for bit.
        ref = self._reference()
        current = self.sess.current
        self.assertEqual('bump', current.kind.name)
        for curve in _session.CURVES['bump']:
            with self.subTest(curve=curve.key):
                np.testing.assert_array_equal(ref[curve.key],
                                              current.data[curve.key])
                xs, ys = current.curve(curve.key)
                np.testing.assert_array_equal(ref['travel'] * 1.e3, xs)
                np.testing.assert_array_equal(ref[curve.key] * curve.scale,
                                              ys)

    def test_adjuster_keeps_the_replaced_curve(self):
        base = self.sess.current
        self.assertTrue(self.sess.set_adjuster('camber_shim', 2.0))
        self.assertIs(base, self.sess.previous)
        self.assertEqual(2.0, self.sess.current.setup['camber_shim'])
        np.testing.assert_array_equal(
            self._reference(camber_shim=2.0)['camber'],
            self.sess.current.data['camber'])
        # The shim spaces the upper ball joint outboard: more negative
        # camber over the whole travel.
        self.assertTrue(np.all(self.sess.current.data['camber']
                               < base.data['camber']))
        # A second change moves the window: the replaced curve is the one
        # of the first change, not the design.
        first = self.sess.current
        self.assertTrue(self.sess.set_adjuster('toe_link', -0.5))
        self.assertIs(first, self.sess.previous)

    def test_failed_sweep_keeps_the_last_valid_result(self):
        self.sess.set_cursor(10)
        valid = self.sess.current
        pose = self.sess.pose()
        # 400 mm of bump pulls the wheel past the reach of the links.
        self.assertFalse(self.sess.set_range(-40.0, 400.0, 12))
        self.assertIs(valid, self.sess.current)
        self.assertEqual(10, self.sess.cursor)
        for body, (pos, quat) in self.sess.pose().items():
            np.testing.assert_array_equal(pose[body][0], pos)
            np.testing.assert_array_equal(pose[body][1], quat)
        error, shown = self.sess.warnings[:2]
        self.assertEqual('error', error['level'])
        self.assertIn("did not converge", error['message'])
        self.assertIn("of 12 points converged", error['message'])
        self.assertIn("showing the last valid result", shown['message'])

    def test_unassemblable_adjuster_keeps_the_last_valid_result(self):
        valid = self.sess.current
        # A tie rod half a meter longer cannot be assembled at all.
        self.assertFalse(self.sess.set_adjuster('toe_link', 500.0))
        self.assertIs(valid, self.sess.current)
        self.assertIsNone(self.sess.previous)
        self.assertEqual(500.0, self.sess.adjusters['toe_link'])
        self.assertIn("toe link +500.00 mm", self.sess.warnings[0]['message'])

    def test_links_keep_their_length_in_the_view(self):
        # The 3D view draws the links from the solved poses; a pose read
        # with the wrong frame convention would stretch them.
        geometry = self.sess.geometry()
        model = geometry.model
        for index in (0, 16, 32):
            segments = geometry.link_segments(self.sess.pose(index))
            for name in model.links:
                start, end = segments[name]
                self.assertAlmostEqual(model.link_length(name) * 1.e3,
                                       np.linalg.norm(end - start),
                                       places=5)
            start, end = segments['pushrod']
            length = np.linalg.norm(model.hp['pushrod_outer']
                                    - model.hp['pushrod_inner']) * 1.e3
            self.assertAlmostEqual(length, np.linalg.norm(end - start),
                                   places=5)

    def test_sweep_pose_keeps_the_chassis_at_design(self):
        geometry = self.sess.geometry()
        trans, quat = geometry.object_poses(self.sess.pose(0))['chassis']
        np.testing.assert_array_equal([0.0, 0.0, 0.0], trans)
        np.testing.assert_array_equal([1.0, 0.0, 0.0, 0.0], quat)

    def test_hardpoints_are_where_the_links_end(self):
        # The 3D pick and readout use hardpoints(); they must be the ends
        # of the drawn links at every pose.
        geometry = self.sess.geometry()
        for index in (0, 32):
            pose = self.sess.pose(index)
            points = geometry.hardpoints(pose)
            segments = geometry.link_segments(pose)
            for name, (_, inner, outer) in geometry.model.links.items():
                start, end = segments[name]
                np.testing.assert_allclose(start, points[inner][1],
                                           atol=1.e-9)
                np.testing.assert_allclose(end, points[outer][1], atol=1.e-9)
            start, end = segments['pushrod']
            np.testing.assert_allclose(start, points['pushrod_outer'][1],
                                       atol=1.e-9)
            np.testing.assert_allclose(end, points['pushrod_inner'][1],
                                       atol=1.e-9)
        # At design travel the current position is the design position.
        middle = self.sess.pose(16)
        for name, (design, now) in geometry.hardpoints(middle).items():
            np.testing.assert_allclose(design, now, atol=1.e-6,
                                       err_msg=name)

    def test_rotation_is_the_engine_convention(self):
        quat = self.sess.pose(0)['upright'][1]
        np.testing.assert_allclose(ENGINE.metrics.rot(quat),
                                   _scene.rotation(quat), atol=1.e-15)

    def test_csv_names_the_model_and_the_units(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "sweep.csv")
            self.sess.export_csv(path)
            with open(path) as fobj:
                lines = fobj.read().splitlines()
        self.assertIn(f"# model file: {self.sess.model_path}", lines)
        units = [line for line in lines if line.startswith("# units:")]
        self.assertIn("camber [deg]", units[0])
        self.assertIn("scrub [mm]", units[0])
        header = lines[len([ln for ln in lines if ln.startswith("#")])]
        self.assertTrue(header.startswith("Wheel travel (+ bump) [mm],"))
        rows = lines[lines.index(header) + 1:]
        self.assertEqual(33, len(rows))
        first = [float(v) for v in rows[0].split(",")]
        self.assertEqual(-40.0, first[0])
        self.assertAlmostEqual(self.sess.current.curve('camber')[1][0],
                               first[1], places=8)

    def test_joints_are_named_for_the_warnings(self):
        rig = ENGINE.corner.kc_rig(ENGINE.mbs, self.sess.model())
        names = set(self.sess._joint_names(rig).values())
        self.assertTrue(set(self.sess.model().links) <= names)
        self.assertIn('wheel travel driver', names)
        self.assertIn('rack driver (steer)', names)
        self.assertEqual(rig[0].get_njoint(), len(names))

    def test_force_arrows_balance_the_upright(self):
        # On a K&C rig the upright is held by the links, the pushrod, and
        # the travel driver; with the driver left out the arrows carry what
        # the driver does not.
        forces = self.sess.forces(32)
        names = [name for name, _, _ in forces]
        self.assertIn('pushrod', names)
        self.assertIn('spring', names)
        self.assertNotIn('wheel travel driver', names)


@requires_engine
class OtherSweepTC(unittest.TestCase):

    def test_steer_and_roll_are_the_package_output(self):
        sess = _session.SuspensionSession(ENGINE)
        self.assertTrue(sess.set_kind('steer'))
        racks = sess.travels()
        ref = ENGINE.kc.steer_sweep(ENGINE.mbs, sess.model(), racks)
        np.testing.assert_array_equal(ref['toe'], sess.current.data['toe'])
        self.assertEqual(25, len(sess.current.poses))
        self.assertTrue(sess.set_kind('roll'))
        travels = sess.travels()
        ref = ENGINE.axle.roll_sweep(ENGINE.mbs, sess.model(), travels)
        np.testing.assert_array_equal(ref['moment'],
                                      sess.current.data['moment'])
        self.assertIsNone(sess.previous)

    def test_outline_shows_units(self):
        sess = _session.SuspensionSession(ENGINE)
        outline = dict(sess.outline())
        self.assertEqual(['Bodies', 'Links', 'Drivers', 'Forces',
                          'Adjusters'], [group for group, _ in
                                         sess.outline()])
        spring = dict(dict(outline['Forces'])['spring'])
        self.assertEqual("150 N/mm", spring['rate'])
        link = dict(dict(outline['Links'])['tie_rod'])
        self.assertTrue(link['length'].endswith(" mm"))


@requires_engine
class RollAxleTC(unittest.TestCase):

    def test_both_corners_are_posed(self):
        sess = _session.SuspensionSession(ENGINE)
        self.assertTrue(sess.set_kind('roll'))
        left, right = sess.geometries()
        self.assertEqual(_session.RIGHT, right.suffix)
        pose = sess.pose(0)
        for body in ('upright', 'rocker', 'rack', 'arb'):
            self.assertIn(body, pose)
            self.assertIn(body + _session.RIGHT, pose)
        # Opposite wheel travel: the wheels move apart in z by twice the
        # swept value.
        dz = (pose['upright'][0][2] - pose['upright_right'][0][2]) * 1.e3
        self.assertAlmostEqual(2.0 * sess.current.x[0], dz, places=6)
        # The mirrored corner keeps its link lengths, which a wrong body
        # number or mirror would break.
        for geom in (left, right):
            segments = geom.link_segments(pose)
            for name in geom.model.links:
                start, end = segments[name + geom.suffix]
                self.assertAlmostEqual(geom.model.link_length(name) * 1.e3,
                                       np.linalg.norm(end - start),
                                       places=5)
        self.assertEqual(2 + 4 * 2, len(_scene.arb_segments(left, right,
                                                            pose)))

    def test_force_arrows_on_both_corners(self):
        sess = _session.SuspensionSession(ENGINE)
        self.assertTrue(sess.set_kind('roll'))
        names = [name for name, _, _ in sess.forces(0)]
        for name in ('tie_rod', 'tie_rod_right', 'pushrod_right', 'spring',
                     'spring_right'):
            self.assertIn(name, names)
        self.assertFalse([n for n in names if n.startswith('wheel travel')])


@requires_engine
class AnalysisTC(unittest.TestCase):

    DEG = 180.0 / np.pi

    def setUp(self):
        self.sess = _session.SuspensionSession(ENGINE)

    def test_statics_is_the_compliance_output(self):
        force = [0.0, 1000.0, -500.0]
        model = self.sess.model()
        sys_obj, _ = ENGINE.compliance.compliant_rig(ENGINE.mbs, model)
        base = ENGINE.compliance.apply_patch_load(ENGINE.mbs, sys_obj, model,
                                                  [0.0] * 3)
        loaded = ENGINE.compliance.apply_patch_load(ENGINE.mbs, sys_obj,
                                                    model, force)
        out = self.sess.run_compliance(force)
        self.assertTrue(out['completed'])
        self.assertEqual((loaded['toe'] - base['toe']) * self.DEG,
                         out['toe'])
        self.assertEqual((loaded['camber'] - base['camber']) * self.DEG,
                         out['camber'])
        np.testing.assert_array_equal(
            (loaded['contact_patch'] - base['contact_patch']) * 1.e3,
            out['patch'])
        # A lateral push at the patch steers the wheel: the change is real.
        self.assertNotEqual(0.0, out['toe'])

    def test_modal_is_the_engine_output(self):
        sys_obj, _ = ENGINE.vehicle.quarter_car(ENGINE.mbs,
                                                self.sess.model(), 200.0)
        self.assertTrue(sys_obj.find_equilibrium(tol=1.e-9)['converged'])
        modal = ENGINE.mbs.modal_analysis(sys_obj)
        out = self.sess.run_modal(200.0)
        self.assertTrue(out['completed'])
        self.assertEqual(['ride', 'wheel hop'],
                         [name for name, _, _ in out['modes']])
        self.assertEqual(sorted(zip(np.asarray(modal.frequencies).tolist(),
                                    np.asarray(modal.damping_ratios)
                                    .tolist())),
                         [(f, z) for _, f, z in out['modes']])
        # A ride mode near 1 to 3 Hz and a wheel hop mode an order above.
        self.assertLess(out['modes'][0][1], 5.0)
        self.assertGreater(out['modes'][1][1], 10.0)

    def test_dynamics_is_the_engine_march(self):
        run = self.sess.quarter_car(200.0, 5.0, 3.0, 0.02, dt=2.e-3)
        self.assertTrue(run.start())
        while run.advance(3):
            pass
        self.assertTrue(run.completed)
        self.assertEqual("completed", run.message)
        self.assertEqual(run.steps + 1, len(run.history['time']))
        mbs = ENGINE.mbs
        rig = _session.make_road_input(ENGINE, self.sess.model(), 'rig',
                                       {'amplitude': 5.0, 'frequency': 3.0,
                                        'duration': 0.02})
        sys_obj, hd = ENGINE.vehicle.quarter_car(mbs, self.sess.model(),
                                                 200.0, road=rig, dt=2.e-3)
        sys_obj.set_integrator('rk45')
        sys_obj.set_tolerances(1e-7, 1e-10)
        z0 = np.asarray(sys_obj.get_body_positions())[1][2]
        body = [0.0]
        force = [mbs.get_tire_state(hd['tire'])['normal_force']]
        for _ in range(run.steps):
            sys_obj.solve()
            body.append((np.asarray(sys_obj.get_body_positions())[1][2]
                         - z0) * 1.e3)
            force.append(mbs.get_tire_state(hd['tire'])['normal_force'])
        self.assertEqual(body, run.history['body'])
        self.assertEqual(force, run.history['force'])
        # The plate starts at rest (faded in) and rises into the tire.
        self.assertEqual(0.0, run.history['road'][0])
        self.assertGreater(run.history['road'][-1], 0.0)

    def test_dynamics_on_a_road(self):
        # A pothole at 40 km/h: the effective road dips, the tire leaves the
        # road (force 0) and lands again; the run lasts its duration.
        run = self.sess.quarter_car(200.0, road=('pothole', {
            'speed': 40.0, 'depth': 60.0, 'length': 300.0,
            'duration': 0.4}), dt=5.e-4)
        self.assertTrue(run.start())
        while run.advance(200):
            pass
        self.assertTrue(run.completed, run.message)
        self.assertIn("pothole 60 mm x 300 mm at 40 km/h", run.road.name)
        self.assertLess(min(run.history['road']), -10.0)
        self.assertGreater(run.airborne_time, 0.0)
        off = [f for f, c in zip(run.history['force'],
                                 run.history['contact']) if not c]
        self.assertEqual([0.0] * len(off), off)
        self.assertGreater(len(run.road_profile()), 100)

    def test_road_duration_is_the_run_time(self):
        # The duration sets the simulated time, also past the road (the
        # car runs on the flat road after the stone), and the status can
        # tell a duration that ends on the profile.
        for duration in (0.05, 0.3):
            run = self.sess.quarter_car(200.0, road=('bump', {
                'speed': 30.0, 'height': 80.0, 'length': 150.0,
                'duration': duration}), dt=2.e-3)
            self.assertTrue(run.start(), run.message)
            self.assertAlmostEqual(duration - 2.e-3, run.duration)
            self.assertGreater(run.pass_time, 0.05)
            self.assertLess(run.pass_time, 0.3)
        while run.advance(50):
            pass
        self.assertTrue(run.completed, run.message)
        self.assertEqual(len(run.history['time']),
                         len(run.history['travel']))
        # The stone hits the bump stop, and the history records it.
        self.assertGreater(run.stop_time(-1), 0.0)
        self.assertGreater(max(run.history['stop']), 0.0)
        self.assertLess(max(run.history['travel']), 70.0)
        run = self.sess.quarter_car(200.0, road=('bump', {'duration': 0.0}))
        self.assertFalse(run.start())
        self.assertIn("duration must be positive", run.message)

    def test_road_cams_use_the_sprung_mass(self):
        # The static tire deflection, and so the cam contact length, comes
        # from the quarter car mass, not a fixed one.
        light = self.sess.road_input('bump', {}, sprung_mass=100.0)
        heavy = self.sess.road_input('bump', {}, sprung_mass=400.0)
        self.assertLess(light.cam.reach, heavy.cam.reach)

    def test_bad_road_input_is_reported(self):
        run = self.sess.quarter_car(200.0, road=('csv', {'path': ''}))
        self.assertFalse(run.start())
        self.assertIn("pick a road profile CSV", run.message)

    def test_dynamics_pose_moves_the_corner_with_the_chassis(self):
        # The 3D view animates the quarter car from run.pose(): the corner
        # bodies are the engine bodies, the chassis pickups ride on the
        # sprung mass, and the links keep their length.
        run = self.sess.quarter_car(200.0, 20.0, 3.0, 0.2, dt=2.e-3)
        self.assertIsNone(run.pose())
        self.assertTrue(run.start())
        # The static equilibrium is the design pose.
        rest = run.pose()['chassis'][0][2]
        self.assertAlmostEqual(0.0, rest, places=8)
        while run.advance(20):
            pass
        pose = run.pose()
        pos = np.asarray(run._sys.get_body_positions())
        quat = np.asarray(run._sys.get_body_quaternions())
        for body, num in (('chassis', 1), ('upright', 2), ('rocker', 3),
                          ('rack', 4)):
            np.testing.assert_array_equal(pos[num], pose[body][0])
            np.testing.assert_array_equal(quat[num], pose[body][1])
        height = run.road.at(run._sys.get_sim_time())[0]
        self.assertAlmostEqual(height, pose['road'][0][2], places=12)
        self.assertAlmostEqual(run.history['body'][-1],
                               (pose['chassis'][0][2] - rest) * 1.e3,
                               places=9)
        self.assertGreater(abs(run.history['body'][-1]), 0.5)
        lift = pose['chassis'][0][2] * 1.e3
        geometry = self.sess.geometry()
        model = geometry.model
        segments = geometry.link_segments(pose)
        for name, (kind, inner, _) in model.links.items():
            start, end = segments[name]
            self.assertAlmostEqual(model.link_length(name) * 1.e3,
                                   np.linalg.norm(end - start), places=5)
            if 'rack' != kind:
                np.testing.assert_allclose(
                    geometry.hp[inner] + [0.0, 0.0, lift], start,
                    atol=1.e-9)
        trans, _ = geometry.object_poses(pose)['chassis']
        np.testing.assert_array_equal(pose['chassis'][0], trans)

    def test_dynamics_cancel_stops_the_march(self):
        run = self.sess.quarter_car(200.0, 5.0, 3.0, 1.0)
        self.assertTrue(run.start())
        run.advance(2)
        run.cancel()
        self.assertTrue(run.finished)
        self.assertFalse(run.completed)
        self.assertTrue(run.message.startswith("cancelled at t = 0.004"))
        self.assertFalse(run.advance(5))
        self.assertEqual(3, len(run.history['time']))

    def test_sweep_points_are_capped(self):
        with self.assertRaisesRegex(ValueError, "2 to 201 points"):
            self.sess.set_range(-40.0, 40.0, 202)


@requires_engine
class SetupSheetTC(unittest.TestCase):
    """The setup sheet of the session (the numbers are validated in
    tests/test_multibody_suspension_tuning.py)."""

    def test_setup_changes_every_analysis(self):
        sess = _session.SuspensionSession(ENGINE)
        sess.load_library('macpherson_coilover')
        keys = [row[0] for row in sess.setup_fields()]
        self.assertEqual(['spring_rate', 'ride_height', 'bump_stop_gap',
                          'bump_clicks', 'rebound_clicks', 'arb_rate',
                          'tire_rate'], keys)
        self.assertTrue(sess.set_setup('spring_rate', 80.0))
        self.assertEqual(80e3, sess.model().spring_k)
        sess.set_setup('ride_height', -20.0)
        out = sess.setup_readouts(400.0)
        self.assertTrue(out['completed'], out['message'])
        self.assertAlmostEqual(20.0, out['current']['static_travel'])
        self.assertAlmostEqual(0.0, out['reference']['static_travel'])
        self.assertGreater(out['current']['ride_frequency'],
                           out['reference']['ride_frequency'])
        # The modal analysis and the ride see the setup.
        modes = sess.run_modal(400.0)['modes']
        self.assertAlmostEqual(out['current']['ride_mode'], modes[0][1],
                               places=3)
        with self.assertRaises(ValueError):
            sess.set_setup('bump_clicks', 40)
        self.assertNotIn('bump_clicks', sess.setup)
        sess.set_setup_reference()
        again = sess.setup_readouts(400.0)
        self.assertAlmostEqual(again['current']['ride_frequency'],
                               again['reference']['ride_frequency'])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'setup.json')
            sess.save_setup(path)
            sess.reset_setup()
            self.assertEqual({}, sess.setup)
            sess.load_setup(path)
            self.assertEqual(80.0, sess.setup['spring_rate'])
            self.assertEqual(-20.0, sess.setup['ride_height'])
        # A new model starts from its own file.
        sess.load_library('five_link_coilover')
        self.assertEqual({}, sess.setup)

    def test_vehicle_setup_and_handling(self):
        sess = _session.SuspensionSession(ENGINE)
        setups = sess.vehicle_setup('coilover_car')
        self.assertEqual(60.0, setups['front']['spring_rate'])
        self.assertIn('arb_rate', setups['rear'])
        base = sess.run_handling('coilover_car', setups)
        self.assertTrue(base['completed'], base['message'])
        stiff = dict(setups, rear=dict(setups['rear'], arb_rate=80.0))
        stiff['front'] = dict(setups['front'], arb_rate=5.0)
        loose = sess.run_handling('coilover_car', stiff)
        self.assertEqual('front', base['limited_by'])
        self.assertEqual('rear', loose['limited_by'])
        ro = sess.vehicle_readouts('coilover_car', setups)
        self.assertTrue(ro['completed'], ro['message'])
        self.assertGreater(ro['front']['roll_stiffness'],
                           ro['front']['roll_stiffness_springs'])


@requires_engine
class LongitudinalTC(unittest.TestCase):
    """Braking and traction of the session (validated in
    tests/test_multibody_suspension_longitudinal.py)."""

    def test_braking_and_traction(self):
        sess = _session.SuspensionSession(ENGINE)
        out = sess.run_longitudinal('coilover_car', 0.65, 0.0)
        self.assertTrue(out['completed'], out['message'])
        self.assertEqual('rear', out['brake']['locks'])
        self.assertEqual('rear', out['traction']['spins'])
        self.assertGreater(out['braking']['pitch'], 0.0)
        self.assertLess(out['driving']['pitch'], 0.0)
        self.assertAlmostEqual(out['braking']['anti'][0],
                               out['formula']['anti_dive'], delta=0.03)
        front = sess.run_longitudinal('coilover_car', 0.85, 1.0)
        self.assertEqual('front', front['brake']['locks'])
        self.assertEqual('front', front['traction']['spins'])
        self.assertLess(front['traction']['limit'],
                        out['traction']['limit'])


@requires_engine
class RideVehicleTC(unittest.TestCase):
    """The ride and vehicle analyses of the session (the numbers are
    validated in tests/test_multibody_suspension_ride.py and _car.py)."""

    def test_ride_response_and_time_run(self):
        sess = _session.SuspensionSession(ENGINE)
        sess.load_library('macpherson_front')
        out = sess.run_ride('C', 60.0, 400.0)
        self.assertTrue(out['completed'], out['message'])
        st = out['stats']
        self.assertGreater(st['body_acc'], st['body_acc_w'])
        self.assertEqual(ENGINE.ride.comfort_reaction(st['body_acc_w']),
                         st['comfort'])
        self.assertLess(st['tire_load_ratio'], 0.3)
        # Per mm of road: the wheel hop peak of the body acceleration.
        self.assertEqual(len(out['freq']), len(out['body_acc']))
        peak = out['freq'][int(np.argmax(out['body_acc']))]
        self.assertTrue(5.0 < peak < 20.0, peak)
        run = sess.run_ride_time('C', 60.0, 400.0, duration=6.0)
        self.assertTrue(run['completed'], run['message'])
        self.assertAlmostEqual(st['body_acc_w'], run['body_acc_w'],
                               delta=0.2 * st['body_acc_w'])
        bad = sess.run_ride('J', 60.0, 400.0)
        self.assertFalse(bad['completed'])
        self.assertIn("road class", bad['message'])

    def test_vehicle_cornering_and_step_steer(self):
        sess = _session.SuspensionSession(ENGINE)
        stems = [stem for stem, _ in sess.vehicles()]
        self.assertIn('road_car', stems)
        out = sess.run_vehicle('road_car', (0.0, 0.05, 0.1))
        self.assertTrue(out['completed'], out['message'])
        self.assertGreater(out['understeer'], 0.0)
        self.assertTrue(0.3 < out['steady']['lltd'] < 0.7)
        run = sess.run_step_steer('road_car', 72.0, 1.0,
                                  understeer=out['understeer'],
                                  duration=3.0)
        self.assertTrue(run['completed'], run['message'])
        self.assertAlmostEqual(run['steady']['yaw_rate'],
                               run['bicycle_yaw_rate'],
                               delta=0.03 * abs(run['bicycle_yaw_rate']))


class EngineTC(unittest.TestCase):

    def test_a_build_without_the_engine_is_reported(self):
        from unittest import mock
        import solvcon.multibody
        with mock.patch.object(solvcon.multibody, 'HAS_MULTIBODY', False):
            with self.assertRaisesRegex(_engine.EngineUnavailable,
                                        "HAS_MULTIBODY is false"):
                _engine.load()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
