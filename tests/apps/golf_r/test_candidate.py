# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Corner and vehicle models from AC data (synthetic fixture, no car data).

* The hardpoints are from the wheel centre on both sides: the kingpin
  inclination and the link lengths equal those of the raw coordinates.
* The static camber and toe keep their AC signs.
* The wheel rates of the model (spring, ARB, damper) equal the AC rates.
* The step steer at 120 km/h and 4 mm of rack stays plausible.
"""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from solvcon import multibody
from solvcon.apps.golf_r import suspension_candidate as sc
from solvcon.apps.golf_r import vehicle_candidate as vc
from solvcon.multibody.suspension import axle, car, kc
from solvcon.multibody.suspension.model import CornerModel, damper_table

SUSPENSIONS = """[HEADER]
VERSION=1
[BASIC]
WHEELBASE=2.60
CG_LOCATION=0.60
[ARB]
FRONT=30000
REAR=15000
[FRONT]
TYPE=STRUT
BASEY=-0.24
TRACK=1.56
HUB_MASS=40
STRUT_CAR=0.25, 0.55, -0.04
STRUT_TYRE=0.07, -0.14, 0.01
WBCAR_BOTTOM_FRONT=0.40, -0.11, 0.10
WBCAR_BOTTOM_REAR=0.41, -0.12, -0.27
WBTYRE_BOTTOM=0.07, -0.14, 0.01
WBCAR_STEER=0.50, -0.03, -0.19
WBTYRE_STEER=0.10, -0.04, -0.13
TOE_OUT=0.0005
STATIC_CAMBER=-2.0
SPRING_RATE=50000
PROGRESSIVE_SPRING_RATE=0
BUMP_STOP_RATE=60000
BUMPSTOP_UP=0.08
BUMPSTOP_DN=0.06
DAMP_BUMP=5000
DAMP_FAST_BUMP=2500
DAMP_FAST_BUMPTHRESHOLD=0.10
DAMP_REBOUND=7000
DAMP_FAST_REBOUND=4000
DAMP_FAST_REBOUNDTHRESHOLD=0.12
[REAR]
TYPE=DWB
BASEY=-0.26
TRACK=1.53
HUB_MASS=35
WBCAR_TOP_FRONT=0.35, 0.09, 0.22
WBCAR_TOP_REAR=0.40, 0.09, 0.03
WBCAR_BOTTOM_FRONT=0.48, -0.12, 0.30
WBCAR_BOTTOM_REAR=0.58, -0.12, -0.12
WBTYRE_TOP=0.07, 0.10, -0.01
WBTYRE_BOTTOM=0.05, -0.13, 0.02
WBCAR_STEER=0.58, -0.10, -0.11
WBTYRE_STEER=0.07, -0.10, -0.10
TOE_OUT=0.0002
STATIC_CAMBER=-1.5
SPRING_RATE=40000
BUMP_STOP_RATE=40000
BUMPSTOP_UP=0.08
BUMPSTOP_DN=-0.00
DAMP_BUMP=4000
DAMP_FAST_BUMP=3500
DAMP_FAST_BUMPTHRESHOLD=0.10
DAMP_REBOUND=4500
DAMP_FAST_REBOUND=4000
DAMP_FAST_REBOUNDTHRESHOLD=0.12
"""

TYRE = """RADIUS=0.32
RATE=250000
DAMP=400
FZ0=3000
LS_EXPY=0.8
LS_EXPX=0.85
DY_REF=1.1
DX_REF=1.15
FRICTION_LIMIT_ANGLE=8.0
FALLOFF_LEVEL=0.86
CX_MULT=1.0
RELAXATION_LENGTH=0.3
"""


def write_fixture(path):
    """Write the synthetic AC files into ``path``."""
    path = Path(path)
    (path / 'suspensions.ini').write_text(SUSPENSIONS)
    (path / 'tyres.ini').write_text(
        '[HEADER]\nVERSION=10\n[FRONT]\n' + TYRE + '[REAR]\n' + TYRE)
    (path / 'car.ini').write_text(
        '[BASIC]\nTOTALMASS=1500\nINERTIA=1.6,1.4,4.3\n')
    (path / 'brakes.ini').write_text('[DATA]\nFRONT_SHARE=0.70\n')


class CandidateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        write_fixture(cls.tmp.name)
        cls.front = sc.candidate('front', cls.tmp.name)
        cls.rear = sc.candidate('rear', cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_one_tire_in_every_model(self):
        # The multibody vehicle, the handling diagram, and the bicycle and
        # wheel models use the same Magic Formula of the AC data.
        from solvcon.apps.golf_r.bicycle import tire_from_data

        vm = vc.vehicle_model(self.tmp.name)
        tire = tire_from_data(self.tmp.name)
        for front in (True, False):
            engine = vm.tire(front)
            self.assertEqual(engine.fz0, tire.model.fz0)
            self.assertEqual(engine.lateral, tire.model.lateral)
            self.assertEqual(engine.longitudinal, tire.model.longitudinal)
        load, alpha = 3500.0, math.radians(3.0)
        self.assertEqual(
            car.tire_side_force(vm, True, alpha, load),
            -tire.model.force(load, 0.0, alpha)[1])
        self.assertEqual(vm.relaxation_length, tire.relaxation_length)

    def test_hardpoints_are_from_the_wheel_centre(self):
        # Car and tyre points share the origin: the lateral distance of a
        # link is the difference of the raw x values.
        link = self.rear['links']['tie_rod']
        dy = link['upright'][1] - link['chassis'][1]
        self.assertAlmostEqual(dy, 1000.0 * (0.58 - 0.07), places=9)
        self.assertAlmostEqual(self.front['links']['lower_front'][
            'chassis'][1], 1000.0 * (0.78 - 0.40), places=9)

    def test_kingpin_inclination_of_the_strut(self):
        res = kc.bump_sweep(multibody, CornerModel(self.front), [0.0])
        expect = math.atan2(0.25 - 0.07, 0.55 + 0.14)
        self.assertAlmostEqual(float(res['kpi'][0]), expect, places=6)
        self.assertLess(math.degrees(expect), 16.0)

    def test_static_camber_and_toe_keep_their_signs(self):
        for model, camber in ((self.front, -2.0), (self.rear, -1.5)):
            res = kc.bump_sweep(multibody, CornerModel(model), [0.0])
            self.assertAlmostEqual(math.degrees(float(res['camber'][0])),
                                   camber, places=6)
            # TOE_OUT > 0 is toe-out: the toe metric is negative.
            self.assertLess(float(res['toe'][0]), 0.0)

    def test_wheel_rates_equal_the_ac_rates(self):
        for model, rate in ((self.front, 50.0), (self.rear, 40.0)):
            res = kc.bump_sweep(multibody, CornerModel(model),
                                [-1e-3, 0.0, 1e-3])
            self.assertAlmostEqual(float(res['wheel_rate'][1]) / 1e3, rate,
                                   delta=0.005 * rate)
        self.assertLess(self.front['wheel_rates']['motion_ratio'], 1.0)
        self.assertEqual(self.rear['wheel_rates']['motion_ratio'], 1.0)

    def test_damper_wheel_slopes_equal_the_ac_slopes(self):
        mr = self.front['wheel_rates']['motion_ratio']
        v, f = (np.asarray(a) for a in damper_table(
            self.front["spring"]["damper"]))
        # Spring-line slopes times mr^2 are the wheel slopes (N.s/m).
        slopes = np.diff(f) / np.diff(v) * mr * mr
        np.testing.assert_allclose(slopes, [2500, 5000, 7000, 4000],
                                   rtol=1e-12)
        # The knees at the AC thresholds (wheel speed times mr).
        np.testing.assert_allclose(v[[1, 3]], [-0.10 * mr, 0.12 * mr],
                                   rtol=1e-12)

    def test_anti_roll_bar_gives_the_ac_roll_stiffness(self):
        for model, rate, track in ((self.front, 30000.0, 1.56),
                                   (self.rear, 15000.0, 1.53)):
            m = CornerModel(model)
            both = axle.roll_sweep(multibody, m, [-1e-3, 0.0, 1e-3], True)
            springs = axle.roll_sweep(multibody, m, [-1e-3, 0.0, 1e-3],
                                      False)
            bar = float(both['roll_stiffness'][1]
                        - springs['roll_stiffness'][1])
            self.assertAlmostEqual(bar, rate * track * track,
                                   delta=1e-6 * rate)

    def test_stops_and_the_droop_stop_at_static(self):
        self.assertEqual(self.front['stops']['bump_travel'], 80.0)
        self.assertEqual(self.front['stops']['droop_travel'], -60.0)
        # BUMPSTOP_DN = 0: the droop stop sits 1 mm below static.
        self.assertEqual(self.rear['stops']['droop_travel'], -1.0)
        self.assertTrue(any('BUMPSTOP_DN' in a
                            for a in self.rear['assumptions']))

    def test_vehicle_masses_cg_and_inertia(self):
        data = vc.vehicle_data(self.tmp.name)
        sprung = 1500.0 - 2 * (40.0 + vc.RACK_MASS) - 2 * 35.0
        self.assertAlmostEqual(data['sprung_mass'], sprung, places=9)
        self.assertAlmostEqual(
            data['front_fraction'],
            (1500.0 * 0.6 - 2 * (40.0 + vc.RACK_MASS)) / sprung, places=12)
        self.assertAlmostEqual(
            data['cg_height'],
            1000.0 * (0.6 * (0.32 + 0.24) + 0.4 * (0.32 + 0.26)), places=9)
        np.testing.assert_allclose(
            data['sprung_inertia'],
            np.array([1.6 ** 2 + 1.4 ** 2, 1.4 ** 2 + 4.3 ** 2,
                      1.6 ** 2 + 4.3 ** 2]) * sprung / 12.0, rtol=1e-12)
        # The Magic Formula of the tyre data: the peak DY_REF FZ0 at
        # FRICTION_LIMIT_ANGLE, the sliding level FALLOFF_LEVEL.
        mf = data['tire']['magic_formula']['front']
        self.assertEqual(mf['fz0'], 3000.0)
        lat = mf['lateral']
        self.assertAlmostEqual(math.sin(0.5 * math.pi * lat['shape']), 0.86,
                               places=12)
        self.assertAlmostEqual(
            lat['stiffness'] * math.tan(math.radians(8.0)),
            math.tan(0.5 * math.pi / lat['shape']), places=12)
        self.assertEqual((lat['mu'], lat['load_exponent']), (1.1, 0.8))
        self.assertEqual(mf['longitudinal']['mu'], 1.15)
        self.assertEqual(data['tire']['relaxation_length'], 0.3)
        vm = vc.vehicle_model(self.tmp.name)
        tire = vm.tire(True)
        self.assertAlmostEqual(
            -tire.force(3000.0, 0.0, math.radians(8.0))[1], 1.1 * 3000.0,
            places=6)

    def test_step_steer_at_120_kmh_is_plausible(self):
        # 4 mm of rack takes this car to its friction limit: it must not
        # spin and must not exceed the friction.
        vm = vc.vehicle_model(self.tmp.name)
        run = car.step_steer(multibody, vm, 120.0 / 3.6, 0.004, start=0.5,
                             ramp=0.3, duration=2.5, dt=0.002)
        self.assertTrue(run['plausible'], run['checks'])
        self.assertLess(np.max(np.abs(run['sideslip'])),
                        car.SPIN_SIDESLIP)

    def test_steady_turn_lateral_acceleration_is_v_r(self):
        vm = vc.vehicle_model(self.tmp.name)
        run = car.step_steer(multibody, vm, 120.0 / 3.6, 0.0015, start=0.5,
                             ramp=0.3, duration=2.5, dt=0.002)
        self.assertTrue(run['plausible'], run['checks'])
        st = run['steady']
        self.assertAlmostEqual(st['lateral_acceleration'],
                               st['speed'] * st['yaw_rate'],
                               delta=0.02 * abs(st['lateral_acceleration']))


class PlausibilityTest(unittest.TestCase):
    def run_of(self, sideslip, ay, fz=4000.0, yaw=0.3):
        t = np.linspace(0.0, 3.0, 301)
        return {'time': t, 'yaw_rate': np.full_like(t, yaw),
                'speed': np.full_like(t, 30.0),
                'lateral_acceleration': np.full_like(t, ay),
                'sideslip': np.full_like(t, sideslip),
                'normal_force': np.full((len(t), 4), fz)}

    def test_a_steady_turn_passes(self):
        self.assertEqual(car.plausibility(self.run_of(0.02, 8.0), 1.0), [])

    def test_a_spin_is_invalid(self):
        checks = car.plausibility(self.run_of(math.radians(25.0), 8.0), 1.0)
        self.assertEqual([c['key'] for c in checks], ['spin'])
        self.assertEqual(checks[0]['level'], 'invalid')

    def test_more_than_the_friction_is_invalid(self):
        checks = car.plausibility(self.run_of(0.02, 1.2 * 9.81), 1.0)
        self.assertEqual([c['key'] for c in checks], ['friction'])

    def test_a_lifted_wheel_warns(self):
        checks = car.plausibility(self.run_of(0.02, 8.0, fz=0.0), 1.0)
        self.assertEqual([(c['key'], c['level']) for c in checks],
                         [('lift', 'warning')])

    def test_a_diverged_run_is_invalid(self):
        run = self.run_of(0.02, 8.0)
        run['yaw_rate'][-1] = np.nan
        self.assertEqual(car.plausibility(run, 1.0)[0]['key'], 'finite')


if __name__ == '__main__':
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
