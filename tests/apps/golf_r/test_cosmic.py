# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""COSMIC axles (Custom Shaders Patch) in the AC conversion, on the
synthetic fixture of tests/data/multibody/ac_synthetic.

* The rear double wishbone of the fixture, rewritten as COSMIC links (in a
  shuffled order) and a coilover on the DWB spring line, has the same
  kinematics as the DWB conversion.
* The coilover rates are on its own axis: RATE in series with INSTALL_RATE,
  the DAMP_* slopes unscaled, the stops at its length limits.
* A front COSMIC axle steers on the rack.
* Axles that are not two wishbones and one steering joint are rejected.
"""

import math
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from solvcon import multibody
from solvcon.apps.golf_r import suspension_candidate as sc
from solvcon.apps.golf_r import vehicle_candidate as vc
from solvcon.multibody.suspension import kc
from solvcon.multibody.suspension.model import CornerModel, damper_table

#: A synthetic car in the AC file layout (see its README).
FIXTURE = Path(__file__).resolve().parents[2] / 'data' / 'multibody' \
    / 'ac_synthetic'


def write_fixture(path):
    """Copy the synthetic AC files into ``path``."""
    for name in ('suspensions.ini', 'tyres.ini', 'car.ini', 'brakes.ini',
                 'drivetrain.ini'):
        shutil.copy(FIXTURE / name, Path(path) / name)


#: The rear DWB of the fixture as COSMIC links: the lower rear, the upper
#: front, the lower front, the upper rear.  The coilover is the DWB spring
#: line (a vertical line at the wheel centre, 300 mm long).
REAR = """[REAR]
TYPE=COSMIC
BASEY=-0.26
TRACK=1.53
HUB_MASS=35
DJ0=LOWER_REAR
DJ0_BODY_A=CHASSIS
DJ0_BODY_B=HUB_L
DJ0_POS0=0.58, -0.12, -0.12
DJ0_POS1=0.05, -0.13, 0.02
DJ1=UPPER_FRONT
DJ1_BODY_A=CHASSIS
DJ1_BODY_B=HUB_L
DJ1_POS0=0.35, 0.09, 0.22
DJ1_POS1=0.07, 0.10, -0.01
DJ2=LOWER_FRONT
DJ2_BODY_A=CHASSIS
DJ2_BODY_B=HUB_L
DJ2_POS0=0.48, -0.12, 0.30
DJ2_POS1=0.05, -0.13, 0.02
DJ3=UPPER_REAR
DJ3_BODY_A=CHASSIS
DJ3_BODY_B=HUB_L
DJ3_POS0=0.40, 0.09, 0.03
DJ3_POS1=0.07, 0.10, -0.01
STEER_JOINT_0=TOE
STEER_JOINT_0_BODY_A=CHASSIS
STEER_JOINT_0_BODY_B=HUB_L
STEER_JOINT_0_POS0=0.58, -0.10, -0.11
STEER_JOINT_0_POS1=0.07, -0.10, -0.10
TOE_OUT=0.0002
STATIC_CAMBER=-1.5

[REAR_COILOVER_0]
BODY_A=CHASSIS
BODY_B=HUB_L
POS_A=0.0, 0.30, 0.0
POS_B=0.0, 0.0, 0.0
MIN_LENGTH=0.20
MAX_LENGTH=0.36
BUMPSTOP_GAP=0.03
END_RATE=500000
INSTALL_RATE=400000
RATE=40000
DAMP_BUMP=4000
DAMP_FAST_BUMP=3500
DAMP_FAST_BUMPTHRESHOLD=0.10
DAMP_REBOUND=4500
DAMP_FAST_REBOUND=4000
DAMP_FAST_REBOUNDTHRESHOLD=0.12
"""

#: A front COSMIC double wishbone on the lower points of the fixture strut.
FRONT = """[FRONT]
TYPE=COSMIC
BASEY=-0.24
TRACK=1.56
HUB_MASS=40
DJ0=UPPER_FRONT
DJ0_BODY_A=CHASSIS
DJ0_BODY_B=HUB_L
DJ0_POS0=0.33, 0.12, 0.08
DJ0_POS1=0.10, 0.13, -0.02
DJ1=UPPER_REAR
DJ1_BODY_A=CHASSIS
DJ1_BODY_B=HUB_L
DJ1_POS0=0.34, 0.12, -0.18
DJ1_POS1=0.10, 0.13, -0.02
DJ2=LOWER_FRONT
DJ2_BODY_A=CHASSIS
DJ2_BODY_B=HUB_L
DJ2_POS0=0.40, -0.11, 0.10
DJ2_POS1=0.07, -0.14, 0.01
DJ3=LOWER_REAR
DJ3_BODY_A=CHASSIS
DJ3_BODY_B=HUB_L
DJ3_POS0=0.41, -0.12, -0.27
DJ3_POS1=0.07, -0.14, 0.01
STEER_JOINT_0=TIE_ROD
STEER_JOINT_0_BODY_A=CHASSIS
STEER_JOINT_0_BODY_B=HUB_L
STEER_JOINT_0_POS0=0.50, -0.03, -0.19
STEER_JOINT_0_POS1=0.10, -0.04, -0.13
TOE_OUT=0.0005
STATIC_CAMBER=-2.0

[FRONT_COILOVER_0]
BODY_A=CHASSIS
BODY_B=HUB_L
POS_A=0.20, 0.40, -0.03
POS_B=0.07, -0.12, 0.01
MIN_LENGTH=0.40
MAX_LENGTH=0.60
BUMPSTOP_GAP=0.04
END_RATE=500000
RATE=60000
DAMP_BUMP=5000
DAMP_FAST_BUMP=2500
DAMP_FAST_BUMPTHRESHOLD=0.10
DAMP_REBOUND=7000
DAMP_FAST_REBOUND=4000
DAMP_FAST_REBOUNDTHRESHOLD=0.12
"""


def replace_section(path, name, text):
    """Put ``text`` in place of the ``[name]`` section of an ini file."""
    old = Path(path).read_text()
    new = re.sub(rf'\[{name}\].*?(?=^\[|\Z)', lambda _: text.rstrip()
                 + '\n', old, flags=re.S | re.M)
    if new == old:
        raise ValueError(f'no section [{name}] in {path}')
    Path(path).write_text(new)


def cosmic_fixture(path, front=False):
    """The fixture with a COSMIC rear (and front)."""
    write_fixture(path)
    ini = Path(path) / 'suspensions.ini'
    replace_section(ini, 'REAR', REAR)
    if front:
        replace_section(ini, 'FRONT', FRONT)


def sweep(model):
    return kc.bump_sweep(multibody, CornerModel(model),
                         [-0.02, -1e-3, 0.0, 1e-3, 0.02])


class CosmicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dwb = Path(cls.tmp.name) / 'dwb'
        cls.cosmic = Path(cls.tmp.name) / 'cosmic'
        for folder in (cls.dwb, cls.cosmic):
            folder.mkdir()
        write_fixture(cls.dwb)
        cosmic_fixture(cls.cosmic, front=True)
        cls.rear_dwb = sc.candidate('rear', cls.dwb)
        cls.rear = sc.candidate('rear', cls.cosmic)
        cls.front = sc.candidate('front', cls.cosmic)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_links_make_the_dwb_kinematics(self):
        self.assertEqual('double_wishbone', self.rear['type'])
        self.assertEqual(self.rear_dwb['links'], self.rear['links'])
        a, b = sweep(self.rear_dwb), sweep(self.rear)
        for key in ('camber', 'toe', 'caster', 'kpi', 'roll_center',
                    'motion_ratio'):
            np.testing.assert_allclose(b[key], a[key], rtol=0, atol=1e-12,
                                       err_msg=key)

    def test_the_coilover_rates_are_on_its_axis(self):
        self.assertAlmostEqual(self.rear['wheel_rates']['motion_ratio'], 1.0,
                               places=12)
        line = 40000.0 * 400000.0 / 440000.0
        res = sweep(self.rear)
        self.assertAlmostEqual(float(res['wheel_rate'][2]), line,
                               delta=1e-6 * line)
        self.assertAlmostEqual(self.rear['wheel_rates']['spring'], line,
                               delta=1e-6 * line)
        v, f = (np.asarray(x) for x in damper_table(
            self.rear['spring']['damper']))
        np.testing.assert_allclose(np.diff(f) / np.diff(v),
                                   [3500, 4000, 4500, 4000], rtol=1e-12)
        np.testing.assert_allclose(v[[1, 3]], [-0.10, 0.12], rtol=1e-12)
        # 300 mm long: the bump stop at 230 mm, the droop stop at 360 mm.
        stops = self.rear['stops']
        self.assertAlmostEqual(stops['bump_travel'], 70.0, places=9)
        self.assertAlmostEqual(stops['droop_travel'], -60.0, places=9)
        self.assertAlmostEqual(stops['k'], 500.0, places=12)
        self.assertTrue(any('INSTALL_RATE' in a
                            for a in self.rear['assumptions']))

    def test_a_front_axle_steers_on_the_rack(self):
        self.assertEqual('double_wishbone', self.front['type'])
        self.assertIn('rack', self.front['links']['tie_rod'])
        self.assertIn('rack', self.front)
        self.assertEqual(self.front['spring']['chassis'],
                         [1000.0 * -0.03, 1000.0 * (0.78 - 0.20),
                          1000.0 * (0.32 + 0.40)])
        res = sweep(self.front)
        self.assertAlmostEqual(math.degrees(float(res['camber'][2])), -2.0,
                               places=6)
        # TOE_OUT > 0 is toe-out: the toe metric is negative.
        self.assertLess(float(res['toe'][2]), 0.0)
        mr = self.front['wheel_rates']['motion_ratio']
        self.assertLess(mr, 1.0)
        self.assertAlmostEqual(float(res['wheel_rate'][2]),
                               self.front['wheel_rates']['spring'],
                               delta=0.01 * 60000.0 * mr * mr)

    def test_the_vehicle_takes_the_car_name(self):
        car = Path(self.cosmic) / 'car.ini'
        text = car.read_text()
        try:
            car.write_text('[INFO]\nSCREEN_NAME=Test Cup\n' + text)
            data = vc.vehicle_data(self.cosmic)
        finally:
            car.write_text(text)
        self.assertEqual('Test Cup (AC data)', data['name'])
        self.assertEqual('Test Cup rear (AC data)', data['rear']['name'])

    def test_unequal_relaxation_lengths_take_the_mean(self):
        tyres = Path(self.cosmic) / 'tyres.ini'
        text = tyres.read_text()
        head, rear = text.split('[REAR]')
        try:
            tyres.write_text(head + '[REAR]' + rear.replace(
                'RELAXATION_LENGTH=0.3', 'RELAXATION_LENGTH=0.4'))
            block, notes = vc.tire_data(self.cosmic)
        finally:
            tyres.write_text(text)
        self.assertAlmostEqual(block['relaxation_length'], 0.35, places=12)
        self.assertTrue(any('RELAXATION_LENGTH' in n for n in notes))


class CosmicRejectTest(unittest.TestCase):
    def _rejects(self, old, new, message):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(tmp)
            replace_section(Path(tmp) / 'suspensions.ini', 'REAR',
                            REAR.replace(old, new))
            with self.assertRaisesRegex(ValueError, message):
                sc.geometry('rear', tmp)

    def test_three_links_are_not_two_wishbones(self):
        self._rejects('DJ3=UPPER_REAR', 'XX3=UPPER_REAR', 'four DJ links')

    def test_a_link_to_another_body(self):
        self._rejects('DJ1_BODY_B=HUB_L', 'DJ1_BODY_B=ROCKER',
                      'only CHASSIS to HUB_L')

    def test_no_steering_joint(self):
        self._rejects('STEER_JOINT_0=TOE', 'STEER_X=TOE',
                      'exactly one STEER_JOINT')

    def test_a_coilover_outside_its_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(tmp)
            replace_section(Path(tmp) / 'suspensions.ini', 'REAR',
                            REAR.replace('MAX_LENGTH=0.36',
                                         'MAX_LENGTH=0.29'))
            with self.assertRaisesRegex(ValueError, 'MAX_LENGTH'):
                sc.candidate('rear', tmp)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
