# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import math
import unittest

import numpy as np

from solvcon import multibody as mb


def _pendulum(theta0, dt=1e-3, length=1.0, mass=1.0):
    """Point-like rod pendulum about the global z axis, gravity along -y."""
    system = mb.DynamicsSystem(dt)
    ground = mb.create_ground(0)
    pos = [length * math.sin(theta0), -length * math.cos(theta0), 0.0]
    zero = [0.0, 0.0, 0.0]
    body = mb.create_mobilized_body(
        1, pos, zero, zero, zero, zero, zero, mass, [1e-6, 1e-6, 1e-6],
        [0.0, -9.81 * mass, 0.0], zero)
    system.Add(ground)
    system.Add(body)
    # Revolute joint: ground origin to the body center at the pivot.
    joint = mb.create_joint(1, zero, [-pos[0], -pos[1], 0.0], zero, zero,
                            [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], ground, body)
    system.Add(joint)
    system.init()
    return system


@unittest.skipUnless(mb.HAS_MULTIBODY, "multibody needs the vendor LAPACK")
class MultibodyTC(unittest.TestCase):

    def test_pendulum_small_angle_period(self):
        # The small-angle period is 2 pi sqrt(L / g); theta0 = 0.01 rad
        # changes it by 6e-6 relative, below the 1e-3 tolerance (dt = 1e-3).
        dt = 1e-3
        system = _pendulum(0.01, dt=dt)
        self.assertEqual(system.get_dof(), 1)
        xs = []
        for _ in range(3500):
            system.solve()
            xs.append(np.asarray(system.get_body_positions())[1][0])
        xs = np.array(xs)
        # Times of the downward zero crossings of x.
        idx = np.where((xs[:-1] > 0.0) & (xs[1:] <= 0.0))[0]
        t = (idx + 1 + xs[idx] / (xs[idx] - xs[idx + 1])) * dt
        period = t[1] - t[0]
        self.assertAlmostEqual(period, 2.0 * math.pi * math.sqrt(1 / 9.81),
                               delta=1e-3)

    def test_constraints_and_reactions(self):
        system = _pendulum(0.3)
        ev = system.evaluate_constraints(0.0)
        self.assertLess(np.abs(ev["C"]).max(), 1e-10)
        self.assertEqual(ev["Cq"].shape, (7 + 5 + 1, 14))
        reac = system.get_joint_reactions()
        self.assertEqual(len(reac), 1)
        # At rest the centripetal term is zero: the rod tension on the body
        # points to the pivot and balances gravity along the rod.
        f_body = np.asarray(reac[0]["force_j"])
        rod = np.array(np.asarray(system.get_body_positions())[1])
        rod /= np.linalg.norm(rod)
        np.testing.assert_allclose(np.dot(f_body, rod),
                                   -9.81 * math.cos(0.3), rtol=1e-8)

    def test_drive_function_and_cam_profile(self):
        f = mb.DriveFunction.sine(2.0, 3.0, 0.0, 1.0)
        c, cd, cdd = f.eval(0.5)
        self.assertAlmostEqual(c, 1.0 + 2.0 * math.sin(1.5), places=14)
        self.assertAlmostEqual(cd, 6.0 * math.cos(1.5), places=14)
        self.assertAlmostEqual(cdd, -18.0 * math.sin(1.5), places=13)
        phi = np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False)
        r = 1.0 + 0.2 * np.cos(phi) - 0.1 * np.sin(2.0 * phi)
        cam = mb.CamProfile.fit(list(phi), list(r), 2)
        self.assertAlmostEqual(cam.eval(0.7)[0],
                               1.0 + 0.2 * math.cos(0.7) - 0.1 * math.sin(1.4),
                               places=12)

    def test_modal_analysis_pendulum(self):
        system = _pendulum(0.0)
        res = mb.modal_analysis(system)
        self.assertEqual(res.n_dof, 1)
        self.assertEqual(len(res.frequencies), 1)
        # Linearized frequency sqrt(g / L) / (2 pi); the finite-difference
        # stiffness is accurate to about 1e-6 relative.
        self.assertAlmostEqual(res.frequencies[0],
                               math.sqrt(9.81) / (2.0 * math.pi), delta=1e-5)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
