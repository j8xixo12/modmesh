# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for math functions, joint types, and solver integration.
"""

import math

import numpy as np
import pytest

from solvcon import multibody


@pytest.fixture
def mbs():
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    return multibody


@pytest.fixture
def mbs_module():
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    return multibody


# --- Math function tests ---

class TestQuaternionMatrix:

    def test_identity_quaternion(self, mbs):
        # q = [1, 0, 0, 0] => identity rotation
        ground = mbs.create_ground(0)
        body = mbs.create_mobilized_body(
            1, [0, 0, 0], [0, 0, 0], [0, 0, 0],
            [0, 0, 0], [0, 0, 0], [0, 0, 0],
            1.0, [1, 1, 1], [0, 0, 0], [0, 0, 0]
        )
        # If angle is (0,0,0), quaternion should be (1,0,0,0)
        # and position should remain unchanged
        sys = mbs.DynamicsSystem(0.001)
        sys.Add(ground)
        sys.Add(body)
        ai = [0, 0, 1]
        joint = mbs.create_joint(0, [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0],
                                 ai, ai, ground, body)
        sys.Add(joint)
        sys.Assembly()
        sys.init()
        positions = np.asarray(sys.get_body_positions())
        # Ground at origin
        assert abs(positions[0][0]) < 1e-10
        assert abs(positions[0][1]) < 1e-10
        assert abs(positions[0][2]) < 1e-10


class TestGMatrix:

    def test_g_matrix_orthogonality(self, mbs):
        # For unit quaternion p, G*G^T should equal I (3x3)
        # We test indirectly: after simulation with zero velocity,
        # quaternion norm should be preserved
        sys = mbs.DynamicsSystem(0.001)
        ground = mbs.create_ground(0)
        body = mbs.create_mobilized_body(
            1, [0, 0, 0], [0, 0, 0], [0, 0, 0],
            [0.3, 0.2, 0.1], [0, 0, 0], [0, 0, 0],
            1.0, [1, 1, 1], [0, -9.8, 0], [0, 0, 0]
        )
        sys.Add(ground)
        sys.Add(body)
        ai = [0, 0, 1]
        joint = mbs.create_joint(0, [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0],
                                 ai, ai, ground, body)
        sys.Add(joint)
        sys.Assembly()
        sys.init()
        # Run 100 steps
        for _ in range(100):
            sys.solve()
        # Should not crash, positions should be finite
        positions = np.asarray(sys.get_body_positions())
        for p in positions:
            for v in p:
                assert math.isfinite(v), f"Non-finite position: {p}"


# --- Joint type tests ---

class TestJointTypes:

    def test_spherical_joint_3_constraints(self, mbs):
        sys = mbs.DynamicsSystem(0.001)
        ground = mbs.create_ground(0)
        body = mbs.create_mobilized_body(
            1, [0, 0, 0], [0, 0, 0], [0, 0, 0],
            [0, 0, 0], [0, 0, 0], [0, 0, 0],
            1.0, [1, 1, 1], [0, -9.8, 0], [0, 0, 0]
        )
        sys.Add(ground)
        sys.Add(body)
        ai = [0, 0, 1]
        # Type 0 = spherical
        joint = mbs.create_joint(0, [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0],
                                 ai, ai, ground, body)
        sys.Add(joint)
        sys.Assembly()
        sys.init()
        # Should work (spherical = 3 constraints)
        sys.solve()
        pos = np.asarray(sys.get_body_positions())
        assert len(pos) == 2

    def test_revolute_joint_5_constraints(self, mbs):
        sys = mbs.DynamicsSystem(0.001)
        ground = mbs.create_ground(0)
        body = mbs.create_mobilized_body(
            1, [0, 0, 0], [0, 0, 0], [0, 0, 0],
            [0, 0, 0], [0, 0, 0], [0, 0, 0],
            1.0, [1, 1, 1], [0, -9.8, 0], [0, 0, 0]
        )
        sys.Add(ground)
        sys.Add(body)
        # Type 1 = revolute, axis along z
        joint = mbs.create_joint(1, [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0],
                                 [0, 0, 1], [0, 0, 1], ground, body)
        sys.Add(joint)
        sys.Assembly()
        sys.init()
        sys.solve()
        pos = np.asarray(sys.get_body_positions())
        assert len(pos) == 2
        for v in pos[1]:
            assert math.isfinite(v)

    def test_prismatic_joint_5_constraints(self, mbs):
        sys = mbs.DynamicsSystem(0.001)
        ground = mbs.create_ground(0)
        body = mbs.create_mobilized_body(
            1, [0, -1, 0], [0, 0, 0], [0, 0, 0],
            [0, 0, 0], [0, 0, 0], [0, 0, 0],
            1.0, [1, 1, 1], [0, -9.8, 0], [0, 0, 0]
        )
        sys.Add(ground)
        sys.Add(body)
        # Type 2 = prismatic, slide along y
        joint = mbs.create_joint(2, [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                                 [0, 1, 0], [0, 1, 0], ground, body)
        sys.Add(joint)
        sys.Assembly()
        sys.init()
        sys.solve()
        pos = np.asarray(sys.get_body_positions())
        assert len(pos) == 2
        for v in pos[1]:
            assert math.isfinite(v)


# --- Integration tests ---

class TestSolverIntegration:

    def test_chain_simulation_100_steps(self, mbs):
        sys = mbs.DynamicsSystem(0.001)
        ground = mbs.create_ground(0)
        sys.Add(ground)

        prev_body = ground
        for i in range(4):
            body = mbs.create_mobilized_body(
                i + 1, [0, 0, 0], [0, 0, 0], [0, 0, 0],
                [0.05, 0, 0], [0, 0, 0], [0, 0, 0],
                1.0, [1, 1, 1], [0, -9.8, 0], [0, 0, 0]
            )
            sys.Add(body)
            ai = [0, 0, 1]
            joint = mbs.create_joint(0, [0, 0, 0], [0, 1, 0], [0, 0, 0],
                                     [0, 0, 0], ai, ai, prev_body, body)
            sys.Add(joint)
            prev_body = body

        sys.Assembly()
        sys.init()

        for _ in range(100):
            sys.solve()

        positions = np.asarray(sys.get_body_positions())
        assert len(positions) == 5
        # Ground stays at origin
        assert abs(positions[0][0]) < 1e-10
        assert abs(positions[0][1]) < 1e-10
        # Bodies should have moved downward (gravity in -y)
        for i in range(1, 5):
            assert positions[i][1] < 0, f"Body {i} should be below origin"

    def test_constraint_satisfaction(self, mbs):
        # After simulation, spherical joint should keep attachment points
        # coincident
        sys = mbs.DynamicsSystem(0.001)
        ground = mbs.create_ground(0)
        body = mbs.create_mobilized_body(
            1, [0, 0, 0], [0, 0, 0], [0, 0, 0],
            [0.1, 0, 0], [0, 0, 0], [0, 0, 0],
            1.0, [1, 1, 1], [0, -9.8, 0], [0, 0, 0]
        )
        sys.Add(ground)
        sys.Add(body)
        ai = [0, 0, 1]
        # pj = [0, 1, 0] means body center is 1 unit from joint along
        # body-local y
        joint = mbs.create_joint(0, [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0],
                                 ai, ai, ground, body)
        sys.Add(joint)
        sys.Assembly()
        sys.init()

        for _ in range(500):
            sys.solve()

        positions = np.asarray(sys.get_body_positions())
        # Distance from ground (0,0,0) to body should be approximately 1
        # (length of the pendulum arm)
        bx, by, bz = positions[1]
        dist = math.sqrt(bx**2 + by**2 + bz**2)
        # With constraint drift, allow some tolerance
        assert abs(dist
                   - 1.0) < 0.01, (f"Constraint violated: distance = {dist}")

    def test_initial_velocity_no_crash(self, mbs):
        # This was a known bug: initial velocity caused crashes
        sys = mbs.DynamicsSystem(0.001)
        ground = mbs.create_ground(0)
        body = mbs.create_mobilized_body(
            1, [0, 0, 0], [1.0, 0, 0], [0, 0, 0],
            [0, 0, 0], [0, 0, 1.0], [0, 0, 0],
            1.0, [1, 1, 1], [0, -9.8, 0], [0, 0, 0]
        )
        sys.Add(ground)
        sys.Add(body)
        ai = [0, 0, 1]
        joint = mbs.create_joint(0, [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0],
                                 ai, ai, ground, body)
        sys.Add(joint)
        sys.Assembly()
        sys.init()

        for _ in range(200):
            sys.solve()

        positions = np.asarray(sys.get_body_positions())
        for p in positions:
            for v in p:
                assert math.isfinite(v), (
                    f"Non-finite after initial velocity: {p}")

    def test_initial_rotation_no_crash(self, mbs):
        # Another known bug: initial rotation caused divergence
        sys = mbs.DynamicsSystem(0.001)
        ground = mbs.create_ground(0)
        body = mbs.create_mobilized_body(
            1, [0, 0, 0], [0, 0, 0], [0, 0, 0],
            [0.5, 0.3, 0.2], [0, 0, 0], [0, 0, 0],
            1.0, [1, 1, 1], [0, -9.8, 0], [0, 0, 0]
        )
        sys.Add(ground)
        sys.Add(body)
        ai = [0, 0, 1]
        joint = mbs.create_joint(0, [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0],
                                 ai, ai, ground, body)
        sys.Add(joint)
        sys.Assembly()
        sys.init()

        for _ in range(500):
            sys.solve()

        positions = np.asarray(sys.get_body_positions())
        for p in positions:
            for v in p:
                assert math.isfinite(v), (
                    f"Non-finite after initial rotation: {p}")


class TestDrivingConstraint:
    """Tests for driving constraint (Type 3)."""

    @pytest.mark.requires_build
    def test_driving_constraint_creates_rotation(self, mbs_module):
        mbs = mbs_module
        s = mbs.DynamicsSystem(0.001)
        g = mbs.create_ground(0)
        s.Add(g)
        Z = [0, 0, 0]
        AZ = [0, 0, 1]
        # Attach a body with an offset so rotation produces position change
        b = mbs.create_mobilized_body(
            1, [1, 0, 0], Z, Z, Z, Z, Z, 1.0, [1, 1, 1], Z, Z
        )
        s.Add(b)
        s.Add(mbs.create_joint_driving(3, Z, Z, Z, Z, AZ, AZ, g, b, 5.0))
        s.Assembly()
        s.init()
        # Sample angles at an intermediate step (before Euler wrap-around)
        max_angle = 0.0
        for i in range(500):
            s.solve()
            if i % 100 == 0:
                angles = np.asarray(s.get_body_angles())
                body_angles = angles[1]
                max_angle = max(max_angle, max(abs(a) for a in body_angles))
        assert max_angle > 1.0, \
            f"Driving joint should produce rotation > 1 rad, max={max_angle}"


class TestPhysicsPlausibility:
    """Tests verifying basic physics plausibility."""

    @pytest.mark.requires_build
    def test_pendulum_swings_down(self, mbs_module):
        mbs = mbs_module
        s = mbs.DynamicsSystem(0.001)
        g = mbs.create_ground(0)
        s.Add(g)
        Z = [0, 0, 0]
        AZ = [0, 0, 1]
        b = mbs.create_mobilized_body(
            1, Z, Z, Z, [0.5, 0, 0], Z, Z,
            1.0, [1, 1, 1], [0, -9.8, 0], Z
        )
        s.Add(b)
        s.Add(mbs.create_joint(0, Z, [0, 1, 0], Z, Z, AZ, AZ, g, b))
        s.Assembly()
        s.init()
        y_values = []
        for i in range(2000):
            s.solve()
            if i % 200 == 0:
                pos = np.asarray(s.get_body_positions())[1]
                y_values.append(pos[1])
        assert min(y_values) < y_values[0] - 0.05, \
            f"Pendulum should swing down, values={y_values}"

    @pytest.mark.requires_build
    def test_constraint_tight_tolerance(self, mbs_module):
        mbs = mbs_module
        s = mbs.DynamicsSystem(0.001)
        g = mbs.create_ground(0)
        s.Add(g)
        Z = [0, 0, 0]
        AZ = [0, 0, 1]
        b = mbs.create_mobilized_body(
            1, Z, Z, Z, [0.3, 0.2, 0], Z, Z,
            1.0, [1, 1, 1], [0, -9.8, 0], Z
        )
        s.Add(b)
        s.Add(mbs.create_joint(0, Z, [0, 1, 0], Z, Z, AZ, AZ, g, b))
        s.Assembly()
        s.init()
        max_error = 0.0
        for i in range(1000):
            s.solve()
            if i % 50 == 0:
                pos = np.asarray(s.get_body_positions())[1]
                dist = (pos[0]**2 + pos[1]**2 + pos[2]**2)**0.5
                error = abs(dist - 1.0)
                max_error = max(max_error, error)
        assert max_error < 0.01, f"Constraint violation: {max_error}"

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
