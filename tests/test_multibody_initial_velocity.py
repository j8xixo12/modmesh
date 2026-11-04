# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for consistent initial velocity computation in ``init()``.

Rules under test:

* If a user gives a non-zero velocity vector for a body, then ``init()``
  keeps that vector, provided the vector is compatible with the joints.
* If a user gives a non-zero angular velocity vector for a body, then
  ``init()`` keeps that vector, provided the vector is compatible with
  the joints.
* Every initial velocity after ``init()`` satisfies the joint velocity
  equations exactly, even if the user input conflicts with the joints.
"""


import numpy as np
import pytest

from solvcon import multibody


Z = [0.0, 0.0, 0.0]
ARM = np.array([1.0, 0.0, 0.0])  # COM position; the joint is at the origin


@pytest.fixture
def mbs():
    """Import the compiled solver module or skip.

    :return: compiled solver module
    :rtype: module
    """
    if not multibody.HAS_MULTIBODY:
        pytest.skip("solvcon is built without the multibody engine")
    return multibody


def build_pendulum(mbs, vel, omega, use_assembly=False):
    """Build a gravity-free spherical pendulum and run ``init()``.

    The joint is at the origin. The body COM is at (1, 0, 0).

    :param mbs: compiled solver module
    :type mbs: module
    :param vel: initial COM velocity given by the user
    :type vel: list
    :param omega: initial angular velocity (inertial frame) given by the user
    :type omega: list
    :param use_assembly: call ``Assembly()`` before ``init()``
    :type use_assembly: bool
    :return: (velocity, angular velocity) of the body after ``init()``
    :rtype: tuple
    """
    s = mbs.DynamicsSystem(1e-4)
    ground = mbs.create_ground(0)
    s.Add(ground)
    body = mbs.create_mobilized_body_quat(
        1, list(ARM), vel, Z, [1.0, 0.0, 0.0, 0.0], omega, Z,
        1.0, [0.1, 0.1, 0.1], Z, Z)
    s.Add(body)
    s.Add(mbs.create_joint(0, Z, [-1.0, 0.0, 0.0], Z, Z,
                           [0, 0, 1], [0, 0, 1], ground, body))
    if use_assembly:
        s.Assembly()
    s.init()
    v = np.array(s.get_body_velocities())[1]
    w = np.array(s.get_body_angular_velocities())[1]
    return v, w


@pytest.mark.parametrize("use_assembly", [False, True])
@pytest.mark.parametrize(
    "vel, omega, expected_v, expected_w",
    [
        # Only omega given: omega is kept, v follows from omega x r.
        (Z, [0, 0, 1], [0, 1, 0], [0, 0, 1]),
        # Only v given: v is kept, omega is the smallest compatible one.
        ([0, 1, 0], Z, [0, 1, 0], [0, 0, 1]),
        # Compatible pair: both are kept.
        ([0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]),
        # Spin about the arm axis is compatible with v = 0.
        (Z, [2, 0, 0], [0, 0, 0], [2, 0, 0]),
    ],
    ids=["omega_only", "vel_only", "consistent_pair", "axial_spin"],
)
def test_init_keeps_user_velocity(mbs, vel, omega, expected_v, expected_w,
                                  use_assembly):
    """Test that ``init()`` keeps compatible user-given velocities."""
    v, w = build_pendulum(mbs, vel, omega, use_assembly)
    np.testing.assert_allclose(v, expected_v, atol=1e-10)
    np.testing.assert_allclose(w, expected_w, atol=1e-10)


def test_init_satisfies_joint_when_user_input_conflicts(mbs):
    """Test that the joint velocity equation wins over conflicting input."""
    v, w = build_pendulum(mbs, [0, 2, 0], [0, 0, 1])
    # Spherical joint at origin: v_com = omega x r_com.
    np.testing.assert_allclose(v, np.cross(w, ARM), atol=1e-10)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
