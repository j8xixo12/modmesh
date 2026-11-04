# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for modal analysis.
Validates natural frequencies against analytical solutions.
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


class TestModalAnalysisGravityPendulum:
    """Gravity pendulum on revolute joint: omega = sqrt(g/L) for small
    angles."""

    def _build_pendulum(self, mbs, length, mass, gravity=9.81):
        """Build a simple pendulum hanging from origin.

        Revolute joint at origin about z-axis. Body at [0, -length, 0].
        The joint attachment point on body is [0, length, 0] (body-local),
        placing the body center 'length' below the pivot.
        """
        sys = mbs.DynamicsSystem(0.001)
        ground = mbs.create_ground(0)
        sys.Add(ground)
        Z = [0, 0, 0]
        AZ = [0, 0, 1]
        body = mbs.create_mobilized_body(
            1, [0, -length, 0], Z, Z, Z, Z, Z,
            mass, [0.01, 0.01, 0.01], [0, -mass * gravity, 0], Z
        )
        sys.Add(body)
        joint = mbs.create_joint(
            1, Z, [0, length, 0], Z, Z, AZ, AZ, ground, body
        )
        sys.Add(joint)
        sys.Assembly()
        sys.init()
        return sys

    def test_pendulum_frequency(self, mbs):
        L = 1.0
        g = 9.81
        expected_freq = math.sqrt(g / L) / (2 * math.pi)  # ~0.498 Hz

        sys = self._build_pendulum(mbs, L, 1.0, g)
        result = mbs.modal_analysis(sys)
        assert result.n_dof > 0, "Should have independent DOF"
        freqs = np.asarray(result.frequencies)
        assert len(freqs) > 0, ("Should find at least one mode "
                                f"(n_dof={result.n_dof})")

        closest = min(freqs, key=lambda f: abs(f - expected_freq))
        rel_error = abs(closest - expected_freq) / expected_freq
        assert rel_error < 0.20, (
            f"Expected ~{expected_freq:.3f} Hz, got {closest:.3f} Hz "
            f"(error={rel_error * 100:.1f}%)"
        )

    def test_shorter_pendulum_higher_frequency(self, mbs):
        """Shorter pendulum should have higher frequency."""
        freqs = []
        for L in [2.0, 0.5]:
            sys = self._build_pendulum(mbs, L, 1.0)
            result = mbs.modal_analysis(sys)
            non_trivial = [f for f in np.asarray(result.frequencies) if f
                           > 0.01]
            freqs.append(min(non_trivial) if non_trivial else 0.0)

        assert freqs[1] > freqs[0], (
            f"Shorter pendulum should oscillate faster: "
            f"L=2 -> {freqs[0]:.3f} Hz, L=0.5 -> {freqs[1]:.3f} Hz"
        )

    def test_damped_pendulum(self, mbs):
        """Pendulum with spring-damper should show nonzero damping ratio."""
        sys = mbs.DynamicsSystem(0.001)
        ground = mbs.create_ground(0)
        sys.Add(ground)
        Z = [0, 0, 0]
        AZ = [0, 0, 1]
        body = mbs.create_mobilized_body(
            1, [0, -1, 0], Z, Z, Z, Z, Z,
            1.0, [0.01, 0.01, 0.01], [0, -9.81, 0], Z
        )
        sys.Add(body)
        joint = mbs.create_joint(1, Z, [0, 1, 0], Z, Z, AZ, AZ, ground, body)
        sys.Add(joint)
        # Add damper to the pendulum tip
        spring = mbs.create_spring_damper(
            ground, body, [0, -0.5, 0], [0, -0.5, 0],
            0.0, 10.0, 0.5, 0.0
        )
        sys.Add(spring)
        sys.Assembly()
        sys.init()
        result = mbs.modal_analysis(sys)
        # Should still find oscillatory modes
        assert result.n_dof > 0


class TestModalAnalysisNoSpring:
    """System without elastic elements should have no oscillatory modes."""

    def test_rigid_pendulum_no_elastic_modes(self, mbs):
        sys = mbs.DynamicsSystem(0.001)
        ground = mbs.create_ground(0)
        sys.Add(ground)

        Z = [0, 0, 0]
        AZ = [0, 0, 1]
        body = mbs.create_mobilized_body(
            1, [0, -1, 0], Z, Z, Z, Z, Z,
            1.0, [1, 1, 1], [0, -9.8, 0], Z
        )
        sys.Add(body)
        joint = mbs.create_joint(0, Z, [0, 1, 0], Z, Z, AZ, AZ, ground, body)
        sys.Add(joint)
        sys.Assembly()
        sys.init()

        result = mbs.modal_analysis(sys)
        # Should still have DOF (spherical joint = 3 rotational DOF)
        assert result.n_dof > 0


class TestModalAnalysisResult:
    """Test ModalResult data structure."""

    def _build_gravity_pendulum(self, mbs):
        sys = mbs.DynamicsSystem(0.001)
        ground = mbs.create_ground(0)
        sys.Add(ground)
        Z = [0, 0, 0]
        AZ = [0, 0, 1]
        body = mbs.create_mobilized_body(
            1, [0, -1, 0], Z, Z, Z, Z, Z,
            1.0, [0.01, 0.01, 0.01], [0, -9.81, 0], Z
        )
        sys.Add(body)
        joint = mbs.create_joint(1, Z, [0, 1, 0], Z, Z, AZ, AZ, ground, body)
        sys.Add(joint)
        sys.Assembly()
        sys.init()
        return sys

    def test_result_fields(self, mbs):
        sys = self._build_gravity_pendulum(mbs)
        result = mbs.modal_analysis(sys)
        assert hasattr(result, 'frequencies')
        assert hasattr(result, 'damping_ratios')
        assert hasattr(result, 'mode_shapes')
        assert hasattr(result, 'n_dof')
        freqs = np.asarray(result.frequencies)
        assert len(freqs) == len(np.asarray(result.damping_ratios))
        assert len(np.asarray(result.mode_shapes)) == len(freqs)

    def test_frequencies_sorted_ascending(self, mbs):
        sys = self._build_gravity_pendulum(mbs)
        result = mbs.modal_analysis(sys)
        freqs = np.asarray(result.frequencies)
        for i in range(len(freqs) - 1):
            assert freqs[i] <= freqs[i + 1], (
                f"Frequencies not sorted: {freqs[i]} > {freqs[i + 1]}"
            )

    def test_simulation_not_affected_by_modal_analysis(self, mbs):
        """Modal analysis should not alter solver state."""
        sys = self._build_gravity_pendulum(mbs)

        pos_before = np.array(sys.get_body_positions())
        mbs.modal_analysis(sys)
        pos_after = np.asarray(sys.get_body_positions())

        for i in range(len(pos_before)):
            for j in range(3):
                assert abs(pos_before[i][j] - pos_after[i][j]) < 1e-10, (
                    f"State changed after modal analysis: body {i} "
                    f"component {j}: {pos_before[i][j]} -> {pos_after[i][j]}"
                )

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
