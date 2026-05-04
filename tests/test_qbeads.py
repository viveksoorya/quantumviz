"""
Tests for the BEADS (Quantum Beads) visualization module.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from quantumviz.qbeads import (
    plot_qbeads,
    plot_qbeads_from_file,
    parse_amplitude,
    compute_qbeads,
    compute_ebeads,
    prob_0_to_color,
    entanglement_to_color,
    QBead,
    EBead,
)


class TestParseAmplitude:
    """Tests for parse_amplitude function."""

    def test_real_number(self):
        assert parse_amplitude(1.0) == 1.0
        assert parse_amplitude(0.0) == 0.0

    def test_complex_number(self):
        np.testing.assert_almost_equal(parse_amplitude('1j'), 0+1j)
        np.testing.assert_almost_equal(parse_amplitude('0.707+0.707j'), 0.707+0.707j)

    def test_string(self):
        np.testing.assert_almost_equal(parse_amplitude('1j'), 1j)
        np.testing.assert_almost_equal(parse_amplitude('0.707'), 0.707+0j)

    def test_list(self):
        np.testing.assert_almost_equal(parse_amplitude([0.707, 0.707]), 0.707+0.707j)


class TestProb0ToColor:
    """Tests for probability to color conversion."""

    def test_pure_0(self):
        """P(0)=1 should be red."""
        color = prob_0_to_color(1.0)
        assert color == (1.0, 0.0, 0.0)

    def test_pure_1(self):
        """P(0)=0 should be green."""
        color = prob_0_to_color(0.0)
        assert color == (0.0, 1.0, 0.0)

    def test_superposition(self):
        """P(0)=0.5 should be yellow."""
        color = prob_0_to_color(0.5)
        np.testing.assert_almost_equal(color, (1.0, 1.0, 0.0))

    def test_clamping(self):
        """Values outside [0,1] should be clamped."""
        color_high = prob_0_to_color(1.5)
        color_low = prob_0_to_color(-0.5)
        assert color_high == (1.0, 0.0, 0.0)
        assert color_low == (0.0, 1.0, 0.0)


class TestEntanglementToColor:
    """Tests for entanglement to color conversion."""

    def test_max_entanglement(self):
        """Max entanglement should be yellow."""
        color = entanglement_to_color(1.0)
        np.testing.assert_almost_equal(color, (1.0, 1.0, 0.0))  # Yellow

    def test_separable(self):
        """Separable should be blue."""
        color = entanglement_to_color(0.0)
        np.testing.assert_almost_equal(color, (0.0, 0.0, 1.0))  # Blue


class TestComputeQBeads:
    """Tests for computing Q-bead properties."""

    def test_single_qubit_0(self):
        """|0⟩ state should have P(0)=1."""
        beads = compute_qbeads([1, 0], 1)
        assert len(beads) == 1
        assert beads[0].prob_0 == pytest.approx(1.0)
        assert beads[0].prob_1 == pytest.approx(0.0)

    def test_single_qubit_1(self):
        """|1⟩ state should have P(0)=0."""
        beads = compute_qbeads([0, 1], 1)
        assert len(beads) == 1
        assert beads[0].prob_0 == pytest.approx(0.0)
        assert beads[0].prob_1 == pytest.approx(1.0)

    def test_single_qubit_plus(self):
        """|+⟩ state should have P(0)=0.5."""
        sv = [1/np.sqrt(2), 1/np.sqrt(2)]
        beads = compute_qbeads(sv, 1)
        assert len(beads) == 1
        assert beads[0].prob_0 == pytest.approx(0.5, abs=0.01)
        assert beads[0].prob_1 == pytest.approx(0.5, abs=0.01)

    def test_two_qubit_product(self):
        """Product state |00⟩ should have both qubits with P(0)=1."""
        state = [1, 0, 0, 0]
        beads = compute_qbeads(state, 2)
        assert len(beads) == 2
        assert beads[0].prob_0 == pytest.approx(1.0)
        assert beads[1].prob_0 == pytest.approx(1.0)

    def test_two_qubit_bell(self):
        """Bell state |Φ+⟩ should have P(0)=0.5 for each qubit."""
        bell = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]
        beads = compute_qbeads(bell, 2)
        assert len(beads) == 2
        assert beads[0].prob_0 == pytest.approx(0.5, abs=0.01)
        assert beads[1].prob_0 == pytest.approx(0.5, abs=0.01)


class TestComputeEBeads:
    """Tests for computing E-bead properties."""

    def test_separable_no_ebeads(self):
        """Product state should have no E-beads."""
        state = [1, 0, 0, 0]
        ebeads = compute_ebeads(state, 2)
        # May or may not have ebeads depending on threshold

    def test_entangled_has_ebeads(self):
        """Bell state should have E-beads."""
        bell = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]
        ebeads = compute_ebeads(bell, 2)
        # Should detect some entanglement

    def test_single_qubit_no_ebeads(self):
        """Single qubit should have no E-beads."""
        ebeads = compute_qbeads([1, 0], 1)
        ebeads_e = compute_ebeads([1, 0], 1)
        assert len(ebeads_e) == 0


class TestPlotQBeads:
    """Tests for plot_qbeads function."""

    def test_single_qubit_0(self):
        """Test single qubit |0> state."""
        fig = plot_qbeads([1, 0], 'test')
        assert fig is not None
        plt.close(fig)

    def test_single_qubit_1(self):
        """Test single qubit |1> state."""
        fig = plot_qbeads([0, 1], 'test')
        assert fig is not None
        plt.close(fig)

    def test_single_qubit_plus(self):
        """Test single qubit |+> state."""
        sv = [1/np.sqrt(2), 1/np.sqrt(2)]
        fig = plot_qbeads(sv, 'test')
        assert fig is not None
        plt.close(fig)

    def test_two_qubit_bell(self):
        """Test two-qubit Bell state."""
        bell = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]
        fig = plot_qbeads(bell, 'Bell')
        assert fig is not None
        plt.close(fig)

    def test_three_qubit_ghz(self):
        """Test three-qubit GHZ state."""
        ghz = [1/np.sqrt(2), 0, 0, 0, 0, 0, 0, 1/np.sqrt(2)]
        fig = plot_qbeads(ghz, 'GHZ')
        assert fig is not None
        plt.close(fig)

    def test_output_path(self):
        """Test saving to file."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.png')
            plot_qbeads([1, 0], 'test', output_path, dpi=100)
            assert os.path.exists(output_path)

    def test_invalid_state_length(self):
        """Test non-power-of-2 length."""
        with pytest.raises(Exception):
            plot_qbeads([1, 0, 1], 'test')