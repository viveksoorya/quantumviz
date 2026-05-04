"""Tests for Bloch Multivector visualization."""
import pytest
import numpy as np

from quantumviz.bloch_multivector import (
    partial_trace,
    bloch_vector_from_rho,
    purity,
    plot_bloch_multivector,
)


class TestPartialTrace:
    """Test partial trace computation."""

    def test_2qubit_trace_q0(self):
        """Trace out qubit 0 from Bell state."""
        bell = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        rho = np.outer(bell, bell.conj())
        rho_0 = partial_trace(rho, 0, 2)
        # Should be maximally mixed
        assert rho_0.shape == (2, 2)
        np.testing.assert_allclose(rho_0, np.eye(2)/2, atol=1e-9)

    def test_2qubit_trace_q1(self):
        """Trace out qubit 1 from Bell state."""
        bell = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        rho = np.outer(bell, bell.conj())
        rho_1 = partial_trace(rho, 1, 2)
        assert rho_1.shape == (2, 2)
        np.testing.assert_allclose(rho_1, np.eye(2)/2, atol=1e-9)


class TestBlochVector:
    """Test Bloch vector computation."""

    def test_pure_state_x(self):
        """|+⟩ state: Bloch vector should point to X."""
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        vec = bloch_vector_from_rho(rho)
        np.testing.assert_allclose(vec, [1.0, 0, 0], atol=1e-9)

    def test_pure_state_z(self):
        """|0⟩ state: Bloch vector should point to Z."""
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        vec = bloch_vector_from_rho(rho)
        np.testing.assert_allclose(vec, [0, 0, 1.0], atol=1e-9)

    def test_maximally_mixed(self):
        """I/2: Bloch vector should be zero."""
        rho = np.eye(2) / 2
        vec = bloch_vector_from_rho(rho)
        np.testing.assert_allclose(vec, [0, 0, 0], atol=1e-9)


class TestPurity:
    """Test purity computation."""

    def test_pure_state(self):
        """Pure state: purity = 1."""
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        assert abs(purity(rho) - 1.0) < 1e-9

    def test_maximally_mixed(self):
        """Maximally mixed: purity = 0.5."""
        rho = np.eye(2) / 2
        assert abs(purity(rho) - 0.5) < 1e-9


class TestPlotBlochMultivector:
    """Test Bloch Multivector plotting."""

    def test_plot_bell_state(self, tmp_path):
        """Test plotting Bell state."""
        bell = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        output = tmp_path / "bloch_mv_bell.png"
        result = plot_bloch_multivector(bell, "Bell Bloch Multivector", str(output), dpi=150)
        assert result is None  # When saving to file
        assert output.exists()

    def test_plot_from_qiskit(self, tmp_path):
        """Test with Qiskit Statevector if available."""
        try:
            from qiskit.quantum_info import Statevector
            sv = Statevector.from_label('00') + Statevector.from_label('11')
            sv = sv / np.sqrt(2)
            output = tmp_path / "bloch_mv_qiskit.png"
            plot_bloch_multivector(sv, "Qiskit Bloch Multivector", str(output), dpi=150)
            assert output.exists()
        except ImportError:
            pytest.skip("Qiskit not installed")
