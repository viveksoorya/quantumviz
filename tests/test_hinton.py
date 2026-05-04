"""Tests for Hinton visualization."""
import pytest
import numpy as np

from quantumviz.hinton import plot_hinton, state_to_density


class TestStateToDensity:
    """Test statevector to density matrix conversion."""

    def test_basic_conversion(self):
        """Test conversion of Bell state."""
        bell = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]
        rho = state_to_density(bell)
        assert rho.shape == (4, 4)
        assert abs(np.trace(rho) - 1.0) < 1e-9


class TestPlotHinton:
    """Test Hinton plotting."""

    def test_plot_bell_state(self, tmp_path):
        """Test plotting Bell state Hinton diagram."""
        bell = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]
        output = tmp_path / "hinton_bell.png"
        result = plot_hinton(bell, "Bell Hinton", str(output), dpi=150)
        assert result is None  # When saving to file
        assert output.exists()

    def test_plot_density_matrix(self, tmp_path):
        """Test plotting density matrix."""
        bell = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]
        rho = state_to_density(bell)
        output = tmp_path / "hinton_rho.png"
        result = plot_hinton(rho, "Density Matrix Hinton", str(output), dpi=150)
        assert result is None
        assert output.exists()

    def test_plot_from_qiskit(self, tmp_path):
        """Test with Qiskit Statevector if available."""
        try:
            from qiskit.quantum_info import Statevector
            sv = Statevector.from_label('00') + Statevector.from_label('11')
            sv = sv / np.sqrt(2)
            output = tmp_path / "hinton_qiskit.png"
            plot_hinton(sv, "Qiskit Hinton", str(output), dpi=150)
            assert output.exists()
        except ImportError:
            pytest.skip("Qiskit not installed")
