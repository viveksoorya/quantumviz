"""Tests for PauliVec visualization."""
import pytest
import numpy as np

from quantumviz.paulivec import pauli_expectations, plot_paulivec


class TestPauliExpectations:
    """Test Pauli expectation computation."""

    def test_bell_state_2qubits(self):
        """Test with Bell state |00⟩+|11⟩ / √2."""
        bell = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        labels, values = pauli_expectations(bell, n_qubits=2)
        
        assert len(labels) == 16  # 4^2
        assert len(values) == 16
        assert -1.0 <= values.min() <= 1.0
        assert -1.0 <= values.max() <= 1.0

    def test_ghz_state_3qubits(self):
        """Test with GHZ state."""
        ghz = np.zeros(8, dtype=complex)
        ghz[0] = 1/np.sqrt(2)
        ghz[7] = 1/np.sqrt(2)
        labels, values = pauli_expectations(ghz, n_qubits=3)
        
        assert len(labels) == 64  # 4^3
        assert len(values) == 64

    def test_single_qubit_zero_state(self):
        """Test with |0⟩ state."""
        # |0⟩ state
        sv = np.array([1, 0], dtype=complex)
        labels, values = pauli_expectations(sv, n_qubits=1)
        
        assert len(labels) == 4  # 4^1
        assert len(values) == 4
        # For |0⟩: ⟨Z⟩ = 1, ⟨I⟩ = 1, ⟨X⟩ = ⟨Y⟩ = 0
        assert abs(values[0] - 1.0) < 1e-9  # I
        assert abs(values[3] - 1.0) < 1e-9  # Z


class TestPlotPauliVec:
    """Test PauliVec plotting."""

    def test_plot_bell(self, tmp_path):
        """Test plotting Bell state."""
        bell = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        output = tmp_path / "paulivec_bell.png"
        result = plot_paulivec(bell, "Bell PauliVec", str(output), dpi=150)
        assert result is None  # When saving to file
        assert output.exists()

    def test_plot_from_statevector(self, tmp_path):
        """Test with Qiskit Statevector if available."""
        try:
            from qiskit.quantum_info import Statevector
            sv = Statevector.from_label('00') + Statevector.from_label('11')
            sv = sv / np.sqrt(2)
            output = tmp_path / "paulivec_qiskit.png"
            plot_paulivec(sv, "Qiskit PauliVec", str(output), dpi=150)
            assert output.exists()
        except ImportError:
            pytest.skip("Qiskit not installed")
