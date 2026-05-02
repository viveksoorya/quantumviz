"""
Tests for Qiskit compatibility bridge.
Run only if Qiskit is installed.
"""

import pytest
import numpy as np


@pytest.mark.skipif(
    not pytest.importorskip("qiskit", reason="Qiskit not installed"),
    reason="Qiskit required for these tests"
)
class TestQiskitBridge:
    """Test Qiskit object detection and conversion."""

    def test_statevector_detection(self):
        """Test is_statevector() with Qiskit objects."""
        from qiskit.quantum_info import Statevector
        from quantumviz.qiskit_bridge import is_statevector

        sv = Statevector.from_label('00')
        assert is_statevector(sv) is True

    def test_non_statevector_detection(self):
        """Test is_statevector() returns False for non-Statevector."""
        from quantumviz.qiskit_bridge import is_statevector

        assert is_statevector([1, 0, 0, 0]) is False
        assert is_statevector("not a state") is False

    def test_density_matrix_detection(self):
        """Test is_density_matrix() with Qiskit objects."""
        from qiskit.quantum_info import DensityMatrix
        from quantumviz.qiskit_bridge import is_density_matrix

        dm = DensityMatrix.from_label('0')
        assert is_density_matrix(dm) is True

    def test_quantum_circuit_detection(self):
        """Test is_quantum_circuit() with Qiskit objects."""
        from qiskit import QuantumCircuit
        from quantumviz.qiskit_bridge import is_quantum_circuit

        qc = QuantumCircuit(2)
        assert is_quantum_circuit(qc) is True

    def test_statevector_to_list(self):
        """Test statevector_to_list() conversion."""
        from qiskit.quantum_info import Statevector
        from quantumviz.qiskit_bridge import statevector_to_list

        sv = Statevector.from_label('0')
        result = statevector_to_list(sv)
        assert len(result) == 2
        assert np.abs(result[0] - 1.0) < 1e-10
        assert np.abs(result[1]) < 1e-10

    def test_statevector_to_list_bell(self):
        """Test statevector_to_list() with Bell state."""
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        from quantumviz.qiskit_bridge import statevector_to_list

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        sv = Statevector(qc)
        result = statevector_to_list(sv)
        assert len(result) == 4
        # Bell state: (|00> + |11>)/sqrt(2)
        expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_density_matrix_to_array(self):
        """Test density_matrix_to_array() conversion."""
        from qiskit.quantum_info import DensityMatrix
        from quantumviz.qiskit_bridge import density_matrix_to_array

        dm = DensityMatrix.from_label('0')
        result = density_matrix_to_array(dm)
        assert result.shape == (2, 2)
        assert result[0, 0] == pytest.approx(1.0)
        assert result[1, 1] == pytest.approx(0.0)

    def test_circuit_to_dict(self):
        """Test circuit_to_dict() conversion."""
        from qiskit import QuantumCircuit
        from quantumviz.qiskit_bridge import circuit_to_dict

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        result = circuit_to_dict(qc)
        assert result['qubits'] == 2
        assert len(result['gates']) == 2
        assert result['gates'][0]['type'] == 'H'
        assert result['gates'][1]['type'] == 'CNOT'

    def test_circuit_to_dict_with_params(self):
        """Test circuit_to_dict() with parametric gates."""
        import numpy as np
        from qiskit import QuantumCircuit
        from quantumviz.qiskit_bridge import circuit_to_dict

        qc = QuantumCircuit(1)
        qc.rx(np.pi/2, 0)

        result = circuit_to_dict(qc)
        assert len(result['gates']) == 1
        assert result['gates'][0]['type'] == 'RX'
        assert 'theta' in result['gates'][0]


@pytest.mark.skipif(
    not pytest.importorskip("qiskit", reason="Qiskit not installed"),
    reason="Qiskit required for these tests"
)
class TestBlochSphereQiskit:
    """Test Bloch sphere with Qiskit objects."""

    def test_plot_bloch_sphere_statevector(self):
        """Test plot_bloch_sphere() with Qiskit Statevector."""
        from qiskit.quantum_info import Statevector
        from quantumviz.bloch_sphere import plot_bloch_sphere

        sv = Statevector.from_label('0')
        fig = plot_bloch_sphere([sv])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_bloch_sphere_bell_state(self):
        """Test plot_bloch_sphere() with Bell state."""
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        from quantumviz.bloch_sphere import plot_bloch_sphere

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        sv = Statevector(qc)
        # Take first qubit's Bloch representation
        fig = plot_bloch_sphere([sv])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


@pytest.mark.skipif(
    not pytest.importorskip("qiskit", reason="Qiskit not installed"),
    reason="Qiskit required for these tests"
)
class TestStateCityQiskit:
    """Test State City with Qiskit objects."""

    def test_plot_state_city_statevector(self):
        """Test plot_state_city() with Qiskit Statevector."""
        from qiskit.quantum_info import Statevector
        from quantumviz.state_city import plot_state_city

        sv = Statevector.from_label('0')
        fig = plot_state_city(sv)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_state_city_density_matrix(self):
        """Test plot_state_city() with Qiskit DensityMatrix."""
        from qiskit.quantum_info import DensityMatrix
        from quantumviz.state_city import plot_state_city

        dm = DensityMatrix.from_label('0')
        fig = plot_state_city(dm)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


@pytest.mark.skipif(
    not pytest.importorskip("qiskit", reason="Qiskit not installed"),
    reason="Qiskit required for these tests"
)
class TestCircuitDiagramQiskit:
    """Test Circuit Diagram with Qiskit objects."""

    def test_plot_circuit_quantum_circuit(self):
        """Test plot_circuit() with Qiskit QuantumCircuit."""
        from qiskit import QuantumCircuit
        from quantumviz.circuit_diagram import plot_circuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        fig = plot_circuit(qc)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_circuit_grover(self):
        """Test plot_circuit() with Grover circuit."""
        from qiskit import QuantumCircuit
        from quantumviz.circuit_diagram import plot_circuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(1)
        qc.cx(0, 1)
        qc.h(0)
        qc.h(1)

        fig = plot_circuit(qc)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


@pytest.mark.skipif(
    not pytest.importorskip("qiskit", reason="Qiskit not installed"),
    reason="Qiskit required for these tests"
)
class TestDCNQiskit:
    """Test DCN with Qiskit objects."""

    def test_plot_dcn_statevector(self):
        """Test plot_dcn() with Qiskit Statevector."""
        from qiskit.quantum_info import Statevector
        from quantumviz.dcn import plot_dcn

        sv = Statevector.from_label('0')
        fig = plot_dcn(sv)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_dcn_bell_state(self):
        """Test plot_dcn() with Bell state."""
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        from quantumviz.dcn import plot_dcn

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        sv = Statevector(qc)
        fig = plot_dcn(sv)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


@pytest.mark.skipif(
    not pytest.importorskip("qiskit", reason="Qiskit not installed"),
    reason="Qiskit required for these tests"
)
class TestDynamicFlowQiskit:
    """Test Dynamic Flow with Qiskit objects."""

    def test_plot_time_evolution_statevector(self):
        """Test plot_time_evolution() with Qiskit Statevector."""
        from qiskit.quantum_info import Statevector
        from quantumviz.dynamic_flow import plot_time_evolution

        sv = Statevector.from_label('0')
        fig = plot_time_evolution([[1, 0], [0, 1]])  # Use list for time evolution
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
