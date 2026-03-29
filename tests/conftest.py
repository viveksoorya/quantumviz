"""
Pytest configuration and shared fixtures for quantumviz tests.
"""

import pytest
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(autouse=True)
def reset_matplotlib():
    """Reset matplotlib after each test to avoid state leakage."""
    yield
    plt.close('all')


@pytest.fixture
def sample_state_vector_1qubit():
    """Sample 1-qubit state vector."""
    return [1, 0]


@pytest.fixture
def sample_state_vector_2qubit():
    """Sample 2-qubit state vector."""
    return [1, 0, 0, 0]


@pytest.fixture
def sample_superposition():
    """Sample superposition state."""
    return [0.707+0j, 0.707+0j]


@pytest.fixture
def sample_bloch_input():
    """Sample Bloch sphere input lines."""
    return ["|0>", "|1>", "|+>", "|->"]


@pytest.fixture
def sample_circuit_grover():
    """Sample Grover algorithm circuit."""
    return {
        "qubits": 2,
        "gates": [
            {"type": "H", "qubit": 0},
            {"type": "H", "qubit": 1},
            {"type": "CNOT", "control": 0, "target": 1},
            {"type": "H", "qubit": 0},
            {"type": "H", "qubit": 1}
        ],
        "name": "Grover Search"
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for output files."""
    return tmp_path


@pytest.fixture
def mock_matplotlib(mocker):
    """Mock matplotlib to avoid actual rendering in tests."""
    return mocker.patch('matplotlib.pyplot.savefig')
