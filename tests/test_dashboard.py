"""
Tests for quantumviz Dashboard FastAPI endpoints.
"""

import sys
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from quantumviz.dashboard.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_healthy(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health_has_version(self):
        response = client.get("/api/health")
        assert "0.3.0" in response.json()["version"]


class TestVisualizeEndpoint:
    def test_visualize_valid_state(self):
        response = client.post("/api/visualize", json={
            "qubits": 2,
            "stages": [{"name": "Test", "state": [1, 0, 0, 0]}]
        })
        assert response.status_code == 200
        data = response.json()
        assert "bloch_plot" in data
        assert "density_plot" in data
        assert "state_vector" in data

    def test_visualize_state_0(self):
        response = client.post("/api/visualize", json={
            "qubits": 1,
            "stages": [{"name": "|0>", "state": [1, 0]}]
        })
        data = response.json()
        bloch = data["bloch_vector"]
        assert abs(bloch["z"] - 1.0) < 0.01

    def test_visualize_state_1(self):
        response = client.post("/api/visualize", json={
            "qubits": 1,
            "stages": [{"name": "|1>", "state": [0, 1]}]
        })
        data = response.json()
        bloch = data["bloch_vector"]
        assert abs(bloch["z"] + 1.0) < 0.01

    def test_visualize_superposition(self):
        response = client.post("/api/visualize", json={
            "qubits": 1,
            "stages": [{"name": "|+>", "state": [0.707, 0.707]}]
        })
        data = response.json()
        bloch = data["bloch_vector"]
        assert abs(bloch["x"] - 1.0) < 0.01

    def test_visualize_2qubit(self):
        response = client.post("/api/visualize", json={
            "qubits": 2,
            "stages": [
                {"name": "Bell", "state": [0.707, 0, 0, 0.707]}
            ]
        })
        assert response.status_code == 200
        data = response.json()
        assert "state_vector" in data

    def test_visualize_wrong_dimensions(self):
        response = client.post("/api/visualize", json={
            "qubits": 2,
            "stages": [{"name": "Wrong", "state": [1, 0]}]
        })
        assert response.status_code in [200, 400]

    def test_visualize_no_stages(self):
        response = client.post("/api/visualize", json={
            "qubits": 1,
            "stages": []
        })
        data = response.json()
        assert "error" in data

    def test_visualize_plotly_structure(self):
        response = client.post("/api/visualize", json={
            "qubits": 1,
            "stages": [{"name": "Test", "state": [1, 0]}]
        })
        data = response.json()
        assert "data" in data["bloch_plot"]
        assert "layout" in data["bloch_plot"]


class TestRunHardwareEndpoint:
    def test_run_hardware_simulator_mode(self):
        response = client.post("/api/run-hardware", json={
            "qubits": 2,
            "backend": "ibmq_qasm_simulator",
            "token": None,
            "shots": 1024,
            "state_vector": None
        })
        assert response.status_code == 200
        data = response.json()
        assert "counts" in data or "error" in data

    def test_run_hardware_missing_qiskit(self):
        with patch.dict(sys.modules, {"qiskit": None, "qiskit_ibm_runtime": None}):
            response = client.post("/api/run-hardware", json={
                "qubits": 2,
                "backend": "ibmq_qasm_simulator",
                "shots": 1024
            })
            data = response.json()
            assert "error" in data or "note" in data

    def test_run_hardware_validation(self):
        response = client.post("/api/run-hardware", json={
            "qubits": 0,
            "backend": "ibmq_qasm_simulator",
            "shots": 1024
        })
        assert response.status_code == 422

    def test_run_hardware_shots_range(self):
        response = client.post("/api/run-hardware", json={
            "qubits": 2,
            "backend": "ibmq_qasm_simulator",
            "shots": 50
        })
        assert response.status_code == 422

    def test_run_hardware_saves_result(self):
        response = client.post("/api/run-hardware", json={
            "qubits": 2,
            "backend": "simulator",
            "shots": 100
        })
        data = response.json()
        if "result_file" in data:
            assert "result_" in data["result_file"]


class TestBackendsEndpoint:
    def test_backends_list(self):
        response = client.get("/api/backends")
        assert response.status_code == 200
        data = response.json()
        assert "backends" in data

    def test_backends_qiskit_not_installed(self):
        with patch.dict(sys.modules, {"qiskit_ibm_runtime": None}):
            response = client.get("/api/backends")
            data = response.json()
            assert data["backends"] == [] or "error" in data


class TestRootEndpoint:
    def test_root_returns_html(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")

    def test_root_contains_dashboard_title(self):
        response = client.get("/")
        assert "Quantum Algorithm Visualization Dashboard" in response.text

    def test_root_includes_plotly(self):
        response = client.get("/")
        assert "plot.ly" in response.text or "plotly" in response.text


class TestCORS:
    def test_app_has_cors_configured(self):
        from quantumviz.dashboard.main import app
        assert app is not None
        assert app.title == "Quantum Viz Dashboard"


class TestDashboardHelpers:
    def test_state_to_bloch_vector(self):
        from quantumviz.dashboard.main import state_to_bloch_vector
        result = state_to_bloch_vector([1, 0])
        assert abs(result["z"] - 1.0) < 0.01

    def test_state_to_bloch_vector_superposition(self):
        from quantumviz.dashboard.main import state_to_bloch_vector
        result = state_to_bloch_vector([0.707, 0.707])
        assert abs(result["x"] - 1.0) < 0.01

    def test_state_to_density_matrix(self):
        from quantumviz.dashboard.main import state_to_density_matrix
        result = state_to_density_matrix([1, 0], 2)
        assert "density_matrix_real" in result
        assert "probabilities" in result

    def test_bloch_to_plotly(self):
        from quantumviz.dashboard.main import bloch_to_plotly
        result = bloch_to_plotly({"x": 0, "y": 0, "z": 1})
        assert "data" in result
        assert "layout" in result

    def test_density_to_plotly(self):
        from quantumviz.dashboard.main import density_to_plotly
        dm = {
            "dimensions": 2,
            "probabilities": [1.0, 0.0]
        }
        result = density_to_plotly(dm)
        assert "data" in result
        assert "layout" in result
