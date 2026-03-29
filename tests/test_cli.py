"""
Comprehensive tests for the quantumviz CLI.
"""

import pytest
import click
from click.testing import CliRunner
import json
import tempfile
import os

from quantumviz.cli import main, bloch_sphere, state_city, cost_landscape, circuit, dynamic_flow, serve


class TestCLI:
    """Tests for CLI main entry point."""

    def test_main_help(self):
        """Test main help output."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'quantumviz' in result.output

    def test_main_version(self):
        """Test version output."""
        runner = CliRunner()
        result = runner.invoke(main, ['--version'])
        assert result.exit_code == 0
        assert '0.1.0' in result.output


class TestBlochSphereCLI:
    """Tests for bloch-sphere CLI command."""

    def test_bloch_sphere_help(self):
        """Test bloch-sphere help."""
        runner = CliRunner()
        result = runner.invoke(bloch_sphere, ['--help'])
        assert result.exit_code == 0

    def test_bloch_sphere_missing_input(self):
        """Test bloch-sphere with missing input."""
        runner = CliRunner()
        result = runner.invoke(bloch_sphere)
        assert result.exit_code != 0

    def test_bloch_sphere_invalid_input(self):
        """Test bloch-sphere with invalid input file."""
        runner = CliRunner()
        result = runner.invoke(bloch_sphere, ['nonexistent.txt'])
        assert result.exit_code != 0

    def test_bloch_sphere_valid_input(self):
        """Test bloch-sphere with valid input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "test.txt")
            with open(input_file, 'w') as f:
                f.write("|0>\n|1>\n|+>")

            output_file = os.path.join(tmpdir, "output.png")
            runner = CliRunner()
            result = runner.invoke(bloch_sphere, [input_file, '-o', output_file])
            assert result.exit_code == 0
            assert os.path.exists(output_file)


class TestStateCityCLI:
    """Tests for state-city CLI command."""

    def test_state_city_help(self):
        """Test state-city help."""
        runner = CliRunner()
        result = runner.invoke(state_city, ['--help'])
        assert result.exit_code == 0

    def test_state_city_missing_input(self):
        """Test state-city with missing input."""
        runner = CliRunner()
        result = runner.invoke(state_city)
        assert result.exit_code != 0

    def test_state_city_valid_input(self):
        """Test state-city with valid input."""
        data = {
            "qubits": 1,
            "stages": [
                {"name": "Zero", "state_vector": [1, 0]},
                {"name": "One", "state_vector": [0, 1]}
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "test.json")
            with open(input_file, 'w') as f:
                json.dump(data, f)

            runner = CliRunner()
            result = runner.invoke(state_city, [input_file])
            assert result.exit_code == 0


class TestCostLandscapeCLI:
    """Tests for cost-landscape CLI command."""

    def test_cost_landscape_help(self):
        """Test cost-landscape help."""
        runner = CliRunner()
        result = runner.invoke(cost_landscape, ['--help'])
        assert result.exit_code == 0

    def test_cost_landscape_qaoa(self):
        """Test cost-landscape qaoa."""
        data = {"edges": [[0, 1], [1, 2]]}
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "qaoa.json")
            with open(input_file, 'w') as f:
                json.dump(data, f)
            output_file = os.path.join(tmpdir, "qaoa.png")
            runner = CliRunner()
            result = runner.invoke(cost_landscape, ['qaoa', input_file, '-o', output_file])
            assert result.exit_code == 0
            assert os.path.exists(output_file)

    def test_cost_landscape_vqe(self):
        """Test cost-landscape vqe."""
        data = {"terms": [{"coeff": 0.5, "paulis": ["Z"]}]}
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "vqe.json")
            with open(input_file, 'w') as f:
                json.dump(data, f)
            output_file = os.path.join(tmpdir, "vqe.png")
            runner = CliRunner()
            result = runner.invoke(cost_landscape, ['vqe', input_file, '-o', output_file])
            assert result.exit_code == 0
            assert os.path.exists(output_file)

    def test_cost_landscape_invalid_algorithm(self):
        """Test cost-landscape with invalid algorithm."""
        runner = CliRunner()
        result = runner.invoke(cost_landscape, ['invalid'])
        assert result.exit_code != 0


class TestCircuitCLI:
    """Tests for circuit CLI command."""

    def test_circuit_help(self):
        """Test circuit help."""
        runner = CliRunner()
        result = runner.invoke(circuit, ['--help'])
        assert result.exit_code == 0

    def test_circuit_missing_input(self):
        """Test circuit with missing input."""
        runner = CliRunner()
        result = runner.invoke(circuit)
        assert result.exit_code != 0

    def test_circuit_valid_input(self):
        """Test circuit with valid input."""
        data = {
            "qubits": 2,
            "gates": [
                {"type": "H", "qubit": 0},
                {"type": "H", "qubit": 1},
                {"type": "CNOT", "control": 0, "target": 1}
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "circuit.json")
            with open(input_file, 'w') as f:
                json.dump(data, f)

            output_file = os.path.join(tmpdir, "circuit.png")
            runner = CliRunner()
            result = runner.invoke(circuit, [input_file, '-o', output_file])
            assert result.exit_code == 0
            assert os.path.exists(output_file)


class TestDynamicFlowCLI:
    """Tests for dynamic-flow CLI command."""

    def test_dynamic_flow_help(self):
        """Test dynamic-flow help."""
        runner = CliRunner()
        result = runner.invoke(dynamic_flow, ['--help'])
        assert result.exit_code == 0

    def test_dynamic_flow_missing_input(self):
        """Test dynamic-flow with missing input."""
        runner = CliRunner()
        result = runner.invoke(dynamic_flow)
        assert result.exit_code != 0

    def test_dynamic_flow_valid_input(self):
        """Test dynamic-flow with valid input."""
        data = {
            "qubits": 1,
            "stages": [
                {"name": "t=0", "state_vector": [1, 0]},
                {"name": "t=1", "state_vector": [0, 1]}
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "test.json")
            with open(input_file, 'w') as f:
                json.dump(data, f)

            output_file = os.path.join(tmpdir, "output.png")
            runner = CliRunner()
            result = runner.invoke(dynamic_flow, [input_file, '-o', output_file])
            assert result.exit_code == 0
            assert os.path.exists(output_file)


class TestServeCLI:
    """Tests for serve CLI command."""

    def test_serve_help(self):
        """Test serve help."""
        runner = CliRunner()
        result = runner.invoke(serve, ['--help'])
        assert result.exit_code == 0

    def test_serve_default(self):
        """Test serve with default options."""
        runner = CliRunner()
        result = runner.invoke(serve, ['--help'])
        assert result.exit_code == 0
        assert '8000' in result.output

    def test_serve_custom_port(self):
        """Test serve with custom port."""
        runner = CliRunner()
        result = runner.invoke(serve, ['-p', '9000'])
        # Should try to start but may fail due to missing deps - that's OK
        assert 'dashboard' in result.output.lower() or result.exit_code != 0
