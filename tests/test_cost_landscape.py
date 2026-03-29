"""
Comprehensive tests for the Cost Landscape visualization module.
"""

import pytest
import numpy as np
import numpy.testing as npt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tempfile
import os

from quantumviz.cost_landscape import (
    qaoa_cost,
    vqe_energy,
    plot_qaoa_landscape,
    plot_vqe_landscape,
)

SIMPLE_GRAPH_EDGES = [(0, 1)]
LINE_GRAPH_EDGES = [(0, 1), (1, 2)]
TRIANGLE_EDGES = [(0, 1), (1, 2), (0, 2)]

H2_TERMS = [
    {"coeff": -1.0, "paulis": []},
    {"coeff": 0.5, "paulis": ["Z"]},
    {"coeff": 0.2, "paulis": ["Z", "Z"]},
]

LIH_TERMS = [
    {"coeff": -7.8, "paulis": []},
    {"coeff": 0.3, "paulis": ["Z"]},
    {"coeff": 0.15, "paulis": ["Z", "Z"]},
]


class TestQAOACost:
    """Tests for qaoa_cost function."""

    def test_qaoa_cost_scalar(self):
        """Test QAOA cost with scalar inputs."""
        result = qaoa_cost(0.5, 0.5, SIMPLE_GRAPH_EDGES)
        assert isinstance(result, (float, np.floating))

    def test_qaoa_cost_array(self):
        """Test QAOA cost with array inputs."""
        gamma = np.array([0, np.pi/4, np.pi/2])
        beta = np.array([0, np.pi/4, np.pi/2])
        result = qaoa_cost(gamma, beta, SIMPLE_GRAPH_EDGES)
        assert len(result) == 3

    def test_qaoa_cost_meshgrid(self):
        """Test QAOA cost with meshgrid."""
        gamma = np.linspace(0, np.pi, 10)
        beta = np.linspace(0, np.pi, 10)
        G, B = np.meshgrid(gamma, beta)
        Z = qaoa_cost(G, B, SIMPLE_GRAPH_EDGES)
        assert Z.shape == (10, 10)

    def test_qaoa_cost_range(self):
        """Test QAOA cost is in valid range."""
        gamma = np.linspace(0, np.pi, 50)
        beta = np.linspace(0, np.pi, 50)
        G, B = np.meshgrid(gamma, beta)
        Z = qaoa_cost(G, B, SIMPLE_GRAPH_EDGES)
        assert Z.min() >= -2
        assert Z.max() <= 2

    def test_qaoa_cost_symmetry_gamma(self):
        """Test QAOA cost is symmetric in gamma."""
        c1 = qaoa_cost(0, 0, SIMPLE_GRAPH_EDGES)
        c2 = qaoa_cost(np.pi, 0, SIMPLE_GRAPH_EDGES)
        npt.assert_almost_equal(c1, c2, decimal=5)

    def test_qaoa_cost_empty_edges(self):
        """Test QAOA cost with no edges returns zeros."""
        result = qaoa_cost(0.5, 0.5, [])
        assert result == 0.0


class TestVQEEnergy:
    """Tests for vqe_energy function."""

    def test_vqe_energy_h2_scalar(self):
        """Test VQE energy H2 with scalar input."""
        result = vqe_energy(0.5, H2_TERMS)
        expected = -1.0 + 0.5 * np.cos(0.5) + 0.2 * np.cos(0.5)**2
        assert result == pytest.approx(expected)

    def test_vqe_energy_lih_scalar(self):
        """Test VQE energy LiH with scalar input."""
        result = vqe_energy(0.5, LIH_TERMS)
        expected = -7.8 + 0.3 * np.cos(0.5) + 0.15 * np.cos(0.5)**2
        assert result == pytest.approx(expected)

    def test_vqe_energy_array(self):
        """Test VQE energy with array input."""
        theta = np.array([0, np.pi/4, np.pi/2])
        result = vqe_energy(theta, H2_TERMS)
        assert len(result) == 3

    def test_vqe_energy_empty_terms(self):
        """Test VQE energy with no terms returns zeros."""
        result = vqe_energy(0.5, [])
        assert result == 0.0

    def test_vqe_energy_single_term(self):
        """Test VQE energy with single term."""
        terms = [{"coeff": 2.0, "paulis": []}]
        result = vqe_energy(0.5, terms)
        assert result == 2.0


class TestQAOALandscape:
    """Tests for plot_qaoa_landscape function."""

    def test_plot_creates_figure(self):
        """Test plotting creates a figure."""
        fig = plot_qaoa_landscape(SIMPLE_GRAPH_EDGES)
        assert fig is not None
        plt.close(fig)

    def test_plot_saves_to_file(self):
        """Test plotting saves to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "qaoa.png")
            plot_qaoa_landscape(SIMPLE_GRAPH_EDGES, output_path=output_path)
            assert os.path.exists(output_path)

    def test_plot_custom_resolution(self):
        """Test plotting with custom resolution."""
        fig = plot_qaoa_landscape(SIMPLE_GRAPH_EDGES, resolution=25)
        assert fig is not None
        plt.close(fig)

    def test_plot_custom_ranges(self):
        """Test plotting with custom ranges."""
        fig = plot_qaoa_landscape(
            SIMPLE_GRAPH_EDGES,
            gamma_range=(0, 2*np.pi),
            beta_range=(0, 2*np.pi)
        )
        assert fig is not None
        plt.close(fig)

    def test_plot_custom_dpi(self):
        """Test plotting with custom DPI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "qaoa.png")
            plot_qaoa_landscape(SIMPLE_GRAPH_EDGES, output_path=output_path, dpi=300)
            assert os.path.exists(output_path)

    def test_plot_triangle_graph(self):
        """Test plotting with triangle graph."""
        fig = plot_qaoa_landscape(TRIANGLE_EDGES)
        assert fig is not None
        plt.close(fig)


class TestVQELandscape:
    """Tests for plot_vqe_landscape function."""

    def test_plot_creates_figure(self):
        """Test plotting creates a figure."""
        fig = plot_vqe_landscape(H2_TERMS)
        assert fig is not None
        plt.close(fig)

    def test_plot_saves_to_file(self):
        """Test plotting saves to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "vqe.png")
            plot_vqe_landscape(H2_TERMS, output_path=output_path)
            assert os.path.exists(output_path)

    def test_plot_custom_resolution(self):
        """Test plotting with custom resolution."""
        fig = plot_vqe_landscape(H2_TERMS, resolution=50)
        assert fig is not None
        plt.close(fig)

    def test_plot_custom_theta_range(self):
        """Test plotting with custom theta range."""
        fig = plot_vqe_landscape(H2_TERMS, theta_range=(0, 4*np.pi))
        assert fig is not None
        plt.close(fig)

    def test_plot_custom_dpi(self):
        """Test plotting with custom DPI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "vqe.png")
            plot_vqe_landscape(H2_TERMS, output_path=output_path, dpi=300)
            assert os.path.exists(output_path)

    def test_plot_lih_terms(self):
        """Test plotting with LiH terms."""
        fig = plot_vqe_landscape(LIH_TERMS)
        assert fig is not None
        plt.close(fig)


class TestCostFunctionProperties:
    """Property-based tests for cost functions."""

    def test_qaoa_cost_periodicity(self):
        """Test QAOA cost is periodic."""
        gamma = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        beta = np.zeros(5)
        costs = qaoa_cost(gamma, beta, SIMPLE_GRAPH_EDGES)
        assert costs[0] == pytest.approx(costs[-1], rel=1e-5)

    def test_vqe_energy_periodicity(self):
        """Test VQE energy is periodic."""
        theta = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        energies = vqe_energy(theta, H2_TERMS)
        assert energies[0] == pytest.approx(energies[-1], rel=1e-5)

    def test_qaoa_cost_gradient_exists(self):
        """Test QAOA cost has finite gradient."""
        gamma = np.linspace(0, np.pi, 20)
        beta = np.linspace(0, np.pi, 20)
        G, B = np.meshgrid(gamma, beta)
        Z = qaoa_cost(G, B, SIMPLE_GRAPH_EDGES)
        assert np.all(np.isfinite(Z))

    def test_vqe_energy_gradient_exists(self):
        """Test VQE energy has finite gradient."""
        theta = np.linspace(0, 2*np.pi, 100)
        energies = vqe_energy(theta, H2_TERMS)
        assert np.all(np.isfinite(energies))


class TestEdgeCases:
    """Tests for edge cases."""

    def test_qaoa_cost_zero_inputs(self):
        """Test QAOA cost with zero inputs."""
        result = qaoa_cost(0, 0, SIMPLE_GRAPH_EDGES)
        assert isinstance(result, (float, np.floating))

    def test_vqe_energy_zero_input(self):
        """Test VQE energy with zero input."""
        result = vqe_energy(0, H2_TERMS)
        assert isinstance(result, (float, np.floating, np.ndarray))

    def test_qaoa_cost_large_inputs(self):
        """Test QAOA cost with large inputs."""
        result = qaoa_cost(100, 100, SIMPLE_GRAPH_EDGES)
        assert np.isfinite(result)

    def test_vqe_energy_large_input(self):
        """Test VQE energy with large input."""
        result = vqe_energy(100, H2_TERMS)
        assert np.isfinite(result)

    def test_qaoa_cost_empty_graph(self):
        """Test QAOA cost with empty graph."""
        gamma = np.array([0, np.pi/4, np.pi/2])
        beta = np.array([0, np.pi/4, np.pi/2])
        Z = qaoa_cost(gamma, beta, [])
        assert np.all(Z == 0)

    def test_vqe_energy_empty_hamiltonian(self):
        """Test VQE energy with empty Hamiltonian."""
        theta = np.array([0, np.pi/4, np.pi/2])
        energies = vqe_energy(theta, [])
        assert np.all(energies == 0)
