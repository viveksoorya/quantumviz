"""
Tests for the DCN (Dimensional Circular Notation) visualization module.
"""

import matplotlib
import numpy as np
import pytest
import tempfile
import os

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from quantumviz.dcn import (
    plot_dcn,
    parse_amplitude,
    plot_dcns_from_file,
)


class TestParseAmplitude:
    """Tests for parse_amplitude function."""

    def test_real_number(self):
        assert parse_amplitude(1.0) == 1.0
        assert parse_amplitude(0.0) == 0.0

    def test_complex_number(self):
        # Note: parse_amplitude expects string format '1j' not Python's 1j
        np.testing.assert_almost_equal(parse_amplitude('1j'), 0+1j)
        np.testing.assert_almost_equal(parse_amplitude('0.707+0.707j'), 0.707+0.707j)

    def test_string(self):
        np.testing.assert_almost_equal(parse_amplitude('1j'), 1j)
        np.testing.assert_almost_equal(parse_amplitude('0.707'), 0.707+0j)

    def test_list(self):
        np.testing.assert_almost_equal(parse_amplitude([0.707, 0.707]), 0.707+0.707j)


class TestPlotDCN:
    """Tests for plot_dcn function."""

    def test_single_qubit_0(self):
        """Test single qubit |0> state."""
        fig = plot_dcn([1, 0], 'test')
        assert fig is not None
        plt.close(fig)

    def test_single_qubit_1(self):
        """Test single qubit |1> state."""
        fig = plot_dcn([0, 1], 'test')
        assert fig is not None
        plt.close(fig)

    def test_single_qubit_plus(self):
        """Test single qubit |+> state."""
        sv = [1/np.sqrt(2), 1/np.sqrt(2)]
        fig = plot_dcn(sv, 'test')
        assert fig is not None
        plt.close(fig)

    def test_two_qubit_bell(self):
        """Test two-qubit Bell state."""
        bell = [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]
        fig = plot_dcn(bell, 'Bell')
        assert fig is not None
        plt.close(fig)

    def test_empty_state_vector(self):
        """Test empty state raises error."""
        with pytest.raises(ValueError):
            plot_dcn([], 'test')

    def test_invalid_state_length(self):
        """Test non-power-of-2 length raises error."""
        with pytest.raises(ValueError):
            plot_dcn([1, 0, 1], 'test')


class TestPlotDCNFromFile:
    """Tests for plot_dcns_from_file."""

    def test_single_stage_file(self):
        """Test loading from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"qubits": 1, "stages": [{"name": "test", "state_vector": [1, 0]}]}')
            temp_file = f.name

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                plot_dcns_from_file(temp_file, tmpdir)
                output_files = os.listdir(tmpdir)
                assert len(output_files) == 1
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])