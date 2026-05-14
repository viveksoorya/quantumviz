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
    is_separable_along_qubit,
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

    def test_3qubit_separable(self):
        """Test 3-qubit separable state |000>."""
        state = [1] + [0]*7
        fig = plot_dcn(state, '3qubit separable')
        assert fig is not None
        plt.close(fig)

    def test_3qubit_ghz(self):
        """Test 3-qubit GHZ state."""
        ghz = [0.707+0j, 0, 0, 0, 0, 0, 0, 0.707+0j]
        fig = plot_dcn(ghz, '3qubit GHZ')
        assert fig is not None
        plt.close(fig)

    def test_3qubit_superposition(self):
        """Test 3-qubit state with multiple amplitudes."""
        state = [0.5+0j, 0, 0, 0, 0, 0.5+0j, 0, 0.5+0j]
        fig = plot_dcn(state, '3qubit superposition')
        assert fig is not None
        plt.close(fig)


class TestIsSeparableAlongQubit:
    """Tests for is_separable_along_qubit function."""

    def test_separable_along_q0(self):
        """Separable along qubit 0: |0+0> state."""
        state = [0.707+0j, 0.707+0j, 0, 0, 0, 0, 0, 0]
        assert is_separable_along_qubit(state, 3, qubit_idx=0) == True

    def test_separable_along_q1(self):
        """Separable along qubit 1."""
        state = [0.707+0j, 0, 0.707+0j, 0, 0, 0, 0, 0]
        assert is_separable_along_qubit(state, 3, qubit_idx=1) == True

    def test_separable_along_q2(self):
        """Separable along qubit 2."""
        state = [0.707+0j, 0, 0, 0, 0.707+0j, 0, 0, 0]
        assert is_separable_along_qubit(state, 3, qubit_idx=2) == True

    def test_entangled_ghz_all(self):
        """GHZ state is entangled along all qubits."""
        ghz = [0.707+0j, 0, 0, 0, 0, 0, 0, 0.707+0j]
        assert is_separable_along_qubit(ghz, 3, qubit_idx=0) == False
        assert is_separable_along_qubit(ghz, 3, qubit_idx=1) == False
        assert is_separable_along_qubit(ghz, 3, qubit_idx=2) == False

    def test_2qubit_bell_entangled(self):
        """Bell state is entangled along both qubits."""
        bell = [0.707+0j, 0, 0, 0.707+0j]
        assert is_separable_along_qubit(bell, 2, qubit_idx=0) == False
        assert is_separable_along_qubit(bell, 2, qubit_idx=1) == False

    def test_2qubit_separable(self):
        """Product state is separable along both qubits."""
        state = [0.5+0j, 0.5+0j, 0.5+0j, 0.5+0j]
        assert is_separable_along_qubit(state, 2, qubit_idx=0) == True
        assert is_separable_along_qubit(state, 2, qubit_idx=1) == True

    def test_separable_near_zero_ref(self):
        """Separable state with very small ref component (numerical robustness)."""
        eps = 1e-8
        state = [eps, 1, 2*eps, 2]
        assert is_separable_along_qubit(state, 2, qubit_idx=0) == True


class TestPlotDCNFromFile:
    """Tests for plot_dcns_from_file."""

    def test_single_stage_file(self):
        """Test loading from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"qubits": 1, "stages": [{"name": "test", "state_vector": [1, 0]}]}')
            temp_file = f.name

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                plot_dcns_from_file(temp_file, tmpdir, fmt="png")
                output_files = os.listdir(tmpdir)
                assert len(output_files) == 1
        finally:
            os.unlink(temp_file)

    def test_single_stage_file_pdf(self):
        """Test loading from JSON file and saving to PDF."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"qubits": 1, "stages": [{"name": "test", "state_vector": [1, 0]}]}')
            temp_file = f.name

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                plot_dcns_from_file(temp_file, tmpdir, fmt="pdf")
                output_files = os.listdir(tmpdir)
                assert len(output_files) == 1
                pdf_file = os.path.join(tmpdir, output_files[0])
                with open(pdf_file, 'rb') as f:
                    assert f.read(4) == b'%PDF'
        finally:
            os.unlink(temp_file)

    def test_3qubit_file(self):
        """Test loading 3-qubit state from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"qubits": 3, "stages": [{"name": "GHZ", "state_vector": [0.707,0,0,0,0,0,0,0.707]}]}')
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
