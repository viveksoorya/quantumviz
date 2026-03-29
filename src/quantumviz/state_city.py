"""
State City Visualization Module

Provides functions for visualizing multi-qubit density matrices
as 3D bar charts (state city plots).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from typing import Union, List, Dict, Any, Optional


def parse_amplitude(amp: Any) -> complex:
    """
    Convert various input formats to a complex number.

    Args:
        amp: Amplitude in various formats (int, float, str, list/tuple)

    Returns:
        Complex amplitude

    Raises:
        ValueError: If amplitude format is unsupported
    """
    if isinstance(amp, (int, float)):
        return complex(amp, 0)
    elif isinstance(amp, str):
        amp = amp.replace(' ', '')
        if amp.endswith('j'):
            return complex(amp)
        else:
            return complex(float(amp), 0)
    elif isinstance(amp, (list, tuple)) and len(amp) == 2:
        return complex(amp[0], amp[1])
    else:
        raise ValueError(f"Unsupported amplitude format: {amp}")


def state_to_density(state_vector: List[complex]) -> np.ndarray:
    """
    Convert a state vector (list of complex numbers) to a density matrix.

    The density matrix is computed as ρ = |ψ⟩⟨ψ|.

    Args:
        state_vector: List of complex amplitudes

    Returns:
        2D numpy array representing the density matrix

    Raises:
        ValueError: If state vector is empty
    """
    if not state_vector:
        raise ValueError("State vector cannot be empty")

    psi = np.array(state_vector, dtype=complex).reshape(-1, 1)
    return psi @ psi.conj().T


def plot_state_city(
    state_vector: List[complex],
    title: str = "Density Matrix",
    output_path: Optional[str] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Create a state city plot (real and imaginary parts of density matrix).

    Args:
        state_vector: List of complex amplitudes (length must be power of 2)
        title: Title for the plot
        output_path: Path to save the figure (if None, returns figure object)
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object if output_path is None, else None
    """
    rho = state_to_density(state_vector)
    dim = rho.shape[0]

    xpos, ypos = np.meshgrid(range(dim), range(dim), indexing='ij')
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    dx = dy = 0.8 * np.ones_like(zpos)
    real_vals = np.real(rho).flatten()
    imag_vals = np.imag(rho).flatten()

    real_colors = ['red' if v >= 0 else 'blue' for v in real_vals]
    imag_colors = ['red' if v >= 0 else 'blue' for v in imag_vals]

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(title, fontsize=16)

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.bar3d(xpos, ypos, zpos, dx, dy, real_vals, color=real_colors, alpha=0.8)
    ax1.set_title('Real part')
    ax1.set_xlabel('Basis state (row)')
    ax1.set_ylabel('Basis state (col)')
    ax1.set_zlabel('Amplitude')
    ax1.set_xticks(range(dim))
    ax1.set_yticks(range(dim))

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.bar3d(xpos, ypos, zpos, dx, dy, imag_vals, color=imag_colors, alpha=0.8)
    ax2.set_title('Imaginary part')
    ax2.set_xlabel('Basis state (row)')
    ax2.set_ylabel('Basis state (col)')
    ax2.set_zlabel('Amplitude')
    ax2.set_xticks(range(dim))
    ax2.set_yticks(range(dim))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi)
        plt.close()
    else:
        return fig


def plot_state_cities_from_file(
    input_file: str,
    output_dir: Optional[str] = None,
    dpi: int = 150
) -> List[str]:
    """
    Plot multiple stages from a JSON input file.

    Args:
        input_file: Path to JSON file with stages
        output_dir: Directory to save output files (if None, uses current dir)
        dpi: Resolution for saved figures

    Returns:
        List of output filenames
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    n_qubits = data['qubits']
    dim = 2 ** n_qubits
    stages = data['stages']

    output_files = []

    for i, stage in enumerate(stages):
        name = stage.get('name', f'Stage {i+1}')
        raw_state = stage['state_vector']
        state_vector = [parse_amplitude(amp) for amp in raw_state]

        if len(state_vector) != dim:
            raise ValueError(f"Stage '{name}': state vector length {len(state_vector)} "
                           f"does not match 2^{n_qubits}={dim}")

        if output_dir:
            filename = f"{output_dir}/stage_{i+1:02d}_{name.replace(' ', '_')}.png"
        else:
            filename = f"stage_{i+1:02d}_{name.replace(' ', '_')}.png"

        plot_state_city(state_vector, name, filename, dpi)
        output_files.append(filename)
        print(f"Saved: {filename}")

    return output_files


def main(args: Optional[List[str]] = None) -> None:
    """
    CLI entry point for State City visualization.

    Args:
        args: Command line arguments (if None, uses sys.argv)
    """
    import sys
    if args is None:
        args = sys.argv[1:]

    if len(args) != 1:
        print("Usage: python -m quantumviz.state_city <input_file.json>")
        sys.exit(1)

    input_file = args[0]
    plot_state_cities_from_file(input_file)


if __name__ == "__main__":
    main()
