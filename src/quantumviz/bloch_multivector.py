"""
Bloch Multivector Visualization Module.

N Bloch spheres, one per qubit's reduced state.
Point position: Bloch vector r_i = (⟨X_i⟩, ⟨Y_i⟩, ⟨Z_i⟩)
Sphere opacity: purity Tr(ρ_i²), with filled=1.0, translucent=0.5
"""

import matplotlib
import numpy as np

matplotlib.use('Agg')
from typing import Any, List, Optional

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from quantumviz.qiskit_bridge import is_statevector


def partial_trace(
    rho: np.ndarray,
    qubit_idx: int,
    n_qubits: int
) -> np.ndarray:
    """Compute reduced density matrix for qubit qubit_idx.

    Args:
        rho: Full density matrix (2^n x 2^n)
        qubit_idx: Qubit to keep (0=LSB, n-1=MSB)
        n_qubits: Total number of qubits

    Returns:
        2x2 reduced density matrix
    """
    # Dimensions: keep qubit_idx, trace out all others
    # Convert to binary index manipulation
    dim = 2 ** n_qubits
    keep_mask = np.zeros(dim, dtype=bool)

    for i in range(dim):
        # Check if qubit_idx is set in i
        if (i >> qubit_idx) & 1:
            # Map to corresponding index in reduced system
            keep_mask[i] = True

    # Trace out: ρ_i = Tr_{not i}(ρ)
    # Simplified: ρ_i[j,k] = Σ_l ρ[j,l;k,l] where l runs over traced qubits
    rho_reduced = np.zeros((2, 2), dtype=complex)

    for i in range(2):
        for j in range(2):
            # For each pair of basis states of qubit qubit_idx
            # Sum over all configurations of other qubits
            total = 0j
            for state in range(dim):
                if ((state >> qubit_idx) & 1) == i:
                    for state2 in range(dim):
                        if ((state2 >> qubit_idx) & 1) == j:
                            # Check if other qubits match
                            if (state & ~(1 << qubit_idx)) == (state2 & ~(1 << qubit_idx)):
                                total += rho[state, state2]
            rho_reduced[i, j] = total

    return rho_reduced


def bloch_vector_from_rho(rho_i: np.ndarray) -> np.ndarray:
    """Compute Bloch vector (⟨X⟩, ⟨Y⟩, ⟨Z⟩) from 2x2 density matrix.

    Args:
        rho_i: 2x2 density matrix

    Returns:
        Array [⟨X⟩, ⟨Y⟩, ⟨Z⟩]
    """
    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)  # noqa: N806
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)  # noqa: N806
    Z = np.array([[1, 0], [0, -1]], dtype=complex)  # noqa: N806

    bloch = np.array([
        np.trace(X @ rho_i).real,
        np.trace(Y @ rho_i).real,
        np.trace(Z @ rho_i).real
    ])

    return bloch


def purity(rho_i: np.ndarray) -> float:
    """Compute purity Tr(ρ²).

    Args:
        rho_i: Density matrix

    Returns:
        Purity value in [0, 1]
    """
    return float(np.trace(rho_i @ rho_i).real)


def draw_bloch_sphere_3d(
    ax: Axes3D,
    bloch_vector: np.ndarray,
    opacity: float = 1.0,
    title: str = ""
) -> None:
    """Draw a Bloch sphere with the given Bloch vector point.

    Args:
        ax: 3D matplotlib axes
        bloch_vector: [⟨X⟩, ⟨Y⟩, ⟨Z⟩]
        opacity: Sphere opacity (0.0 to 1.0)
        title: Qubit label
    """
    # Draw wireframe sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_wireframe(x, y, z, color='gray', alpha=0.3, linewidth=0.5)

    # Draw axes
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'k-', alpha=0.5, linewidth=1)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'k-', alpha=0.5, linewidth=1)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'k-', alpha=0.5, linewidth=1)

    # Draw Bloch vector point
    if bloch_vector is not None:
        x, y, z = bloch_vector
        ax.scatter([x], [y], [z], color='red', s=100, alpha=opacity)
        # Draw line from origin to point
        ax.plot([0, x], [0, y], [0, z], 'r-', alpha=0.8, linewidth=2)

    # Set properties
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title, fontsize=10)

    # Make aspect ratio equal
    ax.set_box_aspect([1, 1, 1])


def plot_bloch_multivector(
    state: Any,
    title: str = "Bloch Multivector",
    output_path: Optional[str] = None,
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    Create Bloch Multivector plot - N Bloch spheres for each qubit.

    Args:
        state: List of complex amplitudes, or Qiskit Statevector object
        title: Title for the plot
        output_path: Path to save figure
        dpi: Resolution

    Returns:
        matplotlib Figure if output_path is None
    """
    # Handle Qiskit objects
    if is_statevector(state):
        from quantumviz.qiskit_bridge import statevector_to_list
        state_vector = statevector_to_list(state)
    else:
        state_vector = state

    # Determine number of qubits
    n = len(state_vector)
    n_qubits = int(np.log2(n))
    if 2 ** n_qubits != n:
        raise ValueError(f"State vector length {n} is not a power of 2")

    # Convert to density matrix
    state_array = np.array(state_vector, dtype=complex)
    rho = np.outer(state_array, state_array.conj())

    # Create figure with N subplots
    n_spheres = n_qubits
    n_cols = min(n_spheres, 4)
    n_rows = (n_spheres + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
    fig.suptitle(title, fontsize=12, y=0.98)

    for i in range(n_spheres):
        # Compute reduced density matrix for qubit i
        # Note: qubit i in our convention (0=LSB) corresponds to (n_qubits-1-i) in some conventions
        qubit_idx = i  # Use 0=Q3/LSB, n_qubits-1=Q1/MSB
        rho_i = partial_trace(rho, qubit_idx, n_qubits)

        # Compute Bloch vector and purity
        bloch_vec = bloch_vector_from_rho(rho_i)
        pur = purity(rho_i)

        # Create 3D subplot
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        draw_bloch_sphere_3d(
            ax,
            bloch_vec,
            opacity=pur,
            title=f'Qubit {n_qubits - 1 - i}'  # Label as Q1, Q2, ... from MSB
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi)
        plt.close()
        return None

    return fig


def plot_bloch_multivectors_from_file(
    input_file: str,
    output_dir: Optional[str] = None,
    dpi: int = 150,
    fmt: str = "png"
) -> List[str]:
    """Plot multiple Bloch Multivector plots from JSON file."""
    import json
    import os

    with open(input_file, 'r') as f:
        data = json.load(f)

    stages = data['stages']

    if output_dir and not output_dir.endswith('/'):
        output_dir += '/'
    if output_dir is None:
        output_dir = './'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_files = []

    for idx, stage in enumerate(stages):
        name = stage.get('name', f'Stage {idx+1}')

        state_vector = []
        for amp in stage['state_vector']:
            if isinstance(amp, (int, float)):
                state_vector.append(complex(amp, 0))
            elif isinstance(amp, str):
                state_vector.append(complex(amp.replace(' ', '')))
            else:
                state_vector.append(complex(amp[0], amp[1]))

        safe_name = name.replace(' ', '_').replace('/', '_')
        stage_num = f'stage_{idx+1:02d}'
        output_path = f'{output_dir}{stage_num}_{safe_name}.{fmt}'

        plot_bloch_multivector(state_vector, name, output_path, dpi)
        output_files.append(output_path)

        print(f'Saved: {output_path}')

    return output_files


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python bloch_multivector.py <input_file> [output_path]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    plot_bloch_multivectors_from_file(input_file, output_path)
