"""
Hinton Visualization Module.

Grid of squares where size = magnitude, color = sign/phase.
For density matrices: shows |ρ_ij| as square area, sign/phase as color.
For statevectors: shows |α_k|² as squares (probability distribution).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Any, List, Optional

from quantumviz.qiskit_bridge import is_statevector, statevector_to_list


def state_to_density(state_vector: List[complex]) -> np.ndarray:
    """Convert statevector to density matrix.
    
    Args:
        state_vector: List of complex amplitudes
        
    Returns:
        2D density matrix
    """
    psi = np.array(state_vector, dtype=complex)
    return np.outer(psi, psi.conj())


def plot_hinton(
    state: Any,
    title: str = "Hinton Diagram",
    output_path: Optional[str] = None,
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    Create Hinton diagram.
    
    Args:
        state: List of complex amplitudes, density matrix, or Qiskit object
        title: Title for the plot
        output_path: Path to save figure
        dpi: Resolution
        
    Returns:
        matplotlib Figure if output_path is None
    """
    # Handle Qiskit objects
    if is_statevector(state):
        state_vector = statevector_to_list(state)
    else:
        state_vector = state
    
    # Convert to numpy array
    state_array = np.array(state_vector, dtype=complex)
    
    # Check if input is statevector or density matrix
    if len(state_array.shape) == 1:  # Statevector
        n = len(state_array)
        if n == 0:
            raise ValueError("State vector cannot be empty")
        
        # Convert to density matrix for visualization
        rho = state_to_density(state_array)
        n_qubits = int(np.log2(n))
        if 2 ** n_qubits != n:
            raise ValueError(f"State vector length {n} is not power of 2")
    else:  # Density matrix
        rho = state_array
        n = rho.shape[0]
        n_qubits = int(np.log2(n))
    
    # Create figure with 2 subplots (real and imaginary)
    fig, (ax_real, ax_imag) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=12, y=0.98)
    
    # Real part
    _plot_hinton_subplot(
        ax_real,
        np.real(rho),
        f'Real Part (n={n_qubits} qubits)',
        cmap='RdBu_r',
        vmin=-1, vmax=1
    )
    
    # Imaginary part
    _plot_hinton_subplot(
        ax_imag,
        np.imag(rho),
        f'Imaginary Part (n={n_qubits} qubits)',
        cmap='RdBu_r',
        vmin=-1, vmax=1
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return None
    
    return fig


def _plot_hinton_subplot(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    cmap: str = 'RdBu_r',
    vmin: float = -1,
    vmax: float = 1
) -> None:
    """Plot single Hinton subplot."""
    n = matrix.shape[0]
    
    # Use imshow for efficiency
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    
    # Add grid
    ax.set_xticks(np.arange(n+1)-0.5, minor=True)
    ax.set_yticks(np.arange(n+1)-0.5, minor=True)
    ax.grid(True, which='minor', color='black', linewidth=0.5, alpha=0.3)
    
    # Labels
    labels = [format(i, f'0{n.bit_length()-1}b') for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Basis State |k⟩')
    ax.set_ylabel('Basis State ⟨k|')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_hintons_from_file(
    input_file: str,
    output_dir: Optional[str] = None,
    dpi: int = 150,
    fmt: str = "png"
) -> List[str]:
    """Plot multiple Hinton diagrams from JSON file."""
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
        
        plot_hinton(state_vector, name, output_path, dpi)
        output_files.append(output_path)
        
        print(f'Saved: {output_path}')
    
    return output_files


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python hinton.py <input_file> [output_path]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_hintons_from_file(input_file, output_path)
