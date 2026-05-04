"""
PauliVec Visualization Module

Bar chart of Pauli operator expectations.
X-axis: Pauli labels (II, IX, IY, ..., ZZ for 2 qubits)
Y-axis: Expectation values in range [-1, 1]
Bar height: Tr(σρ) for each Pauli term σ
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Any, List, Optional, Tuple

from quantumviz.qiskit_bridge import is_statevector, statevector_to_list


def pauli_expectations(state: Any, n_qubits: int) -> Tuple[List[str], np.ndarray]:
    """Compute Pauli expectations for all 4^n terms.
    
    Args:
        state: Statevector (list/ndarray) or DensityMatrix
        n_qubits: Number of qubits
        
    Returns:
        Tuple of (labels, values) where:
        - labels: List of Pauli term strings (e.g., "II", "IX", ...)
        - values: Array of expectation values Tr(σρ)
    """
    # Convert statevector to density matrix if needed
    state_array = np.array(state, dtype=complex)
    
    if len(state_array) == 2 ** n_qubits:  # Statevector
        rho = np.outer(state_array, state_array.conj())
    else:  # Already density matrix
        rho = state_array.reshape((2**n_qubits, 2**n_qubits))
    
    n_terms = 4 ** n_qubits
    labels = []
    values = np.zeros(n_terms)
    
    # Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = [I, X, Y, Z]
    pauli_names = ['I', 'X', 'Y', 'Z']
    
    for i in range(n_terms):
        # Convert i to base-4 to get Pauli indices (LSB = qubit 0)
        indices = []
        temp = i
        for _ in range(n_qubits):
            indices.append(temp % 4)
            temp //= 4
        
        # Build Pauli label and tensor product matrix
        # For qubits: σ = σ_n ⊗ σ_{n-1} ⊗ ... ⊗ σ_0
        label = ''
        matrix = None
        
        for qubit_idx, idx in enumerate(indices):
            label += pauli_names[idx]
            if matrix is None:
                matrix = paulis[idx]
            else:
                matrix = np.kron(paulis[idx], matrix)
        
        labels.append(label)
        # Tr(σρ)
        values[i] = np.trace(rho @ matrix).real
    
    return labels, values


def plot_paulivec(
    state: Any,
    title: str = "PauliVec Visualization",
    output_path: Optional[str] = None,
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    Create PauliVec bar chart.
    
    Args:
        state: Statevector or density matrix
        title: Title for the plot
        output_path: Path to save figure
        dpi: Resolution
        
    Returns:
        matplotlib Figure if output_path is None
    """
    # Handle Qiskit objects
    if is_statevector(state):
        from quantumviz.qiskit_bridge import statevector_to_list
        state = statevector_to_list(state)
    
    # Determine number of qubits
    state_array = np.array(state, dtype=complex)
    if len(state_array.shape) == 1:  # Statevector
        n_qubits = int(np.log2(len(state_array)))
    else:  # Density matrix
        n_qubits = int(np.log2(state_array.shape[0]))
    
    if 2 ** n_qubits != (len(state_array) if len(state_array.shape) == 1 else state_array.shape[0]):
        raise ValueError("State length must be a power of 2")
    
    labels, values = pauli_expectations(state, n_qubits)
    
    # Create figure
    fig_width = max(8, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    fig.suptitle(title, fontsize=12, y=0.95)
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, values, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Set axis properties
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('Pauli Term', fontsize=10)
    ax.set_ylabel('Expectation ⟨σ⟩', fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return None
    
    return fig


def plot_paulivecs_from_file(
    input_file: str,
    output_dir: Optional[str] = None,
    dpi: int = 150,
    fmt: str = "png"
) -> List[str]:
    """Plot multiple PauliVec charts from JSON file."""
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
        
        plot_paulivec(state_vector, name, output_path, dpi)
        output_files.append(output_path)
        print(f'Saved: {output_path}')
    
    return output_files


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python paulivec.py <input_file> [output_path]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_paulivecs_from_file(input_file, output_path)
