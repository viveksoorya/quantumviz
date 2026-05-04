"""
BEADS (Quantum Beads) Visualization Module

Provides visualization of quantum states using the BEADS representation:
- Q-Beads: One colored sphere per qubit (red-green color scheme)
  Red = high |0⟩ probability, Green = high |1⟩ probability
- E-Beads: Yellow-blue spheres representing entanglement between qubits

Based on: Huber & Glaser, "BEADS: A canonical visualization of quantum states
for applications in quantum information processing" (2025)
"""

import json
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Agg')


@dataclass
class QBead:
    """Represents a single Q-Bead (qubit bead)."""
    qubit_index: int
    prob_0: float  # Probability of measuring |0⟩
    prob_1: float  # Probability of measuring |1⟩
    theta: float  # Polar angle (0 = |0⟩, π = |1⟩)
    phi: float    # Azimuthal angle (phase)


@dataclass
class EBead:
    """Represents a single E-Bead (entanglement bead)."""
    qubits: Tuple[int, ...]  # Which qubits are entangled
    entanglement_strength: float  # 0 = separable, 1 = max entanglement
    entanglement_type: str  # "pairwise", "triple", "global"


def parse_amplitude(amp: Any) -> complex:
    """Convert various input formats to a complex number."""
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
    """Convert state vector to density matrix: ρ = |ψ⟩⟨ψ|"""
    psi = np.array(state_vector, dtype=complex).reshape(-1, 1)
    return (psi @ psi.conj().T).astype(complex)


def compute_reduced_density_matrix(
    state_vector: List[complex],
    n_qubits: int,
    qubit_idx: int
) -> np.ndarray:
    """
    Compute reduced density matrix for a single qubit by tracing out others.

    Uses the standard partial trace formula:
    ρ_A = Tr_B(ρ_AB) = Σ_b (I ⊗ ⟨b|) ρ_AB (I ⊗ |b⟩)

    For n_qubits=2 with basis |00⟩=0, |01⟩=1, |10⟩=2, |11⟩=3:
    - Tracing out qubit 0: sum rows/cols where bit 0 changes
    - Tracing out qubit 1: sum rows/cols where bit 1 changes

    Args:
        state_vector: Full system state vector (length 2^n_qubits)
        n_qubits: Total number of qubits
        qubit_idx: Index of qubit to keep (0 = LSB, n_qubits-1 = MSB)

    Returns:
        2x2 density matrix for the reduced system
    """
    rho_full = state_to_density(state_vector)
    dim = 2 ** n_qubits

    # If only 1 qubit, return the full density matrix
    if n_qubits == 1:
        return rho_full

    rho_reduced = np.zeros((2, 2), dtype=complex)

    # More efficient: iterate through full matrix and accumulate
    # For each element rho_full[a, b], determine how it contributes
    # to rho_reduced[i, j] based on qubit_idx bit values
    mask_keep = 1 << qubit_idx  # Which bit we're keeping
    mask_trace = ~mask_keep & ((1 << n_qubits) - 1)  # Which bits to trace out

    for a in range(dim):
        for b in range(dim):
            # Check if they differ only in traced-out qubits
            # Extract the bits that should be the same in reduced matrix
            bit_a_keep = (a >> qubit_idx) & 1
            bit_b_keep = (b >> qubit_idx) & 1
            bits_a_other = (a >> 0) & mask_trace
            bits_b_other = (b >> 0) & mask_trace

            if bits_a_other == bits_b_other:
                rho_reduced[bit_a_keep, bit_b_keep] += rho_full[a, b]

    return rho_reduced


def compute_qbeads(state_vector: List[complex], n_qubits: int) -> List[QBead]:
    """
    Compute Q-bead properties for each qubit.

    Uses reduced density matrices to determine:
    - P(0) and P(1) probabilities for coloring
    - Polar angle θ for phase indication

    Args:
        state_vector: State vector of length 2^n_qubits
        n_qubits: Number of qubits

    Returns:
        List of QBead objects, one per qubit
    """
    qbeads = []

    for q in range(n_qubits):
        rho_red = compute_reduced_density_matrix(state_vector, n_qubits, q)

        # Probability of measuring |0⟩ and |1⟩
        prob_0 = float(np.real(rho_red[0, 0]))
        prob_1 = float(np.real(rho_red[1, 1]))

        # Compute Bloch vector components
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

        rx = float(np.real(np.trace(rho_red @ sigma_x)))
        ry = float(np.real(np.trace(rho_red @ sigma_y)))
        rz = float(np.real(np.trace(rho_red @ sigma_z)))

        # Convert to spherical coordinates
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        if r < 1e-10:
            theta = np.pi / 2  # Undefined, default to equator
            phi = 0.0
        else:
            theta = np.arccos(np.clip(rz / r, -1, 1))
            phi = np.arctan2(ry, rx)
            if phi < 0:
                phi += 2 * np.pi

        qbeads.append(QBead(
            qubit_index=q,
            prob_0=prob_0,
            prob_1=prob_1,
            theta=theta,
            phi=phi
        ))

    return qbeads


def compute_ebeads(state_vector: List[complex], n_qubits: int) -> List[EBead]:
    """
    Compute E-bead (entanglement) properties.

    Uses correlation functions to detect and quantify entanglement.
    For simplicity, uses pairwise concurrence-like measure.

    Args:
        state_vector: State vector of length 2^n_qubits
        n_qubits: Number of qubits

    Returns:
        List of EBead objects representing entanglement
    """
    ebeads = []

    if n_qubits < 2:
        return ebeads

    # Compute pairwise entanglement for each qubit pair
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            # Compute reduced density matrix for the pair
            rho_pair = compute_pair_density_matrix(state_vector, n_qubits, i, j)

            # Compute concurrence (simplified version)
            # For pure states: C = 2 * |c00*c11 - c01*c10|
            # More general: compute eigenvalues of sqrt(rho * rho.flip())
            try:
                # Compute rho ~rho (spin-flipped)
                sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
                rho_tilde = np.kron(sigma_y, sigma_y) @ rho_conj(rho_pair) @ np.kron(sigma_y, sigma_y)

                # Eigenvalues of rho * rho_tilde
                product = rho_pair @ rho_tilde
                eigenvalues = np.linalg.eigvals(product)
                eigenvalues = np.sqrt(np.abs(eigenvalues))
                eigenvalues = np.sort(eigenvalues)[::-1]

                # Concurrence = sum of positive sqrt eigenvalues
                concurrence = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] + eigenvalues[3])
            except Exception:
                concurrence = 0.0

            # Only include if there's meaningful entanglement
            if concurrence > 0.01:
                ebeads.append(EBead(
                    qubits=(i, j),
                    entanglement_strength=concurrence,
                    entanglement_type="pairwise"
                ))

    # Compute global entanglement (measure of total system entanglement)
    # Future: could add higher-order entanglement detection here

    return ebeads


def rho_conj(rho: np.ndarray) -> np.ndarray:
    """Complex conjugate of density matrix."""
    return np.conj(rho).T


def compute_pair_density_matrix(
    state_vector: List[complex],
    n_qubits: int,
    qubit_a: int,
    qubit_b: int
) -> np.ndarray:
    """Compute reduced density matrix for a pair of qubits."""
    rho_full = state_to_density(state_vector)
    dim = 2 ** n_qubits

    # Trace out all qubits except qubit_a and qubit_b
    rho_pair = np.zeros((4, 4), dtype=complex)

    for i in range(4):
        for j in range(4):
            # Convert 2-qubit indices to full system indices
            # Big-endian: qubit_b is more significant than qubit_a
            idx_a = 1 << qubit_a
            idx_b = 1 << qubit_b

            # Sum over all other qubit states
            for other in range(dim):
                # Skip if 'other' bits overlap with a or b
                mask = idx_a | idx_b
                if other & mask != 0:
                    continue

                full_i = i | (other & ~mask)
                full_j = j | (other & ~mask)

                if full_i < dim and full_j < dim:
                    rho_pair[i, j] += rho_full[full_i, full_j]

    return rho_pair


def prob_0_to_color(prob_0: float) -> Tuple[float, float, float]:
    """
    Convert P(0) probability to RGB color.

    Red (1,0,0) = P(0) = 1 (definite |0⟩)
    Yellow (1,1,0) = P(0) = 0.5 (superposition)
    Green (0,1,0) = P(0) = 0 (definite |1⟩)

    Args:
        prob_0: Probability of measuring |0⟩ (0 to 1)

    Returns:
        RGB tuple (0 to 1)
    """
    prob_0 = np.clip(prob_0, 0, 1)

    if prob_0 >= 0.5:
        # Yellow (0.5) to Red (1.0): G goes from 1 to 0
        t = 2 * (prob_0 - 0.5)  # 0 to 1
        r = 1.0
        g = 1.0 - t
        b = 0.0
    else:
        # Green (0) to Yellow (0.5): R goes from 0 to 1
        t = 2 * prob_0  # 0 to 1
        r = t
        g = 1.0
        b = 0.0

    return (r, g, b)


def entanglement_to_color(strength: float) -> Tuple[float, float, float]:
    """
    Convert entanglement strength to E-bead color.

    Yellow (1,1,0) = Maximum entanglement (strength = 1)
    Blue (0,0,1) = Separable state (strength = 0)

    Args:
        strength: Entanglement strength (0 to 1)

    Returns:
        RGB tuple
    """
    strength = np.clip(strength, 0, 1)
    # strength=1 → yellow (1,1,0), strength=0 → blue (0,0,1)
    return (strength, strength, 1 - strength)


def plot_bead_sphere(
    ax: Axes3D,
    center: Tuple[float, float, float],
    radius: float,
    color: Tuple[float, float, float],
    alpha: float = 0.9,
    label: Optional[str] = None
) -> None:
    """Plot a single 3D sphere with given properties."""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    ax.plot_surface(x, y, z, color=color, alpha=alpha, shade=True)

    if label:
        ax.text(center[0], center[1] + radius * 1.3, center[2],
                label, fontsize=10, ha='center')


def plot_qbeads(
    state: Union[List[complex], Any],
    title: str = "BEADS Visualization",
    output_path: Optional[str] = None,
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    Create a BEADS visualization (Q-beads + E-beads).

    Args:
        state: List of complex amplitudes (length = 2^n_qubits),
               or Qiskit Statevector/DensityMatrix
        title: Title for the plot
        output_path: Path to save figure (if None, returns figure)
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure if output_path is None, else None
    """
    from quantumviz.qiskit_bridge import (
        density_matrix_to_array,
        is_density_matrix,
        is_statevector,
        statevector_to_list,
    )

    # Handle Qiskit objects
    if is_statevector(state):
        state_vector = statevector_to_list(state)
        n_qubits = state.num_qubits
    elif is_density_matrix(state):
        rho = density_matrix_to_array(state)
        state_vector = None  # Need to convert
        n_qubits = int(np.log2(rho.shape[0]))
        # Extract pure state from density matrix (simplified)
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        idx = np.argmax(eigenvalues)
        state_vector = eigenvectors[:, idx].tolist()
    else:
        state_vector = state
        n_qubits = int(np.log2(len(state_vector)))

    # Compute beads
    qbeads = compute_qbeads(state_vector, n_qubits)
    ebeads = compute_ebeads(state_vector, n_qubits)

    # Calculate layout
    n_cols = max(n_qubits, 2)
    fig_width = max(14, n_cols * 4)
    fig = plt.figure(figsize=(fig_width, 8))
    fig.suptitle(title, fontsize=16, y=0.95)

    # Q-beads subplot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Q-Beads (Qubit States)", fontsize=12, pad=10)

    # Position Q-beads in a line
    bead_spacing = 3.0
    for i, qbead in enumerate(qbeads):
        x_pos = (i - (n_qubits - 1) / 2) * bead_spacing
        color = prob_0_to_color(qbead.prob_0)
        plot_bead_sphere(ax1, (x_pos, 0, 0), 1.0, color, label=f"Q{i}")
        ax1.text(x_pos, 0, -1.8, f"P(0)={qbead.prob_0:.2f}", ha='center', fontsize=9)

    ax1.set_xlim(-bead_spacing * n_qubits / 2 - 1, bead_spacing * n_qubits / 2 + 1)
    ax1.set_ylim(-2, 2)
    ax1.set_zlim(-2, 2)
    ax1.set_axis_off()

    # E-beads subplot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("E-Beads (Entanglement)", fontsize=12, pad=10)

    # Position E-beads between Q-beads
    if ebeads:
        for ebead in ebeads:
            i, j = ebead.qubits
            x_i = (i - (n_qubits - 1) / 2) * bead_spacing
            x_j = (j - (n_qubits - 1) / 2) * bead_spacing

            # Draw connection line
            ax2.plot([x_i, x_j], [0, 0], [0, 0], 'gray', linewidth=2, alpha=0.5)

            # Draw E-bead sphere at midpoint
            mid_x = (x_i + x_j) / 2
            color = entanglement_to_color(ebead.entanglement_strength)
            plot_bead_sphere(ax2, (mid_x, 0, 0), 0.6, color, label=f"E{i}-{j}")
            ax2.text(mid_x, 0, -1.2, f"strength={ebead.entanglement_strength:.2f}",
                    ha='center', fontsize=9)
    else:
        ax2.text(0, 0, 0, "No entanglement detected", ha='center', fontsize=12)

    # Also show Q-beads in E-bead plot for reference
    for i, qbead in enumerate(qbeads):
        x_pos = (i - (n_qubits - 1) / 2) * bead_spacing
        # Semi-transparent reference
        color = prob_0_to_color(qbead.prob_0)
        plot_bead_sphere(ax2, (x_pos, 0, 0), 0.5, color, alpha=0.3)

    ax2.set_xlim(-bead_spacing * n_qubits / 2 - 1, bead_spacing * n_qubits / 2 + 1)
    ax2.set_ylim(-2, 2)
    ax2.set_zlim(-2, 2)
    ax2.set_axis_off()

    # Add color legend
    ax_legend = fig.add_axes([0.02, 0.02, 0.15, 0.12])
    ax_legend.set_title("Q-Bead Colors", fontsize=9)
    ax_legend.axis('off')

    # Q-bead legend gradient
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    ax_legend.imshow(gradient, aspect='auto', cmap=None,
                    extent=(0, 1, 0, 1), origin='lower')
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.text(0, -0.2, "|0⟩", fontsize=9)
    ax_legend.text(0.5, -0.2, "50/50", fontsize=9)
    ax_legend.text(1, -0.2, "|1⟩", fontsize=9, ha='right')
    ax_legend.text(-0.1, 0.5, "Red", fontsize=8, rotation=90, va='center')
    ax_legend.text(1.1, 0.5, "Green", fontsize=8, rotation=90, va='center')

    # E-bead legend
    ax_legend2 = fig.add_axes([0.18, 0.02, 0.12, 0.12])
    ax_legend2.set_title("E-Bead Colors", fontsize=9)
    ax_legend2.axis('off')
    ax_legend2.text(0.5, 0.8, "Yellow = Entangled", fontsize=8, ha='center')
    ax_legend2.text(0.5, 0.5, "Blue = Separable", fontsize=8, ha='center')
    ax_legend2.text(0.5, 0.2, f"Qubits: {n_qubits}", fontsize=8, ha='center')

    plt.tight_layout(rect=[0, 0.15, 1, 0.92])

    if output_path:
        plt.savefig(output_path, dpi=dpi)
        plt.close()
    else:
        return fig


def plot_qbeads_from_file(
    input_file: str,
    output_dir: Optional[str] = None,
    dpi: int = 150,
    fmt: str = "png"
) -> List[str]:
    """
    Plot BEADS visualization from JSON input file.

    Supports formats:
    - {"states": [[...], ...]}
    - {"state_vector": [...]}
    - {"qubits": N, "stages": [...]}
    """
    try:
        from quantumviz.qiskit_bridge import is_statevector, statevector_to_list
    except ImportError:
        is_statevector = None  # type: ignore
        statevector_to_list = None  # type: ignore

    with open(input_file, 'r') as f:
        data = json.load(f)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_files = []

    # Format 1: Array of states
    if 'states' in data:
        states = data['states']
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        for i, state in enumerate(states):
            state_vec = [parse_amplitude(amp) for amp in state]
            name = f"{base_name}_{i+1:02d}"
            if output_dir:
                filename = f"{output_dir}/{name}.{fmt}"
            else:
                filename = f"{name}.{fmt}"
            plot_qbeads(state_vec, name, filename, dpi)
            output_files.append(filename)
            print(f"Saved: {filename}")

    # Format 2: Single state vector
    elif 'state_vector' in data:
        state_vector = [parse_amplitude(amp) for amp in data['state_vector']]
        name = data.get('name', 'BEADS')
        if output_dir:
            filename = f"{output_dir}/{name.replace(' ', '_')}.{fmt}"
        else:
            filename = f"{name.replace(' ', '_')}.{fmt}"
        plot_qbeads(state_vector, name, filename, dpi)
        output_files.append(filename)
        print(f"Saved: {filename}")

    # Format 3: Stages
    elif 'stages' in data:
        _ = data['qubits']  # Required but not used - n_qubits inferred from state_vector
        stages = data['stages']
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        for i, stage in enumerate(stages):
            name = stage.get('name', f'Stage {i+1}')
            raw_state = stage['state_vector']

            if is_statevector(raw_state):
                state_vector = statevector_to_list(raw_state)
            else:
                state_vector = [parse_amplitude(amp) for amp in raw_state]

            if output_dir:
                filename = f"{output_dir}/stage_{i+1:02d}_{name.replace(' ', '_')}.{fmt}"
            else:
                filename = f"stage_{i+1:02d}_{name.replace(' ', '_')}.{fmt}"

            plot_qbeads(state_vector, name, filename, dpi)
            output_files.append(filename)
            print(f"Saved: {filename}")

    else:
        raise ValueError("Input must contain 'states', 'state_vector', or 'stages'")

    return output_files


def main(args: Optional[List[str]] = None) -> None:
    """CLI entry point for BEADS visualization."""
    import sys
    if args is None:
        args = sys.argv[1:]

    if len(args) != 1:
        print("Usage: python -m quantumviz.qbeads <input_file.json>")
        sys.exit(1)

    input_file = args[0]
    plot_qbeads_from_file(input_file)


if __name__ == "__main__":
    main()