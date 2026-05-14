"""
BEADS (Quantum Beads) Visualization Module

Provides visualization of quantum states using the BEADS representation:
- Q-Beads: One sphere per qubit (red-green color scheme)
  Red = high |0⟩ probability, Green = high |1⟩ probability
- E-Beads: Yellow-blue spheres representing entanglement between qubits
- C-Beads: Compound correlation functions
- T-Beads: Total correlation functions (E + C) with color wheel

Based on: Huber & Glaser, "BEADS: A canonical visualization of quantum states
for applications in quantum information processing" (2025)

Display variants (Table 4):
- A: Q + T-Beads, separate symmetries (default)
- B: Q + T-Beads, single color scheme
- F: Q + E-Beads only (entanglement focus)
- H: Q + E-Beads, symmetric only (simplified)
- I: Q-Beads only (educational)
- J: Q-Beads only, no arcs (minimal)
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

matplotlib.use('Agg')


# ============================================================================
# Display Variant Configuration
# ============================================================================

class DisplayVariant(str, Enum):
    """BEADS display variants from Table 4."""
    A = "A"  # Q + T, separate sym, complete
    B = "B"  # Q + T, single color, complete
    C = "C"  # Q + T, combined sym, complete
    D = "D"  # Q + E + C, separate sym, complete
    E = "E"  # Q + E + C, combined sym, complete
    F = "F"  # Q + E only, separate sym, complete
    G = "G"  # Q + E only, combined sym, complete
    H = "H"  # Q + E, symmetric only, incomplete
    I = "I"  # Q only, with arcs, incomplete
    J = "J"  # Q only, no arcs, incomplete


@dataclass
class BeadsDisplayConfig:
    """Configuration for BEADS display variant."""
    variant: DisplayVariant = DisplayVariant.A
    show_q_beads: bool = True
    show_e_beads: bool = True
    show_c_beads: bool = False
    show_t_beads: bool = False
    show_entanglement_arcs: bool = True
    color_scheme: str = "standard"  # "standard", "continuous", "high_contrast", "colorblind", "grayscale"
    point_symmetry: str = "separate"  # "separate", "combined"
    perm_symmetry: str = "all"  # "all", "symmetric_only", "tau1_only"
    complete: bool = True

    @classmethod
    def from_variant(cls, variant: str) -> "BeadsDisplayConfig":
        """Create config from variant letter (Table 4)."""
        v = DisplayVariant(variant)
        config = cls(variant=v)

        if v == DisplayVariant.A:
            config.show_q_beads = True
            config.show_t_beads = True
            config.show_e_beads = False
            config.show_c_beads = False
            config.point_symmetry = "separate"
            config.perm_symmetry = "all"
            config.complete = True
        elif v == DisplayVariant.B:
            config.show_q_beads = True
            config.show_t_beads = True
            config.show_e_beads = False
            config.show_c_beads = False
            config.point_symmetry = "separate"
            config.perm_symmetry = "all"
            config.complete = True
        elif v == DisplayVariant.F:
            config.show_q_beads = True
            config.show_e_beads = True
            config.show_t_beads = False
            config.show_c_beads = False
            config.point_symmetry = "separate"
            config.perm_symmetry = "all"
            config.complete = True
        elif v == DisplayVariant.H:
            config.show_q_beads = True
            config.show_e_beads = True
            config.show_t_beads = False
            config.show_c_beads = False
            config.point_symmetry = "separate"
            config.perm_symmetry = "symmetric_only"
            config.complete = False
        elif v == DisplayVariant.I:
            config.show_q_beads = True
            config.show_e_beads = False
            config.show_t_beads = False
            config.show_c_beads = False
            config.show_entanglement_arcs = True
            config.complete = False
        elif v == DisplayVariant.J:
            config.show_q_beads = True
            config.show_e_beads = False
            config.show_t_beads = False
            config.show_c_beads = False
            config.show_entanglement_arcs = False
            config.complete = False

        return config


# ============================================================================
# Bead Data Classes
# ============================================================================

@dataclass
class QBead:
    """Represents a single Q-Bead (qubit bead)."""
    qubit_index: int
    prob_0: float  # Probability of measuring |0⟩
    prob_1: float  # Probability of measuring |1⟩
    theta: float  # Polar angle (0 = |0⟩, π = |1⟩)
    phi: float  # Azimuthal angle
    bloch_vector: Tuple[float, float, float]  # (rx, ry, rz)
    lisa_coefficients: Dict[int, Dict[int, complex]] = field(default_factory=dict)


@dataclass
class EBead:
    """Represents an E-Bead (entanglement bead)."""
    qubits: Tuple[int, ...]  # Which qubits are entangled
    strength: float  # 0 = separable, 1 = max entanglement
    bead_type: str  # "pairwise", "triple", "global"
    symmetry: str  # "even" or "odd"
    tau: Optional[str] = None  # tau label if applicable
    lisa_coefficients: Dict[int, Dict[int, complex]] = field(default_factory=dict)
    connected_correlations: Dict[str, float] = field(default_factory=dict)


@dataclass
class CBead:
    """Represents a C-Bead (compound correlation bead)."""
    qubits: Tuple[int, ...]
    strength: float
    lisa_coefficients: Dict[int, Dict[int, complex]] = field(default_factory=dict)


@dataclass
class TBead:
    """Represents a T-Bead (total correlation bead)."""
    qubits: Tuple[int, ...]
    total_correlation: float
    connected_part: float  # E
    compound_part: float  # C
    lisa_coefficients: Dict[int, Dict[int, complex]] = field(default_factory=dict)


# ============================================================================
# Core Computation Functions
# ============================================================================

def state_to_density(state_vector: List[complex]) -> np.ndarray:
    """Convert state vector to density matrix: ρ = |ψ⟩⟨ψ|."""
    psi = np.array(state_vector, dtype=complex).reshape(-1, 1)
    return (psi @ psi.conj().T).astype(complex)


def compute_reduced_density_matrix(
    state_vector: List[complex],
    n_qubits: int,
    qubit_idx: int
) -> np.ndarray:
    """
    Compute reduced density matrix for a single qubit by tracing out others.

    Uses the standard partial trace formula.
    """
    rho_full = state_to_density(state_vector)
    dim = 2 ** n_qubits

    if n_qubits == 1:
        return rho_full

    rho_reduced = np.zeros((2, 2), dtype=complex)

    for a in range(dim):
        for b in range(dim):
            # Check if they differ only in traced-out qubits
            bit_a_keep = (a >> qubit_idx) & 1
            bit_b_keep = (b >> qubit_idx) & 1

            # Sum over traced-out qubits
            match = True
            for q in range(n_qubits):
                if q != qubit_idx:
                    if ((a >> q) & 1) != ((b >> q) & 1):
                        match = False
                        break
            if match:
                rho_reduced[bit_a_keep, bit_b_keep] += rho_full[a, b]

    return rho_reduced


def compute_pair_density_matrix(
    state_vector: List[complex],
    n_qubits: int,
    qubit_a: int,
    qubit_b: int
) -> np.ndarray:
    """
    Compute reduced density matrix for a pair of qubits by tracing out others.

    Uses partial trace: ρ_ab = Tr_others(|ψ⟩⟨ψ|)
    """
    rho_full = state_to_density(state_vector)
    dim = 2 ** n_qubits

    rho_pair = np.zeros((4, 4), dtype=complex)

    # Build mask for qubits a and b
    mask_keep = (1 << qubit_a) | (1 << qubit_b)
    mask_trace = ((1 << n_qubits) - 1) & ~mask_keep

    for a in range(dim):
        for b in range(dim):
            # Check if they match on traced-out qubits
            if (a & mask_trace) != (b & mask_trace):
                continue

            # Extract 2-qubit indices from a and b
            a_ab = 0
            b_ab = 0
            if (a >> qubit_a) & 1:
                a_ab |= 2
            if (a >> qubit_b) & 1:
                a_ab |= 1
            if (b >> qubit_a) & 1:
                b_ab |= 2
            if (b >> qubit_b) & 1:
                b_ab |= 1

            rho_pair[a_ab, b_ab] += rho_full[a, b]

    return rho_pair


def compute_qbeads(
    state_vector: List[complex],
    n_qubits: int,
    N: Optional[int] = None
) -> List[QBead]:
    """
    Compute Q-Bead properties for each qubit using LISA decomposition.

    Args:
        state_vector: State vector of length 2^n_qubits
        n_qubits: Number of qubits
        N: Total qubits in system (defaults to n_qubits)

    Returns:
        List of QBead objects, one per qubit
    """
    if N is None:
        N = n_qubits

    from quantumviz.beads_scaling import qbead_scaling

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

        # Get LISA coefficients
        scaling = qbead_scaling(N, q)
        lisa_coeffs = {
            1: {
                0: complex(rz * np.sqrt(2**N) / np.sqrt(4*np.pi/3), 0),
            }
        }

        qbeads.append(QBead(
            qubit_index=q,
            prob_0=prob_0,
            prob_1=prob_1,
            theta=theta,
            phi=phi,
            bloch_vector=(rx, ry, rz),
            lisa_coefficients=lisa_coeffs
        ))

    return qbeads


def compute_ebeads(
    state_vector: List[complex],
    n_qubits: int,
    N: Optional[int] = None
) -> List[EBead]:
    """
    Compute E-Beads (entanglement beads) using Ursell function separation.

    Uses correlation function separation: E = T - C
    """
    if N is None:
        N = n_qubits

    from quantumviz import ursell
    from quantumviz.beads_scaling import ebead_even_scaling, ebead_odd_scaling

    ebeads = []

    if n_qubits < 2:
        return ebeads

    # Compute pairwise E-Beads
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            rho_pair = compute_pair_density_matrix(state_vector, n_qubits, i, j)
            rho1 = compute_reduced_density_matrix(state_vector, n_qubits, i)
            rho2 = compute_reduced_density_matrix(state_vector, n_qubits, j)

            # Compute connected correlation (zz component)
            E_zz = ursell.bilinear_connected_correlation(rho_pair, rho1, rho2, 'z', 'z')

            # Only include if there's meaningful entanglement
            if abs(E_zz) > 0.01:
                # Even (symmetric) component
                scaling_even = ebead_even_scaling(N, i, j)
                lisa_coeffs_even = {
                    0: {'0': complex(E_zz * 0.5, 0)},
                    2: {'0': complex(E_zz * 0.5, 0)},
                }

                ebeads.append(EBead(
                    qubits=(i, j),
                    strength=abs(E_zz),
                    bead_type="pairwise",
                    symmetry="even",
                    tau=None,
                    lisa_coefficients=lisa_coeffs_even,
                    connected_correlations={'zz': E_zz}
                ))

                # Odd (antisymmetric) component - simplified
                scaling_odd = ebead_odd_scaling(N, i, j)
                E_odd = ursell.bilinear_total_correlation(rho_pair, 'x', 'z') - \
                       ursell.bilinear_compound_correlation(rho1, rho2, 'x', 'z')

                if abs(E_odd) > 0.01:
                    lisa_coeffs_odd = {
                        1: {'0': complex(E_odd, 0)},
                    }
                    ebeads.append(EBead(
                        qubits=(i, j),
                        strength=abs(E_odd),
                        bead_type="pairwise",
                        symmetry="odd",
                        tau=None,
                        lisa_coefficients=lisa_coeffs_odd,
                        connected_correlations={'xz': E_odd}
                    ))

    return ebeads


# ============================================================================
# 3D Rendering Functions
# ============================================================================

def plot_bead_sphere(
    ax: Axes3D,
    center: Tuple[float, float, float],
    radius: float,
    color: Tuple[float, float, float, float],
    alpha: float = 0.9,
    label: Optional[str] = None,
    show_prob: bool = False,
    prob_value: float = 0.5
):
    """Plot a single 3D sphere with given color."""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    ax.plot_surface(x, y, z, color=color[:3], alpha=alpha, shade=True)

    if label:
        ax.text(center[0], center[1] + radius * 1.3, center[2],
                label, fontsize=10, ha='center')

    if show_prob:
        ax.text(center[0], center[1], center[2] - radius * 1.1,
                f'P(0)={prob_value:.2f}', fontsize=8, ha='center')


def plot_bead_with_function(
    ax: Axes3D,
    center: Tuple[float, float, float],
    radius: float,
    coefficients: Dict[int, Dict[int, complex]],
    scaling_factors: Dict[int, float],
    color_scheme: str = "standard",
    bead_type: str = "Q",
    alpha: float = 0.9,
    label: Optional[str] = None,
    n_theta: int = 30,
    n_phi: int = 30
):
    """
    Plot a bead sphere with spherical function-based coloring.

    Uses the BEADS mapping: b(θ, φ) = Σ s_j * c_{j,m} * Y_{j,m}(θ, φ)
    """
    from quantumviz.spherical_beads import bead_surface_with_values
    from quantumviz.beads_colors import value_to_color_array

    # Generate mesh
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    X = center[0] + radius * np.sin(theta_grid) * np.cos(phi_grid)
    Y = center[1] + radius * np.sin(theta_grid) * np.sin(phi_grid)
    Z = center[2] + radius * np.cos(theta_grid)

    # Compute bead function values
    from quantumviz.spherical_beads import bead_function_grid
    values = bead_function_grid(theta_grid, phi_grid, coefficients, scaling_factors)

    # Map values to colors
    colors = value_to_color_array(values, color_scheme, bead_type)

    # Plot with facecolors
    ax.plot_surface(X, Y, Z, facecolors=colors, alpha=alpha, shade=False)

    if label:
        ax.text(center[0], center[1] + radius * 1.3, center[2],
                label, fontsize=10, ha='center')


def plot_entanglement_arc(
    ax: Axes3D,
    qbead1: QBead,
    qbead2: QBead,
    strength: float,
    x_offset: float = 0,
    y_offset: float = 0,
    z_offset: float = 0
):
    """Plot entanglement arc between two Q-Beads."""
    x1 = (qbead1.qubit_index - x_offset) * 3.0
    x2 = (qbead2.qubit_index - x_offset) * 3.0

    # Line width proportional to entanglement strength
    linewidth = 1.0 + 3.0 * strength

    ax.plot([x1, x2], [0, 0], [0, 0], 'gray', linewidth=linewidth, alpha=0.5)


# ============================================================================
# Main Plotting Function
# ============================================================================

def plot_qbeads(
    state: Union[List[complex], Any],
    title: str = "BEADS Visualization",
    output_path: Optional[str] = None,
    dpi: int = 150,
    config: Optional[BeadsDisplayConfig] = None,
    variant: str = "A",
    color_scheme: str = "standard"
) -> Optional[plt.Figure]:
    """
    Create a BEADS visualization (Q-Beads + E-Beads + optional T/C-Beads).

    Args:
        state: List of complex amplitudes, Qiskit Statevector, or DensityMatrix
        title: Title for the plot
        output_path: Path to save figure (if None, returns figure)
        dpi: Resolution for saved figure
        config: Display configuration (overrides variant/color_scheme if provided)
        variant: Display variant (A-J)
        color_scheme: Color scheme name

    Returns:
        matplotlib Figure if output_path is None, else None
    """
    from quantumviz.qiskit_bridge import (
        density_matrix_to_array,
        is_density_matrix,
        is_statevector,
        statevector_to_list,
    )
    from quantumviz.beads_colors import (
        create_colorbar,
        standard_red_green,
        standard_yellow_blue,
    )

    # Parse config
    if config is None:
        config = BeadsDisplayConfig.from_variant(variant)
        config.color_scheme = color_scheme

    # Handle Qiskit objects
    if is_statevector(state):
        state_vector = statevector_to_list(state)
        n_qubits = state.num_qubits
    elif is_density_matrix(state):
        rho = density_matrix_to_array(state)
        # Extract dominant state
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        idx = np.argmax(eigenvalues)
        state_vector = eigenvectors[:, idx].tolist()
        n_qubits = int(np.log2(rho.shape[0]))
    else:
        state_vector = state
        n_qubits = int(np.log2(len(state_vector)))

    # Compute N (total qubits in system)
    N = n_qubits

    # Compute beads
    qbeads_list = compute_qbeads(state_vector, n_qubits, N)
    ebeads_list = compute_ebeads(state_vector, n_qubits, N) if config.show_e_beads else []

    # Calculate layout
    n_cols = max(n_qubits, 2) if config.show_e_beads else n_qubits
    fig_width = max(14, n_cols * 4)
    fig = plt.figure(figsize=(fig_width, 8))
    fig.suptitle(title, fontsize=16, y=0.95)

    # Q-Beads subplot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Q-Beads (Qubit States)", fontsize=12, pad=10)

    # Position Q-Beads in a line
    bead_spacing = 3.0
    x_offset = (n_qubits - 1) / 2

    from quantumviz.beads_colors import prob_0_to_color

    for i, qbead in enumerate(qbeads_list):
        x_pos = (i - x_offset) * bead_spacing
        color = prob_0_to_color(qbead.prob_0)

        if config.variant in ('A', 'B', 'C'):
            # Use spherical function coloring
            from quantumviz.beads_scaling import qbead_scaling
            scaling = qbead_scaling(N, i)
            plot_bead_with_function(
                ax1, (x_pos, 0, 0), 1.0,
                qbead.lisa_coefficients, scaling,
                config.color_scheme, 'Q',
                label=f'Q{i}',
                n_theta=30, n_phi=30
            )
        else:
            # Simple uniform coloring
            plot_bead_sphere(ax1, (x_pos, 0, 0), 1.0,
                            color, label=f'Q{i}')
            ax1.text(x_pos, 0, -1.8, f"P(0)={qbead.prob_0:.2f}",
                     ha='center', fontsize=9)

    ax1.set_xlim(-bead_spacing * n_qubits / 2 - 1, bead_spacing * n_qubits / 2 + 1)
    ax1.set_ylim(-2, 2)
    ax1.set_zlim(-2, 2)
    ax1.set_axis_off()

    # E-Beads subplot (or T-Beads)
    if config.show_e_beads or config.show_t_beads:
        ax2 = fig.add_subplot(122, projection='3d')
        if config.show_t_beads:
            ax2.set_title("T-Beads (Total Correlations)", fontsize=12, pad=10)
        else:
            ax2.set_title("E-Beads (Entanglement)", fontsize=12, pad=10)

        # Position E-Beads between Q-Beads
        if ebeads_list:
            for ebead in ebeads_list:
                if len(ebead.qubits) >= 2:
                    i, j = ebead.qubits[0], ebead.qubits[1]
                    x_i = (i - x_offset) * bead_spacing
                    x_j = (j - x_offset) * bead_spacing

                    # Draw connection line if arcs enabled
                    if config.show_entanglement_arcs:
                        ax2.plot([x_i, x_j], [0, 0], [0, 0],
                                 'gray', linewidth=2, alpha=0.5)

                    # Draw E-Bead sphere at midpoint
                    mid_x = (x_i + x_j) / 2

                    if ebead.strength > 0.01:
                        from quantumviz.beads_colors import standard_yellow_blue
                        color = standard_yellow_blue(ebead.strength)
                        plot_bead_sphere(ax2, (mid_x, 0, 0), 0.6,
                                        color, alpha=0.8,
                                        label=f'E{i}-{j}')
                        ax2.text(mid_x, 0, -1.2,
                                 f'str={ebead.strength:.2f}',
                                 ha='center', fontsize=8)
        else:
            ax2.text(0, 0, 0, "No entanglement detected",
                     ha='center', fontsize=12)

        # Show Q-Beads in E-Bead plot for reference
        for i, qbead in enumerate(qbeads_list):
            x_pos = (i - x_offset) * bead_spacing
            color = prob_0_to_color(qbead.prob_0)
            plot_bead_sphere(ax2, (x_pos, 0, 0), 0.5,
                            color, alpha=0.3)

        ax2.set_xlim(-bead_spacing * n_qubits / 2 - 1, bead_spacing * n_qubits / 2 + 1)
        ax2.set_ylim(-2, 2)
        ax2.set_zlim(-2, 2)
        ax2.set_axis_off()
    else:
        # Single plot if no E-Beads
        fig.delaxes(fig.axes[1])
        ax1.set_position([0.1, 0.1, 0.8, 0.8])

    # Add color legend
    ax_legend = fig.add_axes([0.02, 0.02, 0.15, 0.12])
    ax_legend.set_title("Q-Bead Colors", fontsize=9)
    ax_legend.axis('off')

    # Q-Bead legend gradient
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    ax_legend.imshow(gradient, aspect='auto',
                   extent=(0, 1, 0, 1), origin='lower')
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.text(0, -0.2, "|0⟩", fontsize=9)
    ax_legend.text(0.5, -0.2, "50/50", fontsize=9)
    ax_legend.text(1, -0.2, "|1⟩", fontsize=9, ha='right')
    ax_legend.text(-0.1, 0.5, "Red", fontsize=8, rotation=90, va='center')
    ax_legend.text(1.1, 0.5, "Green", fontsize=8, rotation=90, va='center')

    # E-Bead legend
    if config.show_e_beads:
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


# ============================================================================
# File I/O and CLI Support
# ============================================================================

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


def plot_qbeads_from_file(
    input_file: str,
    output_dir: Optional[str] = None,
    dpi: int = 150,
    fmt: str = "png",
    variant: str = "A",
    color_scheme: str = "standard"
) -> List[str]:
    """
    Plot BEADS visualization from JSON input file.

    Supports formats:
    - {"states": [[...], ...]}
    - {"state_vector": [...]}
    - {"qubits": N, "stages": [...]}
    """
    from quantumviz.qiskit_bridge import is_statevector, statevector_to_list

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
            plot_qbeads(state_vec, name, filename, dpi,
                       variant=variant, color_scheme=color_scheme)
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
        plot_qbeads(state_vector, name, filename, dpi,
                   variant=variant, color_scheme=color_scheme)
        output_files.append(filename)
        print(f"Saved: {filename}")

    # Format 3: Stages
    elif 'stages' in data:
        _ = data['qubits']  # Required but not used
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

            plot_qbeads(state_vector, name, filename, dpi,
                       variant=variant, color_scheme=color_scheme)
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
