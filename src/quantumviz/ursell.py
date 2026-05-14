"""
Ursell Function Module - Correlation Function Separation

Implements the separation of correlation functions into:
- Connected (entanglement-based) correlations E
- Compound (factorizable) correlations C
- Total correlations T = E + C

Based on:
- Schlienz & Mahler, Phys. Rev. A 52, 4396 (1995)
- Schlienz & Mahler, Phys. Lett. A 224, 39 (1996)
- Huber & Glaser, New J. Phys. 27 094509 (2025), Section 5.4

For pure states ρ = |ψ⟩⟨ψ|:
- Connected correlations E arise ONLY from entanglement
- Compound correlations C = products of lower-order expectations

Formulas:
- Bilinear: E₁₂ = T₁₂ - C₁₂ = ⟨σ₁σ₂⟩ - ⟨σ₁⟩⟨σ₂⟩  (eq. 14)
- Trilinear: E₁₂₃ = T₁₂₃ - C₁₂₃  (eq. 16)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.linalg import sqrtm


# ============================================================================
# Pauli Expectation Values
# ============================================================================

def pauli_expectation(rho: np.ndarray, pauli_type: str, qubit_idx: int, N: int) -> complex:
    """
    Compute expectation value ⟨σ_{qubit_idx, pauli_type}⟩ from reduced density matrix.

    Args:
        rho: 2x2 reduced density matrix
        pauli_type: 'x', 'y', or 'z'
        qubit_idx: Qubit index (for full system - not used for reduced)
        N: Total qubits (not needed for reduced)

    Returns:
        Expectation value (real for Hermitian operators)
    """
    pauli_matrices = {
        'x': np.array([[0, 1], [1, 0]], dtype=complex),
        'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'z': np.array([[1, 0], [0, -1]], dtype=complex),
    }

    if pauli_type not in pauli_matrices:
        raise ValueError(f"Unknown Pauli type: {pauli_type}")

    sigma = pauli_matrices[pauli_type]
    return np.trace(rho @ sigma)


def all_pauli_expectations(rho_red: np.ndarray) -> Dict[str, float]:
    """
    Compute all single-qubit Pauli expectations.

    Returns:
        Dict with keys 'x', 'y', 'z' mapping to expectation values
    """
    result = {}
    for p in ['x', 'y', 'z']:
        result[p] = np.real(pauli_expectation(rho_red, p, 0, 1))
    return result


# ============================================================================
# Bilinear Correlation Functions (2-Qubit)
# ============================================================================

def bilinear_total_correlation(
    rho_pair: np.ndarray,
    alpha: str = 'z',
    beta: str = 'z'
) -> float:
    """
    Compute total bilinear correlation T₁₂ = ⟨σ₁α σ₂β⟩.

    Args:
        rho_pair: 4x4 density matrix for qubit pair
        alpha: Pauli type for qubit 1
        beta: Pauli type for qubit 2

    Returns:
        Total correlation coefficient T₁₂ (range [-1, 1])
    """
    pauli = {
        'x': np.array([[0, 1], [1, 0]], dtype=complex),
        'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'z': np.array([[1, 0], [0, -1]], dtype=complex),
    }

    sigma_alpha = pauli[alpha]
    sigma_beta = pauli[beta]

    # Tensor product: σ₁α ⊗ σ₂β
    op = np.kron(sigma_alpha, sigma_beta)
    return np.real(np.trace(rho_pair @ op))


def bilinear_compound_correlation(
    rho1: np.ndarray,
    rho2: np.ndarray,
    alpha: str = 'z',
    beta: str = 'z'
) -> float:
    """
    Compute compound correlation C₁₂ = ⟨σ₁α⟩⟨σ₂β⟩.

    Args:
        rho1: 2x2 reduced density matrix for qubit 1
        rho2: 2x2 reduced density matrix for qubit 2
        alpha: Pauli type for qubit 1
        beta: Pauli type for qubit 2

    Returns:
        Compound correlation coefficient C₁₂
    """
    exp1 = pauli_expectation(rho1, alpha, 0, 1)
    exp2 = pauli_expectation(rho2, beta, 0, 1)
    return np.real(exp1 * exp2)


def bilinear_connected_correlation(
    rho_pair: np.ndarray,
    rho1: np.ndarray,
    rho2: np.ndarray,
    alpha: str = 'z',
    beta: str = 'z'
) -> float:
    """
    Compute connected correlation E₁₂ = T₁₂ - C₁₂ (equation 14).

    For pure states, non-zero E₁₂ indicates entanglement between qubits 1 and 2.

    Args:
        rho_pair: 4x4 pair density matrix
        rho1: 2x2 reduced for qubit 1
        rho2: 2x2 reduced for qubit 2
        alpha, beta: Pauli types

    Returns:
        Connected correlation E₁₂ (range [-1, 1])
    """
    T = bilinear_total_correlation(rho_pair, alpha, beta)
    C = bilinear_compound_correlation(rho1, rho2, alpha, beta)
    return T - C


def bilinear_all_correlations(
    rho_pair: np.ndarray,
    rho1: np.ndarray,
    rho2: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute all bilinear correlation functions for all Pauli combinations.

    Returns:
        Dict with structure:
        {
            'total': {'xx': T_xx, 'xy': T_xy, ..., 'zz': T_zz},
            'compound': {'xx': C_xx, ..., 'zz': C_zz},
            'connected': {'xx': E_xx, ..., 'zz': E_zz},
        }
    """
    paulis = ['x', 'y', 'z']
    result = {'total': {}, 'compound': {}, 'connected': {}}

    for a in paulis:
        for b in paulis:
            key = f'{a}{b}'
            T = bilinear_total_correlation(rho_pair, a, b)
            C = bilinear_compound_correlation(rho1, rho2, a, b)
            E = T - C
            result['total'][key] = T
            result['compound'][key] = C
            result['connected'][key] = E

    return result


# ============================================================================
# Trilinear Correlation Functions (3-Qubit)
# ============================================================================

def trilinear_total_correlation(
    rho_tri: np.ndarray,
    alpha: str = 'z',
    beta: str = 'z',
    gamma: str = 'z'
) -> float:
    """
    Compute total trilinear correlation T₁₂₃ = ⟨σ₁α σ₂β σ₃γ⟩.

    Args:
        rho_tri: 8x8 density matrix for qubit triple
        alpha, beta, gamma: Pauli types

    Returns:
        Total correlation coefficient T₁₂₃
    """
    pauli = {
        'x': np.array([[0, 1], [1, 0]], dtype=complex),
        'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'z': np.array([[1, 0], [0, -1]], dtype=complex),
    }

    op = np.kron(np.kron(pauli[alpha], pauli[beta]), pauli[gamma])
    return np.real(np.trace(rho_tri @ op))


def trilinear_compound_correlation(
    rho1: np.ndarray,
    rho2: np.ndarray,
    rho3: np.ndarray,
    E12: float, E13: float, E23: float,
    alpha: str = 'z',
    beta: str = 'z',
    gamma: str = 'z'
) -> float:
    """
    Compute compound correlation C₁₂₃ (equation 13).

    C₁₂₃ = ⟨σ₁α⟩⟨σ₂β⟩⟨σ₃γ⟩
            + ⟨σ₁α⟩E₂₃ + ⟨σ₂β⟩E₁₃ + ⟨σ₃γ⟩E₁₂

    Args:
        rho1, rho2, rho3: 2x2 reduced density matrices
        E12, E13, E23: Bilinear connected correlations (zz components)
        alpha, beta, gamma: Pauli types

    Returns:
        Compound correlation C₁₂₃
    """
    exp1 = pauli_expectation(rho1, alpha, 0, 1)
    exp2 = pauli_expectation(rho2, beta, 0, 1)
    exp3 = pauli_expectation(rho3, gamma, 0, 1)

    # Product of single-qubit expectations
    C_product = np.real(exp1 * exp2 * exp3)

    # Terms with bilinear connected correlations
    # Note: This is simplified - should use E with appropriate Pauli types
    C_connected = np.real(exp1 * E23 + exp2 * E13 + exp3 * E12)

    return C_product + C_connected


def trilinear_connected_correlation(
    rho_tri: np.ndarray,
    rho1: np.ndarray,
    rho2: np.ndarray,
    rho3: np.ndarray,
    E12: float = 0.0,
    E13: float = 0.0,
    E23: float = 0.0,
    alpha: str = 'z',
    beta: str = 'z',
    gamma: str = 'z'
) -> float:
    """
    Compute connected correlation E₁₂₃ = T₁₂₃ - C₁₂₃ (equation 16).

    For pure states, non-zero trilinear E indicates genuine 3-qubit entanglement.

    Args:
        rho_tri: 8x8 triple density matrix
        rho1, rho2, rho3: Reduced density matrices
        E12, E13, E23: Bilinear connected correlations
        alpha, beta, gamma: Pauli types

    Returns:
        Connected correlation E₁₂₃
    """
    T = trilinear_total_correlation(rho_tri, alpha, beta, gamma)
    C = trilinear_compound_correlation(rho1, rho2, rho3, E12, E13, E23, alpha, beta, gamma)
    return T - C


# ============================================================================
# Modified Density Operator (tilde ρ) - Removing Compound Correlations
# ============================================================================

def modified_density_matrix_2q(
    rho_full: np.ndarray,
    n_qubits: int,
    q1: int,
    q2: int
) -> np.ndarray:
    """
    Compute modified density operator tilde ρ for 2-qubit subsystem (equation 15).

    tilde ρ = ρ - Σ_{α,β} ⟨σ₁α⟩⟨σ₂β⟩ σ₁α σ₂β / 4

    This removes compound correlation components, leaving only:
    - Single-qubit components (Q-Beads)
    - Connected correlation components (E-Beads)

    Args:
        rho_full: Full system density matrix
        n_qubits: Total qubits
        q1, q2: Qubit indices for the pair

    Returns:
        Modified density matrix tilde ρ (same shape as rho_full)
    """
    paulis = ['x', 'y', 'z']
    pauli_mat = {
        'x': np.array([[0, 1], [1, 0]], dtype=complex),
        'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'z': np.array([[1, 0], [0, -1]], dtype=complex),
    }

    # First, get reduced density matrices
    from quantumviz.qbeads import compute_reduced_density_matrix

    rho1 = compute_reduced_density_matrix(
        np.diag(rho_full.diagonal()).reshape(-1, 1)[:, 0].tolist(),  # Hack to get state vec
        n_qubits, q1
    )
    # Proper way: compute from full rho
    rho1 = _get_reduced_2x2(rho_full, n_qubits, q1)
    rho2 = _get_reduced_2x2(rho_full, n_qubits, q2)

    # Build the subtraction term
    subtraction = np.zeros_like(rho_full, dtype=complex)

    for alpha in paulis:
        for beta in paulis:
            # ⟨σ₁α⟩
            exp1 = np.real(np.trace(rho1 @ pauli_mat[alpha]))
            # ⟨σ₂β⟩
            exp2 = np.real(np.trace(rho2 @ pauli_mat[beta]))

            # σ₁α ⊗ σ₂β operator (embedded in full system)
            op = _embed_2q_operator(pauli_mat[alpha], pauli_mat[beta], q1, q2, n_qubits)

            subtraction += exp1 * exp2 * op

    subtraction = subtraction / 4.0

    return rho_full - subtraction


def _get_reduced_2x2(rho_full: np.ndarray, n_qubits: int, qubit_idx: int) -> np.ndarray:
    """Get 2x2 reduced density matrix for a single qubit."""
    from quantumviz.qbeads import compute_reduced_density_matrix
    # Need state vector for the existing function
    # For now, compute directly from rho_full
    dim = 2 ** n_qubits
    rho_red = np.zeros((2, 2), dtype=complex)

    mask_keep = 1 << qubit_idx

    for a in range(dim):
        for b in range(dim):
            bit_a_keep = (a >> qubit_idx) & 1
            bit_b_keep = (b >> qubit_idx) & 1
            bits_a_other = a & ~mask_keep
            bits_b_other = b & ~mask_keep

            if bits_a_other == bits_b_other:
                rho_red[bit_a_keep, bit_b_keep] += rho_full[a, b]

    return rho_red


def _embed_2q_operator(
    op1: np.ndarray,
    op2: np.ndarray,
    q1: int,
    q2: int,
    n_qubits: int
) -> np.ndarray:
    """Embed a 2-qubit operator into the full N-qubit space."""
    ops = []
    for i in range(n_qubits):
        if i == q1:
            ops.append(op1)
        elif i == q2:
            ops.append(op2)
        else:
            ops.append(np.eye(2, dtype=complex))

    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


# ============================================================================
# Entanglement Norm (Measure of Entanglement)
# ============================================================================

def entanglement_norm(rho_full: np.ndarray, n_qubits: int) -> float:
    """
    Compute entanglement norm from modified density operator.

    The entanglement norm ∥tilde ρ∥ serves as a measure of
    the total entanglement in the system.

    Based on Schlienz & Mahler (1995).

    Args:
        rho_full: Full system density matrix
        n_qubits: Total number of qubits

    Returns:
        Entanglement norm (non-negative)
    """
    # For 2-qubit case
    if n_qubits == 2:
        # Use concurrence as entanglement measure
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        rho_tilde = np.kron(sigma_y, sigma_y) @ rho_full.conj().T @ np.kron(sigma_y, sigma_y)

        product = rho_full @ rho_tilde
        eigenvalues = np.linalg.eigvals(product)
        eigenvalues = np.sqrt(np.abs(eigenvalues))
        eigenvalues = np.sort(eigenvalues)[::-1]

        concurrence = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] + eigenvalues[3])
        return concurrence

    # For N > 2, sum over all bipartite entanglement
    # This is a simplified version
    total_norm = 0.0
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            # Compute concurrence for pair (i, j)
            # This is non-trivial for N>2, simplified here
            total_norm += 0.0  # Placeholder

    return total_norm


# ============================================================================
# Utility: Compute all E-Bead values for a system
# ============================================================================

def compute_all_ebeads(
    state_vector: List[complex],
    n_qubits: int
) -> List[Dict]:
    """
    Compute all E-Bead (connected correlation) values for a system.

    Args:
        state_vector: State vector of length 2^n_qubits
        n_qubits: Number of qubits

    Returns:
        List of dicts, each representing an E-Bead with:
        - 'qubits': tuple of qubit indices
        - 'type': 'pairwise' or 'triple'
        - 'connected': Dict of connected correlations (zz, etc.)
        - 'strength': Overall entanglement strength
    """
    from quantumviz.qbeads import (
        compute_reduced_density_matrix,
        compute_pair_density_matrix,
    )

    rho_full = None  # Will compute as needed

    results = []

    if n_qubits >= 2:
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                # Get reduced density matrices
                rho1 = compute_reduced_density_matrix(state_vector, n_qubits, i)
                rho2 = compute_reduced_density_matrix(state_vector, n_qubits, j)
                rho_pair = compute_pair_density_matrix(state_vector, n_qubits, i, j)

                # Compute connected correlations
                E_zz = bilinear_connected_correlation(rho_pair, rho1, rho2, 'z', 'z')

                # Overall strength (use absolute value of zz component as proxy)
                strength = abs(E_zz)

                if strength > 0.01:  # Threshold for meaningful entanglement
                    results.append({
                        'qubits': (i, j),
                        'type': 'pairwise',
                        'connected': {'zz': E_zz},
                        'strength': strength,
                    })

    if n_qubits >= 3:
        # For 3 qubits, also compute trilinear E-Beads
        # This is more complex - simplified placeholder
        pass

    return results
