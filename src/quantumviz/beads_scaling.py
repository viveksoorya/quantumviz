"""
BEADS Scaling Factors Module

Implements the three-part scaling factors for BEADS spherical functions:
    s_j^{(ℓ')}(N, g) = ζ(N) · ξ_j^{(ℓ')}(g) · η_j

Based on:
- ζ(N) = sqrt(2^N) — N-qubit system scaling (equation B.4)
- η_j = sqrt(4π/(2j+1)) — rank-dependent factor (equation B.6)
- ξ_j^{(ℓ')}(g) — rank/linearity-dependent factors (Tables 3, C1)

Two methods for ξ:
1. Canonical: Ensures spherical function values = Pauli expectation values
2. GUB (Global Unitary Bound): Bounds values to [-1, 1]

References:
- Appendix B: Canonical scaling for permutation-symmetric components
- Appendix C: GUB scaling for non-fully-symmetric components
- Table 3: ξ_j^{(ℓ')}(g) for up to trilinear operators
- Table C1: GUB-based scaling factors
"""

import math
import numpy as np
from typing import Dict, Optional, Tuple


# ============================================================================
# Base Scaling Factors
# ============================================================================

def zeta(N: int) -> float:
    """
    System size scaling factor ζ(N) = sqrt(2^N).

    Args:
        N: Total number of qubits in the system

    Returns:
        ζ(N) scaling factor
    """
    return np.sqrt(2 ** N)


def eta(j: int) -> float:
    """
    Rank-dependent scaling factor η_j = sqrt(4π/(2j+1)).

    Args:
        j: Spherical harmonic rank (0, 1, 2, 3, ...)

    Returns:
        η_j scaling factor
    """
    return np.sqrt(4 * np.pi / (2 * j + 1))


# ============================================================================
# Canonical ξ Factors (Appendix B, Table B1)
# ============================================================================

# Canonical ξ_j^{(ℓ')}(g) for fully permutation-symmetric components
# Table B1: Scaling factors for Pauli-Z operators
_CANONICAL_XI: Dict[Tuple[int, int], Dict[int, float]] = {
    # (linearity_g, tau_label) -> {j: xi_j}
    # g=0: Identity
    (0, 0): {0: 1.0},
    # g=1: Single-qubit (Q-Bead)
    (1, 0): {1: 1.0},
    # g=2, τ (fully symmetric even):
    # ξ_0 = sqrt(1/3), ξ_2 = sqrt(2/3)
    (2, 0): {0: np.sqrt(1/3), 2: np.sqrt(2/3)},
    # g=3, τ1 (fully symmetric trilinear):
    # ξ_1 = sqrt(3/5), ξ_3 = sqrt(2/5)
    (3, 1): {1: np.sqrt(3/5), 3: np.sqrt(2/5)},
    # g=4, τ1:
    # ξ_0 = sqrt(7/35), ξ_2 = sqrt(20/35), ξ_4 = sqrt(8/35)
    (4, 1): {0: np.sqrt(7/35), 2: np.sqrt(20/35), 4: np.sqrt(8/35)},
    # g=5, τ1:
    # ξ_1 = sqrt(27/63), ξ_3 = sqrt(28/63), ξ_5 = sqrt(8/63)
    (5, 1): {1: np.sqrt(27/63), 3: np.sqrt(28/63), 5: np.sqrt(8/63)},
    # g=6, τ1:
    # ξ_0 = sqrt(33/231), ξ_2 = sqrt(110/231), ξ_4 = sqrt(72/231), ξ_6 = sqrt(16/231)
    (6, 1): {0: np.sqrt(33/231), 2: np.sqrt(110/231), 4: np.sqrt(72/231), 6: np.sqrt(16/231)},
}

# Canonical ξ for bilinear odd (antisymmetric) - not in Table B1
# Use GUB method instead for non-fully-symmetric


def get_canonical_xi(linearity: int, tau: int, j: int) -> float:
    """
    Get canonical scaling factor ξ_j^{(ℓ')}(g).

    Args:
        linearity: Number of qubits the operator acts on (g)
        tau: Tau permutation label (0=even, 1=τ1, 2=τ2, 3=τ3, 4=τ4)
        j: Spherical harmonic rank

    Returns:
        ξ_j^{(ℓ')}(g) scaling factor

    Raises:
        KeyError: If (linearity, tau, j) combination not in table
    """
    key = (linearity, tau)
    if key in _CANONICAL_XI:
        return _CANONICAL_XI[key].get(j, 0.0)
    return 0.0


# ============================================================================
# GUB (Global Unitary Bound) ξ Factors (Appendix C, Table C1)
# ============================================================================

# GUB-based ξ for components NOT fully permutation-symmetric
# Table C1: ξ_j^{(ℓ')}(g) based on global unitary bounds
_GUB_XI: Dict[str, float] = {
    # Bilinear odd (antisymmetric): ξ_1 = 1/√2
    '2q_odd_1': 1.0 / np.sqrt(2),
    # Trilinear tau2 odd: ξ_1 = 3/(3+√3)
    '3q_tau2_odd_1': 3.0 / (3.0 + np.sqrt(3)),
    # Trilinear tau2 even: ξ_2 = 1/√2
    '3q_tau2_even_2': 1.0 / np.sqrt(2),
    # Trilinear tau3 odd: ξ_1 = 1/√2
    '3q_tau3_odd_1': 1.0 / np.sqrt(2),
    # Trilinear tau3 even: ξ_2 = 1/√2
    '3q_tau3_even_2': 1.0 / np.sqrt(2),
    # Trilinear tau4 even: ξ_0 = 1/√2
    '3q_tau4_even_0': 1.0 / np.sqrt(2),
}


def get_gub_xi(linearity: int, tau: int, j: int, N: int = 3) -> float:
    """
    Get GUB-based scaling factor ξ_j^{(ℓ')}(g).

    Args:
        linearity: Number of qubits (g)
        tau: Tau permutation label
        j: Spherical harmonic rank
        N: Total system qubits (for computing GUB)

    Returns:
        ξ_j^{(ℓ')}(g) GUB scaling factor
    """
    if linearity == 2:
        # Bilinear odd
        if j == 1:
            return 1.0 / np.sqrt(2)
    elif linearity == 3:
        if tau == 2:  # tau2
            if j == 1:
                return 3.0 / (3.0 + np.sqrt(3))
            elif j == 2:
                return 1.0 / np.sqrt(2)
        elif tau == 3:  # tau3
            if j == 1:
                return 1.0 / np.sqrt(2)
            elif j == 2:
                return 1.0 / np.sqrt(2)
        elif tau == 4:  # tau4 (fully antisymmetric)
            if j == 0:
                return 1.0 / np.sqrt(2)

    return 0.0


# ============================================================================
# Combined Scaling Factor s_j^{(ℓ')}(N, g)
# ============================================================================

def scaling_factor(
    j: int,
    linearity: int,
    tau: int,
    N: int,
    method: str = "canonical"
) -> float:
    """
    Compute full scaling factor s_j^{(ℓ')}(N, g) = ζ(N) · ξ_j · η_j.

    Args:
        j: Spherical harmonic rank
        linearity: Number of qubits (g)
        tau: Tau permutation label
        N: Total number of qubits in system
        method: "canonical" or "gub"

    Returns:
        s_j^{(ℓ')}(N, g) scaling factor
    """
    z = zeta(N)
    e = eta(j)

    if method == "canonical":
        xi = get_canonical_xi(linearity, tau, j)
    else:  # "gub"
        xi = get_gub_xi(linearity, tau, j, N)

    return z * xi * e


def get_all_scaling_factors(
    N: int,
    n_qubits: int,
    method: str = "canonical"
) -> Dict[str, Dict[int, float]]:
    """
    Get all scaling factors for an n-qubit system.

    Returns dict keyed by bead label, with sub-dict of {j: s_j}:
    {
        'Q_0': {1: s_1},
        'E_{0,1}_even': {0: s_0, 2: s_2},
        'E_{0,1}_odd': {1: s_1},
        ...
    }
    """
    result = {}

    # Q-Beads (linearity=1, tau=0)
    for k in range(n_qubits):
        label = f'Q_{k}'
        result[label] = {1: scaling_factor(1, 1, 0, N, method)}

    if n_qubits >= 2:
        # Bilinear E-Beads
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                # Even (symmetric) component
                label_even = f'E_{{{i},{j}}}_even'
                result[label_even] = {
                    0: scaling_factor(0, 2, 0, N, method),
                    2: scaling_factor(2, 2, 0, N, method),
                }
                # Odd (antisymmetric) component
                label_odd = f'E_{{{i},{j}}}_odd'
                result[label_odd] = {
                    1: scaling_factor(1, 2, 0, N, "gub"),  # Use GUB for odd
                }

    if n_qubits >= 3:
        # Trilinear E-Beads (fully symmetric τ1)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                for k in range(j + 1, n_qubits):
                    label_tau1 = f'E_{{{i},{j},{k}τ1}}_odd'
                    result[label_tau1] = {
                        1: scaling_factor(1, 3, 1, N, method),
                        3: scaling_factor(3, 3, 1, N, method),
                    }

    return result


# ============================================================================
# Specific Bead-Type Scaling
# ============================================================================

def qbead_scaling(N: int, qubit_idx: int) -> Dict[int, float]:
    """Get scaling for a Q-Bead (linearity=1)."""
    return {1: scaling_factor(1, 1, 0, N, "canonical")}


def ebead_even_scaling(N: int, q1: int, q2: int) -> Dict[int, float]:
    """Get scaling for E-Bead even component (j=0, 2)."""
    return {
        0: scaling_factor(0, 2, 0, N, "canonical"),
        2: scaling_factor(2, 2, 0, N, "canonical"),
    }


def ebead_odd_scaling(N: int, q1: int, q2: int) -> Dict[int, float]:
    """Get scaling for E-Bead odd component (j=1)."""
    return {1: scaling_factor(1, 2, 0, N, "gub")}


def ebead_tau1_scaling(N: int, q1: int, q2: int, q3: int) -> Dict[int, float]:
    """Get scaling for trilinear E-Bead τ1 (j=1, 3)."""
    return {
        1: scaling_factor(1, 3, 1, N, "canonical"),
        3: scaling_factor(3, 3, 1, N, "canonical"),
    }


# ============================================================================
# Pretty Printing / Debugging
# ============================================================================

def print_scaling_table(N: int = 3, n_qubits: int = 3):
    """Print scaling factors in a table format (for debugging)."""
    print(f"Scaling Factors for N={N}, n_qubits={n_qubits}")
    print("=" * 60)

    factors = get_all_scaling_factors(N, n_qubits)

    for label, j_dict in sorted(factors.items()):
        print(f"\n{label}:")
        for j, s in sorted(j_dict.items()):
            xi_method = "canonical" if "tau" not in label or "tau1" in label else "gub"
            xi = s / (zeta(N) * eta(j)) if zeta(N) * eta(j) != 0 else 0
            print(f"  j={j}: s={s:.6f} (ζ={zeta(N):.4f}, ξ={xi:.6f}, η={eta(j):.4f})")


if __name__ == "__main__":
    # Test/debug
    print_scaling_table(2, 2)
    print("\n" + "=" * 60)
    print_scaling_table(3, 3)
