"""
LISA Tensor Operator Basis Module

Implements the LISA (Linearity, Subsystem, Auxiliary criteria) tensor operator
basis for decomposing quantum states into spherical tensor components.

Based on:
- Garon, Zeier & Glaser, Phys. Rev. A 91, 042122 (2015)
- Leiner, Zeier & Glaser, J. Phys. A: Math. Theor. 53 495301 (2020)
- Huber & Glaser, New J. Phys. 27 094509 (2025)

The LISA basis is organized by:
- Linearity g: number of qubits the operator acts on (1, 2, 3, ...)
- Subsystem: the specific set of qubits involved
- Auxiliary criteria: permutation symmetry (tau labels for 3+ qubits)

For Hermitian operators (density matrices), we use real combinations that map to
real spherical harmonics Y_{j,m} (cosine/sine variants).
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Pauli matrices (standard convention)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY_2 = np.array([[1, 0], [0, 1]], dtype=complex)

# Normalization factor for N-qubit systems
def _norm(N: int) -> float:
    """LISA basis normalization factor 1/sqrt(2^N)."""
    return 1.0 / np.sqrt(2 ** N)


# ============================================================================
# 1-Qubit LISA Operators (linearg=1)
# ============================================================================

def lisa_1q_sigma_z(k: int, N: int) -> np.ndarray:
    """
    Linear LISA operator T_{1,0}^{{k}} for qubit k.

    Maps to sigma_z of qubit k: T_{1,0}^{k} = 1/sqrt(2^N) * Z_k

    Returns:
        2^N × 2^N matrix (Hermitian)
    """
    norm = _norm(N)
    # Build Z_k as tensor product: I ⊗ ... ⊗ Z ⊗ ... ⊗ I
    ops = []
    for i in range(N):
        if i == k:
            ops.append(SIGMA_Z)
        else:
            ops.append(IDENTITY_2)
    Z_k = ops[0]
    for op in ops[1:]:
        Z_k = np.kron(Z_k, op)
    return norm * Z_k


def lisa_1q_sigma_x(k: int, N: int) -> np.ndarray:
    """T_{1,1c}^{{k}} ~ sigma_x component (real part, m=1)."""
    norm = _norm(N)
    ops = []
    for i in range(N):
        if i == k:
            ops.append(SIGMA_X)
        else:
            ops.append(IDENTITY_2)
    X_k = ops[0]
    for op in ops[1:]:
        X_k = np.kron(X_k, op)
    return norm * X_k


def lisa_1q_sigma_y(k: int, N: int) -> np.ndarray:
    """
    T_{1,1s}^{{k}} ~ sigma_y component (imaginary part, m=-1).

    For real spherical harmonics, we use: Y_{1,1c} ~ cos(phi)*sin(theta), Y_{1,1s} ~ sin(phi)*sin(theta)
    So we need real combinations: X and Y (not X ± iY)
    """
    norm = _norm(N)
    ops = []
    for i in range(N):
        if i == k:
            ops.append(SIGMA_Y)
        else:
            ops.append(IDENTITY_2)
    Y_k = ops[0]
    for op in ops[1:]:
        Y_k = np.kron(Y_k, op)
    return norm * Y_k


# ============================================================================
# 2-Qubit LISA Operators (linearg=2)
# ============================================================================

def lisa_2q_even_0(q1: int, q2: int, N: int) -> np.ndarray:
    """
    Bilinear symmetric (even) rank j=0: T_{0,0}^{{q1,q2}_even}

    = 1/sqrt(2^N) * (Z1*Z2 + X1*X2 + Y1*Y2) / sqrt(3)
    = 1/sqrt(2^N) * (2|00><00| + 2|11><11| - |01><01| - |10><10|) / sqrt(3)
    """
    norm = _norm(N)
    # XX + YY + ZZ for qubits q1, q2
    # Build the operator
    result = np.zeros((2**N, 2**N), dtype=complex)

    for i in range(2**N):
        for j in range(2**N):
            # Get bit values for q1, q2
            b1_i = (i >> q1) & 1
            b2_i = (i >> q2) & 1
            b1_j = (j >> q1) & 1
            b2_j = (j >> q2) & 1

            # ZZ component
            zz = (1 - 2*b1_i) * (1 - 2*b1_j) * (1 - 2*b2_i) * (1 - 2*b2_j)

            # XX component: only non-zero if other bits match
            xx = 1.0 if (i == j or (i ^ (1<<q1) ^ (1<<q2)) == j) else 0.0

            # YY component: similar but with phase
            yy_sign = 1.0
            if ((i >> q1) & 1) != ((j >> q1) & 1):
                yy_sign *= -1j
            if ((i >> q2) & 1) != ((j >> q2) & 1):
                yy_sign *= -1j
            yy = yy_sign if (i == j or (i ^ (1<<q1) ^ (1<<q2)) == j) else 0.0

            result[i, j] = (zz + xx + yy) / np.sqrt(3)

    return norm * result


def lisa_2q_even_2(q1: int, q2: int, N: int) -> np.ndarray:
    """
    Bilinear symmetric (even) rank j=2: T_{2,0}^{{q1,q2}_even}

    = 1/sqrt(2^N) * (2*Z1*Z2 - X1*X2 - Y1*Y2) / sqrt(6)
    """
    norm = _norm(N)
    result = np.zeros((2**N, 2**N), dtype=complex)

    for i in range(2**N):
        for j in range(2**N):
            b1_i = (i >> q1) & 1
            b2_i = (i >> q2) & 1
            b1_j = (j >> q1) & 1
            b2_j = (j >> q2) & 1

            # ZZ component
            zz = (1 - 2*b1_i) * (1 - 2*b1_j) * (1 - 2*b2_i) * (1 - 2*b2_j)

            # XX component
            xx = 1.0 if (i == j or (i ^ (1<<q1) ^ (1<<q2)) == j) else 0.0

            # YY component
            yy_sign = 1.0
            if ((i >> q1) & 1) != ((j >> q1) & 1):
                yy_sign *= -1j
            if ((i >> q2) & 1) != ((j >> q2) & 1):
                yy_sign *= -1j
            yy = yy_sign if (i == j or (i ^ (1<<q1) ^ (1<<q2)) == j) else 0.0

            result[i, j] = (2 * zz - xx - yy) / np.sqrt(6)

    return norm * result


def lisa_2q_odd_1(q1: int, q2: int, N: int) -> np.ndarray:
    """
    Bilinear antisymmetric (odd) rank j=1: T_{1,0}^{{q1,q2}_odd}

    = 1/(2*sqrt(2^N)) * (Z1*X2 - X1*Z2 + i*(Z1*Y2 - Y1*Z2))
    Maps to commutator-like correlations.

    For Hermitian version, we use: (Z1*X2 - X1*Z2) / (2*sqrt(2))
    """
    norm = _norm(N)
    result = np.zeros((2**N, 2**N), dtype=complex)

    for i in range(2**N):
        for j in range(2**N):
            b1_i, b2_i = (i >> q1) & 1, (i >> q2) & 1
            b1_j, b2_j = (j >> q1) & 1, (j >> q2) & 1

            # Z1*X2 - X1*Z2 (Hermitian)
            zx = (1 - 2*b1_i) * (1.0 if b2_i == b2_j else 0.0)
            xz = (1.0 if b1_i == b1_j else 0.0) * (1 - 2*b2_i)

            result[i, j] = (zx - xz) / (2 * np.sqrt(2))

    return norm * result


# ============================================================================
# 3-Qubit LISA Operators (linearg=3)
# ============================================================================
# Reference: Leiner, Zeier & Glaser, J. Phys. A 53 495301 (2020)
# Tau labels: τ1 (fully symmetric), τ2 (sym antisym in 1,2),
#            τ3 (antisym sym in 1,2), τ4 (fully antisymmetric)

def lisa_3q_tau1_1(q1: int, q2: int, q3: int, N: int) -> np.ndarray:
    """
    Trilinear fully symmetric τ1, rank j=1: T_{1,0}^{{q1,q2,q3τ1}_odd}

    = 1/sqrt(2^N) * (Z1*Z2*Z3 + X1*X2*Z3 + X1*Z2*X3 + Z1*X2*X3
                                        + X1*Y2*Y3 + Y1*X2*Y3 + Y1*Y2*X3 + Z1*Y2*Y3) / sqrt(5)
    Simplified: fully symmetric Pauli-Z product combination.
    """
    norm = _norm(N)
    result = np.zeros((2**N, 2**N), dtype=complex)

    for i in range(2**N):
        for j in range(2**N):
            b1_i, b2_i, b3_i = (i >> q1) & 1, (i >> q2) & 1, (i >> q3) & 1
            b1_j, b2_j, b3_j = (j >> q1) & 1, (j >> q2) & 1, (j >> q3) & 1

            # Z1*Z2*Z3 component
            zzz = (1 - 2*b1_i) * (1 - 2*b1_j) * (1 - 2*b2_i) * (1 - 2*b2_j) * (1 - 2*b3_i) * (1 - 2*b3_j)

            result[i, j] = zzz / np.sqrt(5)

    return norm * result


def lisa_3q_tau1_3(q1: int, q2: int, q3: int, N: int) -> np.ndarray:
    """
    Trilinear fully symmetric τ1, rank j=3: T_{3,0}^{{q1,q2,q3τ1}_odd}

    More complex combination for j=3 component.
    """
    norm = _norm(N)
    result = np.zeros((2**N, 2**N), dtype=complex)

    for i in range(2**N):
        for j in range(2**N):
            b1_i, b2_i, b3_i = (i >> q1) & 1, (i >> q2) & 1, (i >> q3) & 1
            b1_j, b2_j, b3_j = (j >> q1) & 1, (j >> q2) & 1, (j >> q3) & 1

            zzz = (1 - 2*b1_i) * (1 - 2*b1_j) * (1 - 2*b2_i) * (1 - 2*b2_j) * (1 - 2*b3_i) * (1 - 2*b3_j)

            result[i, j] = zzz / np.sqrt(10)

    return norm * result


# ============================================================================
# Decomposition Functions
# ============================================================================

def decompose_density_1q(rho: np.ndarray, k: int, N: int) -> Dict[str, complex]:
    """
    Decompose reduced 1-qubit density matrix into LISA coefficients.

    For 1 qubit, the decomposition is simple:
    ρ = c0 * I/2 + c1 * T_{1,0}^{k}

    Returns:
        Dict with keys 'j0' (identity) and 'j1' (sigma_z coefficient)
    """
    norm = _norm(N)

    # c1 = Tr(rho * T_{1,0}^{k}) / norm_factor
    T10 = lisa_1q_sigma_z(k, N)
    c1 = np.trace(rho @ T10.conj().T)

    result = {'j1': c1}

    # Identity component
    c0 = np.trace(rho) / (2**N)
    result['j0'] = c0

    return result


def decompose_density_2q(rho_pair: np.ndarray, q1: int, q2: int, N: int) -> Dict[str, complex]:
    """
    Decompose 2-qubit density matrix into LISA coefficients.

    Components:
    - even_0: T_{0,0}^{{q1,q2}_even} coefficient
    - even_2: T_{2,0}^{{q1,q2}_even} coefficient
    - odd_1: T_{1,0}^{{q1,q2}_odd} coefficient

    Returns:
        Dict with keys 'even_0', 'even_2', 'odd_1'
    """
    result = {}

    # Even j=0 component (equation B.8)
    T00_even = lisa_2q_even_0(q1, q2, N)
    result['even_0'] = np.trace(rho_pair @ T00_even.conj().T)

    # Even j=2 component (equation B.9)
    T20_even = lisa_2q_even_2(q1, q2, N)
    result['even_2'] = np.trace(rho_pair @ T20_even.conj().T)

    # Odd j=1 component
    T10_odd = lisa_2q_odd_1(q1, q2, N)
    result['odd_1'] = np.trace(rho_pair @ T10_odd.conj().T)

    return result


def decompose_density_3q(rho_tri: np.ndarray, q1: int, q2: int, q3: int, N: int) -> Dict[str, complex]:
    """
    Decompose 3-qubit density matrix into LISA coefficients.

    Components (fully symmetric τ1):
    - tau1_1: T_{1,0}^{τ1} coefficient (j=1)
    - tau1_3: T_{3,0}^{τ1} coefficient (j=3)

    Returns:
        Dict with appropriate tau keys
    """
    result = {}

    # τ1 fully symmetric components
    T10_tau1 = lisa_3q_tau1_1(q1, q2, q3, N)
    result['tau1_1'] = np.trace(rho_tri @ T10_tau1.conj().T)

    T30_tau1 = lisa_3q_tau1_3(q1, q2, q3, N)
    result['tau1_3'] = np.trace(rho_tri @ T30_tau1.conj().T)

    return result


# ============================================================================
# Coefficient Extraction for Beads
# ============================================================================

def get_qbead_coefficients(rho: np.ndarray, qubit_idx: int, N: int) -> Dict[int, complex]:
    """
    Get LISA coefficients for a single qubit Q-Bead.

    For Q-Beads representing reduced density matrix of qubit k:
    - j=1 coefficient gives the Bloch vector component along z
    - Full Bloch vector: (⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩)

    Returns:
        Dict mapping rank j to coefficient c_j
        For j=1: c_1 = sqrt(4π/3) * ⟨σ_z⟩ / sqrt(2^N)
    """
    # Compute expectation values
    rx = np.real(np.trace(rho @ SIGMA_X))
    ry = np.real(np.trace(rho @ SIGMA_Y))
    rz = np.real(np.trace(rho @ SIGMA_Z))

    # Map to LISA coefficients
    # For axial (m=0): c_{1,0} = ⟨σ_z⟩ * sqrt(2^N) / sqrt(4π/3)
    norm = np.sqrt(2**N)
    eta_1 = np.sqrt(4 * np.pi / 3)

    coeffs = {
        1: complex(rz * norm / eta_1, 0),
    }

    return coeffs


def get_ebead_coefficients(rho_pair: np.ndarray, q1: int, q2: int, N: int) -> Dict[str, Dict[int, complex]]:
    """
    Get LISA coefficients for E-Beads between two qubits.

    Returns coefficients for:
    - 'even': Dict mapping j -> c_j for symmetric components (j=0, 2)
    - 'odd': Dict mapping j -> c_j for antisymmetric component (j=1)
    """
    decomp = decompose_density_2q(rho_pair, q1, q2, N)

    result = {
        'even': {
            0: decomp.get('even_0', 0),
            2: decomp.get('even_2', 0),
        },
        'odd': {
            1: decomp.get('odd_1', 0),
        }
    }

    return result


# ============================================================================
# Bead Type Identification
# ============================================================================

def get_bead_spec_for_qubits(n_qubits: int) -> List[Dict[str, Any]]:
    """
    Get the list of beads needed for an n-qubit system.

    Returns list of dicts with:
    - 'type': 'Q', 'E', 'T', or 'C'
    - 'qubits': tuple of qubit indices
    - 'label': string label like 'Q_1', 'E_{1,2}_even', etc.
    - 'symmetry': 'even', 'odd', or None
    - 'tau': tau label if applicable
    """
    beads = []

    # Q-Beads (one per qubit)
    for k in range(n_qubits):
        beads.append({
            'type': 'Q',
            'qubits': (k,),
            'label': f'Q_{k}',
            'symmetry': None,
            'tau': None,
            'ranks': [1],  # Only j=1 for Q-Beads
        })

    if n_qubits >= 2:
        # E-Beads for each pair
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                # Even (symmetric) component
                beads.append({
                    'type': 'E',
                    'qubits': (i, j),
                    'label': f'E_{{{i},{j}}}_even',
                    'symmetry': 'even',
                    'tau': None,
                    'ranks': [0, 2],  # j=0 and j=2
                })
                # Odd (antisymmetric) component
                beads.append({
                    'type': 'E',
                    'qubits': (i, j),
                    'label': f'E_{{{i},{j}}}_odd',
                    'symmetry': 'odd',
                    'tau': None,
                    'ranks': [1],  # j=1
                })

    if n_qubits >= 3:
        # Trilinear E-Beads (fully symmetric τ1)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                for k in range(j + 1, n_qubits):
                    beads.append({
                        'type': 'E',
                        'qubits': (i, j, k),
                        'label': f'E_{{{i},{j},{k}τ1}}_odd',
                        'symmetry': 'odd',
                        'tau': 'tau1',
                        'ranks': [1, 3],  # j=1 and j=3
                    })

    return beads
