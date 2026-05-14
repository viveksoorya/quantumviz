"""
Spherical Beads Module - Spherical Harmonics Mapping for BEADS

Implements the mapping from LISA tensor operator coefficients to
spherical functions b^{(ℓ')}(θ, φ) using real spherical harmonics.

The BEADS spherical function for a bead is:
    b^{(ℓ')}(θ, φ) = Σ_j Σ_m s_j^{(ℓ')} * c_{j,m}^{(ℓ')} * Y_{j,m}(θ, φ)

where:
- s_j^{(ℓ')} = ζ(N) · ξ_j^{(ℓ')} · η_j are the scaling factors
- c_{j,m}^{(ℓ')} are the LISA decomposition coefficients
- Y_{j,m} are real spherical harmonics

For real spherical harmonics (used for Hermitian operators):
- Y_{j,mc} = Y_{j,m} + Y_{j,-m} ∝ cos(m·φ) · P_j^m(cos θ)  (m > 0)
- Y_{j,ms} = Y_{j,m} - Y_{j,-m} ∝ sin(m·φ) · P_j^m(cos θ)  (m > 0)
- Y_{j,0} = Y_{j,0} ∝ P_j(cos θ)

Based on the DROPS/BEADS mapping (equation 7 in paper).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
try:
    from scipy.special import sph_harm_y, lpmv
    # scipy >= 1.15 uses sph_harm_y instead of sph_harm
    def sph_harm(m, j, phi, theta):
        """Wrapper to match old sph_harm(m, j, phi, theta) signature."""
        return sph_harm_y(j, m, theta, phi)
except ImportError:
    try:
        from scipy.special import sph_harm
    except ImportError:
        sph_harm = None
    lpmv = None


# ============================================================================
# Real Spherical Harmonics
# ============================================================================

def real_sph_harm(j: int, m: int, theta: float, phi: float) -> float:
    """
    Compute real spherical harmonic Y_{j,m}(θ, φ).

    For m > 0: Y_{j,mc} = sqrt(2) * Re(Y_{j,m}) = sqrt(2) * P_j^m(cosθ) * cos(mφ)
    For m < 0: Y_{j,ms} = sqrt(2) * Im(Y_{j,|m|}) = sqrt(2) * P_j^{|m|}(cosθ) * sin(|m|φ)
    For m = 0: Y_{j,0} = Y_{j,0}

    Args:
        j: Rank (0, 1, 2, 3, ...)
        m: Order (-j to +j), where sign indicates cosine/sine variant
        theta: Polar angle [0, π] (0 = north pole)
        phi: Azimuthal angle [0, 2π] (from x-axis)

    Returns:
        Real spherical harmonic value
    """
    if m > 0:
        # Cosine variant (m > 0, "c")
        Y_pos = sph_harm(m, j, phi, theta)  # scipy: sph_harm(m, j, phi, theta)
        Y_neg = sph_harm(-m, j, phi, theta)
        return np.sqrt(2) * np.real(Y_pos + Y_neg) / np.sqrt(2)
        # return np.sqrt(2) * np.real(Y_pos)
    elif m < 0:
        # Sine variant (m < 0, "s")
        m_abs = abs(m)
        Y_pos = sph_harm(m_abs, j, phi, theta)
        Y_neg = sph_harm(-m_abs, j, phi, theta)
        return np.sqrt(2) * np.imag(Y_pos - Y_neg) / np.sqrt(2)
        # return np.sqrt(2) * np.imag(Y_pos)
    else:
        # m = 0
        Y = sph_harm(0, j, phi, theta)
        return np.real(Y)


def real_sph_harm_grid(j: int, m: int, theta_grid: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    """
    Vectorized version for meshgrids.

    Args:
        j, m: Rank and order
        theta_grid: 2D array of theta values
        phi_grid: 2D array of phi values

    Returns:
        2D array of spherical harmonic values
    """
    if m > 0:
        Y_pos = sph_harm(m, j, phi_grid, theta_grid)
        Y_neg = sph_harm(-m, j, phi_grid, theta_grid)
        return np.sqrt(2) * np.real(Y_pos)
    elif m < 0:
        m_abs = abs(m)
        Y_pos = sph_harm(m_abs, j, phi_grid, theta_grid)
        Y_neg = sph_harm(-m_abs, j, phi_grid, theta_grid)
        return np.sqrt(2) * np.imag(Y_pos)
    else:
        Y = sph_harm(0, j, phi_grid, theta_grid)
        return np.real(Y)


# ============================================================================
# Bead Spherical Function
# ============================================================================

def bead_function(
    theta: float,
    phi: float,
    coefficients: Dict[int, Dict[int, complex]],
    scaling_factors: Dict[int, float],
    bead_type: str = "Q"
) -> float:
    """
    Compute BEADS spherical function value at direction (θ, φ).

    b^{(ℓ')}(θ, φ) = Σ_j s_j * Σ_m c_{j,m} * Y_{j,m}(θ, φ)

    For axial LISA operators (m=0), we only need Y_{j,0}.
    For full representation, sum over all m for each j.

    Args:
        theta: Polar angle [0, π]
        phi: Azimuthal angle [0, 2π]
        coefficients: Dict {j: {m: c_{j,m}}} - LISA coefficients
        scaling_factors: Dict {j: s_j} - scaling factors for each rank
        bead_type: 'Q', 'E', 'C', or 'T' (affects m-summation)

    Returns:
        Spherical function value b^{(ℓ')}(θ, φ)
    """
    value = 0.0

    for j, m_coeffs in coefficients.items():
        if j not in scaling_factors:
            continue
        s_j = scaling_factors[j]

        # Sum over m for this j
        for m, c_jm in m_coeffs.items():
            Y_jm = real_sph_harm(j, m, theta, phi)
            value += s_j * np.real(c_jm) * Y_jm

    return float(value)


def bead_function_grid(
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
    coefficients: Dict[int, Dict[int, complex]],
    scaling_factors: Dict[int, float],
) -> np.ndarray:
    """
    Vectorized computation of bead function on a (theta, phi) grid.

    Args:
        theta_grid: 2D array of theta values
        phi_grid: 2D array of phi values
        coefficients: Dict {j: {m: c_{j,m}}}
        scaling_factors: Dict {j: s_j}

    Returns:
        2D array of spherical function values
    """
    result = np.zeros_like(theta_grid, dtype=float)

    for j, m_coeffs in coefficients.items():
        if j not in scaling_factors:
            continue
        s_j = scaling_factors[j]

        for m, c_jm in m_coeffs.items():
            Y_jm = real_sph_harm_grid(j, m, theta_grid, phi_grid)
            result += s_j * np.real(c_jm) * Y_jm

    return result


# ============================================================================
# Mesh Generation
# ============================================================================

def generate_sphere_mesh(
    n_theta: int = 30,
    n_phi: int = 30,
    radius: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a spherical mesh for plotting.

    Args:
        n_theta: Number of theta samples
        n_phi: Number of phi samples
        radius: Sphere radius

    Returns:
        Tuple (X, Y, Z, theta_grid, phi_grid)
        X, Y, Z are 2D arrays for surface plotting
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    X = radius * np.sin(theta_grid) * np.cos(phi_grid)
    Y = radius * np.sin(theta_grid) * np.sin(phi_grid)
    Z = radius * np.cos(theta_grid)

    return X, Y, Z, theta_grid, phi_grid


def compute_bead_colors(
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
    coefficients: Dict[int, Dict[int, complex]],
    scaling_factors: Dict[int, float],
    color_scheme: str = "red-green",
    bead_type: str = "Q"
) -> np.ndarray:
    """
    Compute colors for each vertex on the bead surface.

    Args:
        theta_grid: 2D array of theta values
        phi_grid: 2D array of phi values
        coefficients: LISA coefficients
        scaling_factors: Scaling factors per rank
        color_scheme: "red-green", "yellow-blue", "grayscale", etc.
        bead_type: 'Q', 'E', 'C', 'T'

    Returns:
        3D array of RGBA colors (same shape as theta_grid + (4,))
    """
    # Compute bead function values
    values = bead_function_grid(theta_grid, phi_grid, coefficients, scaling_factors)

    # Map values to colors
    from quantumviz.beads_colors import value_to_color_array
    colors = value_to_color_array(values, color_scheme, bead_type)

    return colors


# ============================================================================
# Q-Bead Specific Functions
# ============================================================================

def qbead_coefficients_from_bloch(
    rx: float, ry: float, rz: float,
    scaling_factors: Dict[int, float]
) -> Dict[int, Dict[int, complex]]:
    """
    Create LISA coefficients for a Q-Bead from Bloch vector.

    For a Q-Bead, only j=1 is needed.
    The Bloch vector (rx, ry, rz) maps to:
    - c_{1,0} ∝ rz (z-component)
    - c_{1,1c} ∝ rx (x-component)
    - c_{1,1s} ∝ ry (y-component)

    Args:
        rx, ry, rz: Bloch vector components
        scaling_factors: Scaling factors (should have j=1)

    Returns:
        Coefficients dict: {1: {0: c_{1,0}, 1: c_{1,1c}, -1: c_{1,1s}}}
    """
    # Normalization: For axial (m=0), c_{1,0} = (rz * ζ(N)) / η_1
    # For full sphere: need m=0, +1, -1
    eta_1 = np.sqrt(4 * np.pi / 3)

    if 1 not in scaling_factors:
        return {1: {0: complex(0, 0)}}

    s_1 = scaling_factors[1]

    # c_{j,m} values (before scaling)
    # For bead function: b = s_j * c_{j,m} * Y_{j,m}
    # We want: b(north pole) = rz, b(east) = rx, b(north) = ry
    # At north pole (θ=0): Y_{1,0} = sqrt(3/(4π)), Y_{1,±1} = 0
    # So: rz = s_1 * c_{1,0} * sqrt(3/(4π)) = s_1 * c_{1,0} / η_1
    # => c_{1,0} = rz * η_1 / s_1

    if abs(s_1) > 1e-10:
        c_10 = complex(rz * eta_1 / s_1, 0)
    else:
        c_10 = complex(0, 0)

    # For m=+1 (cosine): at equator, phi=0: Y_{1,1c} ∝ cos(phi) = 1
    # rx = s_1 * c_{1,1} * Y_{1,1c}(θ=π/2, φ=0)
    # Y_{1,1c}(π/2, 0) = sqrt(3/(4π)) * cos(0) / sqrt(2)
    if abs(s_1) > 1e-10:
        c_11 = complex(rx * eta_1 / s_1 / np.sqrt(2), 0)
        c_1m1 = complex(ry * eta_1 / s_1 / np.sqrt(2), 0)  # phi=π/2 gives sin
    else:
        c_11 = complex(0, 0)
        c_1m1 = complex(0, 0)

    return {
        1: {
            0: c_10,
            1: c_11,   # m=+1 → cosine variant
            -1: c_1m1,  # m=-1 → sine variant
        }
    }


# ============================================================================
# E-Bead Specific Functions
# ============================================================================

def ebead_even_coefficients(
    c0: complex, c2: complex,
    scaling_factors: Dict[int, float]
) -> Dict[int, Dict[int, complex]]:
    """
    Create LISA coefficients for E-Bead even (symmetric) component.

    Args:
        c0: Coefficient for j=0 (from decomposition)
        c2: Coefficient for j=2 (from decomposition)
        scaling_factors: Should have j=0 and j=2

    Returns:
        Coefficients dict with j=0 and j=2 entries
    """
    result = {}

    if 0 in scaling_factors:
        result[0] = {0: c0}
    if 2 in scaling_factors:
        # j=2 has m = -2, -1, 0, 1, 2
        # For axial (m=0) only:
        result[2] = {0: c2}

    return result


def ebead_odd_coefficients(
    c1: complex,
    scaling_factors: Dict[int, float]
) -> Dict[int, Dict[int, complex]]:
    """
    Create LISA coefficients for E-Bead odd (antisymmetric) component.

    Args:
        c1: Coefficient for j=1
        scaling_factors: Should have j=1

    Returns:
        Coefficients dict with j=1 entry
    """
    result = {}
    if 1 in scaling_factors:
        result[1] = {0: c1}
    return result


# ============================================================================
# Utility: Convert Reduced Density Matrix to Bead Coefficients
# ============================================================================

def reduced_density_to_qbead_coeffs(
    rho_red: np.ndarray,
    scaling_factors: Dict[int, float],
    N: int
) -> Dict[int, Dict[int, complex]]:
    """
    Convert reduced density matrix to Q-Bead LISA coefficients.

    Args:
        rho_red: 2x2 reduced density matrix for a qubit
        scaling_factors: Scaling factors for j=1
        N: Total qubits in system

    Returns:
        Coefficients dict for Q-Bead
    """
    # Compute Bloch vector from reduced density matrix
    rx = float(np.real(np.trace(rho_red @ np.array([[0, 1], [1, 0]], dtype=complex))))
    ry = float(np.real(np.trace(rho_red @ np.array([[0, -1j], [1j, 0]], dtype=complex))))
    rz = float(np.real(np.trace(rho_red @ np.array([[1, 0], [0, -1]], dtype=complex))))

    return qbead_coefficients_from_bloch(rx, ry, rz, scaling_factors)


# ============================================================================
# Mesh with Bead Function Values (for coloring)
# ============================================================================

def bead_surface_with_values(
    coefficients: Dict[int, Dict[int, complex]],
    scaling_factors: Dict[int, float],
    n_theta: int = 30,
    n_phi: int = 30,
    radius: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate sphere surface with bead function values at each vertex.

    Args:
        coefficients: LISA coefficients
        scaling_factors: Scaling factors per rank
        n_theta, n_phi: Mesh resolution
        radius: Sphere radius

    Returns:
        Tuple (X, Y, Z, values)
        X, Y, Z: 2D coordinate arrays
        values: 2D array of bead function values (for coloring)
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    X = radius * np.sin(theta_grid) * np.cos(phi_grid)
    Y = radius * np.sin(theta_grid) * np.sin(phi_grid)
    Z = radius * np.cos(theta_grid)

    values = bead_function_grid(theta_grid, phi_grid, coefficients, scaling_factors)

    return X, Y, Z, values
