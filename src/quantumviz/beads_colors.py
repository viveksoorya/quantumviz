"""
BEADS Color Schemes Module

Implements all color schemes from Appendix F of the BEADS paper:
- Standard: Red-Green (Q/C) + Yellow-Blue (E)
- Discontinuous: Band-based (0.1 increments)
- Continuous: Smooth gradient
- High contrast: Red-White-Green, Yellow-White-Blue
- Colorblind-friendly: Red-Blue + Yellow-Green
- Grayscale: For printing

Also implements the total correlation color wheel (Figure F2).

Reference:
- Appendix F: BEADS color schemes
- Figure F1: Color scale variants
- Figure F2: Total correlation color wheel
"""

import numpy as np
from typing import Tuple, Optional, Dict
from enum import Enum


# ============================================================================
# Color Scheme Types
# ============================================================================

class ColorScheme(Enum):
    """Enum for different color scheme types."""
    STANDARD = "standard"
    DISCONTINUOUS = "discontinuous"
    CONTINUOUS = "continuous"
    HIGH_CONTRAST = "high_contrast"
    COLORBLIND = "colorblind"
    GRAYSCALE = "grayscale"


# ============================================================================
# Standard Discontinuous Color Scales (Figure F1, Appendix F)
# ============================================================================

def standard_red_green(value: float) -> Tuple[float, float, float, float]:
    """
    Standard red-green discontinuous scale for Q-Beads and compound correlations.

    Ranges:
    - value = +1.0 → Bright Red (1, 0, 0)
    - value = +0.5 → Yellow (1, 1, 0)
    - value =  0.0 → Black (0, 0, 0)
    - value = -0.5 → Cyan (0, 1, 1)
    - value = -1.0 → Bright Green (0, 1, 0)

    Uses 0.1 bands for discontinuous look.

    Args:
        value: Expectation value in [-1, 1]

    Returns:
        RGBA tuple (R, G, B, A) with values in [0, 1]
    """
    value = np.clip(value, -1.0, 1.0)

    # Map to 0.1 bands (discontinuous)
    band = np.round(value * 10) / 10  # Round to nearest 0.1

    if band >= 0.5:
        # Red to Yellow: G increases from 0 to 1
        t = (band - 0.5) * 2  # 0 to 1
        r, g, b = 1.0, t, 0.0
    elif band > -0.5:
        # Yellow to Black to Cyan: R decreases, B increases
        if band >= 0:
            # Yellow (1,1,0) to Black (0,0,0)
            t = band * 2  # 0 to 1
            r, g, b = 1.0 - t, 1.0 - t, 0.0
        else:
            # Black (0,0,0) to Cyan (0,1,1)
            t = abs(band) * 2  # 0 to 1
            r, g, b = 0.0, t, t
    else:
        # Cyan to Green: R stays 0, B decreases
        t = (abs(band) - 0.5) * 2  # 0 to 1
        r, g, b = 0.0, 1.0, 1.0 - t

    return (r, g, b, 1.0)


def standard_yellow_blue(value: float) -> Tuple[float, float, float, float]:
    """
    Standard yellow-blue discontinuous scale for E-Beads (connected correlations).

    Ranges:
    - value = +1.0 → Bright Yellow (1, 1, 0)
    - value =  0.0 → Black (0, 0, 0)
    - value = -1.0 → Bright Blue (0, 0, 1)

    Args:
        value: Correlation value in [-1, 1]

    Returns:
        RGBA tuple
    """
    value = np.clip(value, -1.0, 1.0)

    band = np.round(value * 10) / 10

    if band >= 0:
        # Yellow to Black: R and G decrease
        t = band  # 0 to 1
        r, g, b = 1.0 - t, 1.0 - t, 0.0
    else:
        # Black to Blue: B increases
        t = abs(band)  # 0 to 1
        r, g, b = 0.0, 0.0, t

    return (r, g, b, 1.0)


# ============================================================================
# Continuous Color Scales
# ============================================================================

def continuous_red_green(value: float) -> Tuple[float, float, float, float]:
    """
    Continuous red-green scale (smooth gradient).

    Linear interpolation:
    - value = +1 → Red (1, 0, 0)
    - value =  0 → Black (0, 0, 0)
    - value = -1 → Green (0, 1, 0)
    """
    value = np.clip(value, -1.0, 1.0)

    if value >= 0:
        # Red to Black
        r, g, b = value, 0.0, 0.0
    else:
        # Black to Green
        r, g, b = 0.0, abs(value), 0.0

    return (r, g, b, 1.0)


def continuous_yellow_blue(value: float) -> Tuple[float, float, float, float]:
    """
    Continuous yellow-blue scale.

    - value = +1 → Yellow (1, 1, 0)
    - value =  0 → Black (0, 0, 0)
    - value = -1 → Blue (0, 0, 1)
    """
    value = np.clip(value, -1.0, 1.0)

    if value >= 0:
        r, g, b = value, value, 0.0
    else:
        r, g, b = 0.0, 0.0, abs(value)

    return (r, g, b, 1.0)


# ============================================================================
# High Contrast Scales
# ============================================================================

def high_contrast_red_green(value: float) -> Tuple[float, float, float, float]:
    """
    High contrast red-white-green scale.

    - value = +1 → Bright Red (1, 0, 0)
    - value =  0 → White (1, 1, 1)
    - value = -1 → Bright Green (0, 1, 0)
    """
    value = np.clip(value, -1.0, 1.0)

    if value >= 0:
        r, g, b = 1.0, 1.0 - value, 1.0 - value
    else:
        r, g, b = 1.0 - abs(value), 1.0, 1.0 - abs(value)

    return (r, g, b, 1.0)


def high_contrast_yellow_blue(value: float) -> Tuple[float, float, float, float]:
    """
    High contrast yellow-white-blue scale.

    - value = +1 → Bright Yellow (1, 1, 0)
    - value =  0 → White (1, 1, 1)
    - value = -1 → Bright Blue (0, 0, 1)
    """
    value = np.clip(value, -1.0, 1.0)

    if value >= 0:
        r, g, b = 1.0, 1.0, 1.0 - value
    else:
        r, g, b = 1.0 - abs(value), 1.0 - abs(value), 1.0

    return (r, g, b, 1.0)


# ============================================================================
# Colorblind-Friendly Scales
# ============================================================================

def colorblind_red_blue(value: float) -> Tuple[float, float, float, float]:
    """
    Colorblind-friendly: Red-Blue scale (for Q-Beads/compound).

    - value = +1 → Red (0.8, 0, 0)
    - value =  0 → White (1, 1, 1)
    - value = -1 → Blue (0, 0, 0.8)
    """
    value = np.clip(value, -1.0, 1.0)

    if value >= 0:
        r, g, b = 0.8 + 0.2 * (1 - value), 1.0 - value, 1.0 - value
    else:
        r, g, b = 1.0 - abs(value), 1.0 - abs(value), 0.8 + 0.2 * (1 - abs(value))

    return (r, g, b, 1.0)


def colorblind_yellow_green(value: float) -> Tuple[float, float, float, float]:
    """
    Colorblind-friendly: Yellow-Green scale (for E-Beads/connected).

    - value = +1 → Yellow (1, 1, 0)
    - value =  0 → White (1, 1, 1)
    - value = -1 → Green (0, 0.6, 0)
    """
    value = np.clip(value, -1.0, 1.0)

    if value >= 0:
        r, g, b = 1.0, 1.0, 1.0 - value
    else:
        r, g, b = 1.0 - abs(value), 1.0 + value, 1.0 - abs(value) * 0.4

    return (r, g, b, 1.0)


# ============================================================================
# Grayscale Scale
# ============================================================================

def grayscale(value: float) -> Tuple[float, float, float, float]:
    """
    Grayscale: Black to White.

    - value = +1 → White (1, 1, 1)
    - value =  0 → Gray (0.5, 0.5, 0.5)
    - value = -1 → Black (0, 0, 0)
    """
    value = np.clip(value, -1.0, 1.0)
    gray = (value + 1.0) / 2.0  # Map [-1,1] to [0,1]
    return (gray, gray, gray, 1.0)


# ============================================================================
# Probability Color (P(0) → Red-Green)
# ============================================================================

def prob_0_to_color(prob_0: float) -> Tuple[float, float, float, float]:
    """
    Convert P(0) probability to color (as in paper TikZ example).

    - P(0) = 1.0 → Red
    - P(0) = 0.5 → Black (equator view)
    - P(0) = 0.0 → Green

    Args:
        prob_0: Probability of measuring |0⟩ (0 to 1)

    Returns:
        RGBA tuple
    """
    prob_0 = np.clip(prob_0, 0.0, 1.0)

    if prob_0 >= 0.5:
        # Red to Black: G decreases
        t = 2 * (prob_0 - 0.5)  # 0 to 1
        r, g, b = 1.0, 1.0 - t, 0.0
    else:
        # Black to Green: R increases
        t = 2 * prob_0  # 0 to 1
        r, g, b = t, 1.0, 0.0

    return (r, g, b, 1.0)


def prob_0_to_color_continuous(prob_0: float) -> Tuple[float, float, float, float]:
    """
    Continuous version: Red (P=1) → Yellow (P=0.5) → Green (P=0).

    This matches the TikZ shader{ball color=red} and {ball color=green} behavior.
    """
    prob_0 = np.clip(prob_0, 0.0, 1.0)

    if prob_0 >= 0.5:
        # Red to Yellow
        t = 2 * (prob_0 - 0.5)
        r, g, b = 1.0, t, 0.0
    else:
        # Yellow to Green
        t = 2 * prob_0
        r, g, b = t, 1.0, 0.0

    return (r, g, b, 1.0)


# ============================================================================
# Value to Color Dispatch
# ============================================================================

def value_to_color(
    value: float,
    scheme: str = "standard",
    bead_type: str = "Q"
) -> Tuple[float, float, float, float]:
    """
    Convert a value to color based on scheme and bead type.

    Args:
        value: Value in [-1, 1] (expectation value or correlation)
        scheme: Color scheme name
        bead_type: 'Q' (red-green), 'E' (yellow-blue), 'C' (red-green), 'T' (mixed)

    Returns:
        RGBA tuple
    """
    # Determine which scale to use
    if bead_type in ('Q', 'C'):
        scale_type = 'red_green'
    elif bead_type == 'E':
        scale_type = 'yellow_blue'
    else:
        scale_type = 'red_green'  # Default

    # Dispatch to appropriate function
    if scheme == "standard":
        if scale_type == 'red_green':
            return standard_red_green(value)
        else:
            return standard_yellow_blue(value)
    elif scheme == "continuous":
        if scale_type == 'red_green':
            return continuous_red_green(value)
        else:
            return continuous_yellow_blue(value)
    elif scheme == "high_contrast":
        if scale_type == 'red_green':
            return high_contrast_red_green(value)
        else:
            return high_contrast_yellow_blue(value)
    elif scheme == "colorblind":
        if scale_type == 'red_green':
            return colorblind_red_blue(value)
        else:
            return colorblind_yellow_green(value)
    elif scheme == "grayscale":
        return grayscale(value)
    else:
        # Default to standard
        if scale_type == 'red_green':
            return standard_red_green(value)
        else:
            return standard_yellow_blue(value)


def value_to_color_array(
    values: np.ndarray,
    scheme: str = "standard",
    bead_type: str = "Q"
) -> np.ndarray:
    """
    Vectorized version: apply value_to_color to an array of values.

    Args:
        values: 2D array of values
        scheme: Color scheme name
        bead_type: 'Q', 'E', 'C', or 'T'

    Returns:
        3D array of shape (H, W, 4) with RGBA values
    """
    shape = values.shape
    colors = np.zeros(shape + (4,), dtype=np.float32)

    # Vectorized computation
    if bead_type in ('Q', 'C'):
        scale_type = 'red_green'
    elif bead_type == 'E':
        scale_type = 'yellow_blue'
    else:
        scale_type = 'red_green'

    clipped = np.clip(values, -1.0, 1.0)

    if scheme == "standard":
        if scale_type == 'red_green':
            # Discontinuous: round to 0.1 bands
            bands = np.round(clipped * 10) / 10

            # Red-Green discontinuous
            mask_pos = bands >= 0.5
            mask_mid_pos = (bands >= 0) & (bands < 0.5)
            mask_neg = bands < 0

            # Red to Yellow (bands 0.5 to 1.0)
            t = (bands - 0.5) * 2
            colors[mask_pos, 0] = 1.0
            colors[mask_pos, 1] = np.clip(t[mask_pos], 0, 1)
            colors[mask_pos, 2] = 0.0
            colors[mask_pos, 3] = 1.0

            # Yellow to Black to Cyan (bands 0 to -0.5)
            t_mid = bands[mask_mid_pos] * 2
            colors[mask_mid_pos, 0] = np.clip(1.0 - t_mid, 0, 1)
            colors[mask_mid_pos, 1] = np.clip(1.0 - t_mid, 0, 1)
            colors[mask_mid_pos, 2] = 0.0
            colors[mask_mid_pos, 3] = 1.0

            # Cyan to Green (bands -0.5 to -1.0)
            t_neg = (np.abs(bands) - 0.5) * 2
            colors[mask_neg, 0] = 0.0
            colors[mask_neg, 1] = 1.0
            colors[mask_neg, 2] = np.clip(1.0 - t_neg[mask_neg], 0, 1)
            colors[mask_neg, 3] = 1.0
        else:
            # Yellow-Blue discontinuous
            bands = np.round(clipped * 10) / 10
            mask_pos = bands >= 0
            mask_neg = bands < 0

            t_pos = np.clip(bands[mask_pos], 0, 1)
            colors[mask_pos, 0] = 1.0 - t_pos
            colors[mask_pos, 1] = 1.0 - t_pos
            colors[mask_pos, 2] = 0.0
            colors[mask_pos, 3] = 1.0

            t_neg = np.clip(np.abs(bands[mask_neg]), 0, 1)
            colors[mask_neg, 0] = 0.0
            colors[mask_neg, 1] = 0.0
            colors[mask_neg, 2] = t_neg
            colors[mask_neg, 3] = 1.0
    else:
        # For non-standard schemes, use the scalar function iteratively
        # (Can be optimized later)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                colors[i, j] = value_to_color(values[i, j], scheme, bead_type)

    return colors


# ============================================================================
# Total Correlation Color Wheel (Figure F2)
# ============================================================================

def total_correlation_color(
    T: float,
    E: float,
    C: float,
    scheme: str = "standard"
) -> Tuple[float, float, float, float]:
    """
    Compute T-Bead color using the total correlation color wheel (Equation F.2).

    Args:
        T: Total correlation coefficient
        E: Connected correlation coefficient
        C: Compound correlation coefficient
        scheme: Color scheme

    Returns:
        RGBA tuple for the blended color
    """
    T = np.clip(T, -1.0, 1.0)
    E = np.clip(E, -1.0, 1.0)
    C = np.clip(C, -1.0, 1.0)

    # Get base colors
    if scheme == "standard":
        # Red-green for C, Yellow-blue for E
        Gamma_C = np.array(standard_red_green(C)[:3])
        Gamma_E = np.array(standard_yellow_blue(E)[:3])
    else:
        # Use the appropriate functions
        Gamma_C = np.array(value_to_color(C, scheme, 'C')[:3])
        Gamma_E = np.array(value_to_color(E, scheme, 'E')[:3])

    # Blending angle (equation F.1)
    if abs(C) < 1e-10:
        theta_blend = np.pi / 2  # Default if C=0
    else:
        theta_blend = np.arctan(abs(E) / abs(C))

    # Blend (equation F.2)
    blend_factor = (2 * theta_blend / np.pi)
    blended = Gamma_C + blend_factor * (Gamma_E - Gamma_C)

    return (float(blended[0]), float(blended[1]), float(blended[2]), 1.0)


def generate_color_wheel(
    n_samples: int = 200,
    scheme: str = "standard"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the total correlation color wheel (Figure F2).

    Returns:
        Tuple of (R, G, B) arrays for the color wheel image
    """
    # Create polar grid
    r = np.linspace(0, 1, n_samples)
    phi = np.linspace(0, 2 * np.pi, n_samples)
    R, PHI = np.meshgrid(r, phi)

    # Map to E:C ratio
    # phi_corr = atan(E/C)
    # T = r (radius)

    colors = np.zeros((n_samples, n_samples, 4))

    for i in range(n_samples):
        for j in range(n_samples):
            r_val = R[i, j]
            phi_val = PHI[i, j]

            # E:C ratio from angle
            if phi_val <= np.pi / 2:
                # First quadrant: E>0, C>0
                E = r_val * np.sin(phi_val)
                C = r_val * np.cos(phi_val)
            elif phi_val <= np.pi:
                # Second quadrant: E>0, C<0
                E = r_val * np.sin(phi_val)
                C = r_val * np.cos(phi_val)
            elif phi_val <= 3 * np.pi / 2:
                # Third quadrant: E<0, C<0
                E = r_val * np.sin(phi_val)
                C = r_val * np.cos(phi_val)
            else:
                # Fourth quadrant: E<0, C>0
                E = r_val * np.sin(phi_val)
                C = r_val * np.cos(phi_val)

            T = E + C
            colors[i, j] = total_correlation_color(T, E, C, scheme)

    return colors


# ============================================================================
# Color Bar Creation
# ============================================================================

def create_colorbar(
    ax,
    scheme: str = "standard",
    bead_type: str = "Q",
    label: str = ""
):
    """
    Create a color bar for a given scheme.

    Args:
        ax: Matplotlib axes to draw on
        scheme: Color scheme
        bead_type: 'Q', 'E', 'C', or 'T'
        label: Label for the color bar
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Create gradient
    gradient = np.linspace(-1, 1, 256).reshape(1, -1)
    gradient = np.repeat(gradient, 10, axis=0)

    # Map to colors
    if bead_type in ('Q', 'C'):
        if scheme == "standard":
            cmap = mcolors.LinearSegmentedColormap.from_list(
                'red_green', [(0, 1, 0), (0.5, 0.5, 0), (1, 0, 0)]
            )
        elif scheme == "continuous":
            cmap = mcolors.LinearSegmentedColormap.from_list(
                'red_green_cont', [(0, 1, 0), (0, 0, 0), (1, 0, 0)]
            )
        else:
            cmap = plt.cm.RdYlGn
    else:  # E-Bead
        if scheme == "standard":
            cmap = mcolors.LinearSegmentedColormap.from_list(
                'yellow_blue', [(1, 1, 0), (0, 0, 0), (0, 0, 1)]
            )
        else:
            cmap = plt.cm.RdYlBu

    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[-1, 1, 0, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(label)
    ax.set_yticks([])

    # Add labels
    if bead_type in ('Q', 'C'):
        ax.text(-1, -0.2, "|1⟩ (Green)", fontsize=8, ha='left')
        ax.text(0, -0.2, "0.5", fontsize=8, ha='center')
        ax.text(1, -0.2, "|0⟩ (Red)", fontsize=8, ha='right')
    else:
        ax.text(-1, -0.2, "Anti-correlated (Blue)", fontsize=8, ha='left')
        ax.text(0, -0.2, "0", fontsize=8, ha='center')
        ax.text(1, -0.2, "Correlated (Yellow)", fontsize=8, ha='right')
