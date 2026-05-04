"""
Dimensional Circular Notation (DCN) Visualization Module

A visualization where:
- Each basis state with non-zero amplitude gets a circle
- Position in grid shows qubit values
- Inner circle radius shows magnitude (for 1/2 qubits)
- Inner circle color shows MSB qubit value (for 1/2 qubits)
- Radius line direction shows phase (UP=0°, DOWN=180°, etc.)
- Circle labeled with basis state
"""

import json
import os
from typing import Any, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as MplCircle

matplotlib.use('Agg')


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


def is_separable_along_qubit(state_vector: List[complex], n_qubits: int, qubit_idx: int) -> bool:
    """Check if state is separable along given qubit (0=LSB, n_qubits-1=MSB)."""
    groups = {}
    for i, amp in enumerate(state_vector):
        key = i & ~(1 << qubit_idx)
        q_val = (i >> qubit_idx) & 1
        if key not in groups:
            groups[key] = [0+0j, 0+0j]
        groups[key][q_val] = amp

    # Find reference non-zero group
    ref = None
    for key, amps in groups.items():
        norm = np.linalg.norm(amps)
        if norm > 1e-9:
            ref = np.array(amps)
            break

    if ref is None:
        return True  # All zero

    # Check all non-zero groups are scalar multiples of ref
    for key, amps in groups.items():
        curr = np.array(amps)
        curr_norm = np.linalg.norm(curr)
        if curr_norm < 1e-9:
            continue
        # Find scalar: use first non-zero element of ref
        for j in range(2):
            if abs(ref[j]) > 1e-9:
                scalar = curr[j] / ref[j]
                if not np.allclose(curr, scalar * ref, atol=1e-9):
                    return False
                break
    return True


def plot_dcn(
    state: Union[List[complex], Any],
    title: str = "DCN Visualization",
    output_path: Optional[str] = None,
    dpi: int = 150
) -> Optional[plt.Figure]:
    """
    Create a Dimensional Circular Notation (DCN) plot.

    For n-qubit system: circles arranged in grid.
    Each circle shows one basis state's amplitude:
    - Position: based on qubit values
    - Inner circle (1/2 qubits): radius = magnitude * outer_r, color = MSB qubit value
    - Radius line: direction = phase, length = outer_r
    - Circle labeled with basis state

    For 3 qubits: 2D projection of 3D cube (front face Q3=0, back face Q3=1)

    Args:
        state: List of complex amplitudes, or Qiskit Statevector object
        title: Title for the plot
        output_path: Path to save figure
        dpi: Resolution

    Returns:
        matplotlib Figure if output_path is None
    """
    # Handle Qiskit Statevector objects
    from quantumviz.qiskit_bridge import is_statevector, statevector_to_list
    if is_statevector(state):
        state_vector = statevector_to_list(state)
    else:
        state_vector = state

    n = len(state_vector)
    if n == 0:
        raise ValueError("State vector cannot be empty")

    n_qubits = int(np.log2(n))
    if 2 ** n_qubits != n:
        raise ValueError(f"State vector length {n} is not power of 2")

    # Color map for qubit identification (based on MSB)
    qubit_cmap = plt.cm.tab10  # type: ignore[attr-defined]

    fig_size = 8 if n_qubits == 3 else 4
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    fig.suptitle(title, fontsize=12, y=0.98)

    # Circle radius (outer circle)
    circle_r = 1.0

    # Store positions for drawing lines later (3-qubit case)
    positions = {}

    # TikZ scale factor (consistent across position and axis calculations)
    scale = 0.72

    # For 3-qubit: First pass - collect positions only
    # Second pass - draw circles (after planes)
    if n_qubits == 3:
        # TikZ parameters (scaled for matplotlib)
        ax_dim = 5.5 * scale   # \ax: Q1 horizontal spacing
        ay_dim = 3.0 * scale   # \ay: Q2 vertical spacing
        sx = 2.5 * scale      # \sx: back face horizontal shift
        sy = 2.0 * scale      # \sy: back face vertical shift
        circle_r_scaled = 1.0 * scale  # Circle radius

        # First pass: collect all positions
        for i, amp in enumerate(state_vector):
            magnitude = abs(amp)
            phase = np.angle(amp) if magnitude >= 0.01 else 0

            # Extract bits using big-endian: |Q1 Q2 Q3> where Q1 is MSB
            q1 = (i >> 2) & 1  # MSB (horizontal, x-axis)
            q2 = (i >> 1) & 1  # Middle (vertical, y-axis)
            q3 = i & 1          # LSB (front/back face)

            # Position calculation (matching TikZ exactly)
            x_pos = q1 * ax_dim + (sx if q3 else 0)
            y_pos = q2 * ay_dim + (sy if q3 else 0)

            positions[i] = (x_pos, y_pos, q1, q2, q3, magnitude, phase)

        # Draw planes FIRST (behind everything)
        # Check separability along each qubit
        separable_q1 = is_separable_along_qubit(state_vector, 3, qubit_idx=2)
        separable_q2 = is_separable_along_qubit(state_vector, 3, qubit_idx=1)
        separable_q3 = is_separable_along_qubit(state_vector, 3, qubit_idx=0)

        # Determine which planes to draw (at most 2, ordered by axis Q1,Q2,Q3)
        sep_axes = [q for q in [1, 2, 3] if [separable_q1, separable_q2, separable_q3][q-1]]
        ent_axes = [q for q in [1, 2, 3] if not [separable_q1, separable_q2, separable_q3][q-1]]

        if len(sep_axes) == 3:  # Fully separable
            planes = [(1, 'green', 'sep Q1'), (2, 'green', 'sep Q2')]
        elif len(sep_axes) == 0:  # Fully entangled
            planes = [(1, 'red', 'ent Q1'), (2, 'red', 'ent Q2')]
        else:  # Mixed: 1 sep + 2 ent -> show 1 green + 1 red
            first_sep = next(q for q in [1, 2, 3] if q in sep_axes)
            first_ent = next(q for q in [1, 2, 3] if q in ent_axes)
            planes = [(first_sep, 'green', f'sep Q{first_sep}'),
                      (first_ent, 'red', f'ent Q{first_ent}')]

        # Helper to generate plane points for each axis
        def get_plane_pts(axis, m):
            if axis == 1:  # Q1: vertical plane at x = ax_dim/2
                x_mid = ax_dim / 2
                return [(x_mid, -m), (x_mid, ay_dim + m),
                        (x_mid + sx, ay_dim + sy + m), (x_mid + sx, sy - m)]
            elif axis == 2:  # Q2: horizontal-ish plane at y = ay_dim/2
                y_mid = ay_dim / 2
                return [(-m, y_mid), (ax_dim + m, y_mid),
                        (ax_dim + sx + m, y_mid + sy), (sx - m, y_mid + sy)]
            else:  # Q3: diagonal rectangle at offset (sx/2, sy/2)
                x_off = sx / 2
                y_off = sy / 2
                return [(x_off - m, y_off - m), (ax_dim + x_off + m, y_off - m),
                        (ax_dim + x_off + m, ay_dim + y_off + m),
                        (x_off - m, ay_dim + y_off + m)]

        # Draw planes in axis order (Q1, Q2, Q3) - BEHIND other elements
        margin_small = 0.3 * scale
        for axis, color, label in sorted(planes, key=lambda x: x[0]):
            pts = get_plane_pts(axis, margin_small)
            pts.append(pts[0])  # Close polygon
            px_vals = [p[0] for p in pts]
            py_vals = [p[1] for p in pts]
            ax.fill(px_vals, py_vals, color, alpha=0.2, edgecolor=color, linewidth=2,
                   zorder=1)
            # Position label at center of plane
            if axis == 1:
                label_x = ax_dim/2 + sx/2
                label_y = (ay_dim + sy) / 2
            elif axis == 2:
                label_x = (ax_dim + sx) / 2
                label_y = ay_dim/2 + sy/2
            else:
                label_x = sx/2 + ax_dim/2
                label_y = sy/2 + ay_dim/2
            ax.text(label_x, label_y, label, ha='center', va='center',
                   fontsize=7, color=color, fontweight='bold', zorder=2)

        # Second pass: draw all circles and labels (ON TOP of planes)
        for i, (x_pos, y_pos, q1, q2, q3, magnitude, phase) in positions.items():
            # Determine if front or back face
            is_back_face = (q3 == 1)

            # Outer circle - always draw
            outer_circle = MplCircle((x_pos, y_pos), circle_r_scaled, fill=False,
                                     color='black', linewidth=1.5,
                                     linestyle='--' if is_back_face else '-',
                                     alpha=0.6 if is_back_face else 1.0,
                                     zorder=3)
            ax.add_patch(outer_circle)

            # Basis state label - always draw
            basis_label = format(i, f'0{n_qubits}b')
            ax.text(x_pos, y_pos - circle_r_scaled - 0.1, f'|{basis_label}⟩',
                   ha='center', va='top', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   zorder=4)

            # Only draw amplitude-dependent elements for non-zero states
            if magnitude >= 0.01:
                # Inner circle (filled) - radius proportional to magnitude
                inner_r = circle_r_scaled * magnitude
                if inner_r > 0.01:
                    # Color based on Q3 (front/back face): blue for Q3=0, green for Q3=1
                    inner_color = 'skyblue' if q3 == 0 else 'lightgreen'
                    inner_circle = MplCircle((x_pos, y_pos), inner_r, fill=True,
                                             color=inner_color, linewidth=0, alpha=0.7,
                                             zorder=3)
                    ax.add_patch(inner_circle)

                # Phase line (direction = phase angle)
                endpoint_x = x_pos + circle_r_scaled * np.cos(phase)
                endpoint_y = y_pos + circle_r_scaled * np.sin(phase)
                ax.plot([x_pos, endpoint_x], [y_pos, endpoint_y],
                       color='black', linewidth=2.5,
                       alpha=0.6 if is_back_face else 1.0,
                       zorder=3)

                # Center dot
                ax.plot(x_pos, y_pos, 'ko', markersize=3,
                       alpha=0.6 if is_back_face else 1.0,
                       zorder=3)

    else:
        # 1 or 2 qubit case - draw circles in a loop
        for i, amp in enumerate(state_vector):
            magnitude = abs(amp)
            phase = np.angle(amp) if magnitude >= 0.01 else 0

            if n_qubits == 1:
                x = i
                y = 0
            else:
                x = (i >> 1) & 1
                y = i & 1

            if n_qubits == 1:
                px = 0.6 + x * 2.2
                py = 1.2
            else:
                px = 1.0 + x * 2.2
                py = 0.5 + y * 2.6

            # Draw outer circle
            circle = MplCircle((px, py), circle_r, fill=False, color='black', linewidth=1.5)
            ax.add_patch(circle)

            # Inner circle (filled) - radius = magnitude * outer_r
            inner_r = circle_r * magnitude
            if inner_r > 0.01:
                if n_qubits == 1:
                    color_idx = 0
                else:
                    color_idx = (i >> 1) & 1
                inner_color = qubit_cmap(color_idx * 2)
                inner_circle = MplCircle((px, py), inner_r, fill=True, color=inner_color,
                                         linewidth=0, alpha=0.7)
                ax.add_patch(inner_circle)

            # Basis state label
            basis_label = format(i, f'0{n_qubits}b')[-(n_qubits):]
            ax.text(px, py - circle_r - 0.3, f'|{basis_label}⟩', ha='center', va='top',
                   fontsize=9, fontweight='bold')

            # Phase line
            if magnitude >= 0.01:
                endpoint_x = px + circle_r * np.sin(phase)
                endpoint_y = py + circle_r * np.cos(phase)
                ax.plot([px, endpoint_x], [py, endpoint_y], color='black', linewidth=3)

            # Center dot
            ax.plot(px, py, 'ko', markersize=4)

    # Draw separability plane and connecting lines for 3-qubit case
    if n_qubits == 3:
        # Scale factors (must match the values used above for positioning)
        # scale is already defined at line 115
        ax_dim = 5.5 * scale   # \ax: Q1 horizontal spacing
        ay_dim = 3.0 * scale   # \ay: Q2 vertical spacing
        sx = 2.5 * scale      # \sx: back face horizontal shift
        sy = 2.0 * scale      # \sy: back face vertical shift
        circle_r = 1.0 * scale  # Circle radius

        # Origin point (top-left, where all axes start)
        # TikZ origin: (-3.0, \ay+1.0) = (-3.0, 4.0) (all axes start here)
        origin_x = -3.0 * scale
        origin_y = (ay_dim/scale + 1.0) * scale  # = (3.0 + 1.0) * scale = 4.0 * scale

        # Helper: get point on circle circumference toward target
        def circ_point(center, target, r):
            dx = target[0] - center[0]
            dy = target[1] - center[1]
            dist = np.hypot(dx, dy)
            if dist < 1e-9:
                return center
            return (center[0] + dx/dist * r, center[1] + dy/dist * r)

        # Check separability along each qubit
        # qubit_idx: 2=Q1/MSB, 1=Q2, 0=Q3/LSB
        separable_q1 = is_separable_along_qubit(state_vector, 3, qubit_idx=2)
        separable_q2 = is_separable_along_qubit(state_vector, 3, qubit_idx=1)
        separable_q3 = is_separable_along_qubit(state_vector, 3, qubit_idx=0)

        # Determine which planes to draw (at most 2, ordered by axis Q1,Q2,Q3)
        sep_axes = [q for q in [1, 2, 3] if [separable_q1, separable_q2, separable_q3][q-1]]
        ent_axes = [q for q in [1, 2, 3] if not [separable_q1, separable_q2, separable_q3][q-1]]

        if len(sep_axes) == 3:  # Fully separable
            planes = [(1, 'green', 'sep Q1'), (2, 'green', 'sep Q2')]
        elif len(sep_axes) == 0:  # Fully entangled
            planes = [(1, 'red', 'ent Q1'), (2, 'red', 'ent Q2')]
        else:  # Mixed: 1 sep + 2 ent -> show 1 green + 1 red
            first_sep = next(q for q in [1, 2, 3] if q in sep_axes)
            first_ent = next(q for q in [1, 2, 3] if q in ent_axes)
            planes = [(first_sep, 'green', f'sep Q{first_sep}'),
                      (first_ent, 'red', f'ent Q{first_ent}')]

        # Helper to generate plane points for each axis
        def get_plane_pts(axis, m):
            if axis == 1:  # Q1: vertical plane at x = ax_dim/2
                x_mid = ax_dim / 2
                return [(x_mid, -m), (x_mid, ay_dim + m),
                        (x_mid + sx, ay_dim + sy + m), (x_mid + sx, sy - m)]
            elif axis == 2:  # Q2: horizontal-ish plane at y = ay_dim/2
                y_mid = ay_dim / 2
                return [(-m, y_mid), (ax_dim + m, y_mid),
                        (ax_dim + sx + m, y_mid + sy), (sx - m, y_mid + sy)]
            else:  # Q3: diagonal rectangle at offset (sx/2, sy/2)
                x_off = sx / 2
                y_off = sy / 2
                return [(x_off - m, y_off - m), (ax_dim + x_off + m, y_off - m),
                        (ax_dim + x_off + m, ay_dim + y_off + m),
                        (x_off - m, ay_dim + y_off + m)]

        # Draw planes in axis order (Q1, Q2, Q3)
        margin_small = 0.3 * scale
        for axis, color, label in sorted(planes, key=lambda x: x[0]):
            pts = get_plane_pts(axis, margin_small)
            pts.append(pts[0])  # Close polygon
            px_vals = [p[0] for p in pts]
            py_vals = [p[1] for p in pts]
            ax.fill(px_vals, py_vals, color, alpha=0.2, edgecolor=color, linewidth=2)
            # Position label at center of plane
            if axis == 1:
                label_x = ax_dim/2 + sx/2
                label_y = (ay_dim + sy) / 2
            elif axis == 2:
                label_x = (ax_dim + sx) / 2
                label_y = ay_dim/2 + sy/2
            else:
                label_x = sx/2 + ax_dim/2
                label_y = sy/2 + ay_dim/2
            ax.text(label_x, label_y, label, ha='center', va='center',
                   fontsize=7, color=color, fontweight='bold')

        # Draw connecting lines between adjacent basis states (circumference to circumference)
        for i_curr, (x_curr, y_curr, q1, q2, q3, _, _) in positions.items():
            # Along Q1 dimension (horizontal): connect if q1=0
            if q1 == 0:
                i_next = (1 << 2) | (q2 << 1) | q3
                x_next, y_next = positions[i_next][0], positions[i_next][1]
                p1 = circ_point((x_curr, y_curr), (x_next, y_next), circle_r)
                p2 = circ_point((x_next, y_next), (x_curr, y_curr), circle_r)
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=1.0, alpha=0.8)

            # Along Q2 dimension (vertical): connect if q2=0
            if q2 == 0:
                i_next = (q1 << 2) | (1 << 1) | q3
                x_next, y_next = positions[i_next][0], positions[i_next][1]
                p1 = circ_point((x_curr, y_curr), (x_next, y_next), circle_r)
                p2 = circ_point((x_next, y_next), (x_curr, y_curr), circle_r)
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=1.0, alpha=0.8)

            # Along Q3 dimension (front to back): connect if q3=0
            if q3 == 0:
                i_next = (q1 << 2) | (q2 << 1) | 1
                x_next, y_next = positions[i_next][0], positions[i_next][1]
                p1 = circ_point((x_curr, y_curr), (x_next, y_next), circle_r)
                p2 = circ_point((x_next, y_next), (x_curr, y_curr), circle_r)
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=1.0, alpha=0.8)

        # Draw axis labels (matching TikZ exactly)
        # TikZ: Q1 from (-3.0, 4.0) to (-0.5, 4.0)
        q1_end_x = -0.5 * scale
        q1_end_y = origin_y
        ax.annotate('', xy=(q1_end_x, q1_end_y), xytext=(origin_x, origin_y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.text((origin_x + q1_end_x) / 2, origin_y + 0.15,
               '$Q_1$', ha='center', va='bottom', fontsize=10)

        # Q2 axis (vertical, down) from origin
        # TikZ: from (-3.0, 4.0) to (-3.0, 1.5)
        q2_end_x = origin_x
        q2_end_y = (ay_dim/scale - 1.5) * scale  # = (3.0 - 1.5) * scale = 1.5 * scale
        ax.annotate('', xy=(q2_end_x, q2_end_y), xytext=(origin_x, origin_y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.text(origin_x - 0.3, (origin_y + q2_end_y) / 2,
               '$Q_2$', ha='right', va='center', fontsize=10)

        # Q3 axis (diagonal up-right from origin)
        # TikZ: from (-3.0, 4.0) to (-0.75, 5.5)  [(\ax/2 - 3.5, \sy+\ay+0.5)]
        q3_end_x = (ax_dim/scale/2 - 3.5) * scale  # = (2.75 - 3.5) * scale = -0.75 * scale
        q3_end_y = (sy/scale + ay_dim/scale + 0.5) * scale  # = (2.0 + 3.0 + 0.5) * scale = 5.5 * scale
        ax.annotate('', xy=(q3_end_x, q3_end_y), xytext=(origin_x, origin_y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.text((origin_x + q3_end_x) / 2 - 0.1, (origin_y + q3_end_y) / 2 + 0.1,
               '$Q_3$', ha='center', va='bottom', fontsize=10)

        # Set axis limits to show full TikZ figure with margins
        # Account for: circle radii, labels (extends ~0.2 beyond circles), axes
        margin = 0.5 * scale  # Generous margin for labels and axes
        # Left: origin_x for Q2 axis, minus label space
        left = min(origin_x, origin_x - 0.3) - margin
        # Rightmost: back face Q1=1 at (ax_dim+sx), plus circle radius
        right = ax_dim + sx + circle_r + margin
        # Bottom: front face at y=0, minus circle radius and label space
        bottom = -circle_r - 0.2 - margin
        # Top: back face circles at y=sy+ay_dim, plus circle radius and label
        top = sy + ay_dim + circle_r + 0.2 + margin
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_aspect('equal')
        ax.axis('off')
    else:
        margin = 0.3
        # Calculate axis limits based on actual circle positions
        if n_qubits == 1:
            # 1D grid: x = 0.6 + i*2.2 for i in [0,1], y = 1.2
            x_min = 0.6 - circle_r - margin
            x_max = 0.6 + 1*2.2 + circle_r + margin
            y_min = 1.2 - circle_r - margin
            y_max = 1.2 + circle_r + margin
        else:
            # 2D grid: x = 1.0 + x*2.2, y = 0.5 + y*2.6
            # Increase margin to 0.6 to accommodate labels at x_min-0.5 and y_max+0.5
            margin = 0.6
            x_min = 1.0 - circle_r - margin
            x_max = 1.0 + 1*2.2 + circle_r + margin
            y_min = 0.5 - circle_r - margin
            y_max = 0.5 + 1*2.6 + circle_r + margin
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])

        if n_qubits > 1:
            ax.plot([x_min, x_min], [y_min, y_max], 'k-', linewidth=1.5)
            ax.annotate('', xy=(x_min, y_max), xytext=(x_min, y_max - 0.2),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
            ax.text(x_min - 0.5, (y_min + y_max) / 2, '2nd Qubit',
                   ha='center', va='center', fontsize=9, rotation=90)

            ax.plot([x_min, x_max], [y_max, y_max], 'k-', linewidth=1.5)
            ax.annotate('', xy=(x_max, y_max), xytext=(x_max - 0.2, y_max),
                      arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
            ax.text((x_min + x_max) / 2, y_max + 0.5, '1st Qubit',
                   ha='center', va='bottom', fontsize=9)

        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Respect axis limits set above (bbox_inches='tight' would ignore them)
    if output_path:
        plt.savefig(output_path, dpi=dpi)
        plt.close()
        return None

    return fig


def plot_dcns(
    states: List[List[complex]],
    titles: Optional[List[str]] = None,
    output_dir: str = "./",
    base_filename: str = "dcn",
    dpi: int = 150,
    fmt: str = "png"
) -> List[str]:
    """
    Plot multiple DCN diagrams from an array of states.

    Args:
        states: List of state vectors (each a list of complex amplitudes)
        titles: Optional list of titles for each plot
        output_dir: Directory to save output files
        base_filename: Base name for output files
        dpi: Resolution for saved figures
        fmt: Output format (png, pdf, svg)

    Returns:
        List of output filenames
    """
    if not states:
        raise ValueError("States list cannot be empty")

    if titles is None:
        titles = [f"State {i+1}" for i in range(len(states))]

    if len(titles) < len(states):
        raise ValueError("Not enough titles for all states")

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_files = []

    for i, state in enumerate(states):
        if output_dir:
            filename = f"{output_dir}/{base_filename}_{i+1:02d}_{titles[i].replace(' ', '_')}.{fmt}"
        else:
            filename = f"{base_filename}_{i+1:02d}_{titles[i].replace(' ', '_')}.{fmt}"

        plot_dcn(state, titles[i], filename, dpi)
        output_files.append(filename)
        print(f"Saved: {filename}")

    return output_files


def plot_dcns_from_file(
    input_file: str,
    output_dir: Optional[str] = None,
    dpi: int = 150,
    fmt: str = "png"
) -> List[str]:
    """
    Plot multiple DCN diagrams from a JSON file.

    Supports three formats:
    1. {"states": [[...], [...], ...]} - array of state vectors
    2. {"state_vector": [...]} - single state vector
    3. {"stages": [{"name": "...", "state_vector": [...]}, ...]} - named stages

    Args:
        input_file: Path to JSON file
        output_dir: Directory to save output files
        dpi: Resolution for saved figures
        fmt: Output format (png, pdf, svg)

    Returns:
        List of output filenames
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Format 1: Array of states
    if 'states' in data:
        states = data['states']
        if not isinstance(states, list):
            raise ValueError("'states' must be a list of state vectors")

        parsed_states = []
        for state in states:
            parsed_state = [parse_amplitude(amp) for amp in state]
            parsed_states.append(parsed_state)

        base_name = os.path.splitext(os.path.basename(input_file))[0]
        return plot_dcns(parsed_states, None, output_dir or './', base_name, dpi, fmt)

    # Format 2: Single state vector
    elif 'state_vector' in data:
        state_vector = [parse_amplitude(amp) for amp in data['state_vector']]
        name = data.get('name', 'DCN')

        if output_dir:
            filename = f"{output_dir}/{name.replace(' ', '_')}.{fmt}"
        else:
            filename = f"{name.replace(' ', '_')}.{fmt}"

        plot_dcn(state_vector, name, filename, dpi)
        return [filename]

    # Format 3: Stages (existing format)
    elif 'stages' in data:
        stages = data['stages']

        if output_dir and not output_dir.endswith('/'):
            output_dir += '/'
        if output_dir is None:
            output_dir = './'

        output_files = []

        for idx, stage in enumerate(stages):
            name = stage.get('name', f'Stage {idx+1}')

            state_vector = []
            for amp in stage['state_vector']:
                state_vector.append(parse_amplitude(amp))

            safe_name = name.replace(' ', '_').replace('/', '_')
            stage_num = f'stage_{idx+1:02d}'
            output_path = f'{output_dir}{stage_num}_{safe_name}.{fmt}'

            plot_dcn(state_vector, name, output_path, dpi)
            output_files.append(output_path)
            print(f'Saved: {output_path}')

        return output_files

    else:
        raise ValueError("Input file must contain 'states', 'state_vector', or 'stages' key")
