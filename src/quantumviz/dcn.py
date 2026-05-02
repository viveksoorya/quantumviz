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

    fig.suptitle(title, fontsize=12, y=0.95)

    # Circle radius (outer circle)
    circle_r = 1.0

    # Store positions for drawing lines later (3-qubit case)
    positions = {}

    # TikZ scale factor (consistent across position and axis calculations)
    scale = 0.72

    # Draw each basis state's marker
    for i, amp in enumerate(state_vector):
        magnitude = abs(amp)
        phase = np.angle(amp) if magnitude >= 0.01 else 0

        if n_qubits == 3:
            # Extract bits using big-endian: |Q1 Q2 Q3> where Q1 is MSB
            # i = Q1*4 + Q2*2 + Q3 (big-endian)
            q1 = (i >> 2) & 1  # MSB (horizontal, x-axis)
            q2 = (i >> 1) & 1  # Middle (vertical, y-axis)
            q3 = i & 1          # LSB (front/back face)

            # TikZ parameters (scaled for matplotlib)
            ax_dim = 5.5 * scale   # \ax: Q1 horizontal spacing
            ay_dim = 3.0 * scale   # \ay: Q2 vertical spacing
            sx = 2.5 * scale      # \sx: back face horizontal shift
            sy = 2.0 * scale      # \sy: back face vertical shift
            circle_r = 1.0 * scale  # Circle radius

            # Position calculation (matching TikZ exactly)
            # TikZ: Front face (Q3=0) at (Q1*5.5, Q2*3.0), Back face (Q3=1) offset by (2.5, 2.0)
            x_pos = q1 * ax_dim + (sx if q3 else 0)
            y_pos = q2 * ay_dim + (sy if q3 else 0)

            positions[i] = (x_pos, y_pos, q1, q2, q3)

            if magnitude < 0.01:
                continue

            # Determine if front or back face
            is_back_face = (q3 == 1)

            # Outer circle - use matplotlib.patches.Circle
            outer_circle = MplCircle((x_pos, y_pos), circle_r, fill=False,
                                     color='black', linewidth=1.5,
                                     linestyle='--' if is_back_face else '-',
                                     alpha=0.6 if is_back_face else 1.0)
            ax.add_patch(outer_circle)

            # Inner circle (filled) - radius proportional to magnitude
            inner_r = circle_r * magnitude
            if inner_r > 0.01:
                # Color based on Q3 (front/back face): blue for Q3=0, green for Q3=1 (matches TikZ)
                inner_color = 'skyblue' if q3 == 0 else 'lightgreen'
                inner_circle = MplCircle((x_pos, y_pos), inner_r, fill=True,
                                         color=inner_color, linewidth=0, alpha=0.7)
                ax.add_patch(inner_circle)

            # Phase line (direction = phase angle)
            if magnitude >= 0.01:
                endpoint_x = x_pos + circle_r * np.cos(phase)
                endpoint_y = y_pos + circle_r * np.sin(phase)
                ax.plot([x_pos, endpoint_x], [y_pos, endpoint_y],
                       color='black', linewidth=2.5,
                       alpha=0.6 if is_back_face else 1.0)

            # Center dot
            ax.plot(x_pos, y_pos, 'ko', markersize=3,
                   alpha=0.6 if is_back_face else 1.0)

            # Basis state label (below circle)
            basis_label = format(i, f'0{n_qubits}b')
            ax.text(x_pos, y_pos - circle_r - 0.1, f'|{basis_label}⟩',
                   ha='center', va='top', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        else:
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
        # TikZ origin: (-1.0, \ay+1.0) = (-1.0, 4.0)
        origin_x = -1.0 * scale
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
        separable_q1 = is_separable_along_qubit(state_vector, 3, qubit_idx=2)  # Q1

        # Draw separability plane for Q1 (green parallelogram midway between Q1=0 and Q1=1)
        if separable_q1:
            plane_x_mid = ax_dim / 2  # Midway between Q1=0 (x=0) and Q1=1 (x=ax_dim)
            plane_pts = [
                (plane_x_mid, -0.3),
                (plane_x_mid, ay_dim + 0.3),
                (plane_x_mid + sx, ay_dim + sy + 0.3),
                (plane_x_mid + sx, sy - 0.3)
            ]
            plane_pts.append(plane_pts[0])  # Close polygon
            px_vals = [p[0] for p in plane_pts]
            py_vals = [p[1] for p in plane_pts]
            ax.fill(px_vals, py_vals, 'green', alpha=0.2, edgecolor='green', linewidth=2)
            ax.text(plane_x_mid + sx/2, (ay_dim + sy)/2, 'separability',
                   ha='center', va='center', fontsize=8, color='green', fontweight='bold')

        # Draw connecting lines between adjacent basis states (circumference to circumference)
        for i_curr, (x_curr, y_curr, q1, q2, q3) in positions.items():
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
        # TikZ coordinates (scaled): origin (-1.0, 4.0), Q1 ends at (6.0, 4.0)
        q1_end_x = 6.0 * scale
        q1_end_y = origin_y
        ax.annotate('', xy=(q1_end_x, q1_end_y), xytext=(origin_x, origin_y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.text((origin_x + q1_end_x) / 2, origin_y + 0.15,
               '$Q_1$', ha='center', va='bottom', fontsize=10)

        # Q2 axis (vertical, top to bottom) from origin
        # TikZ: from (-1.0, 4.0) to (-1.0, -0.5)
        q2_end_x = origin_x
        q2_end_y = -0.5 * scale
        ax.annotate('', xy=(q2_end_x, q2_end_y), xytext=(origin_x, origin_y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.text(origin_x - 0.3, (origin_y + q2_end_y) / 2,
               '$Q_2$', ha='right', va='center', fontsize=10)

        # Q3 axis (diagonal UP and RIGHT from origin - "behind" the plane)
        # TikZ: (-1.0, 4.0) -> (2.75, 6.0)
        q3_end_x = 2.75 * scale
        q3_end_y = 6.0 * scale
        ax.annotate('', xy=(q3_end_x, q3_end_y), xytext=(origin_x, origin_y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.text((origin_x + q3_end_x) / 2 - 0.1, (origin_y + q3_end_y) / 2 + 0.1,
               '$Q_3$', ha='center', va='bottom', fontsize=10)

        # Set axis limits to show full TikZ figure with margins
        # TikZ bounds: x from -1.0 to ax+0.5+sx = 8.5, y from -0.5 to sy+ay+1.0 = 6.0
        left = -1.0 * scale - 0.3 * scale
        right = (5.5 + 0.5 + 2.5) * scale + 0.3 * scale  # = 8.5 * scale + margin
        bottom = -0.5 * scale - 0.3 * scale
        top = (6.0) * scale + 0.3 * scale  # sy+ay+1.0 = 6.0
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_aspect('equal')
        ax.axis('off')
    else:
        dim = 2 ** n_qubits
        margin = 0.3
        ax.set_xlim(0.5 - circle_r - margin, dim - 0.5 + circle_r + margin)
        ax.set_ylim(0.5 - circle_r - margin, dim - 0.5 + circle_r + margin)
        ax.set_xticks([])
        ax.set_yticks([])

        if n_qubits > 1:
            y_min = 0.5 - circle_r
            y_max = dim - 0.5 + circle_r
            x_min = 0.5 - circle_r
            x_max = dim - 0.5 + circle_r

            ax.plot([y_min, y_min], [y_min, y_max], 'k-', linewidth=1.5)
            ax.annotate('', xy=(y_min, y_max), xytext=(y_min, y_max - 0.2),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
            ax.text(y_min - 0.15, (y_min + y_max) / 2, '2nd Qubit',
                   ha='center', va='center', fontsize=9, rotation=90)

            ax.plot([x_min, x_max], [y_max, y_max], 'k-', linewidth=1.5)
            ax.annotate('', xy=(x_max, y_max), xytext=(x_max - 0.2, y_max),
                      arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
            ax.text((x_min + x_max) / 2, y_max + 0.15, '1st Qubit',
                   ha='center', va='bottom', fontsize=9)

        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # bbox_inches='tight' in savefig handles the layout

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return None

    return fig


def plot_dcns_from_file(
    input_file: str,
    output_dir: Optional[str] = None,
    dpi: int = 150,
    fmt: str = "png"
) -> List[str]:
    """Plot multiple DCN stages from JSON file."""
    with open(input_file, 'r') as f:
        data = json.load(f)

    stages = data['stages']

    if output_dir and not output_dir.endswith('/'):
        output_dir += '/'
    if output_dir is None:
        output_dir = './'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
