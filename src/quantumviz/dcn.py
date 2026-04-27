"""
Dimensional Circular Notation (DCN) Visualization Module

A simple visualization where:
- Each basis state with non-zero amplitude gets a circle
- Circle position in 2D grid: x = 1st qubit, y = 2nd qubit
- Radius line direction shows phase (UP=0°, DOWN=180°, etc.)
- Radius line length shows magnitude
- Circle labeled with basis state
"""

import matplotlib
import numpy as np

matplotlib.use('Agg')
import json
from typing import Any, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle
from matplotlib.lines import Line2D


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


def plot_dcn(
    state_vector: List[complex],
    title: str = "DCN Visualization",
    output_path: Optional[str] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Create a Dimensional Circular Notation (DCN) plot.
    
    For n-qubit system: circles arranged in 2^n grid.
    Each circle shows one basis state's amplitude:
    - Position: (1st qubit on x-axis, 2nd qubit on y-axis)
    - Radius line: direction = phase, length = magnitude
    - Circle labeled with basis state
    
    Args:
        state_vector: List of complex amplitudes
        title: Title for the plot
        output_path: Path to save figure
        dpi: Resolution
    
    Returns:
        matplotlib Figure if output_path is None
    """
    n = len(state_vector)
    if n == 0:
        raise ValueError("State vector cannot be empty")
    
    n_qubits = int(np.log2(n))
    if 2 ** n_qubits != n:
        raise ValueError(f"State vector length {n} is not power of 2")
    
    # Number of rows/cols in grid
    dim = 2 ** n_qubits  # 2, 4, 8, 16 for 1, 2, 3, 4 qubits
    
    # Figure size - compact
    fig_size = 4
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    fig.suptitle(title, fontsize=12, y=0.95)
    
    # Circle radius - larger for multi-qubit systems
    if n_qubits >= 2:
        circle_r = 0.70
    else:
        circle_r = 0.30
    
    # Draw each basis state's circle
    for i, amp in enumerate(state_vector):
        magnitude = abs(amp)
        if magnitude < 0.01:
            continue
        
        # Phase angle: UP = 0, RIGHT = 90, DOWN = 180, LEFT = 270
        phase = np.angle(amp)  # Returns -π to π
        phase_deg = np.degrees(phase)  # Convert to degrees
        
        # Position in grid: x = 1st qubit, y = 2nd qubit
        # For 2 qubits: basis states |00⟩, |01⟩, |10⟩, |11⟩
        # |00⟩ (i=0): x=0, y=0 → bottom-left (origin)
        # |01⟩ (i=1): x=0, y=1 → top-left  
        # |10⟩ (i=2): x=1, y=0 → bottom-right
        # |11⟩ (i=3): x=1, y=1 → top-right
        x = (i >> (n_qubits - 1)) & 1  # First qubit (MSB)
        y = i & 1  # Second qubit (LSB)
        
        # For more qubits, use bit extraction
        for q in range(n_qubits - 2, -1, -1):
            if q == 0:
                x = (i >> 1) & 1
                y = i & 1
            else:
                bit = (i >> q) & 1
                if q == n_qubits - 1:
                    x = bit
                elif q == n_qubits - 2:
                    y = bit
        
        # Actually: for n qubits, read bits from MSB to LSB
        # i in binary: bit_{n-1}...bit_1 bit_0
        # x = bit_{n-1} (1st qubit), y = bit_{n-2} (2nd qubit)
        # For 2 qubits: i = b1*2 + b0, x=b1, y=b0
        if n_qubits == 1:
            x = i  # Display horizontally: |0> at x=0, |1> at x=1
            y = 0  # Same row
        elif n_qubits == 2:
            x = (i >> 1) & 1  # MSB
            y = i & 1        # LSB
        else:
            # More qubits: use first two bits
            x = (i >> (n_qubits - 1)) & 1
            y = (i >> (n_qubits - 2)) & 1
        
        # Convert to plot coordinates
        # For 1 qubit: display horizontally |0> |1> from left to right
        # For 2 qubits: grid layout with x=1st qubit, y=2nd qubit
        if n_qubits == 1:
            px = 0.6 + x * 0.8  # 0.4 for |0>, 1.2 for |1>
            py = 1.2
        else:
            px = 1.0 + x * 2.2
            py = 0.5 + y * 2.2
        
        # Draw circle
        circle = MplCircle(
            (px, py), circle_r,
            fill=False, color='black', linewidth=1.5
        )
        ax.add_patch(circle)
        
        # Draw basis state label below circle
        basis_label = format(i, f'0{n_qubits}b')[-(n_qubits):]
        ax.text(px, py - circle_r - 0.12, f'|{basis_label}⟩', ha='center', va='top',
               fontsize=9, fontweight='bold')
        
        # Draw radius line (from center to circle edge, direction = phase)
        # UP = 0° (phase=0), RIGHT = 90°, DOWN = 180°, LEFT = 270°
        # Extend from center (px, py) to circle edge
        endpoint_x = px + circle_r * np.sin(phase)
        endpoint_y = py + circle_r * np.cos(phase)
        
        # Color: black for radius line
        line_color = 'black'
        
        ax.plot([px, endpoint_x], [py, endpoint_y],
               color=line_color, linewidth=3)
        
        # Draw center dot
        ax.plot(px, py, 'ko', markersize=4)
    
    # Set axis limits and remove ticks
    # Include all circles with margin
    margin = 0.3
    ax.set_xlim(0.5 - circle_r - margin, dim - 0.5 + circle_r + margin)
    ax.set_ylim(0.5 - circle_r - margin, dim - 0.5 + circle_r + margin)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Only draw axes if more than 1 qubit
    if n_qubits > 1:
        # Calculate axis positions
        y_min = 0.5 - circle_r
        y_max = dim - 0.5 + circle_r
        x_min = 0.5 - circle_r
        x_max = dim - 0.5 + circle_r
        
        # Y-axis at left with arrow
        ax.plot([y_min, y_min], [y_min, y_max], 'k-', linewidth=1.5)
        ax.annotate('', xy=(y_min, y_max), xytext=(y_min, y_max - 0.2),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.text(y_min - 0.15, (y_min + y_max) / 2, '2nd Qubit', ha='center', va='center', fontsize=9, rotation=90)
        
        # X-axis at top with arrow
        ax.plot([x_min, x_max], [y_max, y_max], 'k-', linewidth=1.5)
        ax.annotate('', xy=(x_max, y_max), xytext=(x_max - 0.2, y_max),
                  arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.text((x_min + x_max) / 2, y_max + 0.15, '1st Qubit', ha='center', va='bottom', fontsize=9)
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi)
        plt.close()
    else:
        return fig


def plot_dcns_from_file(
    input_file: str,
    output_dir: Optional[str] = None,
    dpi: int = 150
) -> List[str]:
    """Plot multiple DCN stages from JSON file."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    n_qubits = data['qubits']
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
        output_path = f'{output_dir}{stage_num}_{safe_name}.png'
        
        plot_dcn(state_vector, name, output_path, dpi)
        output_files.append(output_path)
        
        print(f'Saved: {output_path}')
    
    return output_files
