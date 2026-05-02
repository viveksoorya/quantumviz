"""
Circuit Diagram Visualization Module

Provides functions for rendering quantum circuit diagrams
as PNG images.
"""

import json

import matplotlib

matplotlib.use('Agg')
from typing import Any, Dict, List, Optional, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt

GATE_COLORS = {
    'H': '#FF6B6B',
    'X': '#4ECDC4',
    'Y': '#45B7D1',
    'Z': '#96CEB4',
    'CNOT': '#FFEAA7',
    'CZ': '#DDA0DD',
    'T': '#98D8C8',
    'S': '#F7DC6F',
    'RX': '#BB8FCE',
    'RY': '#85C1E9',
    'RZ': '#F8B500',
    'U': '#AED6F1',
    'measure': '#E74C3C',
    'default': '#D5D8DC'
}


def draw_qubit_line(ax, y: float, num_wires: int, color: str = 'black') -> None:
    """Draw a horizontal qubit wire."""
    ax.plot([0, num_wires * 2], [y, y], color=color, linewidth=1.5)


def draw_single_gate(ax, gate: str, x: float, y: float, label: Optional[str] = None) -> None:
    """Draw a single-qubit gate."""
    color = GATE_COLORS.get(gate, GATE_COLORS['default'])
    rect = patches.Rectangle((x - 0.3, y - 0.3), 0.6, 0.6,
                             linewidth=1.5, edgecolor='black',
                             facecolor=color, zorder=3)
    ax.add_patch(rect)
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=8, zorder=4)
    else:
        ax.text(x, y, gate, ha='center', va='center', fontsize=8, fontweight='bold', zorder=4)


def draw_cnot(ax, ctrl_x: float, ctrl_y: float, target_x: float, target_y: float) -> None:
    """Draw a CNOT gate."""
    ax.plot([ctrl_x, ctrl_x], [ctrl_y, ctrl_y - 0.5], color='black', linewidth=1.5)
    ax.plot([ctrl_x, target_x], [ctrl_y - 0.5, ctrl_y - 0.5], color='black', linewidth=1.5)
    ax.plot([target_x, target_x], [ctrl_y - 0.5, target_y], color='black', linewidth=1.5)
    ax.scatter([ctrl_x], [ctrl_y], color='black', s=80, zorder=5, marker='o')
    rect = patches.Rectangle((target_x - 0.3, target_y - 0.3), 0.6, 0.6,
                             linewidth=1.5, edgecolor='black',
                             facecolor=GATE_COLORS['CNOT'], zorder=3)
    ax.add_patch(rect)
    ax.text(target_x, target_y, 'X', ha='center', va='center', fontsize=8, fontweight='bold', zorder=4)


def draw_controlled_gate(ax, ctrl_x: float, ctrl_y: float, target_x: float, target_y: float, gate: str = 'Z') -> None:
    """Draw a controlled gate."""
    ax.plot([ctrl_x, ctrl_x], [ctrl_y, ctrl_y - 0.5], color='black', linewidth=1.5)
    ax.plot([ctrl_x, target_x], [ctrl_y - 0.5, ctrl_y - 0.5], color='black', linewidth=1.5)
    ax.plot([target_x, target_x], [ctrl_y - 0.5, target_y], color='black', linewidth=1.5)
    ax.scatter([ctrl_x], [ctrl_y], color='black', s=80, zorder=5, marker='o')
    rect = patches.Rectangle((target_x - 0.3, target_y - 0.3), 0.6, 0.6,
                             linewidth=1.5, edgecolor='black',
                             facecolor=GATE_COLORS.get(gate, GATE_COLORS['default']), zorder=3)
    ax.add_patch(rect)
    ax.text(target_x, target_y, gate, ha='center', va='center', fontsize=8, fontweight='bold', zorder=4)


def draw_measure(ax, x: float, y: float) -> None:
    """Draw a measurement gate."""
    rect = patches.Rectangle((x - 0.3, y - 0.3), 0.6, 0.6,
                             linewidth=1.5, edgecolor='black',
                             facecolor=GATE_COLORS['measure'], zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, 'M', ha='center', va='center', fontsize=8, fontweight='bold', zorder=4)
    ax.plot([x + 0.3, x + 0.7], [y, y + 0.3], color='black', linewidth=1)
    ax.plot([x + 0.7, x + 0.7], [y + 0.3, y + 0.4], color='black', linewidth=1)


def parse_circuit(data: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """
    Parse circuit data into layers for rendering.

    Args:
        data: Circuit dictionary with 'qubits' and 'gates'

    Returns:
        List of layers, each containing a list of gates
    """
    layers = []
    current_layer = []
    for gate in data['gates']:
        if gate['type'] == 'CNOT':
            if current_layer:
                layers.append(current_layer)
                current_layer = []
            layers.append([gate])
        else:
            current_layer.append(gate)
    if current_layer:
        layers.append(current_layer)
    return layers


def plot_circuit(data: Union[Dict[str, Any], Any], output_path: Optional[str] = None, dpi: int = 150) -> plt.Figure:
    """
    Draw a quantum circuit diagram.

    Args:
        data: Circuit dictionary with 'qubits', 'gates', and optional 'name',
              or Qiskit QuantumCircuit object
        output_path: Path to save the figure (if None, returns figure object)
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object if output_path is None, else None
    """
    # Convert Qiskit QuantumCircuit to dict format
    from quantumviz.qiskit_bridge import circuit_to_dict, is_quantum_circuit
    if is_quantum_circuit(data):
        data = circuit_to_dict(data)

    qubits = data['qubits']
    layers = parse_circuit(data)
    n_layers = len(layers)
    n_gates = sum(len(layer) for layer in layers)

    fig, ax = plt.subplots(figsize=(max(10, n_gates * 1.2), max(4, qubits * 1.5)))

    for i in range(qubits):
        draw_qubit_line(ax, i * 2, n_layers)
        ax.text(-0.5, i * 2, f'|{i}⟩', ha='right', va='center', fontsize=10)

    layer_idx = 0
    for layer in layers:
        x_pos = layer_idx * 2 + 1

        if len(layer) == 1 and layer[0]['type'] == 'CNOT':
            gate = layer[0]
            ctrl_q = gate['control']
            target_q = gate['target']
            draw_cnot(ax, x_pos, ctrl_q * 2, x_pos, target_q * 2)
        else:
            for gate in layer:
                q = gate['qubit']
                gate_type = gate['type']
                if gate_type == 'H':
                    draw_single_gate(ax, 'H', x_pos, q * 2)
                elif gate_type == 'X':
                    draw_single_gate(ax, 'X', x_pos, q * 2)
                elif gate_type == 'measure':
                    draw_measure(ax, x_pos, q * 2)
                elif gate_type in ['RX', 'RY', 'RZ']:
                    theta = gate.get('theta', 0)
                    label = f"{gate_type}\nθ={theta:.2f}"
                    draw_single_gate(ax, gate_type, x_pos, q * 2, label)
                elif gate_type in ['P', 'U']:
                    phi = gate.get('phi', 0)
                    draw_single_gate(ax, 'U', x_pos, q * 2, f"P\nφ={phi:.2f}")
                elif gate_type == 'R':
                    phi = gate.get('phi', 0)
                    draw_single_gate(ax, 'R', x_pos, q * 2, f"R\nφ={phi:.2f}")
                else:
                    draw_single_gate(ax, gate_type, x_pos, q * 2)

        layer_idx += 1

    ax.set_xlim(-1, n_gates * 2 + 1)
    ax.set_ylim(-1, (qubits - 1) * 2 + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(data.get('name', 'Quantum Circuit'), fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        return fig


def main(args: Optional[List[str]] = None) -> None:
    """
    CLI entry point for Circuit Diagram visualization.

    Args:
        args: Command line arguments (if None, uses sys.argv)
    """
    import sys

    if args is None:
        args = sys.argv[1:]

    if len(args) != 1:
        print("Usage: python -m quantumviz.circuit_diagram <circuit.json>")
        sys.exit(1)

    input_file = args[0]

    with open(input_file, 'r') as f:
        data = json.load(f)

    base_name = input_file.split('/')[-1].replace('.json', '')
    output_file = f"{base_name}_circuit.png"
    plot_circuit(data, output_file)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
