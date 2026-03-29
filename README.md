# quantumviz - Quantum Algorithm Visualization Library

A Python library for visualizing quantum algorithm states including Bloch spheres, density matrices, cost landscapes, circuit diagrams, and dynamic time evolution.

## Features

- **Bloch Sphere Visualization** - Visualize single-qubit states on the Bloch sphere
- **State City** - 3D bar chart visualization of density matrices for multi-qubit systems
- **Cost Landscape** - Plot optimization landscapes for QAOA and VQE algorithms
- **Circuit Diagrams** - Render quantum circuit diagrams as PNG images
- **Dynamic Flow** - Visualize time evolution and Rabi oscillations
- **Interactive Dashboard** - Web-based interface for quantum state visualization
- **Hardware Integration** - Run circuits on IBM Quantum hardware (requires Qiskit)

## Installation

### Core Package

```bash
pip install quantumviz
```

### With Dashboard

```bash
pip install quantumviz[dashboard]
```

### With Quantum Hardware Support

```bash
pip install quantumviz[all]
```

## Quick Start

### Command Line Interface

```bash
# Plot Bloch sphere
quantumviz bloch-sphere input.txt -o output.png

# Plot State City
quantumviz state-city input.json -o output/

# Plot circuit diagram
quantumviz circuit circuit.json -o circuit.png

# Plot QAOA cost landscape (requires input file)
quantumviz cost-landscape qaoa graph.json -o landscape.png

# Plot VQE energy landscape (requires input file)
quantumviz cost-landscape vqe hamiltonian.json -o energy.png

# Start dashboard
quantumviz serve
```

### Python API

```python
from quantumviz import plot_bloch_sphere, plot_state_city, state_to_density

# Bloch sphere
plot_bloch_sphere("input.txt", "output.png")

# State city
plot_state_city([1, 0, 0, 0], "Density Matrix", "output.png")

# Density matrix
rho = state_to_density([1/np.sqrt(2), 1/np.sqrt(2)])
```

## Input Formats

### Bloch Sphere (TXT)

```
|0>                    # Ket notation
theta=60 deg, phi=45   # Bloch angles
(x,y,z)               # Cartesian coordinates
```

### State City / Circuit / Dynamic Flow (JSON)

```json
{
  "qubits": 2,
  "stages": [
    {"name": "Initial", "state_vector": [1, 0, 0, 0]}
  ]
}
```

### QAOA Cost Landscape (JSON)

Defines a MaxCut problem via graph edges.

```json
{
  "edges": [[0, 1], [1, 2], [0, 2]]
}
```

Each edge `[u, v]` represents a connection between qubits `u` and `v` (0-indexed).

### VQE Energy Landscape (JSON)

Defines a molecular Hamiltonian as a sum of Pauli terms.

```json
{
  "terms": [
    {"coeff": -1.0, "paulis": []},
    {"coeff": 0.5, "paulis": ["Z"]},
    {"coeff": 0.25, "paulis": ["Z", "Z"]}
  ]
}
```

Each term has:
- `coeff`: Real number coefficient
- `paulis`: List of Pauli operators (`I`, `X`, `Y`, `Z`) - one per qubit

### Example Files

See the `examples/` directory for complete input file examples:
- `examples/qaoa_maxcut_2qubit.json` - Simple 2-qubit graph
- `examples/qaoa_maxcut_triangle.json` - Triangle graph
- `examples/vqe_h2.json` - H2 molecule Hamiltonian
- `examples/vqe_lih.json` - LiH molecule Hamiltonian

## Development

### Install for Development

```bash
git clone https://github.com/yourusername/quantumviz.git
cd quantumviz
pip install -e ".[dev,all]"
```

### Run Tests

```bash
pytest
pytest --cov=quantumviz --cov-report=html
```

### Code Quality

```bash
ruff check .
mypy .
```

## Requirements

- Python 3.9+
- numpy >= 1.20
- matplotlib >= 3.5

### Optional Dependencies

- fastapi, uvicorn, pydantic (dashboard)
- qiskit, qiskit-ibm-runtime (hardware)

## License

MIT License - see LICENSE file for details.

## Credits

Developed for quantum algorithm visualization and education.
