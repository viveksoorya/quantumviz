# quantumviz - Quantum Algorithm Visualization Library

A Python library for visualizing quantum algorithm states including Bloch spheres, density matrices, cost landscapes, circuit diagrams, dynamic time evolution, and dimensional circular notation (DCN).

## Features

- **Bloch Sphere Visualization** - Visualize single-qubit states on the Bloch sphere
- **State City** - 3D bar chart visualization of density matrices for multi-qubit systems
- **Cost Landscape** - Plot optimization landscapes for QAOA and VQE algorithms
- **Circuit Diagrams** - Render quantum circuit diagrams as PNG images
- **Dynamic Flow** - Visualize time evolution and Rabi oscillations
- **Dimensional Circular Notation (DCN)** - Visualize quantum states with circular notation showing magnitude and phase
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
# Plot Bloch sphere (PNG default)
quantumviz bloch-sphere input.txt -o output.png

# Plot Bloch sphere (PDF output)
quantumviz bloch-sphere input.txt -o output.pdf -f pdf

# Plot State City (multiple PNG files)
quantumviz state-city input.json -o output/

# Plot State City (multiple PDF files)
quantumviz state-city input.json -o output/ -f pdf

# Plot circuit diagram
quantumviz circuit circuit.json -o circuit.png

# Plot circuit diagram (PDF)
quantumviz circuit circuit.json -o circuit.pdf -f pdf

# Plot QAOA cost landscape
quantumviz cost-landscape qaoa graph.json -o landscape.png

# Plot VQE energy landscape
quantumviz cost-landscape vqe hamiltonian.json -o energy.png

# Plot DCN visualization (PNG)
quantumviz dcn input.json -o output.png

# Plot DCN visualization (PDF)
quantumviz dcn input.json -o output.pdf -f pdf

# Start dashboard
quantumviz serve
```

### Output Format Options
- Use `-f` or `--format` to specify output format: `png` (default), `pdf`, or `svg`
- Output path extension is ignored in favor of the `--format` option

### Python API

```python
from quantumviz import plot_bloch_sphere, plot_state_city, state_to_density
from quantumviz.dcn import plot_dcn

# Bloch sphere (PNG)
plot_bloch_sphere("input.txt", "output.png")

# Bloch sphere (PDF)
plot_bloch_sphere("input.txt", "output.pdf")

# State city (PNG)
plot_state_city([1, 0, 0, 0], "Density Matrix", "output.png")

# State city (PDF)
plot_state_city([1, 0, 0, 0], "Density Matrix", "output.pdf")

# DCN visualization
plot_dcn("input.json", "output.png")

# Density matrix
rho = state_to_density([1/np.sqrt(2), 1/np.sqrt(2)])
```

### Using with Qiskit

quantumviz accepts Qiskit objects directly, complementing Qiskit's built-in visualizations:

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from quantumviz import plot_bloch_sphere, plot_dcn, plot_state_city, plot_circuit

# Create circuit with Qiskit
qc = QuantumCircuit(2)
qc.h(0); qc.cx(0, 1)

# quantumviz's unique visualizations (not in Qiskit)
state = Statevector(qc)
plot_dcn(state, "bell_dcn.png")           # Dimensional Circular Notation
plot_bloch_sphere([state], "bloch.png")   # Accepts Statevector directly
plot_state_city(state, "state_city.png")     # Accepts Statevector directly

# Pass QuantumCircuit directly to circuit visualization
plot_circuit(qc, "circuit.png")           # Converts to quantumviz format
```

**Qiskit is optional** — install with: `pip install quantumviz[all]`

## Input Formats

### Bloch Sphere (TXT)

```
|0>                    # Ket notation
theta=60 deg, phi=45   # Bloch angles
(x,y,z)               # Cartesian coordinates
```

### State City / Circuit / Dynamic Flow / DCN (JSON)

```json
{
  "qubits": 2,
  "stages": [
    {"name": "Initial", "state_vector": [1, 0, 0, 0]}
  ]
}
```

DCN visualization supports 1-3 qubit systems:
- **1-2 qubits**: 2D grid layout with inner circles showing magnitude and phase
- **3 qubits**: 3D visualization with x=qubit0, y=qubit2 (depth), z=qubit1

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

See the `examples/algs/` directory for complete input file examples:
- `examples/algs/bloch_sphere/` - Single-qubit state examples
- `examples/algs/state_city/` - Multi-qubit density matrix examples
- `examples/algs/circuit_diagram/` - Quantum circuit examples
- `examples/algs/cost_landscape/inputs/` - QAOA and VQE examples
- `examples/algs/dcn/` - DCN visualization examples (Bell, GHZ, W states)
- `examples/algs/dynamic_flow/` - Time evolution examples

## Development

### Install for Development

```bash
git clone https://github.com/viveksoorya/quantumviz.git
cd quantumviz
pip install -e ".[dev]"        # Dev deps (pytest, ruff, mypy)
pip install -e ".[dashboard]"  # + FastAPI, uvicorn, pydantic
pip install -e ".[all]"        # + qiskit, qiskit-ibm-runtime
```

### Run Tests

```bash
pytest tests/ --cov=quantumviz    # All tests with coverage
pytest tests/test_dcn.py -v      # Single file (verbose)
pytest -k "pattern"              # Match pattern
```

### Code Quality

```bash
ruff check src/    # Ruff config in pyproject.toml
mypy src/          # Python 3.9 target
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
