# AGENTS.md - quantumviz

## Commands

```bash
# Test (always verbose: addopts = "-v --tb=short" in pyproject.toml)
pytest tests/ --cov=quantumviz        # All tests with coverage
pytest tests/test_dcn.py              # Single file (no -v needed)
pytest -k "pattern"                   # Match pattern

# Lint & Typecheck
ruff check src/                       # Ruff config in pyproject.toml
mypy src/                             # Python 3.9 target

# Install
pip install -e ".[dev]"                # Dev deps (pytest, ruff, mypy)
pip install -e ".[dashboard]"          # + FastAPI, uvicorn, pydantic
pip install -e ".[all]"               # + qiskit, qiskit-ibm-runtime
```

## CLI

```bash
quantumviz bloch-sphere input.txt -o out.png
quantumviz bloch-sphere input.txt -o out.pdf -f pdf
quantumviz state-city input.json -o out/ -f pdf    # Multiple files
quantumviz circuit input.json -o out.png
quantumviz cost-landscape qaoa input.json -o out.png
quantumviz cost-landscape vqe input.json -o out.png
quantumviz dynamic-flow input.json -o out.png
quantumviz dcn input.json -o out.pdf -f pdf
quantumviz gui                                  # Launch desktop GUI app
quantumviz serve                               # Dashboard at localhost:8000 (web)
```

### Output Format Options
- `-f, --format`: Output format (png, pdf, svg) - defaults to `png`
- Output path extension is ignored in favor of the `--format` option
- `--dpi`: DPI for saved figures (default: 150)

## Dependencies

- **click**: Used by CLI (`src/quantumviz/cli.py`) but missing from `pyproject.toml` dependencies. Install manually: `pip install click`
- CI installs it explicitly: `pip install pytest pytest-cov pytest-mock click`

## Conventions

- **Basis ordering**: Big-endian |q₁q₀⟩ (q₁ = MSB)
- **Complex amplitudes**: `j` suffix (`0.707+0.707j`)
- **State vectors**: Must be normalized, length must be power of 2
- **Matplotlib**: Tests auto-configure via `tests/conftest.py` (`Agg` backend, auto `plt.close()`)

## Structure

- `src/quantumviz/` - Visualization modules (one per type)
- `src/quantumviz/dashboard/` - FastAPI web interface
- `tests/test_*.py` - One test file per module
- `tests/conftest.py` - Shared fixtures, matplotlib config
- `examples/` - Input files organized by algorithm

## Qiskit Compatibility

quantumviz accepts Qiskit objects directly:

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from quantumviz import plot_bloch_sphere, plot_dcn, plot_state_city

# Create circuit with Qiskit
qc = QuantumCircuit(2)
qc.h(0); qc.cx(0, 1)

# Pass Statevector directly
state = Statevector(qc)
plot_dcn(state, "bell_dcn.png")          # quantumviz's unique DCN viz
plot_bloch_sphere([state], "bloch.png")  # Accepts Statevector
plot_state_city(state, "state_city.png")  # Accepts Statevector

# Pass QuantumCircuit directly
plot_circuit(qc, "circuit.png")         # Converts to quantumviz format
```

Qiskit is optional — install with: `pip install quantumviz[all]`

## References

- Input formats: `CONTEXT.md`
- Ruff/mypy config: `pyproject.toml`
