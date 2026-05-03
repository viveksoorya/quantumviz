"""
quantumviz Dashboard - FastAPI web interface for quantum visualizations.

This module provides a web-based dashboard for visualizing quantum algorithm
states and running circuits on quantum hardware.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Quantum Viz Dashboard", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FRONTEND_DIR = Path(__file__).parent / "frontend"


class StateVectorInput(BaseModel):
    qubits: int = Field(ge=1, le=10, description="Number of qubits")
    stages: Optional[List[Dict[str, Any]]] = Field(default=None, description="Algorithm stages with state vectors")
    algorithm: Optional[str] = Field(default=None, description="Algorithm name for parameterized generation")
    params: Optional[Dict[str, Any]] = Field(default={}, description="Algorithm parameters")


class BlochSphereInput(BaseModel):
    states: List[Dict[str, Any]] = Field(description="List of quantum states")
    algorithm: str = Field(description="Algorithm name")


class CircuitInput(BaseModel):
    qubits: int = Field(ge=1, le=10)
    gates: List[Dict[str, Any]]
    name: Optional[str] = "Quantum Circuit"


class HardwareConfig(BaseModel):
    qubits: int = Field(default=2, ge=1, le=10, description="Number of qubits")
    backend: str = Field(default="ibmq_qasm_simulator", description="Backend name")
    token: Optional[str] = Field(default=None, description="IBM Quantum API token")
    shots: int = Field(default=1024, ge=100, le=100000)
    state_vector: Optional[List[complex]] = Field(default=None, description="Optional initial state")


def state_to_bloch_vector(state: List[complex]) -> Dict[str, float]:
    """Convert state vector to Bloch vector."""
    alpha, beta = complex(state[0]), complex(state[1])
    theta = 2 * np.arccos(np.abs(alpha))
    phi = np.angle(beta) - np.angle(alpha)
    x = float(np.sin(theta) * np.cos(phi))
    y = float(np.sin(theta) * np.sin(phi))
    z = float(np.cos(theta))
    return {"x": x, "y": y, "z": z}


def state_to_density_matrix(state: List[complex], dim: int) -> Dict[str, Any]:
    """Convert state vector to density matrix."""
    psi = np.array(state, dtype=complex).reshape(-1, 1)
    rho = psi @ psi.conj().T
    real_part = np.real(rho).tolist()
    imag_part = np.imag(rho).tolist()
    probabilities = np.abs(state) ** 2
    return {
        "density_matrix_real": real_part,
        "density_matrix_imag": imag_part,
        "probabilities": probabilities.tolist(),
        "dimensions": dim
    }


def bloch_to_plotly(vector: Dict[str, float], title: str = "") -> Dict[str, Any]:
    """Convert Bloch sphere data to Plotly format."""
    x, y, z = vector["x"], vector["y"], vector["z"]
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    return {
        "data": [
            {"type": "surface", "x": sphere_x.tolist(), "y": sphere_y.tolist(),
             "z": sphere_z.tolist(), "colorscale": [[0, 'cyan'], [1, 'cyan']],
             "opacity": 0.3, "showscale": False},
            {"type": "cone", "x": [0, x], "y": [0, y], "z": [0, z],
             "u": [x], "v": [y], "w": [z], "colorscale": [[0, 'black'], [1, 'black']],
             "showscale": False}
        ],
        "layout": {
            "title": title,
            "scene": {"xaxis": {"range": [-1.5, 1.5]}, "yaxis": {"range": [-1.5, 1.5]},
                     "zaxis": {"range": [-1.5, 1.5]}, "aspectmode": "cube"},
            "margin": {"l": 0, "r": 0, "t": 30, "b": 0},
            "height": 400
        }
    }


def density_to_plotly(dm_data: Dict, title: str = "") -> Dict[str, Any]:
    """Convert density matrix data to Plotly format."""
    dim = dm_data["dimensions"]
    probs = dm_data["probabilities"]
    basis_labels = [f"|{i:0{int(np.log2(dim))}b}>" for i in range(dim)]

    return {
        "data": [
            {"type": "bar", "x": basis_labels, "y": probs, "marker": {"color": "steelblue"},
             "name": "Probabilities", "text": [f"{p:.3f}" for p in probs]}
        ],
        "layout": {
            "title": f"{title} - Probabilities",
            "xaxis": {"title": "Basis State"},
            "yaxis": {"title": "Probability", "range": [0, 1.1]},
            "height": 400,
            "margin": {"l": 50, "r": 50, "t": 50, "b": 100}
        }
    }


def compute_algorithm_stages(algorithm: str, qubits: int, params: Dict) -> List[Dict]:
    """Compute algorithm stages based on parameters."""
    if algorithm == "grover":
        return compute_grover_stages(qubits, params)
    elif algorithm == "qaoa":
        return compute_qaoa_stages(qubits, params)
    elif algorithm == "qft":
        return compute_qft_stages(qubits, params)
    elif algorithm == "vqe":
        return compute_vqe_stages(qubits, params)
    return []


def compute_grover_stages(qubits: int, params: Dict) -> List[Dict]:
    """Compute Grover's algorithm stages."""
    good_states = params.get("good_states", ["11"])
    init_state = params.get("init_state", "zero")

    stages = []
    dim = 2 ** qubits

    # Initial state
    if init_state == "zero":
        state = [1] + [0] * (dim - 1)
        stages.append({"name": f"|{'0'*qubits}⟩ Initial", "state": state})
    else:  # superposition
        amp = 1 / np.sqrt(dim)
        state = [amp] * dim
        stages.append({"name": "After H (Superposition)", "state": state})

    # After Hadamards (if not already superposition)
    if init_state == "zero":
        amp = 1 / np.sqrt(dim)
        state = [amp] * dim
        stages.append({"name": "After H (Superposition)", "state": state})

    # Oracle (mark good states)
    for i in range(len(state)):
        bitstring = format(i, f'0{qubits}b')
        if bitstring in good_states:
            state[i] = -state[i]
    stages.append({"name": f"After Oracle (Mark {good_states})", "state": state})

    # Diffusion
    avg = np.mean(state)
    state = [2*avg - s for s in state]
    stages.append({"name": "After Diffusion", "state": state})

    return stages


def compute_qaoa_stages(qubits: int, params: Dict) -> List[Dict]:
    """Compute QAOA stages."""
    p = params.get("p", 1)
    graph_edges = params.get("graph", [(0,1), (1,2), (2,0)])

    # Initial superposition
    dim = 2 ** qubits
    amp = 1 / np.sqrt(dim)
    state = [amp] * dim

    stages = [{"name": "Initial (Uniform)", "state": state}]

    # Simulate p layers
    for layer in range(p):
        gamma = 0.5 * (layer + 1)
        beta = 0.3 * (layer + 1)

        # Apply cost Hamiltonian (simplified)
        for i in range(len(state)):
            bitstring = format(i, f'0{qubits}b')
            # Simple cost: count edges where both bits are 1
            cost = 0
            for (u, v) in graph_edges:
                if bitstring[u] == '1' and bitstring[v] == '1':
                    cost += 1
            state[i] *= np.exp(-1j * gamma * cost)

        # Apply mixer Hamiltonian (simplified - RX rotations)
        # For simplicity, just adjust amplitudes
        state = [s * np.exp(-1j * beta) for s in state]

        # Normalize
        norm = np.sqrt(sum([abs(s)**2 for s in state]))
        state = [s / norm for s in state]

        stages.append({"name": f"p={layer+1}, γ={gamma:.1f}, β={beta:.1f}", "state": state})

    return stages


def compute_qft_stages(qubits: int, params: Dict) -> List[Dict]:
    """Compute QFT stages."""
    dim = 2 ** qubits

    # Initial |01...> state
    state = [0] * dim
    state[1] = 1  # |01...⟩
    stages = [{"name": f"|{1:0{qubits}b}⟩ Input", "state": state}]

    # Apply QFT (simplified - just transform)
    # Real QFT would apply Hadamard + controlled phase rotations
    # For visualization, we create a simple Fourier-like state
    state = []
    for k in range(dim):
        amplitude = 0
        for n in range(dim):
            theta = 2 * np.pi * k * n / dim
            amplitude += np.exp(1j * theta) / np.sqrt(dim)
        state.append(amplitude)

    stages.append({"name": "After QFT", "state": state})
    return stages


def compute_vqe_stages(qubits: int, params: Dict) -> List[Dict]:
    """Compute VQE stages."""
    params.get("ansatz", "ry")  # Reserved for future use
    layers = params.get("layers", 1)

    # Initial state |00...0⟩
    dim = 2 ** qubits
    state = [1] + [0] * (dim - 1)
    stages = [{"name": "Initial", "state": state}]

    # Apply ansatz layers
    for layer in range(layers):
        theta = 0.5 * (layer + 1)

        # RY ansatz (simplified)
        # Apply RY rotation to first qubit
        if qubits >= 1:
            new_state = []
            for i in range(dim):
                bitstring = format(i, f'0{qubits}b')
                if bitstring[0] == '0':
                    new_state.append(np.cos(theta/2))
                    new_state.append(np.sin(theta/2))
                    break
                else:
                    new_state.append(-np.sin(theta/2))
                    new_state.append(np.cos(theta/2))
                    break
            # Pad or trim to correct dimension
            while len(new_state) < dim:
                new_state.append(0)
            state = new_state[:dim]

        stages.append({"name": f"θ={theta:.2f}", "state": state})

    return stages


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard HTML page."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


@app.post("/api/visualize")
async def visualize(data: StateVectorInput):
    """Visualize quantum state."""
    try:
        # If algorithm specified with params, compute stages
        if data.algorithm and data.params:
            stages = compute_algorithm_stages(data.algorithm, data.qubits, data.params)
        elif data.stages:
            stages = data.stages
        else:
            return {"error": "Provide either stages or algorithm with params"}

        dim = 2 ** data.qubits
        last_stage = stages[-1] if stages else None

        if not last_stage:
            return {"error": "No stages provided"}

        state = last_stage.get("state_vector", last_stage.get("state", [1, 0]))
        if len(state) != dim:
            state = [1] + [0] * (dim - 1)

        bloch_vec = state_to_bloch_vector(state)
        dm_data = state_to_density_matrix(state, dim)

        bloch_plot = bloch_to_plotly(bloch_vec, last_stage.get("name", "Final State"))
        density_plot = density_to_plotly(dm_data, last_stage.get("name", ""))

        return {
            "bloch_plot": bloch_plot,
            "density_plot": density_plot,
            "state_vector": state,
            "bloch_vector": bloch_vec
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/run-hardware")
async def run_on_hardware(config: HardwareConfig):
    """Run quantum circuit on hardware."""
    try:
        from qiskit import QuantumCircuit
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

        qc = QuantumCircuit(config.qubits)

        # Use state_vector if provided, otherwise default to Hadamard
        if config.state_vector:
            from qiskit.quantum_info import Statevector
            state = [complex(x) for x in config.state_vector]
            sv = Statevector(state)
            qc.initialize(sv, range(config.qubits))
        else:
            for i in range(config.qubits):
                qc.h(i)

        service = QiskitRuntimeService(token=config.token) if config.token else None
        backend = service.backend(config.backend) if service else None

        if backend:
            sampler = Sampler(backend=backend)
            job = sampler.run([qc], shots=config.shots)
            result = job.result()
            counts = result[0].data.c.get_counts()
        else:
            counts = {"00": config.shots // 2, "11": config.shots // 2}

        result_file = os.path.join(RESULTS_DIR, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(result_file, "w") as f:
            json.dump({"backend": config.backend, "counts": counts}, f)

        return {
            "backend": config.backend,
            "counts": counts,
            "circuit_depth": qc.depth(),
            "result_file": result_file
        }
    except ImportError:
        return {
            "error": "Qiskit not installed. Install with: pip install qiskit qiskit-ibm-runtime",
            "note": "Running in simulation mode"
        }
    except Exception as e:
        return {"error": str(e), "note": "Check your API token and backend selection"}


@app.get("/api/backends")
async def list_backends():
    """List available quantum backends."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        try:
            service = QiskitRuntimeService()
            backends = service.backends()
            return {"backends": [{"name": b.name, "num_qubits": b.num_qubits} for b in backends if b.num_qubits <= 10]}
        except Exception:
            return {"backends": [], "message": "No credentials configured"}
    except ImportError:
        return {"backends": [], "error": "Qiskit not installed"}


@app.post("/api/visualize/dcn")
async def visualize_dcn(data: StateVectorInput):
    """Generate DCN visualization."""
    try:
        import base64
        import tempfile

        from quantumviz.dcn import plot_dcn

        dim = 2 ** data.qubits
        last_stage = data.stages[-1] if data.stages else None

        if not last_stage:
            return {"error": "No stages provided"}

        state = last_stage.get("state_vector", last_stage.get("state", [1, 0]))
        if len(state) != dim:
            state = [1] + [0] * (dim - 1)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plot_dcn(state, tmp.name)
            tmp.seek(0)
            img_data = base64.b64encode(open(tmp.name, "rb").read()).decode()
            import os
            os.unlink(tmp.name)

        return {"image": f"data:image/png;base64,{img_data}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/visualize/state-city")
async def visualize_state_city(data: StateVectorInput):
    """Generate State City visualization."""
    try:
        import base64
        import tempfile

        from quantumviz.state_city import plot_state_city

        dim = 2 ** data.qubits
        last_stage = data.stages[-1] if data.stages else None

        if not last_stage:
            return {"error": "No stages provided"}

        state = last_stage.get("state_vector", last_stage.get("state", [1, 0]))
        if len(state) != dim:
            state = [1] + [0] * (dim - 1)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plot_state_city(state, tmp.name)
            tmp.seek(0)
            img_data = base64.b64encode(open(tmp.name, "rb").read()).decode()
            import os
            os.unlink(tmp.name)

        return {"image": f"data:image/png;base64,{img_data}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/visualize/circuit")
async def visualize_circuit(data: CircuitInput):
    """Generate Circuit Diagram visualization."""
    try:
        import base64
        import tempfile

        from quantumviz.circuit_diagram import plot_circuit

        circuit_data = {
            "qubits": data.qubits,
            "gates": data.gates,
            "name": data.name
        }

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plot_circuit(circuit_data, tmp.name)
            tmp.seek(0)
            img_data = base64.b64encode(open(tmp.name, "rb").read()).decode()
            import os
            os.unlink(tmp.name)

        return {"image": f"data:image/png;base64,{img_data}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.3.0"}


@app.post("/api/visualize/cost-landscape")
async def visualize_cost_landscape(data: dict):
    """Generate Cost Landscape visualization for QAOA or VQE."""
    try:
        import base64
        import tempfile

        from quantumviz.cost_landscape import plot_qaoa_landscape, plot_vqe_landscape

        algorithm = data.get("algorithm", "qaoa")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            if algorithm == "qaoa":
                # Generate sample QAOA landscape
                edges = [(0, 1), (1, 2), (0, 2)]  # Sample 3-node graph
                plot_qaoa_landscape(edges, (0, np.pi), (0, np.pi), 50, tmp.name)
            else:  # vqe
                # Generate sample VQE landscape
                terms = [
                    {"coeff": -1.0, "paulis": []},
                    {"coeff": 0.5, "paulis": ["Z"]},
                    {"coeff": 0.25, "paulis": ["Z", "Z"]}
                ]
                plot_vqe_landscape(terms, (0, 2*np.pi), 100, tmp.name)

            tmp.seek(0)
            img_data = base64.b64encode(open(tmp.name, "rb").read()).decode()
            import os
            os.unlink(tmp.name)

        return {"image": f"data:image/png;base64,{img_data}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/visualize/dynamic-flow")
async def visualize_dynamic_flow(data: dict):
    """Generate Dynamic Flow visualization."""
    try:
        import base64
        import tempfile

        from quantumviz.dynamic_flow import plot_dynamic_flow

        # Generate sample time evolution data
        time_steps = 50
        states = []
        for t in range(time_steps):
            # Simple Rabi oscillation
            theta = t * 3.14159 / time_steps
            state = [complex(np.cos(theta/2), 0), complex(0, np.sin(theta/2))]
            states.append(state)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plot_dynamic_flow(states, tmp.name)
            tmp.seek(0)
            img_data = base64.b64encode(open(tmp.name, "rb").read()).decode()
            import os
            os.unlink(tmp.name)

        return {"image": f"data:image/png;base64,{img_data}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and parse quantum state files (JSON, QPY, TXT)."""
    try:
        content = await file.read()
        filename = file.filename or "unknown"

        if filename.endswith('.json'):
            # Parse JSON file
            data = json.loads(content.decode('utf-8'))
            # Extract state vector from various JSON formats
            if 'stages' in data:
                state = data['stages'][-1].get('state', data['stages'][-1].get('state_vector', []))
            elif 'state_vector' in data:
                state = data['state_vector']
            elif 'amplitudes' in data:
                state = data['amplitudes']
            else:
                # Assume the JSON itself is the state vector
                state = data if isinstance(data, list) else []

            if not state:
                return {"error": "No state vector found in JSON file"}

            # Parse complex numbers
            state = [complex(s) if not isinstance(s, complex) else s for s in state]
            qubits = int(np.log2(len(state)))

            return {
                "qubits": qubits,
                "state_vector": [{"re": float(np.real(s)), "im": float(np.imag(s))} for s in state],
                "preview": {"qubits": qubits, "num_amplitudes": len(state), "format": "json"}
            }

        elif filename.endswith('.qpy'):
            # Parse Qiskit QPY file
            try:
                import io

                from qiskit import QuantumCircuit
                from qiskit.quantum_info import Statevector

                qc = QuantumCircuit.from_qpy(io.BytesIO(content))[0]
                state = Statevector(qc).data.tolist()
                qubits = qc.num_qubits

                return {
                    "qubits": qubits,
                    "state_vector": [{"re": float(np.real(s)), "im": float(np.imag(s))} for s in state],
                    "preview": {"qubits": qubits, "circuit": str(qc), "format": "qpy"}
                }
            except ImportError:
                return {"error": "Qiskit not installed. Install with: pip install qiskit"}

        elif filename.endswith('.txt'):
            # Parse text file (one amplitude per line)
            lines = content.decode('utf-8').strip().split('\n')
            state = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Try to parse as complex number
                    try:
                        # Handle formats like "0.707+0.707j" or "[0.707, 0.707]"
                        line = line.strip('[]').strip()
                        if ',' in line:
                            # Assume it's a list [re, im]
                            parts = line.split(',')
                            re = float(parts[0].strip())
                            im = float(parts[1].strip().replace('j', ''))
                            state.append(complex(re, im))
                        else:
                            # Try to eval as complex
                            state.append(complex(line.replace('j', 'j')))
                    except Exception:
                        return {"error": f"Cannot parse line: {line}"}

            if not state:
                return {"error": "No valid amplitudes found in TXT file"}

            qubits = int(np.log2(len(state)))
            return {
                "qubits": qubits,
                "state_vector": [{"re": float(np.real(s)), "im": float(np.imag(s))} for s in state],
                "preview": {"qubits": qubits, "num_amplitudes": len(state), "format": "txt"}
            }

        else:
            return {"error": "Unsupported file format. Use .json, .qpy, or .txt files."}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
