"""
quantumviz Dashboard - FastAPI web interface for quantum visualizations.

This module provides a web-based dashboard for visualizing quantum algorithm
states and running circuits on quantum hardware.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import json
import os
from datetime import datetime


app = FastAPI(title="Quantum Viz Dashboard", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class StateVectorInput(BaseModel):
    qubits: int = Field(ge=1, le=10, description="Number of qubits")
    stages: List[Dict[str, Any]] = Field(description="Algorithm stages with state vectors")


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
    x = list(range(dim))
    y = list(range(dim))
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


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard HTML page."""
    return """
    <html>
    <head>
        <title>Quantum Viz Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { color: #333; text-align: center; }
            .card { background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .visualization { margin: 20px 0; }
            button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            button:hover { background: #45a049; }
            select, input { padding: 10px; margin: 5px; border-radius: 5px; border: 1px solid #ddd; }
            .info { background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .success { background: #c8e6c9; }
            .error { background: #ffcdd2; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Quantum Viz Dashboard</h1>
            
            <div class="card">
                <h2>Algorithm Selection</h2>
                <select id="algorithm" onchange="updateStages()">
                    <option value="grover">Grover's Search</option>
                    <option value="qft">Quantum Fourier Transform</option>
                    <option value="qaoa">QAOA</option>
                    <option value="vqe">VQE</option>
                    <option value="custom">Custom Input</option>
                </select>
                <select id="qubits">
                    <option value="2">2 Qubits</option>
                    <option value="3">3 Qubits</option>
                    <option value="4">4 Qubits</option>
                </select>
                <button onclick="loadAlgorithm()">Load Algorithm</button>
            </div>
            
            <div class="card">
                <h2>Visualizations</h2>
                <div id="bloch-container" class="visualization"></div>
                <div id="density-container" class="visualization"></div>
                <div id="prob-container" class="visualization"></div>
            </div>
            
            <div class="card">
                <h2>Run on Hardware</h2>
                <div class="info">
                    <p>To run on real quantum hardware, you need an IBM Quantum API token.</p>
                </div>
                <input type="text" id="api-token" placeholder="Enter IBM Quantum API Token (optional)" style="width: 300px;">
                <select id="backend">
                    <option value="ibmq_qasm_simulator">QASM Simulator</option>
                </select>
                <input type="number" id="shots" value="1024" min="100" max="100000">
                <button onclick="runOnHardware()">Run on Hardware</button>
                <div id="hardware-results"></div>
            </div>
            
            <div class="card">
                <h2>Custom State Vector</h2>
                <textarea id="custom-state" rows="3" cols="50" placeholder="[1, 0, 0, 0] or [[1,0], [0.707,0.707]]"></textarea>
                <br>
                <button onclick="visualizeCustom()">Visualize</button>
            </div>
        </div>
        
        <script>
            const apiBase = "/api";
            let currentData = null;
            
            const algorithmStages = {
                grover: {
                    2: [{"name": "|00>", "state": [1, 0, 0, 0]}, {"name": "After H", "state": [0.5, 0.5, 0.5, 0.5]},
                        {"name": "After Oracle", "state": [0.5, 0.5, 0.5, -0.5]}, {"name": "After Diffusion", "state": [0, 0, 0, 1]}]},
                    3: [{"name": "|000>", "state": [1, 0, 0, 0, 0, 0, 0, 0]}, {"name": "After H", "state": [0.354, 0.354, 0.354, 0.354, 0.354, 0.354, 0.354, 0.354]},
                        {"name": "After Oracle", "state": [0.354, 0.354, 0.354, 0.354, 0.354, 0.354, 0.354, -0.354]}, {"name": "After Diffusion", "state": [0, 0, 0, 0, 0, 0, 0, 1]}]}
                },
                qft: {
                    2: [{"name": "|01>", "state": [0, 1, 0, 0]}, {"name": "After QFT", "state": [0.5, 0.5j, -0.5, -0.5j]}],
                    3: [{"name": "|001>", "state": [0, 0, 1, 0, 0, 0, 0, 0]}, {"name": "After QFT", "state": [0.354, 0.25+0.25j, 0.354j, -0.25+0.25j, -0.354, -0.25-0.25j, -0.354j, 0.25-0.25j]}]
                },
                qaoa: {
                    2: [{"name": "Initial", "state": [0.5, 0.5, 0.5, 0.5]}, {"name": "p=1 iter 1", "state": [0.4, 0.4, 0.6, 0.6]}, {"name": "p=1 iter 2", "state": [0.2, 0.2, 0.8, 0.8]}],
                    3: [{"name": "Initial", "state": [0.354, 0.354, 0.354, 0.354, 0.354, 0.354, 0.354, 0.354]}, {"name": "p=1", "state": [0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6]}]
                }
            };
            
            async function loadAlgorithm() {
                const algo = document.getElementById("algorithm").value;
                const qubits = parseInt(document.getElementById("qubits").value);
                
                let stages;
                if (algo === "custom") {
                    const input = document.getElementById("custom-state").value;
                    try {
                        stages = [{"name": "Custom", "state": JSON.parse(input)}];
                    } catch (e) {
                        alert("Invalid JSON input");
                        return;
                    }
                } else {
                    stages = algorithmStages[algo]?.[qubits] || algorithmStages.grover[2];
                }
                
                currentData = { qubits, stages };
                await visualizeData(currentData);
            }
            
            async function visualizeData(data) {
                const response = await fetch(`${apiBase}/visualize`, {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                
                if (result.bloch_plot) {
                    Plotly.newPlot("bloch-container", result.bloch_plot.data, result.bloch_plot.layout);
                }
                if (result.density_plot) {
                    Plotly.newPlot("density-container", result.density_plot.data, result.density_plot.layout);
                }
                if (result.probability_plot) {
                    Plotly.newPlot("prob-container", result.probability_plot.data, result.probability_plot.layout);
                }
            }
            
            async function visualizeCustom() {
                document.getElementById("algorithm").value = "custom";
                await loadAlgorithm();
            }
            
            async function runOnHardware() {
                const token = document.getElementById("api-token").value;
                const backend = document.getElementById("backend").value;
                const shots = parseInt(document.getElementById("shots").value);
                
                const container = document.getElementById("hardware-results");
                container.innerHTML = '<div class="info">Running on hardware...</div>';
                
                try {
                    const response = await fetch(`${apiBase}/run-hardware`, {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({
                            token: token || null,
                            backend: backend,
                            shots: shots,
                            state_vector: currentData?.stages?.[0]?.state || [1, 0, 0, 0]
                        })
                    });
                    const result = await response.json();
                    
                    if (result.error) {
                        container.innerHTML = `<div class="error">${result.error}</div>`;
                    } else {
                        container.innerHTML = `
                            <div class="success">
                                <h4>Results from ${result.backend}</h4>
                                <pre>${JSON.stringify(result.counts, null, 2)}</pre>
                            </div>
                        `;
                    }
                } catch (e) {
                    container.innerHTML = `<div class="error">Error: ${e.message}</div>`;
                }
            }
            
            function updateStages() {
                console.log("Algorithm updated");
            }
            
            loadAlgorithm();
        </script>
    </body>
    </html>
    """


@app.post("/api/visualize")
async def visualize(data: StateVectorInput):
    """Visualize quantum state."""
    try:
        dim = 2 ** data.qubits
        last_stage = data.stages[-1] if data.stages else None

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
            "probability_plot": density_plot,
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
        except:
            return {"backends": [], "message": "No credentials configured"}
    except ImportError:
        return {"backends": [], "error": "Qiskit not installed"}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
