"""Background workers for quantumviz GUI."""

import os
import tempfile
import textwrap
import traceback
from typing import Any, Optional

from PyQt6.QtCore import QThread, pyqtSignal


class VisualizationWorker(QThread):
    """Worker thread for generating visualizations without blocking the UI."""

    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, viz_type: str, input_file: str, output_format: str = "png", dpi: int = 150):
        super().__init__()
        self.viz_type = viz_type
        self.input_file = input_file
        self.output_format = output_format
        self.dpi = dpi
        self.temp_dir: Optional[str] = None

    def run(self):
        try:
            self.progress.emit(f"Generating {self.viz_type} visualization...")

            self.temp_dir = tempfile.mkdtemp(prefix="quantumviz_")
            output_file = os.path.join(self.temp_dir, f"output.{self.output_format}")

            if self.viz_type == "bloch-sphere":
                self._generate_bloch(output_file)
            elif self.viz_type == "state-city":
                self._generate_state_city(output_file)
            elif self.viz_type == "circuit":
                self._generate_circuit(output_file)
            elif self.viz_type == "dcn":
                self._generate_dcn(output_file)
            elif self.viz_type == "cost-landscape-qaoa":
                self._generate_cost_landscape_qaoa(output_file)
            elif self.viz_type == "cost-landscape-vqe":
                self._generate_cost_landscape_vqe(output_file)
            elif self.viz_type == "dynamic-flow":
                self._generate_dynamic_flow(output_file)
            else:
                raise ValueError(f"Unknown visualization type: {self.viz_type}")

            self.finished.emit(output_file, self.viz_type)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)
        finally:
            self.progress.emit("")

    def _generate_bloch(self, output_file: str):
        from quantumviz.bloch_sphere import plot_bloch_sphere
        if self.input_file.endswith('.qpy'):
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            circuits = QuantumCircuit.from_qpy(self.input_file)
            qc = circuits[0] if circuits else None
            sv = Statevector.from_circuit(qc)
            plot_bloch_sphere([sv], output_file, self.dpi)
        else:
            plot_bloch_sphere(self.input_file, output_file, self.dpi)

    def _generate_state_city(self, output_file: str):
        from quantumviz.state_city import plot_state_city
        if self.input_file.endswith('.qpy'):
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            circuits = QuantumCircuit.from_qpy(self.input_file)
            qc = circuits[0] if circuits else None
            sv = Statevector.from_circuit(qc)
            plot_state_city(sv, "State City", output_file, self.dpi)
        else:
            from quantumviz.state_city import plot_state_cities_from_file
            plot_state_cities_from_file(self.input_file, self.temp_dir, self.dpi, self.output_format)

    def _generate_circuit(self, output_file: str):
        from quantumviz.circuit_diagram import plot_circuit
        if self.input_file.endswith('.qpy'):
            from qiskit import QuantumCircuit
            circuits = QuantumCircuit.from_qpy(self.input_file)
            qc = circuits[0] if circuits else None
            plot_circuit(qc, output_file, self.dpi)
        else:
            import json
            with open(self.input_file, 'r') as f:
                data = json.load(f)
            plot_circuit(data, output_file, self.dpi)

    def _generate_dcn(self, output_file: str):
        from quantumviz.dcn import plot_dcn
        if self.input_file.endswith('.qpy'):
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            circuits = QuantumCircuit.from_qpy(self.input_file)
            qc = circuits[0] if circuits else None
            sv = Statevector.from_circuit(qc)
            plot_dcn(sv, "DCN Visualization", output_file, self.dpi)
        else:
            from quantumviz.dcn import plot_dcns_from_file
            plot_dcns_from_file(self.input_file, self.temp_dir, self.dpi, self.output_format)

    def _generate_cost_landscape_qaoa(self, output_file: str):
        import json

        from quantumviz.cost_landscape import plot_qaoa_landscape, validate_qaoa_input

        with open(self.input_file, 'r') as f:
            data = json.load(f)
        edges = validate_qaoa_input(data)
        plot_qaoa_landscape(edges, output_path=output_file, dpi=self.dpi)

    def _generate_cost_landscape_vqe(self, output_file: str):
        import json

        from quantumviz.cost_landscape import plot_vqe_landscape, validate_vqe_input

        with open(self.input_file, 'r') as f:
            data = json.load(f)
        terms = validate_vqe_input(data)
        plot_vqe_landscape(terms, output_path=output_file, dpi=self.dpi)

    def _generate_dynamic_flow(self, output_file: str):
        from quantumviz.dynamic_flow import plot_dynamic_flow
        if self.input_file.endswith('.qpy'):
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            circuits = QuantumCircuit.from_qpy(self.input_file)
            qc = circuits[0] if circuits else None
            sv = Statevector.from_circuit(qc)
            from quantumviz.dynamic_flow import plot_time_evolution
            plot_time_evolution([sv], "Time Evolution", output_file, self.dpi)
        else:
            plot_dynamic_flow(self.input_file, output_file, self.dpi)


class CodeVisualizationWorker(QThread):
    """Worker thread for executing Qiskit code and generating visualizations."""

    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    output = pyqtSignal(str)

    def __init__(self, viz_type: str, code: str, output_format: str = "png", dpi: int = 150):
        super().__init__()
        self.viz_type = viz_type
        self.code = code
        self.output_format = output_format
        self.dpi = dpi
        self.temp_dir: Optional[str] = None

    def run(self):
        try:
            self.progress.emit(f"Executing code and generating {self.viz_type}...")

            self.temp_dir = tempfile.mkdtemp(prefix="quantumviz_")
            output_file = os.path.join(self.temp_dir, f"output.{self.output_format}")

            qiskit_obj = self._execute_code()

            if self.viz_type == "bloch-sphere":
                self._viz_bloch(qiskit_obj, output_file)
            elif self.viz_type == "state-city":
                self._viz_state_city(qiskit_obj, output_file)
            elif self.viz_type == "circuit":
                self._viz_circuit(qiskit_obj, output_file)
            elif self.viz_type == "dcn":
                self._viz_dcn(qiskit_obj, output_file)
            elif self.viz_type == "dynamic-flow":
                self._viz_dynamic_flow(qiskit_obj, output_file)
            else:
                raise ValueError(f"Code mode not supported for: {self.viz_type}")

            self.finished.emit(output_file, self.viz_type)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)
        finally:
            self.progress.emit("")

    def _execute_code(self) -> Any:
        namespace = {
            "__builtins__": __builtins__,
        }

        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import DensityMatrix, Statevector

            namespace["QuantumCircuit"] = QuantumCircuit
            namespace["Statevector"] = Statevector
            namespace["DensityMatrix"] = DensityMatrix
        except ImportError:
            raise ImportError("Qiskit is required for code execution. Install with: pip install qiskit")

        exec(textwrap.dedent(self.code), namespace)

        self.output.emit("Code executed successfully.")

        result = None
        for name, obj in namespace.items():
            if name.startswith("_"):
                continue
            if self._is_statevector(obj) or self._is_quantum_circuit(obj) or self._is_density_matrix(obj):
                result = obj
                self.output.emit(f"Found Qiskit object: {name} ({type(obj).__name__})")

        if result is None:
            raise ValueError("No Qiskit object (Statevector, QuantumCircuit, DensityMatrix) found in code.")

        return result

    def _is_statevector(self, obj) -> bool:
        try:
            from qiskit.quantum_info import Statevector
            return isinstance(obj, Statevector)
        except ImportError:
            return False

    def _is_quantum_circuit(self, obj) -> bool:
        try:
            from qiskit import QuantumCircuit
            return isinstance(obj, QuantumCircuit)
        except ImportError:
            return False

    def _is_density_matrix(self, obj) -> bool:
        try:
            from qiskit.quantum_info import DensityMatrix
            return isinstance(obj, DensityMatrix)
        except ImportError:
            return False

    def _viz_bloch(self, obj, output_file: str):
        from quantumviz.bloch_sphere import plot_bloch_sphere
        if self._is_statevector(obj):
            plot_bloch_sphere([obj], output_file, self.dpi)
        else:
            raise ValueError("Bloch sphere requires a Statevector object")

    def _viz_state_city(self, obj, output_file: str):
        from quantumviz.state_city import plot_state_city
        if self._is_statevector(obj):
            plot_state_city(obj, "State City", output_file, self.dpi)
        elif self._is_density_matrix(obj):
            plot_state_city(obj, "State City", output_file, self.dpi)
        else:
            raise ValueError("State City requires a Statevector or DensityMatrix object")

    def _viz_circuit(self, obj, output_file: str):
        from quantumviz.circuit_diagram import plot_circuit
        if self._is_quantum_circuit(obj):
            plot_circuit(obj, output_file, self.dpi)
        else:
            raise ValueError("Circuit diagram requires a QuantumCircuit object")

    def _viz_dcn(self, obj, output_file: str):
        from quantumviz.dcn import plot_dcn
        if self._is_statevector(obj):
            plot_dcn(obj, "DCN Visualization", output_file, self.dpi)
        else:
            raise ValueError("DCN requires a Statevector object")

    def _viz_dynamic_flow(self, obj, output_file: str):
        from quantumviz.dynamic_flow import plot_time_evolution
        if self._is_statevector(obj):
            plot_time_evolution([obj], "Time Evolution", output_file, self.dpi)
        elif isinstance(obj, list) and all(self._is_statevector(x) for x in obj):
            plot_time_evolution(obj, "Time Evolution", output_file, self.dpi)
        else:
            raise ValueError("Dynamic Flow requires a Statevector or list of Statevectors")
