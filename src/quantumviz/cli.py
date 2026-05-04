"""
CLI (Command Line Interface) for quantumviz package.

Provides unified command-line interface for all visualization tools.
"""

import os
import sys
from typing import Optional

import click

from quantumviz.bloch_sphere import plot_bloch_sphere as _plot_bloch_sphere
from quantumviz.circuit_diagram import plot_circuit as _draw_circuit
from quantumviz.cost_landscape import (
    get_examples_dir,
    validate_qaoa_input,
    validate_vqe_input,
)
from quantumviz.cost_landscape import (
    plot_qaoa_landscape as _plot_qaoa,
)
from quantumviz.cost_landscape import (
    plot_vqe_landscape as _plot_vqe,
)
from quantumviz.dcn import plot_dcns_from_file as _plot_dcn
from quantumviz.qbeads import plot_qbeads_from_file as _plot_qbeads


def _output_path(input_file: str, suffix: str, fmt: str, output_arg: Optional[str] = None) -> str:
    """Generate output path with correct extension."""
    if output_arg:
        return output_arg
    base = os.path.splitext(input_file)[0]
    return f"{base}{suffix}.{fmt}"


@click.group()
@click.version_option(version="0.3.0")
def main():
    """quantumviz - Quantum Algorithm Visualization Library"""
    pass


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", "output_file", help="Output file path")
@click.option("-f", "--format", "fmt", type=click.Choice(["png", "pdf", "svg"], case_sensitive=False), default="png", help="Output format (png, pdf, svg)")
@click.option("--dpi", default=150, help="DPI for saved figure")
def bloch_sphere(input_file, output_file, fmt, dpi):
    """Plot Bloch sphere visualization from input file or Qiskit object."""
    output_file = _output_path(input_file, "", fmt, output_file)

    # Check for Qiskit QPY format
    if input_file.endswith('.qpy'):
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            circuits = QuantumCircuit.from_qpy(input_file)
            qc = circuits[0] if circuits else None
            sv = Statevector.from_circuit(qc)
            _plot_bloch_sphere([sv], output_file, dpi)
        except ImportError:
            click.echo("Error: Qiskit not installed. Install with: pip install qiskit", err=True)
            sys.exit(1)
    else:
        _plot_bloch_sphere(input_file, output_file, dpi)

    click.echo(f"Saved: {output_file}")


@main.command()
@click.argument("input_files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("-o", "--output-dir", "output_dir", help="Output directory")
@click.option("-f", "--format", "fmt", type=click.Choice(["png", "pdf", "svg"], case_sensitive=False), default="png", help="Output format (png, pdf, svg)")
@click.option("--dpi", default=150, help="DPI for saved figures")
def state_city(input_files, output_dir, fmt, dpi):
    """Plot State City visualization from JSON input file(s) or Qiskit object.

    Supports multiple input files: quantumviz state-city file1.json file2.json

    Input formats supported:
    - {"states": [[...], [...]]} - array of state vectors
    - {"state_vector": [...]} - single state vector
    - {"qubits": N, "stages": [...]} - stages format
    """
    from quantumviz.state_city import plot_state_cities_from_file

    for input_file in input_files:
        # Check for Qiskit QPY format
        if input_file.endswith('.qpy'):
            try:
                from qiskit import QuantumCircuit
                from qiskit.quantum_info import Statevector
                circuits = QuantumCircuit.from_qpy(input_file)
                qc = circuits[0] if circuits else None
                sv = Statevector.from_circuit(qc)
                if output_dir:
                    output_file = f"{output_dir}/state_city.{fmt}"
                else:
                    output_file = f"state_city.{fmt}"
                from quantumviz.state_city import plot_state_city
                plot_state_city(sv, "State City", output_file, dpi)
                click.echo(f"Saved: {output_file}")
            except ImportError:
                click.echo("Error: Qiskit not installed. Install with: pip install qiskit", err=True)
                sys.exit(1)
        else:
            output_files = plot_state_cities_from_file(input_file, output_dir, dpi, fmt)
            for f in output_files:
                click.echo(f"Saved: {f}")


@main.command()
@click.argument("algorithm", type=click.Choice(["qaoa", "vqe"]))
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", "output_file", help="Output file path")
@click.option("-f", "--format", "fmt", type=click.Choice(["png", "pdf", "svg"], case_sensitive=False), default="png", help="Output format (png, pdf, svg)")
@click.option("--dpi", default=150, help="DPI for saved figure")
def cost_landscape(algorithm, input_file, output_file, fmt, dpi):
    """Plot QAOA or VQE cost landscape from input file.

    QAOA format:     {"edges": [[0, 1], [1, 2], ...]}
    VQE format:      {"terms": [{"coeff": 0.5, "paulis": ["Z"]}, ...]}

    See examples in: quantumviz/examples/
    """
    import json

    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid JSON in {input_file}: {e}", err=True)
        sys.exit(1)

    if not isinstance(data, dict):
        click.echo(f"Error: Input must be a JSON object, got {type(data).__name__}.", err=True)
        sys.exit(1)

    output_file = _output_path(input_file, f'_{algorithm}', fmt, output_file)

    if algorithm == "qaoa":
        try:
            edges = validate_qaoa_input(data)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            click.echo(f"\nSee example files: {get_examples_dir()}/")
            sys.exit(1)
        _plot_qaoa(edges, output_path=output_file, dpi=dpi)
    else:
        try:
            terms = validate_vqe_input(data)
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            click.echo(f"\nSee example files: {get_examples_dir()}/")
            sys.exit(1)
        _plot_vqe(terms, output_path=output_file, dpi=dpi)

    click.echo(f"Saved: {output_file}")


@main.command()
@click.argument("input_files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("-o", "--output-dir", "output_dir", help="Output directory")
@click.option("-f", "--format", "fmt", type=click.Choice(["png", "pdf", "svg"], case_sensitive=False), default="png", help="Output format (png, pdf, svg)")
@click.option("--dpi", default=150, help="DPI for saved figure")
def circuit(input_files, output_dir, fmt, dpi):
    """Plot quantum circuit diagram from JSON or QPY input file(s).

    Supports multiple input files: quantumviz circuit file1.json file2.json

    Input formats supported:
    - {"qubits": N, "gates": [...]} - single circuit
    - [{"qubits": N, "gates": [...]}, ...] - array of circuits
    """
    from quantumviz.circuit_diagram import plot_circuit, plot_circuits

    for input_file in input_files:
        # Check for Qiskit QPY format
        if input_file.endswith('.qpy'):
            try:
                from qiskit import QuantumCircuit
                circuits = QuantumCircuit.from_qpy(input_file)
                qc = circuits[0] if circuits else None
                output_file = _output_path(input_file, '_circuit', fmt, output_dir)
                _draw_circuit(qc, output_file, dpi)
                click.echo(f"Saved: {output_file}")
            except ImportError:
                click.echo("Error: Qiskit not installed. Install with: pip install qiskit", err=True)
                sys.exit(1)
        else:
            import json
            with open(input_file, 'r') as f:
                data = json.load(f)

            # Check if it's an array of circuits
            if isinstance(data, list):
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                circuits = data
                output_files = plot_circuits(circuits, output_dir or './', base_name, dpi, fmt)
                for f in output_files:
                    click.echo(f"Saved: {f}")
            else:
                output_file = _output_path(input_file, '_circuit', fmt, output_dir)
                plot_circuit(data, output_file, dpi)
                click.echo(f"Saved: {output_file}")


@main.command()
@click.argument("input_files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("-o", "--output-dir", "output_dir", help="Output directory")
@click.option("-f", "--format", "fmt", type=click.Choice(["png", "pdf", "svg"], case_sensitive=False), default="png", help="Output format (png, pdf, svg)")
@click.option("--dpi", default=150, help="DPI for saved figure")
def dynamic_flow(input_files, output_dir, fmt, dpi):
    """Plot dynamic flow/time evolution from JSON or QPY input file(s).

    Supports multiple input files: quantumviz dynamic-flow file1.json file2.json
    """
    from quantumviz.dynamic_flow import plot_dynamic_flow

    for input_file in input_files:
        output_file = _output_path(input_file, '_flow', fmt, output_dir)

        # Check for Qiskit QPY format
        if input_file.endswith('.qpy'):
            try:
                from qiskit import QuantumCircuit
                from qiskit.quantum_info import Statevector
                circuits = QuantumCircuit.from_qpy(input_file)
                qc = circuits[0] if circuits else None
                sv = Statevector.from_circuit(qc)
                from quantumviz.dynamic_flow import plot_time_evolution
                plot_time_evolution([sv], "Time Evolution", output_file, dpi)
            except ImportError:
                click.echo("Error: Qiskit not installed. Install with: pip install qiskit", err=True)
                sys.exit(1)
        else:
            plot_dynamic_flow(input_file, output_file, dpi)

        click.echo(f"Saved: {output_file}")


@main.command()
@click.argument("input_files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("-o", "--output-dir", "output_dir", help="Output directory")
@click.option("-f", "--format", "fmt", type=click.Choice(["png", "pdf", "svg"], case_sensitive=False), default="png", help="Output format (png, pdf, svg)")
@click.option("--dpi", default=150, help="DPI for saved figure")
def dcn(input_files, output_dir, fmt, dpi):
    """Plot Dimensional Circular Notation (DCN) from JSON or QPY input file(s).

    Supports multiple input files: quantumviz dcn file1.json file2.json

    Input formats supported:
    - {"states": [[...], [...]]} - array of state vectors
    - {"state_vector": [...]} - single state vector
    - {"qubits": N, "stages": [...]} - stages format
    """
    from quantumviz.dcn import plot_dcns_from_file

    for input_file in input_files:
        if output_dir is None:
            # Determine output based on input
            if input_file.endswith('.json'):
                output_file = input_file.replace('.json', f'_dcn.{fmt}')
            elif input_file.endswith('.qpy'):
                output_file = input_file.replace('.qpy', f'_dcn.{fmt}')
            else:
                output_file = input_file + f'_dcn.{fmt}'
        else:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = f"{output_dir}/{base_name}_dcn.{fmt}"

        # Check for Qiskit QPY format
        if input_file.endswith('.qpy'):
            try:
                from qiskit import QuantumCircuit
                from qiskit.quantum_info import Statevector
                circuits = QuantumCircuit.from_qpy(input_file)
                qc = circuits[0] if circuits else None
                sv = Statevector.from_circuit(qc)
                from quantumviz.dcn import plot_dcn
                plot_dcn(sv, "DCN Visualization", output_file, dpi)
            except ImportError:
                click.echo("Error: Qiskit not installed. Install with: pip install qiskit", err=True)
                sys.exit(1)
        else:
            # Check if input is a directory (for multiple stages) or single file
            if os.path.isdir(input_file):
                _plot_dcn(input_file, output_file, dpi, fmt)
            else:
                output_files = plot_dcns_from_file(input_file, output_dir or './', dpi, fmt)
                for f in output_files:
                    click.echo(f"Saved: {f}")
                continue

        click.echo(f"Saved: {output_file}")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", "output_file", help="Output file path")
@click.option("-f", "--format", "fmt", type=click.Choice(["png", "pdf", "svg"], case_sensitive=False), default="png", help="Output format (png, pdf, svg)")
@click.option("--dpi", default=150, help="DPI for saved figure")
def paulivec(input_file, output_file, fmt, dpi):
    """Plot PauliVec visualization (bar chart) from JSON or QPY input file."""
    from quantumviz.paulivec import plot_paulivecs_from_file as _plot_paulivec

    if input_file.endswith('.qpy'):
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            circuits = QuantumCircuit.from_qpy(input_file)
            qc = circuits[0] if circuits else None
            sv = Statevector.from_circuit(qc)
            from quantumviz.paulivec import plot_paulivec
            if output_file is None:
                output_file = f"paulivec.{fmt}"
            plot_paulivec(sv, "PauliVec", output_file, dpi)
        except ImportError:
            click.echo("Error: Qiskit not installed. Install with: pip install qiskit", err=True)
            sys.exit(1)
    else:
        output_dir = os.path.dirname(output_file) if output_file else None
        _plot_paulivec(input_file, output_dir, dpi, fmt)
    click.echo(f"Saved: {output_file if output_file else 'multiple files'}")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", "output_file", help="Output file path")
@click.option("-f", "--format", "fmt", type=click.Choice(["png", "pdf", "svg"], case_sensitive=False), default="png", help="Output format (png, pdf, svg)")
@click.option("--dpi", default=150, help="DPI for saved figure")
def bloch_multivector(input_file, output_file, fmt, dpi):
    """Plot Bloch Multivector (N Bloch spheres) from JSON or QPY input file."""
    from quantumviz.bloch_multivector import plot_bloch_multivectors_from_file as _plot_bloch

    if input_file.endswith('.qpy'):
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            circuits = QuantumCircuit.from_qpy(input_file)
            qc = circuits[0] if circuits else None
            sv = Statevector.from_circuit(qc)
            from quantumviz.bloch_multivector import plot_bloch_multivector
            if output_file is None:
                output_file = f"bloch_multivector.{fmt}"
            plot_bloch_multivector(sv, "Bloch Multivector", output_file, dpi)
        except ImportError:
            click.echo("Error: Qiskit not installed. Install with: pip install qiskit", err=True)
            sys.exit(1)
    else:
        output_dir = os.path.dirname(output_file) if output_file else None
        _plot_bloch(input_file, output_dir, dpi, fmt)
    click.echo(f"Saved: {output_file if output_file else 'multiple files'}")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", "output_file", help="Output file path")
@click.option("-f", "--format", "fmt", type=click.Choice(["png", "pdf", "svg"], case_sensitive=False), default="png", help="Output format (png, pdf, svg)")
@click.option("--dpi", default=150, help="DPI for saved figure")
def hinton(input_file, output_file, fmt, dpi):
    """Plot Hinton diagram from JSON or QPY input file."""
    from quantumviz.hinton import plot_hintons_from_file as _plot_hinton

    if input_file.endswith('.qpy'):
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            circuits = QuantumCircuit.from_qpy(input_file)
            qc = circuits[0] if circuits else None
            sv = Statevector.from_circuit(qc)
            from quantumviz.hinton import plot_hinton
            if output_file is None:
                output_file = f"hinton.{fmt}"
            plot_hinton(sv, "Hinton Diagram", output_file, dpi)
        except ImportError:
            click.echo("Error: Qiskit not installed. Install with: pip install qiskit", err=True)
            sys.exit(1)
    else:
        output_dir = os.path.dirname(output_file) if output_file else None
        _plot_hinton(input_file, output_dir, dpi, fmt)
    click.echo(f"Saved: {output_file if output_file else 'multiple files'}")


@main.command()
@click.argument("input_files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("-o", "--output-dir", "output_dir", help="Output directory")
@click.option("-f", "--format", "fmt", type=click.Choice(["png", "pdf", "svg"], case_sensitive=False), default="png", help="Output format (png, pdf, svg)")
@click.option("--dpi", default=150, help="DPI for saved figures")
def qbeads(input_files, output_dir, fmt, dpi):
    """Plot BEADS (Quantum Beads) visualization from JSON or QPY input file(s).

    Supports multiple input files: quantumviz qbeads file1.json file2.json

    Input formats supported:
    - {"states": [[...], [...]]} - array of state vectors
    - {"state_vector": [...]} - single state vector
    - {"qubits": N, "stages": [...]} - stages format
    """
    for input_file in input_files:
        if output_dir is None:
            if input_file.endswith('.json'):
                output_file = input_file.replace('.json', f'_qbeads.{fmt}')
            elif input_file.endswith('.qpy'):
                output_file = input_file.replace('.qpy', f'_qbeads.{fmt}')
            else:
                output_file = input_file + f'_qbeads.{fmt}'
        else:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = f"{output_dir}/{base_name}_qbeads.{fmt}"

        if input_file.endswith('.qpy'):
            try:
                from qiskit import QuantumCircuit
                from qiskit.quantum_info import Statevector
                circuits = QuantumCircuit.from_qpy(input_file)
                qc = circuits[0] if circuits else None
                sv = Statevector.from_circuit(qc)
                from quantumviz.qbeads import plot_qbeads
                plot_qbeads(sv, "BEADS Visualization", output_file, dpi)
            except ImportError:
                click.echo("Error: Qiskit not installed. Install with: pip install qiskit", err=True)
                sys.exit(1)
        else:
            _plot_qbeads(input_file, output_dir, dpi, fmt)

        click.echo(f"Saved: {output_file}")


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("-p", "--port", default=8000, help="Port to bind to")
def serve(host, port):
    """Start the quantumviz dashboard web server."""
    try:
        import uvicorn

        from quantumviz.dashboard.main import app
        click.echo(f"Starting dashboard at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        click.echo("Error: Dashboard dependencies not installed.", err=True)
        click.echo("Install with: pip install quantumviz[dashboard]", err=True)
        sys.exit(1)


@main.command()
def gui():
    """Launch the quantumviz desktop GUI application."""
    try:
        import sys

        from PyQt6.QtWidgets import QApplication

        from quantumviz.gui import QuantumVizGUI

        app = QApplication(sys.argv)
        app.setApplicationName("QuantumViz")
        app.setApplicationVersion("0.3.0")

        window = QuantumVizGUI()
        window.show()

        sys.exit(app.exec())
    except ImportError:
        click.echo("Error: GUI dependencies not installed.", err=True)
        click.echo("Install with: pip install quantumviz[gui]", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
