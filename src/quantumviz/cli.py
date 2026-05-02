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
from quantumviz.dynamic_flow import plot_dynamic_flow as _plot_dynamic_flow
from quantumviz.state_city import plot_state_cities_from_file as _plot_state_cities


def _output_path(input_file: str, suffix: str, fmt: str, output_arg: Optional[str] = None) -> str:
    """Generate output path with correct extension."""
    if output_arg:
        return output_arg
    base = os.path.splitext(input_file)[0]
    return f"{base}{suffix}.{fmt}"


@click.group()
@click.version_option(version="0.1.0")
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
            qc = QuantumCircuit.from_qpy(input_file)
            sv = Statevector.from_circuit(qc)
            _plot_bloch_sphere([sv], output_file, dpi)
        except ImportError:
            click.echo("Error: Qiskit not installed. Install with: pip install qiskit", err=True)
            sys.exit(1)
    else:
        _plot_bloch_sphere(input_file, output_file, dpi)

    click.echo(f"Saved: {output_file}")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output-dir", "output_dir", help="Output directory")
@click.option("-f", "--format", "fmt", type=click.Choice(["png", "pdf", "svg"], case_sensitive=False), default="png", help="Output format (png, pdf, svg)")
@click.option("--dpi", default=150, help="DPI for saved figures")
def state_city(input_file, output_dir, fmt, dpi):
    """Plot State City visualization from JSON input file or Qiskit object."""
    # Check for Qiskit QPY format
    if input_file.endswith('.qpy'):
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            qc = QuantumCircuit.from_qpy(input_file)
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
        output_files = _plot_state_cities(input_file, output_dir, dpi, fmt)
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
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", "output_file", help="Output file path")
@click.option("-f", "--format", "fmt", type=click.Choice(["png", "pdf", "svg"], case_sensitive=False), default="png", help="Output format (png, pdf, svg)")
@click.option("--dpi", default=150, help="DPI for saved figure")
def circuit(input_file, output_file, fmt, dpi):
    """Plot quantum circuit diagram from JSON or QPY input file."""
    output_file = _output_path(input_file, '_circuit', fmt, output_file)

    # Check for Qiskit QPY format
    if input_file.endswith('.qpy'):
        try:
            from qiskit import QuantumCircuit
            qc = QuantumCircuit.from_qpy(input_file)
            _draw_circuit(qc, output_file, dpi)
        except ImportError:
            click.echo("Error: Qiskit not installed. Install with: pip install qiskit", err=True)
            sys.exit(1)
    else:
        import json
        with open(input_file, 'r') as f:
            data = json.load(f)
        _draw_circuit(data, output_file, dpi)

    click.echo(f"Saved: {output_file}")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", "output_file", help="Output file path")
@click.option("-f", "--format", "fmt", type=click.Choice(["png", "pdf", "svg"], case_sensitive=False), default="png", help="Output format (png, pdf, svg)")
@click.option("--dpi", default=150, help="DPI for saved figure")
def dynamic_flow(input_file, output_file, fmt, dpi):
    """Plot dynamic flow/time evolution from JSON or QPY input file."""
    output_file = _output_path(input_file, '_flow', fmt, output_file)

    # Check for Qiskit QPY format
    if input_file.endswith('.qpy'):
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            qc = QuantumCircuit.from_qpy(input_file)
            sv = Statevector.from_circuit(qc)
            from quantumviz.dynamic_flow import plot_time_evolution
            plot_time_evolution([sv], "Time Evolution", output_file, dpi)
        except ImportError:
            click.echo("Error: Qiskit not installed. Install with: pip install qiskit", err=True)
            sys.exit(1)
    else:
        _plot_dynamic_flow(input_file, output_file, dpi)

    click.echo(f"Saved: {output_file}")


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", "output_file", help="Output file or directory path")
@click.option("-f", "--format", "fmt", type=click.Choice(["png", "pdf", "svg"], case_sensitive=False), default="png", help="Output format (png, pdf, svg)")
@click.option("--dpi", default=150, help="DPI for saved figure")
def dcn(input_file, output_file, fmt, dpi):
    """Plot Dimensional Circular Notation (DCN) from JSON or QPY input file."""
    if output_file is None:
        # Determine output based on input
        if input_file.endswith('.json'):
            output_file = input_file.replace('.json', f'_dcn.{fmt}')
        elif input_file.endswith('.qpy'):
            output_file = input_file.replace('.qpy', f'_dcn.{fmt}')
        else:
            output_file = input_file + f'_dcn.{fmt}'

    # Check for Qiskit QPY format
    if input_file.endswith('.qpy'):
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            qc = QuantumCircuit.from_qpy(input_file)
            sv = Statevector.from_circuit(qc)
            from quantumviz.dcn import plot_dcn
            plot_dcn(sv, "DCN Visualization", output_file, dpi)
        except ImportError:
            click.echo("Error: Qiskit not installed. Install with: pip install qiskit", err=True)
            sys.exit(1)
    else:
        # Check if input is a directory (for multiple stages) or single file
        import os
        if os.path.isdir(input_file):
            _plot_dcn(input_file, output_file, dpi, fmt)
        else:
            # Single stage - determine if output is file or directory
            if output_file and output_file.endswith('/'):
                # Treat as directory
                _plot_dcn(input_file, output_file, dpi, fmt)
            else:
                # Treat as file - extract filename and call plot_dcn directly
                # Create temp dir, save there, then rename
                import tempfile

                from quantumviz.dcn import plot_dcns_from_file
                with tempfile.TemporaryDirectory() as tmpdir:
                    plot_dcns_from_file(input_file, tmpdir, dpi, fmt)
                    # Find the generated file and rename to desired output
                    import glob
                    generated = glob.glob(f'{tmpdir}/*')
                    if generated and output_file:
                        import shutil
                        shutil.move(generated[0], output_file)
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


if __name__ == "__main__":
    main()
