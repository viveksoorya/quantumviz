"""
Cost Landscape Visualization Module

Provides functions for visualizing parameter optimization landscapes
for QAOA (Quantum Approximate Optimization Algorithm) and VQE
(Variational Quantum Eigensolver).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any
import json

VALID_PAULIS = {"I", "X", "Y", "Z"}


def get_examples_dir() -> str:
    """Get the path to the examples directory."""
    module_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(module_dir), "examples")


def validate_qaoa_input(data: Dict[str, Any]) -> List[tuple[int, int]]:
    """
    Validate and parse QAOA input data.

    Args:
        data: Input dictionary from JSON file

    Returns:
        List of edges as (u, v) tuples

    Raises:
        ValueError: If input is invalid
    """
    if "edges" not in data:
        examples_dir = get_examples_dir()
        raise ValueError(
            f"Missing required key 'edges' in input data.\n\n"
            f"QAOA input format:\n"
            f'  {{"edges": [[0, 1], [1, 2], ...]}}\n\n'
            f"Each edge is [u, v] where u and v are qubit indices (0-indexed).\n\n"
            f"See examples: {examples_dir}/"
        )

    edges_raw = data["edges"]
    if not isinstance(edges_raw, list):
        raise ValueError(
            f"'edges' must be a list, got {type(edges_raw).__name__}.\n\n"
            f"Example: {{\"edges\": [[0, 1], [1, 2]]}}"
        )

    edges = []
    for i, edge in enumerate(edges_raw):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError(
                f"Edge {i} must be a pair [u, v], got {edge}.\n\n"
                f"Example: {{\"edges\": [[0, 1], [1, 2]]}}"
            )
        u, v = edge
        if not isinstance(u, int) or not isinstance(v, int):
            raise ValueError(
                f"Edge {i} has non-integer vertices: [{u}, {v}].\n\n"
                f"Vertices must be integers (qubit indices)."
            )
        if u < 0 or v < 0:
            raise ValueError(
                f"Edge {i} has negative vertex: [{u}, {v}].\n\n"
                f"Vertices must be non-negative integers (0-indexed)."
            )
        edges.append((u, v))

    return edges


def validate_vqe_input(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validate and parse VQE input data.

    Args:
        data: Input dictionary from JSON file

    Returns:
        List of Hamiltonian terms

    Raises:
        ValueError: If input is invalid
    """
    if "terms" not in data:
        examples_dir = get_examples_dir()
        raise ValueError(
            f"Missing required key 'terms' in input data.\n\n"
            f"VQE input format:\n"
            f'  {{"terms": [{{"coeff": 0.5, "paulis": ["Z"]}}, ...]}}\n\n'
            f"Each term has:\n"
            f"  - 'coeff': a real number (coefficient)\n"
            f"  - 'paulis': list of Pauli operators (I, X, Y, Z)\n\n"
            f"See examples: {examples_dir}/"
        )

    terms_raw = data["terms"]
    if not isinstance(terms_raw, list):
        raise ValueError(
            f"'terms' must be a list, got {type(terms_raw).__name__}.\n\n"
            f"Example: {{\"terms\": [{{\"coeff\": 0.5, \"paulis\": [\"Z\"]}}]}}"
        )

    terms = []
    for i, term in enumerate(terms_raw):
        if not isinstance(term, dict):
            raise ValueError(
                f"Term {i} must be an object with 'coeff' and 'paulis', got {type(term).__name__}.\n\n"
                f"Example: {{\"coeff\": 0.5, \"paulis\": [\"Z\"]}}"
            )

        coeff = term.get("coeff")
        if coeff is None:
            raise ValueError(
                f"Term {i} missing required key 'coeff'.\n\n"
                f"Example: {{\"coeff\": 0.5, \"paulis\": [\"Z\"]}}"
            )
        if not isinstance(coeff, (int, float)):
            raise ValueError(
                f"Term {i} 'coeff' must be a number, got {type(coeff).__name__}."
            )

        paulis = term.get("paulis", [])
        if not isinstance(paulis, list):
            raise ValueError(
                f"Term {i} 'paulis' must be a list, got {type(paulis).__name__}.\n\n"
                f"Supported Pauli operators: I, X, Y, Z"
            )

        for j, pauli in enumerate(paulis):
            if not isinstance(pauli, str):
                raise ValueError(
                    f"Term {i}, Pauli {j} must be a string, got {type(pauli).__name__}."
                )
            if pauli.upper() not in VALID_PAULIS:
                raise ValueError(
                    f"Term {i}, Pauli '{pauli}' is invalid.\n\n"
                    f"Supported Pauli operators: I, X, Y, Z"
                )

        terms.append({"coeff": float(coeff), "paulis": [p.upper() for p in paulis]})

    return terms


def qaoa_cost(
    gamma: np.ndarray,
    beta: np.ndarray,
    edges: List[tuple[int, int]]
) -> np.ndarray:
    """
    Compute QAOA cost function for a MaxCut problem.

    Args:
        gamma: Array of gamma parameters
        beta: Array of beta parameters
        edges: List of edges in the graph, each edge is (u, v)

    Returns:
        Array of cost values
    """
    g = np.asarray(gamma, dtype=float)
    b = np.asarray(beta, dtype=float)
    n_edges = len(edges)
    if n_edges == 0:
        return np.zeros_like(g, dtype=float)

    cost = np.zeros_like(g, dtype=float)
    for u, v in edges:
        cost += 0.5 - 0.5 * (np.cos(2*g) * (np.cos(2*b)**2 - np.sin(2*b)**2) +
                             np.sin(2*g) * 2*np.sin(2*b))
    return cost / n_edges


def vqe_energy(
    theta: np.ndarray,
    terms: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Compute VQE energy landscape for a molecular Hamiltonian.

    Args:
        theta: Array of variational parameters
        terms: List of Hamiltonian terms, each with "coeff" (real) and "paulis"

    Returns:
        Array of energy values
    """
    t = np.asarray(theta, dtype=float)
    n_terms = len(terms)
    if n_terms == 0:
        return np.zeros_like(t, dtype=float)

    energy = np.zeros_like(t, dtype=float)
    for term in terms:
        coeff = term.get("coeff", 1.0)
        paulis = term.get("paulis", [])
        product = np.ones_like(t)
        for pauli in paulis:
            op = pauli.upper() if isinstance(pauli, str) else "I"
            if op == "Z":
                product *= np.cos(t)
            elif op == "X":
                product *= np.sin(t)
            elif op == "Y":
                product *= np.ones_like(t)
            elif op == "I":
                product *= np.ones_like(t)
        energy += coeff * product
    return energy


def plot_qaoa_landscape(
    edges: List[tuple[int, int]],
    gamma_range: Tuple[float, float] = (0, np.pi),
    beta_range: Tuple[float, float] = (0, np.pi),
    resolution: int = 50,
    output_path: Optional[str] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Plot QAOA cost landscape for a MaxCut problem.

    Args:
        edges: List of edges in the graph, each edge is (u, v)
        gamma_range: Range of gamma values (min, max)
        beta_range: Range of beta values (min, max)
        resolution: Number of points in each dimension
        output_path: Path to save the figure (if None, returns figure object)
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object if output_path is None, else None
    """
    gammas = np.linspace(gamma_range[0], gamma_range[1], resolution)
    betas = np.linspace(beta_range[0], beta_range[1], resolution)
    G, B = np.meshgrid(gammas, betas)
    Z = qaoa_cost(G, B, edges)

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(G, B, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Cost (Cut Value)')
    plt.xlabel('γ (problem parameter)')
    plt.ylabel('β (mixer parameter)')
    n_qubits = max(max(e) for e in edges) + 1 if edges else 0
    plt.title(f'QAOA Cost Landscape: MaxCut ({n_qubits} qubits, {len(edges)} edges)')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi)
        plt.close()
    else:
        return plt.gcf()


def plot_vqe_landscape(
    terms: List[Dict[str, Any]],
    theta_range: Tuple[float, float] = (0, 2*np.pi),
    resolution: int = 100,
    output_path: Optional[str] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Plot VQE energy landscape for a molecular Hamiltonian.

    Args:
        terms: List of Hamiltonian terms with "coeff" and "paulis"
        theta_range: Range of theta values (min, max)
        resolution: Number of points
        output_path: Path to save the figure (if None, returns figure object)
        dpi: Resolution for saved figure

    Returns:
        matplotlib Figure object if output_path is None, else None
    """
    thetas = np.linspace(theta_range[0], theta_range[1], resolution)
    energies = vqe_energy(thetas, terms)

    plt.figure(figsize=(10, 6))
    plt.plot(thetas, energies, 'b-', linewidth=2)
    plt.xlabel('θ (variational parameter)')
    plt.ylabel('Energy')
    plt.title(f'VQE Energy Landscape: {len(terms)} terms')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi)
        plt.close()
    else:
        return plt.gcf()


def main(args: Optional[list] = None) -> None:
    """
    CLI entry point for Cost Landscape visualization.

    Args:
        args: Command line arguments (if None, uses sys.argv)
    """
    import sys

    if args is None:
        args = sys.argv[1:]

    if len(args) < 2:
        examples_dir = get_examples_dir()
        print("Usage: python -m quantumviz.cost_landscape <qaoa|vqe> <input_file> [output_file]")
        print()
        print("QAOA Input Format:")
        print('  {"edges": [[0, 1], [1, 2], ...]}')
        print("  - Each edge [u, v] connects qubits u and v (0-indexed)")
        print()
        print("VQE Input Format:")
        print('  {"terms": [{"coeff": 0.5, "paulis": ["Z"]}, ...]}')
        print("  - 'coeff': real number coefficient")
        print("  - 'paulis': list of Pauli operators (I, X, Y, Z)")
        print()
        print(f"Example files: {examples_dir}/")
        sys.exit(1)

    algorithm = args[0].lower()
    input_file = args[1]
    output_file = args[2] if len(args) > 2 else None

    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_file}: {e}")
        sys.exit(1)

    if not isinstance(data, dict):
        print(f"Error: Input must be a JSON object, got {type(data).__name__}.")
        sys.exit(1)

    if algorithm == "qaoa":
        try:
            edges = validate_qaoa_input(data)
        except ValueError as e:
            print(f"Error: Invalid QAOA input: {e}")
            sys.exit(1)
        if output_file is None:
            output_file = input_file.replace('.json', '_qaoa.png')
        plot_qaoa_landscape(edges, output_path=output_file)
        print(f"Saved: {output_file}")
    elif algorithm == "vqe":
        try:
            terms = validate_vqe_input(data)
        except ValueError as e:
            print(f"Error: Invalid VQE input: {e}")
            sys.exit(1)
        if output_file is None:
            output_file = input_file.replace('.json', '_vqe.png')
        plot_vqe_landscape(terms, output_path=output_file)
        print(f"Saved: {output_file}")
    else:
        print(f"Error: Unknown algorithm '{algorithm}'. Use 'qaoa' or 'vqe'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
