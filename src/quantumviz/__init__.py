"""
quantumviz - Quantum Algorithm Visualization Library

A Python library for visualizing quantum algorithm states including
Bloch spheres, density matrices, cost landscapes, and circuit diagrams.
"""

from quantumviz._version import __version__
from quantumviz.bloch_multivector import (
    plot_bloch_multivector,
    plot_bloch_multivectors_from_file,
)
from quantumviz.bloch_sphere import (
    parse_angles,
    parse_bloch_vector,
    parse_complex_pair,
    parse_ket,
    plot_bloch_sphere,
)
from quantumviz.circuit_diagram import (
    GATE_COLORS,
)
from quantumviz.circuit_diagram import (
    plot_circuit as plot_circuit,
)
from quantumviz.cost_landscape import (
    plot_qaoa_landscape,
    plot_vqe_landscape,
    qaoa_cost,
    vqe_energy,
)
from quantumviz.dcn import (
    plot_dcn,
    plot_dcns_from_file,
)
from quantumviz.dynamic_flow import (
    plot_dynamic_flow,
    plot_rabi_oscillation,
    plot_time_evolution,
)
from quantumviz.hinton import (
    plot_hinton,
    plot_hintons_from_file,
)
from quantumviz.paulivec import (
    plot_paulivec,
    plot_paulivecs_from_file,
)
from quantumviz.state_city import (
    parse_amplitude,
    plot_state_city,
    state_to_density,
)
from quantumviz.qbeads import (
    plot_qbeads,
    plot_qbeads_from_file,
)

__all__ = [
    "__version__",
    "plot_bloch_sphere",
    "parse_ket",
    "parse_bloch_vector",
    "parse_angles",
    "parse_complex_pair",
    "plot_state_city",
    "parse_amplitude",
    "state_to_density",
    "plot_qaoa_landscape",
    "plot_vqe_landscape",
    "qaoa_cost",
    "vqe_energy",
    "plot_circuit",
    "GATE_COLORS",
    "plot_dynamic_flow",
    "plot_rabi_oscillation",
    "plot_time_evolution",
    "plot_dcn",
    "plot_dcns_from_file",
    "plot_paulivec",
    "plot_paulivecs_from_file",
    "plot_bloch_multivector",
    "plot_bloch_multivectors_from_file",
    "plot_hinton",
    "plot_hintons_from_file",
    "plot_qbeads",
    "plot_qbeads_from_file",
]

# Qiskit bridge (optional, requires qiskit installed)
try:
    from quantumviz import qiskit_bridge  # noqa: F401
except ImportError:
    pass
