{
  "_comment": "Example input for VQE Hamiltonian",
  "_description": "Structure your Hamiltonian as a sum of Pauli terms:",
  "format": {
    "terms": [
      {
        "coeff": "real number - coefficient for this term",
        "paulis": ["list of Pauli operators: I, X, Y, or Z (one per qubit)"]
      }
    ]
  },
  "example": {
    "terms": [
      {"coeff": -1.0, "paulis": []},
      {"coeff": 0.5, "paulis": ["Z"]},
      {"coeff": 0.25, "paulis": ["Z", "Z"]},
      {"coeff": 0.1, "paulis": ["X", "Y"]}
    ]
  },
  "supported_paulis": ["I (identity)", "X (Pauli-X)", "Y (Pauli-Y)", "Z (Pauli-Z)"]
}
