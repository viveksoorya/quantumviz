"""Code editor widget for writing and executing Qiskit code."""

from typing import Optional

from PyQt6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat
from PyQt6.QtWidgets import (  # noqa: I001
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class PythonHighlighter(QSyntaxHighlighter):
    """Basic Python syntax highlighter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#0000FF"))
        keywords = [
            "def", "class", "return", "if", "elif", "else", "for", "while",
            "import", "from", "as", "try", "except", "with", "yield", "pass",
            "break", "continue", "and", "or", "not", "in", "is", "lambda",
        ]
        for kw in keywords:
            pattern = f"\\b{kw}\\b"
            self.highlighting_rules.append((pattern, keyword_format))

        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#008000"))
        self.highlighting_rules.append(('"[^"]*"', string_format))
        self.highlighting_rules.append(("'[^']*'", string_format))

        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#808080"))
        self.highlighting_rules.append(("#[^\n]*", comment_format))

        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#FF0000"))
        self.highlighting_rules.append(("\\b\\d+\\b", number_format))

    def highlightBlock(self, text):  # noqa: N802
        for pattern, fmt in self.highlighting_rules:
            import re
            for match in re.finditer(pattern, text):
                start = match.start()
                length = match.end() - start
                self.setFormat(start, length, fmt)


class CodeEditorWidget(QWidget):
    """Widget for editing and executing Qiskit code."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.init_ui()
        self.load_template(0)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        template_layout = QHBoxLayout()
        template_label = QLabel("Template:")
        self.template_combo = QComboBox()
        self.template_combo.addItems([
            "Bell State (Bloch Sphere)",
            "Quantum Circuit (2-qubit)",
            "Multi-qubit State (DCN)",
            "State City (3-qubit GHZ)",
            "Dynamic Flow (Single Qubit)",
        ])
        self.template_combo.currentIndexChanged.connect(self.load_template)
        template_layout.addWidget(template_label)
        template_layout.addWidget(self.template_combo, 1)

        self.load_template_btn = QPushButton("Load")
        self.load_template_btn.clicked.connect(
            lambda: self.load_template(self.template_combo.currentIndex())
        )
        template_layout.addWidget(self.load_template_btn)
        layout.addLayout(template_layout)

        code_group = QGroupBox("Qiskit Code")
        code_layout = QVBoxLayout(code_group)

        self.code_editor = QPlainTextEdit()
        self.code_editor.setFont(QFont("Courier", 10))
        self.code_editor.setMinimumHeight(200)
        self.highlighter = PythonHighlighter(self.code_editor.document())
        code_layout.addWidget(self.code_editor)

        self.run_btn = QPushButton("Run Code & Generate Visualization")
        self.run_btn.setMinimumHeight(35)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1e7e34;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        code_layout.addWidget(self.run_btn)
        layout.addWidget(code_group)

        output_group = QGroupBox("Output / Errors")
        output_layout = QVBoxLayout(output_group)

        self.output_console = QTextEdit()
        self.output_console.setFont(QFont("Courier", 9))
        self.output_console.setMaximumHeight(150)
        self.output_console.setReadOnly(True)
        output_layout.addWidget(self.output_console)
        layout.addWidget(output_group)

    def load_template(self, index: int):
        templates = [
            # Bell State for Bloch Sphere
            """from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Create a Bell state
qc = QuantumCircuit(1)
qc.h(0)
# For Bloch sphere, we need a single-qubit state
# Let's create a superposition
state = Statevector(qc)
""",
            # Quantum Circuit
            """from qiskit import QuantumCircuit

# Create a 2-qubit quantum circuit
qc = QuantumCircuit(2)
qc.h(0)        # Hadamard on qubit 0
qc.cx(0, 1)    # CNOT from qubit 0 to 1
qc.rz(1.57, 1) # Rotation around Z axis on qubit 1
""",
            # Multi-qubit State for DCN
            """from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Create a 3-qubit GHZ state
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
state = Statevector(qc)
""",
            # State City (3-qubit GHZ)
            """from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Create a 3-qubit GHZ state for State City visualization
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
state = Statevector(qc)
""",
            # Dynamic Flow
            """from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np

# Create a time-evolving single qubit state
# This creates a list of states at different times
states = []
for theta in np.linspace(0, np.pi, 10):
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    states.append(Statevector(qc))
""",
        ]

        if 0 <= index < len(templates):
            self.code_editor.setPlainText(templates[index])

    def get_code(self) -> str:
        return str(self.code_editor.toPlainText())

    def get_last_statement_result(self) -> str:
        code = self.get_code()
        lines = code.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("from") and not line.startswith("import"):
                return line
        return ""

    def append_output(self, text: str):
        self.output_console.append(text)

    def clear_output(self):
        self.output_console.clear()

    def set_run_callback(self, callback):
        self.run_btn.clicked.connect(callback)
