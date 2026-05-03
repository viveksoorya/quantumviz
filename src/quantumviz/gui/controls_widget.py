"""Controls widget for quantumviz GUI."""

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ControlsWidget(QWidget):
    """Widget containing input controls for visualization generation."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.input_file_edit: QLineEdit
        self.viz_type_combo: QComboBox
        self.dpi_spin: QSpinBox
        self.format_combo: QComboBox
        self.generate_btn: QPushButton
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        viz_group = QGroupBox("Visualization")
        viz_layout = QFormLayout(viz_group)

        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            "bloch-sphere",
            "state-city",
            "circuit",
            "dcn",
            "cost-landscape-qaoa",
            "cost-landscape-vqe",
            "dynamic-flow",
        ])
        viz_layout.addRow("Type:", self.viz_type_combo)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["png", "pdf", "svg"])
        viz_layout.addRow("Format:", self.format_combo)

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(150)
        self.dpi_spin.setSingleStep(50)
        viz_layout.addRow("DPI:", self.dpi_spin)

        layout.addWidget(viz_group)

        file_group = QGroupBox("Input File")
        file_layout = QVBoxLayout(file_group)

        file_select_layout = QHBoxLayout()
        self.input_file_edit = QLineEdit()
        self.input_file_edit.setPlaceholderText("Select input file...")
        self.input_file_edit.setReadOnly(True)
        file_select_layout.addWidget(self.input_file_edit)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_file)
        file_select_layout.addWidget(self.browse_btn)

        file_layout.addLayout(file_select_layout)

        self.file_type_label = QLabel("Supported: .json, .qpy")
        self.file_type_label.setStyleSheet("color: #666; font-size: 11px;")
        file_layout.addWidget(self.file_type_label)

        layout.addWidget(file_group)

        self.generate_btn = QPushButton("Generate Visualization")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        layout.addWidget(self.generate_btn)

        layout.addStretch()

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input File",
            str(Path.home()),
            "All Supported (*.json *.qpy);;JSON Files (*.json);;QPY Files (*.qpy);;All Files (*)",
        )
        if file_path:
            self.input_file_edit.setText(file_path)
            self.file_type_label.setText(f"Selected: {Path(file_path).name}")

    def get_input_file(self) -> str:
        return self.input_file_edit.text()

    def get_viz_type(self) -> str:
        return self.viz_type_combo.currentText()

    def get_format(self) -> str:
        return self.format_combo.currentText()

    def get_dpi(self) -> int:
        return self.dpi_spin.value()

    def set_generate_callback(self, callback):
        self.generate_btn.clicked.connect(callback)
