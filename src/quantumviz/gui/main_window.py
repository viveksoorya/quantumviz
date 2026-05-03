"""Main window for quantumviz desktop GUI."""

import shutil
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from quantumviz.gui.code_editor_widget import CodeEditorWidget
from quantumviz.gui.controls_widget import ControlsWidget
from quantumviz.gui.display_widget import DisplayWidget
from quantumviz.gui.workers import CodeVisualizationWorker, VisualizationWorker


class QuantumVizGUI(QMainWindow):
    """Main application window for quantumviz desktop GUI."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.worker: Optional[VisualizationWorker] = None
        self.current_output_path: Optional[str] = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("QuantumViz - Quantum Algorithm Visualization")
        self.setMinimumSize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.input_tabs = QTabWidget()
        self.input_tabs.setMaximumWidth(450)
        self.input_tabs.setMinimumWidth(350)

        self.controls = ControlsWidget()
        self.controls.set_generate_callback(self.generate_visualization)
        self.input_tabs.addTab(self.controls, "File Input")

        self.code_editor = CodeEditorWidget()
        self.code_editor.set_run_callback(self.run_code)
        self.input_tabs.addTab(self.code_editor, "Code Input")

        splitter.addWidget(self.input_tabs)

        self.display = DisplayWidget()
        splitter.addWidget(self.display)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        self._create_menu_bar()
        self._create_status_bar()

    def _create_menu_bar(self):
        menu_bar = QMenuBar()

        file_menu = menu_bar.addMenu("&File")
        open_action = QAction("&Open Input File...", self)
        open_action.triggered.connect(self.controls.browse_file)
        file_menu.addAction(open_action)

        export_action = QAction("&Export Image...", self)
        export_action.triggered.connect(self.export_image)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        viz_menu = menu_bar.addMenu("&Visualization")
        for viz_type in [
            "bloch-sphere",
            "state-city",
            "circuit",
            "dcn",
            "cost-landscape-qaoa",
            "cost-landscape-vqe",
            "dynamic-flow",
        ]:
            action = QAction(viz_type.replace("-", " ").title(), self)
            action.triggered.connect(
                lambda checked, v=viz_type: self._set_viz_type(v)
            )
            viz_menu.addAction(action)

        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        self.setMenuBar(menu_bar)

    def _create_status_bar(self):
        self.status_bar = QStatusBar()
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        self.setStatusBar(self.status_bar)

    def _set_viz_type(self, viz_type: str):
        index = self.controls.viz_type_combo.findText(viz_type)
        if index >= 0:
            self.controls.viz_type_combo.setCurrentIndex(index)

    def generate_visualization(self):
        if self.input_tabs.currentIndex() == 1:
            self.run_code()
            return

        input_file = self.controls.get_input_file()
        if not input_file or not Path(input_file).exists():
            QMessageBox.warning(
                self, "No Input File", "Please select a valid input file first."
            )
            return

        if self.worker and self.worker.isRunning():
            QMessageBox.information(
                self, "Busy", "A visualization is already being generated."
            )
            return

        self.controls.generate_btn.setEnabled(False)
        self.status_label.setText("Generating visualization...")

        viz_type = self.controls.get_viz_type()
        output_format = self.controls.get_format()
        dpi = self.controls.get_dpi()

        self.worker = VisualizationWorker(viz_type, input_file, output_format, dpi)
        self.worker.finished.connect(self.on_visualization_done)
        self.worker.error.connect(self.on_visualization_error)
        self.worker.progress.connect(self.status_label.setText)
        self.worker.start()

    def run_code(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(
                self, "Busy", "A visualization is already being generated."
            )
            return

        code = self.code_editor.get_code()
        if not code.strip():
            QMessageBox.warning(self, "No Code", "Please enter some Qiskit code first.")
            return

        viz_type = self.controls.get_viz_type()
        output_format = self.controls.get_format()
        dpi = self.controls.get_dpi()

        self.code_editor.clear_output()
        self.code_editor.append_output("Running code...")
        self.code_editor.run_btn.setEnabled(False)
        self.controls.generate_btn.setEnabled(False)
        self.status_label.setText("Executing code and generating visualization...")

        self.worker = CodeVisualizationWorker(viz_type, code, output_format, dpi)
        self.worker.finished.connect(self.on_code_visualization_done)
        self.worker.error.connect(self.on_code_visualization_error)
        self.worker.progress.connect(self.status_label.setText)
        self.worker.output.connect(self.code_editor.append_output)
        self.worker.start()

    def on_visualization_done(self, output_path: str, viz_type: str):
        self.current_output_path = output_path
        self.display.display_image(output_path)
        self.status_label.setText(f"Generated {viz_type} visualization")
        self.controls.generate_btn.setEnabled(True)

    def on_visualization_error(self, error_msg: str):
        self.status_label.setText("Error generating visualization")
        QMessageBox.critical(self, "Visualization Error", error_msg)
        self.controls.generate_btn.setEnabled(True)

    def on_code_visualization_done(self, output_path: str, viz_type: str):
        self.current_output_path = output_path
        self.display.display_image(output_path)
        self.status_label.setText(f"Generated {viz_type} visualization from code")
        self.code_editor.append_output("Visualization generated successfully!")
        self.code_editor.run_btn.setEnabled(True)
        self.controls.generate_btn.setEnabled(True)

    def on_code_visualization_error(self, error_msg: str):
        self.status_label.setText("Error generating visualization")
        self.code_editor.append_output(f"ERROR:\n{error_msg}")
        self.code_editor.run_btn.setEnabled(True)
        self.controls.generate_btn.setEnabled(True)

    def export_image(self):
        if not self.current_output_path or not Path(self.current_output_path).exists():
            QMessageBox.warning(
                self, "No Image", "Generate a visualization first before exporting."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            str(Path.home() / f"quantumviz_export.{self.controls.get_format()}"),
            f"Images (*.{self.controls.get_format()});;All Files (*)",
        )

        if file_path:
            try:
                shutil.copy2(self.current_output_path, file_path)
                self.status_label.setText(f"Exported to {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def show_about(self):
        QMessageBox.about(
            self,
            "About QuantumViz",
            "QuantumViz Desktop GUI\n"
            "Version 0.3.0\n\n"
            "Quantum Algorithm Visualization Library\n"
            "© 2024 Vivek Soorya Maadoori",
        )

    def closeEvent(self, event):  # noqa: N802
        if self.worker and self.worker.isRunning():
            self.worker.wait()
        event.accept()
