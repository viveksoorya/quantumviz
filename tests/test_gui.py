"""Tests for quantumviz GUI module."""

import pytest
from unittest.mock import patch, MagicMock


class TestGUIImports:
    """Test that GUI module imports correctly."""

    def test_import_quantumviz_gui(self):
        """Test that QuantumVizGUI can be imported."""
        from quantumviz.gui import QuantumVizGUI

        assert QuantumVizGUI is not None

    def test_import_workers(self):
        """Test that VisualizationWorker can be imported."""
        from quantumviz.gui.workers import VisualizationWorker

        assert VisualizationWorker is not None

    def test_import_display_widget(self):
        """Test that DisplayWidget can be imported."""
        from quantumviz.gui.display_widget import DisplayWidget

        assert DisplayWidget is not None

    def test_import_controls_widget(self):
        """Test that ControlsWidget can be imported."""
        from quantumviz.gui.controls_widget import ControlsWidget

        assert ControlsWidget is not None


class TestVisualizationWorker:
    """Test the VisualizationWorker class."""

    def test_worker_creation(self):
        """Test that worker can be created with valid parameters."""
        from quantumviz.gui.workers import VisualizationWorker

        worker = VisualizationWorker("bloch-sphere", "test.json", "png", 150)
        assert worker.viz_type == "bloch-sphere"
        assert worker.input_file == "test.json"
        assert worker.output_format == "png"
        assert worker.dpi == 150

    def test_worker_signals_exist(self):
        """Test that worker has the required signals."""
        from quantumviz.gui.workers import VisualizationWorker

        worker = VisualizationWorker("bloch-sphere", "test.json")
        assert hasattr(worker, "finished")
        assert hasattr(worker, "error")
        assert hasattr(worker, "progress")


class TestCLIIntegration:
    """Test CLI integration for gui command."""

    def test_gui_command_exists(self):
        """Test that gui command is available in CLI."""
        from click.testing import CliRunner
        from quantumviz.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "gui" in result.output
        assert "Launch the quantumviz desktop GUI" in result.output
