"""Display widget for showing visualization results."""

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class DisplayWidget(QWidget):
    """Widget for displaying visualization images."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_pixmap: Optional[QPixmap] = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_label = QLabel("No visualization generated yet.\nSelect a visualization type and input file, then click Generate.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 14px;
                padding: 40px;
            }
        """)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)

        self.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
        """)

    def display_image(self, image_path: str):
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_label.setText(f"Failed to load image: {Path(image_path).name}")
            return

        self.current_pixmap = pixmap
        self._update_display()

    def _update_display(self):
        if self.current_pixmap is None:
            return

        scaled_pixmap = self.current_pixmap.scaled(
            self.scroll_area.viewport().size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled_pixmap)

    def clear(self):
        self.current_pixmap = None
        self.image_label.setText("No visualization generated yet.\nSelect a visualization type and input file, then click Generate.")
        self.image_label.setPixmap(QPixmap())

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._update_display()
