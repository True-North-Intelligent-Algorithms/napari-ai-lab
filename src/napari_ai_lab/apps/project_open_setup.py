"""
Project open dialog + setup logic for ND AI Lab.

The dialog collects:
- Project directory
- Viewer type ("none", "stacked", "sequence")
- axes_to_collapse (e.g. "C" or empty)
- axis_types (e.g. "NYXC" or empty for auto-detect)

``apply_project_setup`` then wires everything onto an already-shown NDAILab.

Keep this module separate from nd_easy_label so the open flow can be swapped
out (different dialog, headless config, etc.) without touching the label UI.
"""

from pathlib import Path

from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
)

from ..models import ImageDataModel
from ..nd_sequence_viewer import NDSequenceViewer
from ..nd_stacked_sequence_viewer import NDStackedSequenceViewer


class ProjectOpenDialog(QDialog):
    """Dialog asking for project directory, viewer type, axes_to_collapse, axis_types."""

    VIEWER_TYPES = ["none", "stacked", "sequence"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Open Project")

        form = QFormLayout(self)

        # Directory + Browse
        dir_row = QHBoxLayout()
        self.dir_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse)
        dir_row.addWidget(self.dir_edit)
        dir_row.addWidget(browse_btn)
        form.addRow("Project directory:", dir_row)

        # Viewer type
        self.viewer_combo = QComboBox()
        self.viewer_combo.addItems(self.VIEWER_TYPES)
        form.addRow("Viewer type:", self.viewer_combo)

        # axes_to_collapse
        self.axes_to_collapse_edit = QLineEdit()
        self.axes_to_collapse_edit.setPlaceholderText(
            'e.g. "C" (leave blank for none)'
        )
        form.addRow("Axes to collapse:", self.axes_to_collapse_edit)

        # axis_types
        self.axis_types_edit = QLineEdit()
        self.axis_types_edit.setPlaceholderText(
            'e.g. "NYXC" (leave blank for auto)'
        )
        form.addRow("Axis types:", self.axis_types_edit)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _on_browse(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Project Directory"
        )
        if directory:
            self.dir_edit.setText(directory)

    def get_values(self) -> dict:
        """Return the dialog selections as a dict."""
        return {
            "directory": self.dir_edit.text().strip(),
            "viewer_type": self.viewer_combo.currentText(),
            "axes_to_collapse": self.axes_to_collapse_edit.text().strip()
            or None,
            "axis_types": self.axis_types_edit.text().strip() or None,
        }


def apply_project_setup(nd_ai_lab, values: dict):
    """
    Apply the dialog values to an already-shown NDAILab widget.

    Mirrors the setup steps performed by launch_nd_ai_lab.py:
    - Create ImageDataModel for the chosen directory
    - Configure annotation / prediction IO if a stacked viewer is used
    - Attach sequence viewer (if requested)
    - Otherwise load first image and run _set_image_layer

    Args:
        nd_ai_lab: The NDAILab widget to configure.
        values: Dict from ProjectOpenDialog.get_values().
    """
    directory = values["directory"]
    if not directory:
        return

    viewer_type = values["viewer_type"]
    axes_to_collapse = values["axes_to_collapse"]
    axis_types = values["axis_types"]

    # Create model
    model = ImageDataModel(Path(directory))
    model.axis_types = axis_types

    # Stacked-viewer mode needs stacked IO + YX save granularity
    if viewer_type == "stacked":
        model.set_annotation_io_type(
            "stacked_sequence", axes_to_collapse=axes_to_collapse
        )
        model.set_prediction_io_type(
            "stacked_sequence", axes_to_collapse=axes_to_collapse
        )
        model.set_annotation_save_granularity("YX")
        model.set_prediction_save_granularity("YX")

    # Push model + axes_to_collapse into the widget
    nd_ai_lab.axes_to_collapse = axes_to_collapse
    nd_ai_lab.label_widget.axes_to_collapse = axes_to_collapse
    nd_ai_lab.augment_widget.axes_to_collapse = axes_to_collapse
    nd_ai_lab.segment_widget.axes_to_collapse = axes_to_collapse
    nd_ai_lab.set_image_data_model(model)

    # Attach sequence viewer or load first image directly
    viewer = nd_ai_lab.viewer
    if viewer_type in ("stacked", "sequence"):
        seq = (
            NDStackedSequenceViewer(viewer)
            if viewer_type == "stacked"
            else NDSequenceViewer(viewer)
        )
        viewer.window.add_dock_widget(
            seq, name="Sequence Viewer", area="bottom"
        )
        nd_ai_lab.connect_sequence_viewer(seq)
        seq.set_image_data_model(model)
    else:
        image_data = model.load_image(0)
        image_layer = viewer.add_image(
            image_data, name="Image", scale=model.get_scale()
        )
        nd_ai_lab._set_image_layer(image_layer)

    print(f"✅ Project loaded: {directory} (viewer={viewer_type})")
