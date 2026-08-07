"""
Helper launcher for NDAILab used by scripts.

Expose a simple function `launch_nd_ai_lab(viewer, *, parent_dir, viewer_type,
axes_to_collapse=None, axis_types=None)` that performs the model creation,
IO configuration, attaches sequence viewer if requested, and inserts the
NDAILab widget into the provided napari Viewer. This keeps the top-level
`src/launch_nd_ai_lab.py` script small and makes it easy for third-party
projects to reuse.
"""

from pathlib import Path

from napari.viewer import Viewer

from ..apps.nd_ai_lab import NDAILab
from ..models import ImageDataModel
from ..nd_sequence_viewer import NDSequenceViewer
from ..nd_stacked_sequence_viewer import NDStackedSequenceViewer
from .register_all import register_all as _register_all


def launch_nd_ai_lab(
    viewer: Viewer,
    parent_dir: Path | str,
    viewer_type: str = "none",
    axes_to_collapse: str | None = None,
    axis_types: str | None = None,
    register_all: bool = False,
):
    """Create ImageDataModel, configure IO, attach viewers, and show NDAILab.

    Returns the tuple (nd_ai_lab_widget, sequence_viewer_or_None, model).
    """
    parent_dir = Path(parent_dir)

    # Optionally register all augmenters/segmenters so the UI lists them.
    # Many scripts prefer selective registration; set register_all=True for the
    # convenience mode that mirrors the napari-plugin behavior.
    if register_all:
        _register_all()

    # Create model
    model = ImageDataModel(parent_dir)
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

    # Create combined widget WITH model
    nd_ai_lab_widget = NDAILab(
        viewer, model, axes_to_collapse=axes_to_collapse
    )
    viewer.window.add_dock_widget(
        nd_ai_lab_widget, area="right", name="AI Lab"
    )

    seq_widget = None
    # Attach sequence viewer if requested
    if viewer_type in ("stacked", "sequence"):
        seq_widget = (
            NDStackedSequenceViewer(viewer)
            if viewer_type == "stacked"
            else NDSequenceViewer(viewer)
        )
        viewer.window.add_dock_widget(
            seq_widget, name="Sequence Viewer", area="bottom"
        )
        nd_ai_lab_widget.connect_sequence_viewer(seq_widget)
        seq_widget.set_image_data_model(model)
    else:
        # Load first image and set central image layer
        image_data = model.load_image(0)
        image_layer = viewer.add_image(
            image_data, name="Image", scale=model.get_scale()
        )
        nd_ai_lab_widget._set_image_layer(image_layer)

    nd_ai_lab_widget.segment_widget.automatic_mode_btn.setChecked(True)

    return nd_ai_lab_widget, seq_widget, model
