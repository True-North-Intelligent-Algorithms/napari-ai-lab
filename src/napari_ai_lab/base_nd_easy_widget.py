"""
Base class for NDEasy segmentation widgets.

This module provides a base class containing common functionality shared between
NDEasyLabel and NDEasySegment widgets.
"""

import napari
from qtpy.QtWidgets import (
    QWidget,
)

from .writers import get_writer


class BaseNDEasyWidget(QWidget):
    """
    Base class for ND Easy segmentation widgets.

    This class contains common functionality shared between NDEasyLabel and NDEasySegment,
    including segmenter management, image loading, layer setup, and label persistence.

    Common attributes and methods will be moved here progressively to reduce code duplication.
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """
        Initialize the base widget with common setup.

        Args:
            viewer: The napari viewer instance.
        """
        super().__init__()
        self.viewer = viewer

        # Initialize layer references (common to both widgets)
        self.image_layer = None
        self.label_layer = None
        self.points_layer = None
        self.shapes_layer = None

        # Initialize label counter (common to both widgets)
        self.current_label_num = 1

        # Track current image context (common to both widgets)
        self.current_image_path = None
        self.current_parent_directory = None

        # Signal processing state protection (common to both widgets)
        self._processing_image_change = False

        # Initialize label writer (common to both widgets)
        self.label_writer = get_writer("numpy")

        # Segmenter management (common to both widgets)
        self.segmenter = None

    # === COMMON METHODS TO BE IMPLEMENTED ===
    # These methods exist in both NDEasyLabel and NDEasySegment with similar/identical implementations

    def _populate_segmenter_combo(self):
        """Populate the segmenter combo box with registered frameworks."""
        # TODO: Move implementation from both widgets
        raise NotImplementedError("To be implemented in next step")

    def _on_segmenter_changed(self, segmenter_name):
        """Handle changes to the segmenter selection."""
        # TODO: Move implementation from both widgets
        raise NotImplementedError("To be implemented in next step")

    def _on_parameters_changed(self, parameters):
        """Handle changes to segmenter parameters."""
        # TODO: Move implementation from both widgets
        raise NotImplementedError("To be implemented in next step")

    def _on_points_changed(self, event):
        """Handle points layer data changes - interactive segmentation."""
        # TODO: Move implementation from both widgets
        raise NotImplementedError("To be implemented in next step")

    def _on_open_directory(self):
        """Open a file dialog to select an image directory."""
        # TODO: Move implementation from both widgets
        raise NotImplementedError("To be implemented in next step")

    def load_image_directory(self, directory):
        """Load images from the selected directory into napari."""
        # TODO: Move implementation from both widgets
        raise NotImplementedError("To be implemented in next step")

    def _set_image_layer(self, image_layer):
        """Set up annotation layers based on the provided image layer."""
        # TODO: Move implementation from both widgets
        raise NotImplementedError("To be implemented in next step")

    def _load_existing_labels(self, image_shape):
        """Load existing labels or create empty ones."""
        # TODO: Move implementation from both widgets
        raise NotImplementedError("To be implemented in next step")

    def _save_current_labels(self):
        """Save the current labels using the configured writer."""
        # TODO: Move implementation from both widgets (nd_easy_segment version)
        raise NotImplementedError("To be implemented in next step")

    def save_labels_now(self):
        """Public method to manually save current labels immediately."""
        # TODO: Move implementation from nd_easy_label
        raise NotImplementedError("To be implemented in next step")

    def set_writer(self, writer_type: str, **kwargs):
        """Change the label writer type."""
        # TODO: Move implementation from nd_easy_label
        raise NotImplementedError("To be implemented in next step")

    def connect_sequence_viewer(self, sequence_viewer):
        """Connect to sequence viewer for automatic layer updates."""
        # TODO: Move implementation from both widgets
        raise NotImplementedError("To be implemented in next step")

    def _on_sequence_image_changed(
        self, image_layer, image_path, parent_directory
    ):
        """Handle sequence viewer image changes."""
        # TODO: Move implementation from both widgets
        raise NotImplementedError("To be implemented in next step")

    # === METHODS UNIQUE TO SPECIFIC WIDGETS ===
    # These methods will remain in the individual widget classes

    # NDEasyLabel specific:
    # - _on_shapes_changed()
    # - _cleanup_existing_layers()
    # - _process_image_change()
    # - _on_click()

    # NDEasySegment specific:
    # - _setup_ui()
    # - _on_mode_changed()
    # - _update_mode_ui()
    # - is_interactive_mode()
    # - _filter_segmenters_for_mode()
    # - _on_segment_current()
    # - _on_segment_all()
    # - _segment_image_automatically()
    # - _setup_interactive_layers()

    # === COMMON ATTRIBUTES THAT WILL BE MANAGED HERE ===
    # - self.viewer
    # - self.image_layer, self.label_layer, self.points_layer, self.shapes_layer
    # - self.current_label_num
    # - self.current_image_path, self.current_parent_directory
    # - self._processing_image_change
    # - self.label_writer
    # - self.segmenter

    # === WIDGET-SPECIFIC ATTRIBUTES ===
    # NDEasyLabel:
    # - self.parameter_form (SegmenterWidget)
    # - self.segmenter_combo, self.segmenter_label
    # - self.dir_btn, btn

    # NDEasySegment:
    # - self.parameter_form (SegmenterWidget)
    # - self.segmenter_combo, self.segmenter_label
    # - self.dir_btn
    # - Mode-specific UI elements (radio buttons, groups, etc.)
