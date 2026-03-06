"""
ND AI Lab - Combined plugin with tabbed interface.

This module combines NDEasyLabel, NDEasyAugment, and NDEasySegment
into a single tabbed interface with shared model and viewer.
"""

import napari
from qtpy.QtWidgets import (
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..models import ImageDataModel
from .nd_easy_augment import NDEasyAugment
from .nd_easy_label import NDEasyLabel
from .nd_easy_segment import NDEasySegment


class NDAILab(QWidget):
    """
    Combined AI Lab widget with tabbed interface.

    Provides Label, Augment, and Segment functionality in separate tabs,
    all sharing the same napari viewer and ImageDataModel.
    """

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        image_data_model: ImageDataModel = None,
    ):
        """
        Initialize the combined AI Lab widget.

        Args:
            viewer: The napari viewer instance.
            image_data_model: Optional ImageDataModel. If not provided, can be set later.
        """
        super().__init__()
        self.viewer = viewer
        self.image_data_model = image_data_model

        # Create sub-apps WITHOUT model initially
        # Model can be set later via set_image_data_model()
        # TODO: Add embedded=True parameter in Phase 2
        self.label_widget = NDEasyLabel(viewer)
        self.augment_widget = NDEasyAugment(viewer)
        self.segment_widget = NDEasySegment(viewer)

        # If model provided, set it now
        if image_data_model is not None:
            self.set_image_data_model(image_data_model)

        # Setup UI
        self._setup_ui()

    def set_image_data_model(self, image_data_model: ImageDataModel):
        """
        Set the shared image data model and propagate to all sub-apps.

        Args:
            image_data_model: The ImageDataModel instance to share.
        """
        self.image_data_model = image_data_model

        # Propagate to all sub-apps
        self.label_widget.set_image_data_model(image_data_model)
        self.augment_widget.set_image_data_model(image_data_model)
        self.segment_widget.set_image_data_model(image_data_model)

        print("✅ Shared image data model set for all tabs")

    def _setup_ui(self):
        """Setup the tabbed user interface."""
        layout = QVBoxLayout(self)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.addTab(self.label_widget, "Label")
        self.tabs.addTab(self.augment_widget, "Augment")
        self.tabs.addTab(self.segment_widget, "Segment")
        layout.addWidget(self.tabs)

        # TODO Phase 2: Add top-level controls (shared directory button)
        # TODO Phase 3: Add central layer management
        # TODO Phase 4: Add sequence viewer connection
