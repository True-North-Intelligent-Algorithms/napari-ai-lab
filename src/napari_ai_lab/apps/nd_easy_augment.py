"""
ND Easy Augment - Simple augmentation interface.

This module provides a minimal interface to explore and test the augmentation widget.
Currently displays the augmentation widget without additional functionality.
More design thinking needed before implementing full augmentation workflow.
"""

import napari
from qtpy.QtWidgets import QVBoxLayout

from ..models import ImageDataModel
from ..widgets.augmentation_widget import AugmentationParametersGroup
from .base_nd_app import BaseNDApp


class NDEasyAugment(BaseNDApp):
    """
    Simple augmentation app for exploring augmentation parameters.

    Currently just displays the augmentation widget.
    Future work will integrate with the augmentation pipeline.
    """

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        image_data_model: "ImageDataModel",
    ):
        super().__init__(viewer, image_data_model)
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        self.setLayout(QVBoxLayout())

        # Add the augmentation parameters widget
        self.augmentation_widget = AugmentationParametersGroup()
        self.layout().addWidget(self.augmentation_widget)

        # TODO: Connect button signals to actual functionality
        # For now, just showing the widget to explore the UI
        # More thinking needed about the augmentation workflow:
        # - How to integrate with ImageDataModel
        # - Where augmented patches should be saved
        # - How to preview augmentations
        # - Connection to training pipeline

    def _set_image_layer(self, image_layer):
        """
        Set up annotation layer based on the provided image layer.

        For augmentation, we only need the annotations layer (not predictions).
        Augmentation works with annotated data to create training patches.
        """
        self.image_layer = image_layer
        image_data = image_layer.data

        # Load existing labels or create empty ones
        labels_data = self.image_data_model.load_existing_annotations(
            image_data.shape, self.current_image_index
        )

        self.annotation_layer = self.viewer.add_labels(
            labels_data, name="Labels (Persistent)"
        )

        print(
            f"Successfully set up annotation layer for image: {image_layer.name}"
        )

    def get_augmentation_parameters(self):
        """Get current augmentation parameters from the widget."""
        return self.augmentation_widget.get_parameters()
