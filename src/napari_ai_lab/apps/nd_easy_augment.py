"""
ND Easy Augment - Augmentation interface with dynamic parameter controls.

This module provides an interface for data augmentation with support for
different augmentation frameworks (SimpleAugmenter, AlbumentationsAugmenter, etc.).
Uses nd_operation_widget for dynamic parameter controls based on selected augmenter.
"""

import napari
from qtpy.QtWidgets import QComboBox, QLabel, QVBoxLayout

from ..Augmenters import AugmenterBase
from ..models import ImageDataModel
from ..widgets.nd_operation_widget import NDOperationWidget
from .base_nd_app import BaseNDApp


class NDEasyAugment(BaseNDApp):
    """
    Augmentation app with dynamic parameter controls.

    Provides combo box selection of different augmenters and displays
    their parameters using nd_operation_widget.
    """

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        image_data_model: "ImageDataModel",
    ):
        super().__init__(viewer, image_data_model)
        self.augmenter_cache = {}  # Cache for augmenter instances
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        self.setLayout(QVBoxLayout())

        # Add Augmenter selection
        self.augmenter_label = QLabel("Augmenter:")
        self.layout().addWidget(self.augmenter_label)

        self.augmenter_combo = QComboBox()
        self.augmenter_combo.currentTextChanged.connect(
            self._on_augmenter_changed
        )
        self.layout().addWidget(self.augmenter_combo)

        # Create augmentation parameter form (separate from base parameter_form)
        # This parses "augmentation" param_type instead of "inference"
        self.augmentation_form = NDOperationWidget(
            param_type_to_parse="augmentation"
        )
        self.layout().addWidget(self.augmentation_form)

        # Populate augmenter combo with registered frameworks
        self._populate_augmenter_combo()

    def _populate_augmenter_combo(self):
        """Populate the augmenter combo box with registered frameworks."""
        # Clear existing items
        self.augmenter_combo.clear()

        # Get registered frameworks
        frameworks = AugmenterBase.get_registered_frameworks()

        if frameworks:
            # Sort augmenter names alphabetically
            framework_names = sorted(frameworks.keys())
            self.augmenter_combo.addItems(framework_names)

            # Trigger selection of first augmenter to populate the form
            if self.augmenter_combo.count() > 0:
                self.augmenter_combo.setCurrentIndex(0)
                self._on_augmenter_changed(self.augmenter_combo.currentText())
        else:
            self.augmenter_combo.addItem("No augmenters available")

    def _on_augmenter_changed(self, augmenter_name: str):
        """Handle augmenter selection change."""
        if augmenter_name == "No augmenters available":
            return

        print(f"Selected augmenter: {augmenter_name}")

        # Check if we already have this augmenter in cache
        if augmenter_name in self.augmenter_cache:
            augmenter = self.augmenter_cache[augmenter_name]
            print(f"Using cached augmenter: {augmenter_name}")
        else:
            # Get the augmenter class from registry
            augmenter_class = AugmenterBase.get_framework(augmenter_name)

            if augmenter_class is None:
                print(f"Warning: Could not find augmenter: {augmenter_name}")
                return

            # Create new instance and cache it
            augmenter = augmenter_class()
            self.augmenter_cache[augmenter_name] = augmenter
            print(f"Created new augmenter instance: {augmenter_name}")
            print(f"Augmenter type: {type(augmenter)}")
            print(
                f"Augmenter has get_parameters_dict: {hasattr(augmenter, 'get_parameters_dict')}"
            )
            if hasattr(augmenter, "get_parameters_dict"):
                print(
                    f"Augmenter parameters: {augmenter.get_parameters_dict()}"
                )

        # Store current augmenter
        self.augmenter = augmenter

        print("About to set augmenter in augmentation form...")
        # Set the new augmenter in the form - this rebuilds the UI with new parameters
        self.augmentation_form.set_nd_operation(self.augmenter)
        print("Augmentation form updated successfully")

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
        """Get current augmentation parameters from the selected augmenter."""
        if hasattr(self, "augmenter") and self.augmenter is not None:
            return self.augmenter.get_parameters_dict()
        return {}
