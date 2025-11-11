"""
Unified ND Easy Segmentation Widget.

This module provides a unified interface for both interactive (point/shape-based)
and automatic (full plane/volume) segmentation workflows.
"""

import napari
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from .base_nd_easy_widget import BaseNDEasyWidget
from .models import ImageDataModel
from .Segmenters.GlobalSegmenters import GlobalSegmenterBase


class NDEasySegment(BaseNDEasyWidget):
    """
    Unified segmentation widget supporting both interactive and automatic modes.

    Modes:
    - Interactive Mode: Point/shape-based segmentation (like nd_easy_label)
    - Automatic Mode: Full plane/volume segmentation (like nd_easy_segment)
    """

    def __init__(
        self, viewer: "napari.viewer.Viewer", image_data_model: ImageDataModel
    ):
        super().__init__(viewer, image_data_model)

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        """Setup the unified user interface."""
        main_layout = QVBoxLayout(self)

        # === Mode Selection Group ===
        mode_group = QGroupBox("Segmentation Mode")
        mode_layout = QHBoxLayout(mode_group)

        self.interactive_mode_btn = QRadioButton("Interactive (Points/Shapes)")
        self.automatic_mode_btn = QRadioButton("Automatic (Full Image)")
        self.interactive_mode_btn.setChecked(True)  # Default to interactive

        self.interactive_mode_btn.toggled.connect(self._on_mode_changed)
        self.automatic_mode_btn.toggled.connect(self._on_mode_changed)

        mode_layout.addWidget(self.interactive_mode_btn)
        mode_layout.addWidget(self.automatic_mode_btn)
        main_layout.addWidget(mode_group)

        # === Common Controls ===
        # Directory selection
        self.dir_btn = QPushButton("Open Image Directory")
        self.dir_btn.clicked.connect(self._on_open_directory)
        main_layout.addWidget(self.dir_btn)

        # Segmenter selection
        self.segmenter_label = QLabel("Segmenter:")
        main_layout.addWidget(self.segmenter_label)

        self.segmenter_combo = QComboBox()
        self.segmenter_combo.currentTextChanged.connect(
            self._on_segmenter_changed
        )
        main_layout.addWidget(self.segmenter_combo)

        # Parameter form widget (from base)
        main_layout.addWidget(self._create_parameter_form())

        # === Mode-Specific Controls ===
        # Automatic mode controls
        self.auto_controls_group = QGroupBox("Automatic Segmentation")
        auto_layout = QVBoxLayout(self.auto_controls_group)

        self.segment_current_btn = QPushButton("Segment Current Image")
        self.segment_current_btn.clicked.connect(self._on_segment_current)
        auto_layout.addWidget(self.segment_current_btn)

        self.segment_all_btn = QPushButton("Segment All Images")
        self.segment_all_btn.clicked.connect(self._on_segment_all)
        auto_layout.addWidget(self.segment_all_btn)

        main_layout.addWidget(self.auto_controls_group)

        # Interactive mode info (no additional controls needed - uses napari layers)
        self.interactive_info_group = QGroupBox("Interactive Segmentation")
        interactive_layout = QVBoxLayout(self.interactive_info_group)

        info_label = QLabel(
            "Click points or draw shapes on the image to segment."
        )

        interactive_layout.addWidget(info_label)

        main_layout.addWidget(self.interactive_info_group)

        # Initialize
        self._populate_segmenter_combo()
        self._update_mode_ui()

    def _on_mode_changed(self):
        """Handle mode change between Interactive and Automatic."""
        self._update_mode_ui()
        print(
            f"Mode changed to: {'Interactive' if self.is_interactive_mode() else 'Automatic'}"
        )

    def _update_mode_ui(self):
        """Update UI visibility based on current mode."""
        is_interactive = self.is_interactive_mode()

        # Show/hide mode-specific controls
        self.auto_controls_group.setVisible(not is_interactive)
        self.interactive_info_group.setVisible(is_interactive)

        # Update segmenter filtering if needed
        self._filter_segmenters_for_mode()

    def is_interactive_mode(self):
        """Check if currently in interactive mode."""
        return self.interactive_mode_btn.isChecked()

    def _filter_segmenters_for_mode(self):
        """Filter available segmenters based on current mode."""
        # For now, show all segmenters in both modes
        # Could be extended to filter segmenters by capabilities

    def _populate_segmenter_combo(self):
        """Populate the segmenter combo box with registered frameworks."""
        self.segmenter_combo.clear()
        frameworks = GlobalSegmenterBase.get_registered_frameworks()

        if frameworks:
            # Custom ordering with Square2D first for interactive mode
            framework_names = list(frameworks.keys())
            if "Square2D" in framework_names:
                framework_names.remove("Square2D")
                ordered_names = ["Square2D"] + sorted(framework_names)
            else:
                ordered_names = sorted(framework_names)

            for name in ordered_names:
                self.segmenter_combo.addItem(name)

            if self.segmenter_combo.count() > 0:
                self.segmenter_combo.setCurrentIndex(0)
                self._on_segmenter_changed(self.segmenter_combo.currentText())
        else:
            self.segmenter_combo.addItem("No segmenters available")
            self.segmenter_combo.setEnabled(False)

    # === Interactive Mode Methods ===
    def _on_points_changed(self, event):
        """Handle points layer data changes - interactive segmentation."""
        if not self.is_interactive_mode():
            return

        points_layer = event.source
        if event.action == "added" and len(points_layer.data) > 0:
            latest_point = points_layer.data[-1]
            print(f"Point added at location: {latest_point}")

            if self.image_layer is None:
                print("No image layer available")
                return

            image_data = self.image_layer.data

            # Ensure segmenter is synced with current parameters
            if hasattr(self, "segmenter") and self.segmenter is not None:
                self.segmenter = self.parameter_form.sync_segmenter_instance(
                    self.segmenter
                )

            try:
                mask = self.segmenter.segment(
                    image_data,
                    points=[latest_point],
                    shapes=None,
                    parent_directory=self.image_data_model.parent_directory,
                )

                self.label_layer.data[mask] = self.current_label_num
                print(
                    f"Added segmentation with label {self.current_label_num}"
                )
                self.current_label_num += 1
                self.label_layer.refresh()

            except (
                AttributeError,
                ValueError,
                TypeError,
                RuntimeError,
                IndexError,
            ) as e:
                print(f"Error during segmentation: {e}")

    def _get_current_slice_indices(self, selected_axis):
        """Get indices for current slice based on selected axis mode."""
        if selected_axis == "YX":
            return self.viewer.dims.current_step[:-2] + (
                slice(None),
                slice(None),
            )
        elif selected_axis == "ZYX":
            return self.viewer.dims.current_step[:-3] + (
                slice(None),
                slice(None),
                slice(None),
            )
        else:
            return self.viewer.dims.current_step[:-2] + (
                slice(None),
                slice(None),
            )

    # === Automatic Mode Methods ===
    def _on_segment_current(self):
        """Segment the current image automatically."""
        if self.image_layer is None:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return

        print("Segmenting current image...")

        # Print the axis mode the user chose
        selected_axis = self.parameter_form.get_selected_axis()
        print(f"User selected axis mode: {selected_axis}")

        # Extract current slice based on selected axis mode
        indices = self._get_current_slice_indices(selected_axis)
        current_yx_slice = self.image_layer.data[indices]

        print(f"Current YX slice shape: {current_yx_slice.shape}")

        self._segment_image_automatically(current_yx_slice)

    def _on_segment_all(self):
        """Segment all images in the directory automatically."""
        if self.image_layer is None:
            QMessageBox.warning(self, "Warning", "No images loaded")
            return

        print("Segmenting all images...")
        # Implementation for batch processing would go here
        QMessageBox.information(
            self, "Info", "Batch segmentation not yet implemented"
        )

    def _segment_image_automatically(self, image_data):
        """Perform automatic segmentation on image data."""
        if not hasattr(self, "segmenter") or self.segmenter is None:
            QMessageBox.warning(self, "Warning", "No segmenter selected")
            return

        try:
            # Ensure segmenter is synced with current parameters
            self.segmenter = self.parameter_form.sync_segmenter_instance(
                self.segmenter
            )

            # Call segmenter without points/shapes for automatic segmentation
            mask = self.segmenter.segment(
                image_data,
                points=None,
                shapes=None,
                parent_directory=self.image_data_model.parent_directory,
            )

            # Apply mask to labels layer
            if self.label_layer is not None:
                selected_axis = self.parameter_form.get_selected_axis()
                indices = self._get_current_slice_indices(selected_axis)
                self.label_layer.data[indices] = mask  # self.current_label_num
                self.current_label_num += 1
                self.label_layer.refresh()
                print("Automatic segmentation completed")
            else:
                print("No label layer available")

        except (
            AttributeError,
            ValueError,
            TypeError,
            RuntimeError,
            IndexError,
        ) as e:
            print(f"Error during automatic segmentation: {e}")
            QMessageBox.critical(
                self, "Error", f"Segmentation failed: {str(e)}"
            )

    # === Common Methods (from original nd_easy_label) ===
    # _on_open_directory and load_image_directory inherited from BaseNDEasyWidget

    def _set_image_layer(self, image_layer):
        """Set up annotation layers based on the provided image layer."""
        try:
            self.image_layer = image_layer
            image_data = image_layer.data

            # Load existing labels or create empty ones
            labels_data = self._load_existing_annotations(image_data.shape)
            self.label_layer = self.viewer.add_labels(
                labels_data, name="Labels (Persistent)"
            )

            # Only create interactive layers if in interactive mode
            if self.is_interactive_mode():
                self._setup_interactive_layers(image_data)

            print(f"Successfully set up layers for image: {image_layer.name}")

            # move image layer to bottom
            # self.viewer.layers.move(self.image_layer, len(self.viewer.layers)-1)

        except (
            AttributeError,
            ValueError,
            TypeError,
            RuntimeError,
            OSError,
        ) as e:
            print(f"Error setting up image layer: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while setting up layers: {str(e)}",
            )

    def _setup_interactive_layers(self, image_data):
        """Setup interactive annotation layers (points and shapes)."""
        # Points layer
        self._point_choices = ["positive", "negative"]
        LABEL_COLOR_CYCLE = ["red", "blue"]

        self.points_layer = self.viewer.add_points(
            name="Point Layer",
            property_choices={"label": self._point_choices},
            border_color="label",
            border_color_cycle=LABEL_COLOR_CYCLE,
            symbol="o",
            face_color="transparent",
            border_width=0.5,
            size=1,
            ndim=len(image_data.shape),
        )
        self.points_layer.events.data.connect(self._on_points_changed)

        # Shapes layer
        annotation_ndim = min(len(image_data.shape), 3)
        self.shapes_layer = self.viewer.add_shapes(
            name="Shapes Layer",
            edge_color="green",
            face_color="transparent",
            edge_width=2,
            ndim=annotation_ndim,
        )
