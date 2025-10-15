"""
Unified ND Easy Segmentation Widget.

This module provides a unified interface for both interactive (point/shape-based)
and automatic (full plane/volume) segmentation workflows.
"""

import contextlib

import napari
import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import ensure_main_thread

from .Segmenters.GlobalSegmenters import GlobalSegmenterBase
from .utility import load_images_from_directory, pad_to_largest
from .widgets import SegmenterWidget
from .writers import get_writer


class NDEasySegment(QWidget):
    """
    Unified segmentation widget supporting both interactive and automatic modes.

    Modes:
    - Interactive Mode: Point/shape-based segmentation (like nd_easy_label)
    - Automatic Mode: Full plane/volume segmentation (like nd_easy_segment)
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        # Initialize layer references
        self.image_layer = None
        self.label_layer = None
        self.points_layer = None
        self.shapes_layer = None

        # Initialize label counter
        self.current_label_num = 1

        # Track current image context
        self.current_image_path = None
        self.current_parent_directory = None
        self._processing_image_change = False

        # Initialize label writer
        self.label_writer = get_writer("numpy")

        # Current segmenter instance
        self.segmenter = None

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

        # Parameter form widget
        self.parameter_form = SegmenterWidget()
        self.parameter_form.parameters_changed.connect(
            self._on_parameters_changed
        )
        main_layout.addWidget(self.parameter_form)

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

    def _on_segmenter_changed(self, segmenter_name):
        """Handle changes to the segmenter selection."""
        if not segmenter_name or segmenter_name == "No segmenters available":
            self.parameter_form.clear_form()
            return

        segmenter_class = GlobalSegmenterBase.get_framework(segmenter_name)
        if segmenter_class:
            self.parameter_form.set_segmenter_class(segmenter_class)
            print(f"Selected segmenter: {segmenter_name}")
            print(f"Supported axes: {segmenter_class().supported_axes}")
        else:
            print(
                f"Warning: Segmenter '{segmenter_name}' not found in registry"
            )
            self.parameter_form.clear_form()

        # Create segmenter instance
        self.segmenter = GlobalSegmenterBase.get_framework(segmenter_name)()

    def _on_parameters_changed(self, parameters):
        """Handle changes to segmenter parameters."""
        segmenter_name = self.segmenter_combo.currentText()
        print(f"Parameters changed for {segmenter_name}: {parameters}")

        # Sync the current segmenter instance with new parameter values
        if hasattr(self, "segmenter") and self.segmenter is not None:
            self.segmenter = self.parameter_form.sync_segmenter_instance(
                self.segmenter
            )
            print("Synced segmenter instance with new parameters")

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
                    image_data, points=[latest_point], shapes=None
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

        # Extract current YX slice (last two dimensions are Y,X)

        indices = self.viewer.dims.current_step[:-2] + (
            slice(None),
            slice(None),
        )
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
            mask = self.segmenter.segment(image_data, points=None, shapes=None)

            # Apply mask to labels layer
            if self.label_layer is not None:
                indices = self.viewer.dims.current_step[:-2] + (
                    slice(None),
                    slice(None),
                )
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
    def _on_open_directory(self):
        """Open a file dialog to select an image directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Image Directory",
            "...",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )

        if directory:
            print(f"Selected directory: {directory}")
            self.load_image_directory(directory)
        else:
            print("No directory selected")

    def load_image_directory(self, directory):
        """Load images from the selected directory into napari."""
        try:
            images, axis_infos, image_paths = load_images_from_directory(
                directory
            )

            if images is None:
                QMessageBox.information(
                    self,
                    "Error",
                    "No images found or could be loaded from the selected directory.",
                )
                return

            print("Processing images with pad_to_largest...")
            padded_images = pad_to_largest(
                images, axis_infos, force8bit=True, normalize_per_channel=False
            )

            self.image_layer = self.viewer.add_image(
                padded_images, name=f"Image Stack ({len(images)} images)"
            )

            self._set_image_layer(self.image_layer)

        except (OSError, ValueError, ImportError, RuntimeError) as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while loading images: {str(e)}",
            )
            print(f"Error loading images: {e}")

    def _set_image_layer(self, image_layer):
        """Set up annotation layers based on the provided image layer."""
        try:
            self.image_layer = image_layer
            image_data = image_layer.data

            # Load existing labels or create empty ones
            labels_data = self._load_existing_labels(image_data.shape)
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

    def _save_current_labels(self):
        """Save the current labels using the configured writer."""
        if not all(
            [
                self.current_image_path,
                self.current_parent_directory,
                self.label_layer,
            ]
        ):
            print(
                "✗ Cannot save labels - missing image context or label layer"
            )
            return False

        return self.label_writer.save_labels(
            self.label_layer.data,
            self.current_image_path,
            self.current_parent_directory,
        )

    def _load_existing_labels(self, image_shape):
        """Load existing labels or create empty ones."""
        if not all([self.current_image_path, self.current_parent_directory]):
            print("No image context for loading labels")
            return np.zeros(image_shape, dtype=np.uint16)

        return self.label_writer.load_labels(
            self.current_image_path, self.current_parent_directory, image_shape
        )

    def connect_sequence_viewer(self, sequence_viewer):
        """Connect to sequence viewer for automatic layer updates."""
        sequence_viewer.image_changed.connect(self._on_sequence_image_changed)
        print("Connected to sequence viewer for automatic layer updates")

    @ensure_main_thread
    def _on_sequence_image_changed(
        self, image_layer, image_path, parent_directory
    ):
        """Handle sequence viewer image changes with simple processing lock to prevent crashes."""
        # If we're already processing a signal, ignore this one to prevent conflicts
        if self._processing_image_change:
            print(
                "Signal received while processing - ignoring to prevent conflicts"
            )
            return

        self._processing_image_change = True

        # Process the image change immediately
        self._process_image_change(image_layer, image_path, parent_directory)

    def _process_image_change(self, image_layer, image_path, parent_directory):
        """Process the image change with simple processing lock."""
        # Set processing flag to prevent re-entrant calls

        try:
            print(f"Processing image change: {image_path}")

            # Save current labels before switching (if we have a current context)
            if (
                self.current_image_path
                and self.current_parent_directory
                and self.label_layer
            ):
                print(f"Saving current labels for: {self.current_image_path}")
                self._save_current_labels()
            else:
                print("No current labels to save (first image or no context)")

            print("Switching images")

            if image_layer and image_path and parent_directory:
                print(f"Setting up new image: {image_layer.name}")
                print(f"New image path: {image_path}")
                print(f"New parent directory: {parent_directory}")

                # Clean up existing layers BEFORE updating context
                print("Cleaning up old layers...")
                self._cleanup_existing_layers()

                # Update current context AFTER cleanup
                print("Updating context...")
                self.current_image_path = image_path
                self.current_parent_directory = parent_directory

                # Small delay to let Napari properly release layer resources
                import time

                time.sleep(0.05)  # 50ms delay

                print("Creating new layers with fresh labels...")
                self._set_image_layer(image_layer)
            else:
                print("Received invalid image data from sequence viewer")
                self._cleanup_existing_layers()
                # Clear current context
                self.current_image_path = None
                self.current_parent_directory = None

        except (
            OSError,
            ValueError,
            RuntimeError,
            AttributeError,
            KeyError,
        ) as e:
            print(f"Error processing image change: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Always clear the processing flag
            self._processing_image_change = False
            print("Finished processing image change")
            print("==============================")

    def _cleanup_existing_layers(self):
        """Remove existing annotation layers from viewer safely."""
        print("Starting layer cleanup...")

        # NOTE: Label saving is now handled in _process_image_change before calling this method
        # This prevents duplicate saves and context confusion

        # Disconnect event handlers first to prevent callbacks during cleanup
        if self.points_layer:
            with contextlib.suppress(Exception):
                self.points_layer.events.data.disconnect(
                    self._on_points_changed
                )

        if self.shapes_layer:
            with contextlib.suppress(Exception):
                self.shapes_layer.events.data.disconnect(
                    self._on_shapes_changed
                )

        # Remove layers one by one with proper error handling
        layers_to_remove = [
            ("label", self.label_layer),
            ("points", self.points_layer),
            ("shapes", self.shapes_layer),
        ]

        for layer_name, layer in layers_to_remove:
            if layer:
                try:
                    if layer in self.viewer.layers:
                        print(f"Removing {layer_name} layer: {layer.name}")
                        self.viewer.layers.remove(layer)
                        print(f"Successfully removed {layer_name} layer")
                    else:
                        print(f"{layer_name} layer not in viewer")
                except (
                    ValueError,
                    KeyError,
                    AttributeError,
                    RuntimeError,
                ) as e:
                    print(f"Error removing {layer_name} layer: {e}")

        # Reset references
        self.label_layer = None
        self.points_layer = None
        self.shapes_layer = None

        print("Completed layer cleanup")
