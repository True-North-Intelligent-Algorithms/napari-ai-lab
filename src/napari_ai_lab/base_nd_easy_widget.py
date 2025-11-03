"""
Base class for NDEasy segmentation widgets.

This module provides a base class containing common functionality shared between
NDEasyLabel and NDEasySegment widgets.
"""

import contextlib

import napari
from qtpy.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QWidget,
)
from superqt.utils import ensure_main_thread

from .models import ImageDataModel
from .Segmenters.GlobalSegmenters import GlobalSegmenterBase
from .Segmenters.InteractiveSegmenters import InteractiveSegmenterBase
from .utility import load_images_from_directory, pad_to_largest


class BaseNDEasyWidget(QWidget):
    """
    Base class for ND Easy segmentation widgets.

    This class contains common functionality shared between NDEasyLabel and NDEasySegment,
    including segmenter management, image loading, layer setup, and label persistence.

    Common attributes and methods will be moved here progressively to reduce code duplication.
    """

    def __init__(
        self, viewer: "napari.viewer.Viewer", image_data_model: ImageDataModel
    ):
        """
        Initialize the base widget with common setup.

        Args:
            viewer: The napari viewer instance.
        """
        super().__init__()
        self.viewer = viewer
        self.image_data_model = image_data_model
        self.segmenter_cache = {}  # Cache for segmenter instances

        # Initialize layer references (common to both widgets)
        self.image_layer = None
        self.label_layer = None
        self.points_layer = None
        self.shapes_layer = None

        # Initialize label counter (common to both widgets)
        self.current_label_num = 1

        # Track current image context (common to both widgets)
        self.current_image_path = None
        self.current_image_index = 0

        # Signal processing state protection (common to both widgets)
        self._processing_image_change = False

        # Initialize label writer from model (common to both widgets)
        self.label_writer = self.image_data_model.get_label_writer()

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
        if not segmenter_name or segmenter_name == "No segmenters available":
            self.parameter_form.clear_form()
            return

        # Check if segmenter is already in cache
        if segmenter_name in self.segmenter_cache:
            self.segmenter = self.segmenter_cache[segmenter_name]
        else:
            # Create new segmenter instance and cache it
            self.segmenter = self._create_segmenter_instance(segmenter_name)
            if self.segmenter:
                self.segmenter_cache[segmenter_name] = self.segmenter

        if self.segmenter:
            # Update parameter form with segmenter instance
            self._update_parameter_form(self.segmenter)
            print(f"Selected segmenter: {segmenter_name}")
            print(f"Supported axes: {self.segmenter.supported_axes}")
        else:
            print(
                f"Warning: Segmenter '{segmenter_name}' not found in registry"
            )
            self.parameter_form.clear_form()

        # Handle post-selection logic (like predictor initialization)
        self._post_segmenter_selection()

    def _create_segmenter_instance(self, segmenter_name):
        """Create segmenter instance from either Interactive or Global registry."""
        # Try Interactive segmenters first
        segmenter_class = InteractiveSegmenterBase.get_framework(
            segmenter_name
        )
        if segmenter_class:
            return segmenter_class()

        # Try Global segmenters if not found in Interactive
        segmenter_class = GlobalSegmenterBase.get_framework(segmenter_name)
        if segmenter_class:
            return segmenter_class()

        return None

    def _update_parameter_form(self, segmenter):
        """Update parameter form with segmenter instance."""
        self.parameter_form.set_segmenter(segmenter)

    def _post_segmenter_selection(self):
        """Handle post-selection logic - to be implemented by derived classes."""
        # Default: do nothing

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
            return

    def load_image_directory(self, directory):
        """Load images from the selected directory into napari."""
        try:
            # Load images from directory using utility function
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

            # Process images with pad_to_largest
            print("Processing images with pad_to_largest...")
            padded_images = pad_to_largest(
                images, axis_infos, force8bit=True, normalize_per_channel=False
            )

            # Add the processed image stack to napari viewer and store reference
            self.image_layer = self.viewer.add_image(
                padded_images, name=f"Image Stack ({len(images)} images)"
            )

            # Initialize the rest of the layers based on the image layer
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
        # TODO: Move implementation from both widgets
        raise NotImplementedError("To be implemented in next step")

    def connect_sequence_viewer(self, sequence_viewer):
        """Connect to sequence viewer for automatic layer updates."""
        sequence_viewer.image_changed.connect(self._on_sequence_image_changed)
        print("Connected to sequence viewer for automatic layer updates")

    @ensure_main_thread
    def _on_sequence_image_changed(self, image_layer, image_index):
        """Handle sequence viewer image changes with simple processing lock to prevent crashes."""
        # If we're already processing a signal, ignore this one to prevent conflicts
        if self._processing_image_change:
            print(
                "Signal received while processing - ignoring to prevent conflicts"
            )
            return

        self._processing_image_change = True

        # Process the image change immediately
        self._process_image_change(image_layer, image_index)

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

    def _process_image_change(self, image_layer, image_index):
        """Process image change with simple processing lock."""
        # Set processing flag to prevent re-entrant calls

        try:
            print(f"Processing image change: index {image_index}")

            # Get image path from model using index
            image_paths = self.image_data_model.get_image_paths()
            if 0 <= image_index < len(image_paths):
                image_path = str(image_paths[self.current_image_index])
                print(f"Image path from model: {image_path}")
            else:
                print(f"Invalid image index: {self.current_image_index}")
                image_path = None

            # Save current labels before switching (if we have a current context)
            if (
                self.current_image_path
                and self.image_data_model.parent_directory
                and self.label_layer
            ):
                self._save_current_labels()
            else:
                print("No current labels to save (first image or no context)")

            print("Switching images")
            # Set the current image index from the signal
            self.current_image_index = image_index

            if image_layer and image_path:
                print(f"Setting up new image: {image_layer.name}")
                print(f"New image path: {image_path}")

                # Clean up existing layers BEFORE updating context
                print("Cleaning up old layers...")
                self._cleanup_existing_layers()

                # Update current context AFTER cleanup
                print("Updating context...")
                self.current_image_path = image_path
                # parent_directory is managed by image_data_model, no need to store locally

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
                # parent_directory is managed by image_data_model

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

    def _load_existing_labels(self, image_shape):
        """Load existing labels or create empty ones."""
        # Get annotation directory from model
        annotation_dir = self.image_data_model.get_base_annotations_directory(
            "class_0"
        )
        annotation_dir.mkdir(parents=True, exist_ok=True)

        image_paths = self.image_data_model.get_image_paths()
        image_name = image_paths[self.current_image_index].stem

        # Load using simplified writer interface
        data = self.label_writer.load(str(annotation_dir), image_name)

        # If empty or no data, create empty labels
        if data.size == 0:
            import numpy as np

            return np.zeros(image_shape, dtype=np.uint16)

        return data

    def _save_current_labels(self):
        """Save the current labels using the configured writer."""
        # Get annotation directory from model
        annotation_dir = self.image_data_model.get_base_annotations_directory(
            "class_0"
        )
        annotation_dir.mkdir(parents=True, exist_ok=True)

        # Get current image name from model using current_image_index
        image_paths = self.image_data_model.get_image_paths()
        if self.current_image_index < len(image_paths):
            image_name = image_paths[self.current_image_index].stem
        else:
            # Fallback to current_image_path if index is invalid
            from pathlib import Path

            image_name = (
                Path(self.current_image_path).stem
                if self.current_image_path
                else "unknown"
            )

        # Convert to uint16 before saving
        import numpy as np

        labels_data = self.label_layer.data.astype(np.uint16)

        # Save using simplified writer interface
        return self.label_writer.save(
            str(annotation_dir), image_name, labels_data
        )

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

    # === METHODS UNIQUE TO SPECIFIC WIDGETS ===
    # These methods will remain in the individual widget classes

    # NDEasyLabel specific:
    # - _on_shapes_changed()
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
    # - self.current_image_path (parent_directory managed by image_data_model)
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
