"""
Base class for ND App widgets.

This module provides a base class containing common functionality shared between
NDEasyLabel and NDEasySegment widgets.
"""

import contextlib

import napari
import numpy as np
from qtpy.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QPushButton,
    QWidget,
)
from superqt.utils import ensure_main_thread

from ..models import ImageDataModel
from ..Segmenters.GlobalSegmenters import GlobalSegmenterBase
from ..Segmenters.InteractiveSegmenters import InteractiveSegmenterBase
from ..utility import (
    get_supported_axes_from_shape,
    load_images_from_directory,
    pad_to_largest,
)
from ..widgets import NDOperationWidget


class BaseNDApp(QWidget):
    """
    Base class for ND App widgets.

    This class contains common functionality shared between NDEasyLabel and NDEasySegment,
    including segmenter management, image loading, layer setup, and label persistence.

    Common attributes and methods will be moved here progressively to reduce code duplication.
    """

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        image_data_model: ImageDataModel = None,
    ):
        """
        Initialize the base widget with common setup.

        Args:
            viewer: The napari viewer instance.
            image_data_model: Optional ImageDataModel. Can be set later via set_image_data_model().
        """
        super().__init__()
        self.viewer = viewer
        self.image_data_model = image_data_model
        # Use the model-owned segmenter cache so caching is centralized.
        # Keep a reference here for backwards compatibility with external code.
        self.segmenter_cache = (
            self.image_data_model.segmenter_cache if image_data_model else {}
        )

        # Initialize layer references (common to both widgets)
        self.image_layer = None
        self.annotation_layer = None  # The *active* annotation layer.
        # Collection of all annotation layers, keyed by their layer name.
        # ``annotation_layer`` is always a pointer into this dict (or None
        # when no image is loaded).  Mirrors ``predictions_layers``.
        self.annotations_layers = {}
        # Scratch labels layer used by interactive segmenters; contents are
        # promoted to ``annotation_layer`` on Commit.  Created lazily by
        # widgets that need it (e.g. NDEasyLabel).
        self.working_layer = None
        self.predictions_layers = (
            {}
        )  # dict: segmenter_name -> napari labels layer
        self.points_layer = None
        self.shapes_layer = None

        # Initialize label counter (common to both widgets)
        self.current_label_num = 1

        # Track current image context (common to both widgets)
        self.current_image_index = 0

        # Signal processing state protection (common to both widgets)
        self._processing_image_change = False

        # Segmenter management (common to both widgets)
        self.segmenter = None

        # Create save annotations button
        self.save_annotations_btn = QPushButton("Save Annotations")
        self.save_annotations_btn.clicked.connect(self._on_save_annotations)

        # Connect to viewer close event - only install once across all BaseNDApp instances
        # sharing the same viewer (e.g. sub-apps inside NDAILab).
        try:
            qt_window = self.viewer.window._qt_window
            if not getattr(qt_window, "_nd_close_handler_installed", False):
                qt_window.closeEvent = self._create_close_event_handler(
                    qt_window.closeEvent
                )
                qt_window._nd_close_handler_installed = True
                print("Connected to viewer close event")
            else:
                print(
                    "Viewer close event handler already installed - skipping"
                )
        except (AttributeError, RuntimeError) as e:
            print(f"Could not connect to viewer closing: {e}")

    def _create_close_event_handler(self, original_close_event):
        """Create a close event handler that wraps the original."""

        def close_event_handler(event):
            """Handle viewer window close event."""
            print("Viewer closing detected via closeEvent")

            # Ask user if they want to save annotations
            if self.annotation_layer:
                reply = QMessageBox.question(
                    self,
                    "Save Annotations?",
                    "Do you want to save annotations before closing?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                    QMessageBox.Yes,
                )

                if reply == QMessageBox.Cancel:
                    # User cancelled, don't close
                    event.ignore()
                    return
                elif reply == QMessageBox.Yes:
                    # Save every annotation layer in the collection.
                    self._save_all_annotations()
                    print("Annotations saved before closing")

                    # Save boxes at the same time
                    self._save_boxes()

                    # Save label patches if boxes exist (NDEasyLabel only)
                    if callable(
                        getattr(self, "_save_label_patches_on_close", None)
                    ):
                        self._save_label_patches_on_close()

            # Call original close event handler
            if original_close_event:
                original_close_event(event)

        return close_event_handler

    def _on_viewer_closing(self):
        """Handle napari viewer closing - deprecated, using closeEvent now."""
        print("Viewer closing detected via destroyed signal")
        QMessageBox.information(None, "Closing", "Closing widget")

    def _save_all_annotations(self, current_step=None):
        """Save every annotation layer in :attr:`annotations_layers`.

        Each layer is saved into ``annotations/<layer_name>/`` so multiple
        named annotation collections coexist (mirroring the predictions
        layout).  Falls back to a single-layer save against
        :attr:`annotation_layer` if the collection is empty (e.g. older
        widgets that never registered into it).
        """
        if self.image_data_model is None:
            return
        layers = dict(self.annotations_layers)
        if not layers and self.annotation_layer is not None:
            layers = {self.annotation_layer.name: self.annotation_layer}
        if not layers:
            print("No annotations to save")
            return
        for name, layer in layers.items():
            try:
                self.image_data_model.save_annotations(
                    layer.data,
                    self.current_image_index,
                    subdirectory=name,
                    current_step=current_step,
                )
                print(
                    f"Saved annotations '{name}' for image index "
                    f"{self.current_image_index}"
                )
            except (OSError, ValueError, RuntimeError) as e:
                print(f"Failed to save annotations '{name}': {e}")
                QMessageBox.warning(
                    self,
                    "Save Error",
                    f"Failed to save annotations '{name}': {e}",
                )

    def _on_save_annotations(self):
        """Save current annotations explicitly."""
        self._save_all_annotations()
        # Save boxes at the same time as annotations
        self._save_boxes()

    def _save_boxes(self):
        """Save all boxes in boxes_layer to CSV via the model (called alongside annotation saves)."""
        boxes_layer = getattr(self, "boxes_layer", None)
        if boxes_layer is None or self.image_data_model is None:
            return
        try:
            self.image_data_model.save_boxes(
                boxes_layer.data,
                self.current_image_index,
            )
        except (OSError, ValueError, RuntimeError) as e:
            print(f"Failed to save boxes: {e}")

    def set_image_data_model(self, image_data_model: ImageDataModel):
        """
        Set or update the image data model.

        This allows the model to be injected after widget creation,
        useful for combined widgets that create the model later.

        Args:
            image_data_model: The ImageDataModel instance to use.
        """
        self.image_data_model = image_data_model
        self.segmenter_cache = image_data_model.segmenter_cache
        print(f"{self.__class__.__name__}: Image data model set")

        # Re-fire segmenter selection now that the model is available
        # (the combo's initial firing was a no-op before the model existed).
        if (
            hasattr(self, "segmenter_combo")
            and self.segmenter_combo.count() > 0
        ):
            self._on_segmenter_changed(self.segmenter_combo.currentText())

    # === Shared ROI / bounding-box helpers ===========================
    # Used by NDEasyLabel (preview-crop second viewer) and NDEasySegment
    # (Segment ROI + optional ROI preview viewer).

    def _get_active_box(self, layer):
        """Return the active (selected) box on `layer`, or the last box
        if none is selected. Returns ``None`` if the layer is empty/missing.
        """
        if layer is None or len(layer.data) == 0:
            return None
        selected = getattr(layer, "selected_data", None)
        try:
            sel_indices = list(selected) if selected else []
        except TypeError:
            sel_indices = []
        idx = sel_indices[0] if sel_indices else len(layer.data) - 1
        try:
            return np.asarray(layer.data[idx])
        except (IndexError, TypeError):
            return None

    def _compute_crop_slice(self, box_layer, target_layer):
        """Build a numpy indexing tuple that crops ``target_layer.data``
        to the active box on ``box_layer``.

        The box's last 2 (Shapes) or 3 (BoundingBoxLayer) columns are
        treated as spatial extents (sliced); preceding columns are taken
        as non-spatial integer indices (T, P, sequence, ...).  The
        resulting tuple is aligned to ``target_layer.data.shape``:

          * Trailing channel axis on the target (target ndim ==
            non_spatial + spatial + 1) → ``slice(None)`` is appended.
          * Leading non-spatial dims are taken from the trailing end of
            the box's non-spatial indices; any extra leading dims on the
            target are padded with ``0``.

        Returns ``None`` if the box is missing, malformed, or yields an
        empty crop.
        """
        if box_layer is None or target_layer is None:
            return None
        box = self._get_active_box(box_layer)
        if box is None or box.ndim != 2:
            return None

        # Detect spatial dimensionality from the box layer type.
        try:
            from ..vendored.napari_bbox import BoundingBoxLayer

            is_3d_bb = isinstance(box_layer, BoundingBoxLayer)
        except ImportError:
            is_3d_bb = False
        n_spatial = 3 if is_3d_bb else 2
        if box.shape[1] < n_spatial:
            return None

        spatial = box[:, -n_spatial:]
        non_spatial_idx = tuple(int(v) for v in box[0, :-n_spatial])
        mins = np.floor(spatial.min(axis=0)).astype(int)
        maxs = np.ceil(spatial.max(axis=0)).astype(int)

        target_shape = target_layer.data.shape
        target_ndim = target_layer.data.ndim

        # Trailing channel axis: target carries exactly 1 extra dim.
        n_non = len(non_spatial_idx)
        has_trailing_ch = target_ndim == n_non + n_spatial + 1
        sp_dim_start = -(n_spatial + 1) if has_trailing_ch else -n_spatial
        sp_dim_end = -1 if has_trailing_ch else None
        sp_shape = target_shape[sp_dim_start:sp_dim_end]

        mins = np.clip(mins, 0, np.array(sp_shape) - 1)
        maxs = np.clip(maxs, 1, np.array(sp_shape))
        if np.any(maxs <= mins):
            return None

        spatial_slices = tuple(
            slice(int(mins[i]), int(maxs[i])) for i in range(n_spatial)
        )

        # Align leading non-spatial dims to whatever the target has room for.
        n_leading = target_ndim - n_spatial - (1 if has_trailing_ch else 0)
        if n_leading > 0:
            if len(non_spatial_idx) >= n_leading:
                prefix = non_spatial_idx[-n_leading:]
            else:
                prefix = (0,) * (
                    n_leading - len(non_spatial_idx)
                ) + non_spatial_idx
        else:
            prefix = ()

        idx = prefix + spatial_slices
        if has_trailing_ch:
            idx = idx + (slice(None),)
        return idx

    # === COMMON METHODS TO BE IMPLEMENTED ===
    # These methods exist in both NDEasyLabel and NDEasySegment with similar/identical implementations

    def _create_segmenter_parameter_form(self):
        """Create the shared parameter form and connect change signal."""
        self.segmenter_parameter_form = NDOperationWidget(
            param_type_to_parse="inference"
        )
        self.segmenter_parameter_form.parameters_changed.connect(
            self._on_segmenter_parameters_changed
        )
        return self.segmenter_parameter_form

    def _populate_segmenter_combo(self):
        """Populate the segmenter combo box with registered frameworks."""
        # TODO: Move implementation from both widgets
        raise NotImplementedError("To be implemented in next step")

    def _on_segmenter_changed(self, segmenter_name):
        """Handle changes to segmenter selection."""
        if not segmenter_name or segmenter_name == "No segmenters available":
            self.segmenter_parameter_form.clear_form()
            return

        # No model yet (plugin opened without a project) — nothing to do.
        # The combo will be re-fired by setup code once a project is loaded.
        if self.image_data_model is None:
            return

        # Use model to get (and cache) segmenter instances
        self.segmenter = self.image_data_model.get_segmenter(segmenter_name)

        if self.segmenter:
            print(f"Potential axes: {self.segmenter.potential_axes}")
            print(
                f"Supported axes (before filtering): {self.segmenter.supported_axes}"
            )

            # Filter potential axes based on current image shape
            if self.image_layer is not None:
                image_shape = self.image_layer.data.shape
                filtered_axes = get_supported_axes_from_shape(
                    image_shape,
                    self.segmenter.potential_axes,
                    self.image_data_model.axis_types,
                )
                self.segmenter.supported_axes = filtered_axes
                print(
                    f"Updated supported axes based on image shape: {filtered_axes}"
                )

                # Ensure selected_axis is valid after filtering
                if (
                    hasattr(self.segmenter, "selected_axis")
                    and self.segmenter.selected_axis not in filtered_axes
                ):
                    # Set to first available axis if current selection is invalid
                    if filtered_axes:
                        self.segmenter.selected_axis = filtered_axes[0]
                        print(
                            f"Updated selected_axis to: {self.segmenter.selected_axis}"
                        )
                    else:
                        print(
                            "Warning: No supported axes available for current image shape"
                        )
            self.segmenter.model_save_dir = (
                self.image_data_model.get_models_directory()
            )
            # Update parameter form with segmenter instance
            self._update_segmenter_parameter_form(self.segmenter)
            print(f"Selected segmenter: {segmenter_name}")
            print(f"Supported axes: {self.segmenter.supported_axes}")
        else:
            print(
                f"Warning: Segmenter '{segmenter_name}' not found in registry"
            )
            self.segmenter_parameter_form.clear_form()

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

    def _update_segmenter_parameter_form(self, segmenter):
        """Update parameter form with segmenter instance."""
        self.segmenter_parameter_form.set_nd_operation(segmenter)

    def _post_segmenter_selection(self):
        """Handle post-selection logic - to be implemented by derived classes."""
        # Default: do nothing

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
            images, axes_infos, image_paths = load_images_from_directory(
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
                images, axes_infos, force8bit=True, normalize_per_channel=False
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
            ("working", self.working_layer),
            ("points", self.points_layer),
            ("shapes", self.shapes_layer),
        ]

        # Add all annotation layers from the collection (falling back to
        # the singular ``annotation_layer`` ref when the dict is empty so
        # we don't leak the initial labels layer).
        annot_layers = dict(self.annotations_layers) or (
            {self.annotation_layer.name: self.annotation_layer}
            if self.annotation_layer is not None
            else {}
        )
        for annot_name, annot_layer in annot_layers.items():
            layers_to_remove.append((f"label ({annot_name})", annot_layer))

        # Add all prediction layers from the dictionary
        for segmenter_name, pred_layer in self.predictions_layers.items():
            layers_to_remove.append(
                (f"predictions ({segmenter_name})", pred_layer)
            )

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
        self.annotation_layer = None
        self.annotations_layers = {}
        self.working_layer = None
        self.predictions_layers = {}
        self.points_layer = None
        self.shapes_layer = None

        print("Completed layer cleanup")

    def _process_image_change(self, image_layer, image_index):
        """Process image change with simple processing lock."""
        # Set processing flag to prevent re-entrant calls

        try:
            print(f"Processing image change: index {image_index}")

            # Save current labels before switching (if we have a current context)
            if self.image_data_model.parent_directory and (
                self.annotations_layers or self.annotation_layer
            ):
                self._save_all_annotations(
                    current_step=self.viewer.dims.current_step
                )
            else:
                print("No current labels to save (first image or no context)")

            print("Switching images")
            # Set the current image index from the signal
            self.current_image_index = image_index

            if image_layer:
                print(f"Setting up new image: {image_layer.name}")

                # Clean up existing layers BEFORE updating context
                print("Cleaning up old layers...")
                self._cleanup_existing_layers()

                # Update current context AFTER cleanup
                print("Updating context...")
                # parent_directory is managed by image_data_model, no need to store locally

                print("Creating new layers with fresh labels...")
                self._set_image_layer(image_layer)
            else:
                print("Received invalid image data from sequence viewer")
                self._cleanup_existing_layers()
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

    def _on_segmenter_parameters_changed(self, parameters):
        """Handle changes to segmenter parameters."""
        segmenter_name = self.segmenter_combo.currentText()
        print(f"Parameters changed for {segmenter_name}: {parameters}")

        # Sync the current segmenter instance with new parameter values
        if hasattr(self, "segmenter") and self.segmenter is not None:
            # If model_path changed, refresh model combo to include the new custom model
            if "model_path" in parameters and hasattr(
                self.segmenter, "get_model_axis_map"
            ):
                # Sync the model_path first
                self.segmenter = (
                    self.segmenter_parameter_form.sync_nd_operation_instance(
                        self.segmenter
                    )
                )
                # Get the model name from the path that was just set
                custom_model_name = self.segmenter.inference_model_name
                if custom_model_name:
                    print(
                        f"📂 Custom model loaded: {custom_model_name}, refreshing model combo..."
                    )
                    # Refresh model combo in both forms to include the custom model
                    self.segmenter_parameter_form.refresh_model_combo(
                        select_name=custom_model_name
                    )
                    if hasattr(self, "training_parameter_form"):
                        self.training_parameter_form.refresh_model_combo(
                            select_name=custom_model_name
                        )
                recommended_axis = self.segmenter.get_recommended_axis()

                self.segmenter_parameter_form.set_selected_axis(
                    recommended_axis
                )

                return

            # If inference_model_name changed for StardistSegmenter, validate before syncing
            if "inference_model_name" in parameters and hasattr(
                self.segmenter, "get_recommended_axis"
            ):
                # Store old value before sync
                old_inference_model_name = getattr(
                    self.segmenter, "inference_model_name", None
                )

                # Temporarily sync to get the recommended axis
                temp_segmenter = (
                    self.segmenter_parameter_form.sync_nd_operation_instance(
                        self.segmenter
                    )
                )
                recommended_axis = temp_segmenter.get_recommended_axis()
                print(
                    f"Model changed to {parameters['inference_model_name']}, recommended axis: {recommended_axis}"
                )

                # Check if the recommended axis is available for the current image
                if recommended_axis not in self.segmenter.supported_axes:
                    # Axis not compatible with current image - show warning and revert
                    from qtpy.QtWidgets import QMessageBox

                    model_name = parameters["inference_model_name"]
                    if "C" in recommended_axis:
                        msg = (
                            f"⚠️ Model '{model_name}' requires a channel dimension (axis: {recommended_axis})\n\n"
                            f"Your current image does not have a channel dimension.\n"
                            f"Please load an RGB/multi-channel image to use this model.\n\n"
                            f"Keeping previous model: {old_inference_model_name}"
                        )
                    elif "Z" in recommended_axis:
                        msg = (
                            f"\u26a0\ufe0f Model '{model_name}' requires a Z dimension (axis: {recommended_axis})\n\n"
                            f"Your current image is 2D.\n"
                            f"Please load a 3D image to use this model.\n\n"
                            f"Keeping previous model: {old_inference_model_name}"
                        )
                    else:
                        msg = (
                            f"\u26a0\ufe0f Model '{model_name}' requires axis: {recommended_axis}\n\n"
                            f"This is not compatible with your current image.\n"
                            f"Available axes: {', '.join(self.segmenter.supported_axes)}\n\n"
                            f"Keeping previous model: {old_inference_model_name}"
                        )

                    QMessageBox.warning(None, "Incompatible Model", msg)
                    print(
                        f"❌ Cannot use model '{model_name}': requires {recommended_axis}, but only {self.segmenter.supported_axes} available"
                    )

                    # Revert the inference_model_name combo to the old value
                    if old_inference_model_name:
                        self.segmenter_parameter_form.set_parameter(
                            "inference_model_name", old_inference_model_name
                        )

                    return  # Don't proceed with sync
                else:
                    # Valid selection - update axis automatically
                    self.segmenter = temp_segmenter
                    print("Synced segmenter instance with new parameters")
                    self.segmenter_parameter_form.set_selected_axis(
                        recommended_axis
                    )
                    return

            # For non-inference_model_name changes or segmenters without get_recommended_axis
            self.segmenter = (
                self.segmenter_parameter_form.sync_nd_operation_instance(
                    self.segmenter
                )
            )
            print("Synced segmenter instance with new parameters")
