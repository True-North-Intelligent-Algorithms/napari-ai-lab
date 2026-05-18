"""
ND AI Lab - Combined plugin with tabbed interface.

This module combines NDEasyLabel, NDEasyAugment, and NDEasySegment
into a single tabbed interface with shared model and viewer.
"""

from pathlib import Path

import napari
import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread

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
        axes_to_collapse: str | list[str] | None = None,
    ):
        """
        Initialize the combined AI Lab widget.

        Args:
            viewer: The napari viewer instance.
            image_data_model: Optional ImageDataModel. If not provided, can be set later.
            axes_to_collapse: Axis names to collapse when creating annotations/predictions
                            (e.g., "C" for channels). Passed to model load/save methods.
        """
        super().__init__()
        self.viewer = viewer
        self.image_data_model = image_data_model
        self.axes_to_collapse = axes_to_collapse

        # Tracking for sequence viewer changes
        self._processing_image_change = False
        self.current_image_index = 0

        # Create sub-apps in EMBEDDED mode (no individual directory buttons)
        # Model can be set later via set_image_data_model()
        self.label_widget = NDEasyLabel(
            viewer,
            image_data_model,
            embedded=True,
            axes_to_collapse=axes_to_collapse,
        )
        self.augment_widget = NDEasyAugment(
            viewer,
            image_data_model,
            embedded=True,
            axes_to_collapse=axes_to_collapse,
        )
        self.segment_widget = NDEasySegment(
            viewer,
            image_data_model,
            embedded=True,
            training_widget_mode="embedded",  # Use embedded training form, not dialog
            axes_to_collapse=axes_to_collapse,
        )

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

    def _set_image_layer(self, image_layer):
        """
        Central layer management - creates ALL layers needed by all sub-apps.

        This prevents duplicate layer creation and ensures consistency.
        Layers are created once here and distributed to sub-apps via direct assignment.

        Args:
            image_layer: The napari image layer to set up annotations for.
        """
        # Store image layer reference
        self.image_layer = image_layer

        # Load existing data or create empty with axis collapsing
        labels_data = self.image_data_model.load_existing_annotations(
            image_layer.data.shape,
            self.current_image_index,
            axes_to_collapse=self.axes_to_collapse,
        )

        # Per-axis scale for annotation layers (collapsed axes dropped)
        annotation_scale = (
            self.image_data_model.get_scale(
                axes_to_collapse=self.axes_to_collapse
            )
            or None
        )

        # Create shared layers ONCE

        self.annotations_layer = self.viewer.add_labels(
            labels_data,
            name="Labels (Persistent)",
            scale=annotation_scale,
        )

        # Dictionary to hold prediction layers for different segmenters
        # Key: segmenter name (e.g., "CellposeSegmenter", "StarDist")
        # Value: napari labels layer
        # Populated on demand when segmentation runs or existing predictions are loaded
        self.segment_widget._load_existing_prediction_layers(
            self.image_layer.data.shape
        )
        self.predictions_layers = self.segment_widget.predictions_layers

        # Create points layer for interactive segmentation
        self._point_choices = ["positive", "negative"]
        LABEL_COLOR_CYCLE = ["red", "blue"]
        annotation_ndim = len(self.annotations_layer.data.shape)

        self.points_layer = self.viewer.add_points(
            name="Point Layer",
            property_choices={"label": self._point_choices},
            border_color="label",
            border_color_cycle=LABEL_COLOR_CYCLE,
            symbol="o",
            face_color="transparent",
            border_width=0.5,
            size=1,
            ndim=annotation_ndim,
            scale=annotation_scale,
        )

        # Create shapes layer for segment widget (if in interactive mode)
        self.shapes_layer = self.viewer.add_shapes(
            name="Shapes Layer",
            edge_color="green",
            face_color="transparent",
            edge_width=2,
            ndim=annotation_ndim,
            scale=annotation_scale,
        )

        # Create boxes layer for label widget (bounding-box annotations)
        self.boxes_layer = self.viewer.add_shapes(
            ndim=annotation_ndim,
            name="Label box",
            face_color="transparent",
            edge_color="blue",
            edge_width=5,
            text={"string": "{split_set}", "size": 15, "color": "green"},
            scale=annotation_scale,
        )

        from napari_ai_lab.vendored.napari_bbox import BoundingBoxLayer

        self.boxes_3D_layer = BoundingBoxLayer(
            name="3D Bounding Boxes",
            edge_color="magenta",
            face_color="transparent",
            edge_width=5,
            ndim=annotation_ndim,
            scale=annotation_scale,
        )

        self.viewer.add_layer(self.boxes_3D_layer)

        # Distribute layers to sub-apps (direct assignment, not calling their _set_image_layer)
        self._distribute_layers_to_sub_apps()

        print(f"✅ Central layer setup complete for image: {image_layer.name}")

    def _distribute_layers_to_sub_apps(self):
        """
        Distribute the centrally-created layers to each sub-app.

        Uses direct attribute assignment instead of calling sub-apps' _set_image_layer()
        to avoid duplicate layer creation.
        """
        # Label widget needs: image, labels, points, boxes
        self.label_widget.image_layer = self.image_layer
        self.label_widget.annotation_layer = self.annotations_layer
        self.label_widget.points_layer = self.points_layer
        self.label_widget.boxes_layer = self.boxes_layer

        # Connect points layer events for label widget
        if self.points_layer and hasattr(
            self.label_widget, "_on_points_changed"
        ):
            self.points_layer.events.data.connect(
                self.label_widget._on_points_changed
            )

        # Connect boxes layer events for label widget
        if self.boxes_layer and hasattr(
            self.label_widget, "_on_boxes_changed"
        ):
            self.boxes_layer.events.data.connect(
                self.label_widget._on_boxes_changed
            )

        # Load existing boxes into the boxes_layer from CSV (BEFORE connecting nd_ai_lab event)
        if hasattr(self.label_widget, "_load_existing_boxes"):
            self.label_widget._load_existing_boxes()

        # Connect boxes layer events for nd_ai_lab (for prediction copying)
        # This is done AFTER loading to avoid triggering dialog on startup
        if self.boxes_layer and hasattr(self, "_on_boxes_changed"):
            self.boxes_layer.events.data.connect(self._on_boxes_changed)

        # Distribute the 3D boxes layer to label and segment widgets and
        # wire up their _on_3D_boxes_changed handlers.
        self.label_widget.boxes_3D_layer = self.boxes_3D_layer
        self.segment_widget.boxes_3D_layer = self.boxes_3D_layer
        if hasattr(self.label_widget, "_on_3D_boxes_changed"):
            self.boxes_3D_layer.events.data.connect(
                self.label_widget._on_3D_boxes_changed
            )
        if hasattr(self.segment_widget, "_on_3D_boxes_changed"):
            self.boxes_3D_layer.events.data.connect(
                self.segment_widget._on_3D_boxes_changed
            )

        # Augment widget needs: image, labels
        self.augment_widget.image_layer = self.image_layer
        self.augment_widget.annotation_layer = self.annotations_layer
        self.augment_widget.boxes_layer = (
            self.boxes_layer
        )  # Provide boxes layer for augmentation if needed

        # Segment widget needs: image, labels, predictions, points, shapes
        self.segment_widget.image_layer = self.image_layer
        self.segment_widget.annotation_layer = self.annotations_layer
        self.segment_widget.predictions_layers = self.predictions_layers
        self.segment_widget.points_layer = self.points_layer
        self.segment_widget.shapes_layer = self.shapes_layer

        # Connect interactive layers events for segment widget (if in interactive mode)
        if self.segment_widget.is_interactive_mode():
            if hasattr(self.segment_widget, "_on_points_changed"):
                self.points_layer.events.data.connect(
                    self.segment_widget._on_points_changed
                )
            if hasattr(self.segment_widget, "_on_shapes_changed"):
                self.shapes_layer.events.data.connect(
                    self.segment_widget._on_shapes_changed
                )

        print("   → Layers distributed to all sub-apps")

    def _setup_ui(self):
        """Setup the tabbed user interface."""
        layout = QVBoxLayout(self)

        # Top-level controls (shared across all tabs)
        self.dir_btn = QPushButton("Open Image Directory")
        self.dir_btn.clicked.connect(self._on_open_directory)
        layout.addWidget(self.dir_btn)

        def _make_scroll(widget):
            scroll = QScrollArea()
            scroll.setWidget(widget)
            scroll.setWidgetResizable(True)
            return scroll

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.addTab(_make_scroll(self.label_widget), "Label")
        self.tabs.addTab(_make_scroll(self.augment_widget), "Augment")
        self.tabs.addTab(_make_scroll(self.segment_widget), "Segment")

        # Add training tab - shows training view of segment widget
        # Same widget, different controls (segmenter combo + training params only)
        self.tabs.addTab(
            _make_scroll(self.segment_widget.get_training_widget()), "Train"
        )

        layout.addWidget(self.tabs)

        # Sync patch_size_xy between Augment and Train tabs
        self._cross_connect_patch_size()

    def _cross_connect_patch_size(self):
        """Sync patch_size_xy spinbox (Augment) with train_patch_size_xy (Train)."""
        aug_spin = self.augment_widget.patch_size_xy_spinbox
        trn_form = self.segment_widget.training_parameter_form

        def _aug_to_train(val):
            trn_form.set_parameter("train_patch_size_xy", val)

        def _train_to_aug(params):
            if "train_patch_size_xy" in params:
                aug_spin.blockSignals(True)
                aug_spin.setValue(params["train_patch_size_xy"])
                aug_spin.blockSignals(False)

        aug_spin.valueChanged.connect(_aug_to_train)
        trn_form.parameters_changed.connect(_train_to_aug)

    def _on_open_directory(self):
        """Open directory and create/set model for all sub-apps."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Image Directory",
            "...",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )

        if directory:
            print(f"📁 Loading directory: {directory}")

            # Create model from directory
            parent_dir = Path(directory)
            self.image_data_model = ImageDataModel(parent_dir)

            # Propagate to all tabs
            self.set_image_data_model(self.image_data_model)

            print("✅ Model created and shared across all tabs")
            # TODO Phase 3: Load images and create layers

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

    def _process_image_change(self, image_layer, image_index):
        """
        Process image change from sequence viewer.

        Handles cleanup and recreation of layers centrally for all sub-apps.
        """
        print(f"🔄 nd_ai_lab: Processing image change to index {image_index}")

        # Save current annotations before switching (delegate to sub-apps)
        # Only save from the currently active tab to avoid duplicate saves
        active_widget_name = self.tabs.tabText(self.tabs.currentIndex())

        if (
            active_widget_name == "Label"
            and hasattr(self, "labels_layer")
            and self.annotations_layer
            and self.image_data_model.parent_directory
        ):
            try:
                self.image_data_model.save_annotations(
                    self.annotations_layer.data,
                    self.current_image_index,
                    current_step=self.viewer.dims.current_step,
                    axes_to_collapse=self.axes_to_collapse,
                )
                print("   Saved annotations from Label tab")
            except (OSError, ValueError, RuntimeError) as e:
                print(f"   Failed to save annotations: {e}")

            # Save boxes at the same time as annotations
            self._save_boxes()
            self._save_label_patches_on_close()

        # Update current image index
        self.current_image_index = image_index

        if image_layer:
            print(f"   Setting up new image: {image_layer.name}")

            # Cleanup existing layers centrally
            self._cleanup_layers()

            # Create new layers centrally (this also distributes to sub-apps)
            self._set_image_layer(image_layer)

            print("✅ nd_ai_lab: Image change complete")
        else:
            print("⚠️  Received invalid image data from sequence viewer")
            self._cleanup_layers()

        self._processing_image_change = False

    def _save_label_patches_on_close(self):
        """Delegate label patch saving to label_widget (called by base close handler)."""
        if hasattr(self, "label_widget"):
            self.label_widget._save_label_patches_on_close()

    def _cleanup_layers(self):
        """
        Cleanup existing layers before switching images.

        Central cleanup for combined app - removes all layers from viewer.
        """
        print("   🧹 Cleaning up existing layers...")

        layers_to_remove = []

        # Collect layers to remove
        if hasattr(self, "labels_layer") and self.annotations_layer:
            layers_to_remove.append(("Labels", self.annotations_layer))

        # Handle predictions_layers dictionary (multiple prediction layers)
        if hasattr(self, "predictions_layers") and self.predictions_layers:
            for segmenter_name, layer in self.predictions_layers.items():
                layers_to_remove.append(
                    (f"Predictions ({segmenter_name})", layer)
                )

        if hasattr(self, "points_layer") and self.points_layer:
            layers_to_remove.append(("Points", self.points_layer))
        if hasattr(self, "shapes_layer") and self.shapes_layer:
            layers_to_remove.append(("Shapes", self.shapes_layer))
        if hasattr(self, "boxes_layer") and self.boxes_layer:
            layers_to_remove.append(("Boxes", self.boxes_layer))

        # Remove layers from viewer
        for layer_name, layer in layers_to_remove:
            try:
                if layer in self.viewer.layers:
                    self.viewer.layers.remove(layer)
                    print(f"      Removed {layer_name} layer")
            except (ValueError, KeyError, RuntimeError) as e:
                print(f"      Error removing {layer_name}: {e}")

        # Clear references
        if hasattr(self, "labels_layer"):
            self.annotations_layer = None
        if hasattr(self, "predictions_layers"):
            self.predictions_layers = {}
        if hasattr(self, "points_layer"):
            self.points_layer = None
        if hasattr(self, "shapes_layer"):
            self.shapes_layer = None
        if hasattr(self, "boxes_layer"):
            self.boxes_layer = None

        print("   ✅ Layer cleanup complete")

    def _on_boxes_changed(self, event):
        """Handle boxes layer data changes - show dialog to copy predictions to new ROI."""

        # Only respond to 'added' events (same check as in nd_easy_label)
        if event.action != "added":
            return

        boxes_layer = event.source
        if len(boxes_layer.data) == 0:
            return

        # Check if we have any predictions layers to copy from
        if (
            not hasattr(self, "predictions_layers")
            or not self.predictions_layers
        ):
            print("No predictions layers available to copy from")
            return

        # Get the most recently added box
        box = boxes_layer.data[-1]

        # Extract spatial coordinates (last 2 columns are Y and X)
        # Use floor for start and ceil for end to ensure end > start
        ystart = int(np.floor(np.min(box[:, -2])))
        yend = int(np.ceil(np.max(box[:, -2])))
        xstart = int(np.floor(np.min(box[:, -1])))
        xend = int(np.ceil(np.max(box[:, -1])))

        # Extract ND indices (all columns before the last 2)
        # These are the indices in ND space (e.g., T, Z, S for TSZYX)
        nd_indices = tuple(int(box[0, i]) for i in range(box.shape[1] - 2))

        # Check each prediction layer for data at this location
        available_predictions = {}
        for segmenter_name, pred_layer in self.predictions_layers.items():
            pred_data = pred_layer.data

            # Build the indexing tuple: nd_indices + (slice(ystart, yend), slice(xstart, xend))
            roi_slice = nd_indices + (slice(ystart, yend), slice(xstart, xend))

            # Check if predictions exist at this location
            try:
                pred_roi = pred_data[roi_slice]

                # Check if there's any non-zero data
                if np.any(pred_roi):
                    available_predictions[segmenter_name] = pred_layer
                    print(f"✓ {segmenter_name}: Predictions found in ROI")
                else:
                    print(
                        f"✗ {segmenter_name}: No non-zero predictions in ROI"
                    )
            except (IndexError, ValueError) as e:
                print(
                    f"✗ {segmenter_name}: No predictions at this location ({e})"
                )

        # If no predictions available, inform user and return
        if not available_predictions:
            print("⚠️ No predictions exist at this location")
            return

        # Create dialog with only available predictions
        dialog = QDialog(self)
        dialog.setWindowTitle("Copy Predictions to ROI")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)

        # Message label
        message_label = QLabel(
            "Choose predictions layer and press 'copy' to copy predictions to new roi"
        )
        layout.addWidget(message_label)

        # Combo box with available predictions only
        combo_layout = QHBoxLayout()
        combo_label = QLabel("Predictions Layer:")
        combo_layout.addWidget(combo_label)

        predictions_combo = QComboBox()
        for segmenter_name in available_predictions:
            predictions_combo.addItem(segmenter_name)
        combo_layout.addWidget(predictions_combo)

        layout.addLayout(combo_layout)

        # Buttons
        button_layout = QHBoxLayout()

        copy_button = QPushButton("Copy")
        cancel_button = QPushButton("Cancel")

        # Copy functionality
        def copy_predictions():
            selected_name = predictions_combo.currentText()
            selected_pred_layer = available_predictions[selected_name]
            pred_data = selected_pred_layer.data

            # Get the prediction ROI
            roi_slice = nd_indices + (slice(ystart, yend), slice(xstart, xend))
            pred_roi = pred_data[roi_slice]

            self.annotations_layer.data[roi_slice] = pred_roi
            self.annotations_layer.refresh()  # Force napari to update the display
            print(f"✓ Copied {selected_name} predictions to labels layer")

            dialog.accept()

        copy_button.clicked.connect(copy_predictions)
        cancel_button.clicked.connect(dialog.reject)

        button_layout.addWidget(copy_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        # Show dialog
        dialog.exec_()
