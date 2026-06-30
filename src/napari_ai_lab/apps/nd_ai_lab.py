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

        # Back-reference so child widgets can call into NDAILab
        # (used by "Open Project" button, "Copy predictions to labels", ...).
        # Set here (not in _distribute_layers_to_sub_apps) so it is available
        # even before the first image is loaded.
        self.label_widget.ai_lab = self
        self.augment_widget.ai_lab = self
        self.segment_widget.ai_lab = self

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

        # Per-axis scale for annotation layers (collapsed axes dropped)
        annotation_scale = (
            self.image_data_model.get_scale(
                axes_to_collapse=self.axes_to_collapse
            )
            or None
        )

        # Multiple named annotation collections are supported: each
        # subdirectory under ``annotations/`` becomes a separate labels
        # layer.  Falls back to a single ``"Labels (Persistent)"`` layer
        # when no subdirectories exist yet.
        annotation_names = (
            self.image_data_model.list_annotation_subdirectories()
            or ["Labels (Persistent)"]
        )
        self.annotations_layers = {}
        for ann_name in annotation_names:
            data = self.image_data_model.load_existing_annotations(
                self.current_image_index,
                image_layer.data.shape,
                subdirectory=ann_name,
                axes_to_collapse=self.axes_to_collapse,
            )
            self.annotations_layers[ann_name] = self.viewer.add_labels(
                data,
                name=ann_name,
                scale=annotation_scale,
            )
        # Pick the first as the active layer (backwards-compatible ref).
        self.annotations_layer = next(iter(self.annotations_layers.values()))

        # Working / scratch labels layer for interactive segmenter output.
        # Always uses WORKING_LABEL_INDEX (7) so its colour is consistent.
        import numpy as _np

        working_data = _np.zeros_like(self.annotations_layer.data)
        self.working_layer = self.viewer.add_labels(
            working_data,
            name="Labels (Working)",
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
            text={"string": "{split_set}", "size": 15, "color": "green"},
        )

        # self.boxes_3D_layer.feature_defaults["split_set"] = ""

        self.viewer.add_layer(self.boxes_3D_layer)

        # Distribute layers to sub-apps (direct assignment, not calling their _set_image_layer)
        self._distribute_layers_to_sub_apps()

        print(f"✅ Central layer setup complete for image: {image_layer.name}")

        # sync augmenter parameters with current image shape
        self.augment_widget.sync_augmenter_parameters()

        # Todo: this logic could fail if the new image is part of a sequence and we want to remember the last selected axis.
        # For now, just reset to the first supported axis.
        self.augment_widget.augmenter.selected_axis = (
            self.augment_widget.augmenter.supported_axes[0]
        )

        self.augment_widget.augmentation_form.set_axis_combo(
            self.augment_widget.augmenter.selected_axis
        )

    def _distribute_layers_to_sub_apps(self):
        """
        Distribute the centrally-created layers to each sub-app.

        Uses direct attribute assignment instead of calling sub-apps' _set_image_layer()
        to avoid duplicate layer creation.
        """
        # Label widget needs: image, labels, points, boxes
        self.label_widget.image_layer = self.image_layer
        self.label_widget.annotation_layer = self.annotations_layer
        self.label_widget.annotations_layers = self.annotations_layers
        self.label_widget.working_layer = self.working_layer
        self.label_widget.points_layer = self.points_layer
        self.label_widget.boxes_layer = self.boxes_layer
        # Sync the label widget's Active Annotation combo with the
        # centrally-loaded collection.
        if hasattr(self.label_widget, "_refresh_active_annotation_combo"):
            self.label_widget._refresh_active_annotation_combo()

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

        # Distribute the 3D boxes layer to label and segment widgets.
        # Note: this is the annotation/labeling 3D boxes layer.  Interactive
        # segmentation is driven by each widget's own "Interactive 3D Boxes"
        # layer (created on demand from the widget UI), so we do NOT wire
        # this layer's events to _on_3D_boxes_changed here.
        self.label_widget.boxes_3D_layer = self.boxes_3D_layer
        self.segment_widget.boxes_3D_layer = self.boxes_3D_layer
        # Back-reference so child widgets (e.g. label widget's
        # "Copy predictions to labels" button) can call into NDAILab.
        self.label_widget.ai_lab = self
        self.segment_widget.ai_lab = self
        # Segment widget also needs the 2D "Label box" layer for its
        # ROI-source combo ("Label Box" option).
        self.segment_widget.boxes_layer = self.boxes_layer

        # Load any previously saved 3D boxes from labels3d/boxes.csv BEFORE
        # connecting the interactive handler, so populating the layer doesn't
        # trigger segmentation on every saved box.
        if hasattr(self.label_widget, "_load_existing_3D_boxes"):
            self.label_widget._load_existing_3D_boxes()

        # Wire shared 3D boxes layer to the label widget's 3D handler so it
        # can be chosen from the Interactive Layer combo ("3D Bounding Boxes").
        # The handler itself ignores the event unless the combo selects it.
        if hasattr(self.label_widget, "_on_3D_boxes_changed"):
            self.boxes_3D_layer.events.data.connect(
                self.label_widget._on_3D_boxes_changed
            )

        # Refresh the label widget's interactive-layer combo so it lists the
        # newly-distributed Label box / 3D Bounding Boxes layers.
        if hasattr(self.label_widget, "_refresh_interactive_layer_combo"):
            self.label_widget._refresh_interactive_layer_combo()

        # Augment widget needs: image, labels
        self.augment_widget.image_layer = self.image_layer
        self.augment_widget.annotation_layer = self.annotations_layer
        self.augment_widget.annotations_layers = self.annotations_layers
        self.augment_widget.boxes_layer = (
            self.boxes_layer
        )  # Provide boxes layer for augmentation if needed

        # Segment widget needs: image, labels, predictions, points, shapes
        self.segment_widget.image_layer = self.image_layer
        self.segment_widget.annotation_layer = self.annotations_layer
        self.segment_widget.annotations_layers = self.annotations_layers
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
        """Handle 2D boxes layer data changes - show dialog to copy predictions to new ROI."""

        # Only respond to 'added' events (same check as in nd_easy_label)
        if event.action != "added":
            return

        boxes_layer = event.source
        if len(boxes_layer.data) == 0:
            return

        if (
            not hasattr(self, "predictions_layers")
            or not self.predictions_layers
        ):
            print("No predictions layers available to copy from")
            return

        available = self._find_available_predictions(boxes_layer)
        if not available:
            print("⚠️ No predictions exist at this location")
            return

        self._show_copy_predictions_dialog(
            box_layer=boxes_layer,
            available_predictions=available,
            allow_source_choice=False,
        )

    # ------------------------------------------------------------------
    # Shared helpers for "copy predictions to labels" (2D and 3D paths)
    # ------------------------------------------------------------------
    def _find_available_predictions(self, box_layer):
        """Return ``{name: pred_layer}`` for predictions with non-zero data
        inside the active box on ``box_layer``.

        Uses the shared ``_compute_crop_slice`` helper on
        :class:`BaseNDApp` (inherited by ``label_widget``) so this works
        uniformly for 2D Shapes and 3D BoundingBoxLayer sources.
        """
        available = {}
        if box_layer is None:
            return available
        if (
            not hasattr(self, "predictions_layers")
            or not self.predictions_layers
        ):
            return available
        # _compute_crop_slice lives on BaseNDApp; reuse via label_widget
        compute = getattr(self.label_widget, "_compute_crop_slice", None)
        if compute is None:
            return available
        for name, pred_layer in self.predictions_layers.items():
            try:
                slc = compute(box_layer, pred_layer)
                if slc is None:
                    print(f"✗ {name}: no slice for ROI")
                    continue
                pred_roi = pred_layer.data[slc]
                if np.any(pred_roi):
                    available[name] = pred_layer
                    print(f"✓ {name}: predictions found in ROI")
                else:
                    print(f"✗ {name}: no non-zero predictions in ROI")
            except (IndexError, ValueError, TypeError) as e:
                print(f"✗ {name}: no predictions at this location ({e})")
        return available

    def _copy_predictions_into_labels(self, box_layer, pred_layer):
        """Copy ``pred_layer`` ROI into ``annotations_layer`` ROI using
        per-target slices from ``_compute_crop_slice``."""
        compute = getattr(self.label_widget, "_compute_crop_slice", None)
        if compute is None or self.annotations_layer is None:
            print("✗ Cannot copy: missing helper or annotations layer")
            return False
        preds_slice = compute(box_layer, pred_layer)
        labels_slice = compute(box_layer, self.annotations_layer)
        if preds_slice is None or labels_slice is None:
            print("✗ Cannot copy: failed to compute crop slice")
            return False
        try:
            self.annotations_layer.data[labels_slice] = pred_layer.data[
                preds_slice
            ]
            self.annotations_layer.refresh()
            return True
        except (IndexError, ValueError) as e:
            print(f"✗ Copy failed: {e}")
            return False

    def _show_copy_predictions_dialog(
        self,
        box_layer=None,
        available_predictions=None,
        allow_source_choice=False,
    ):
        """Show the copy-predictions dialog.

        - When ``allow_source_choice`` is False (the 2D auto-trigger path):
          ``box_layer`` and ``available_predictions`` must be supplied and
          only the predictions combo is shown.
        - When ``allow_source_choice`` is True (the 3D button path): a
          Source ROI combo is also shown (Label box / 3D Bounding Boxes),
          and available predictions are recomputed when the source ROI
          changes.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Copy Predictions to ROI")
        dialog.setMinimumWidth(420)

        layout = QVBoxLayout(dialog)
        layout.addWidget(
            QLabel(
                "Choose predictions layer and press 'Copy' to copy "
                "predictions into the labels layer."
            )
        )

        # Optional source ROI combo
        source_combo = None
        source_layers = {}
        if allow_source_choice:
            if (
                hasattr(self, "boxes_3D_layer")
                and self.boxes_3D_layer is not None
            ):
                source_layers["3D Bounding Boxes"] = self.boxes_3D_layer
            if hasattr(self, "boxes_layer") and self.boxes_layer is not None:
                source_layers["Label box"] = self.boxes_layer
            if not source_layers:
                print("⚠️ No ROI source layers available")
                return
            row = QHBoxLayout()
            row.addWidget(QLabel("Source ROI:"))
            source_combo = QComboBox()
            for name in source_layers:
                source_combo.addItem(name)
            row.addWidget(source_combo)
            layout.addLayout(row)

        # Predictions combo
        row = QHBoxLayout()
        row.addWidget(QLabel("Predictions Layer:"))
        predictions_combo = QComboBox()
        row.addWidget(predictions_combo)
        layout.addLayout(row)

        status_label = QLabel("")
        layout.addWidget(status_label)

        # Buttons
        button_row = QHBoxLayout()
        copy_button = QPushButton("Copy")
        cancel_button = QPushButton("Cancel")
        button_row.addWidget(copy_button)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)

        # Mutable state shared with refresh()
        state = {
            "box_layer": box_layer,
            "available": dict(available_predictions or {}),
        }

        def refresh_predictions():
            if allow_source_choice and source_combo is not None:
                state["box_layer"] = source_layers.get(
                    source_combo.currentText()
                )
                state["available"] = self._find_available_predictions(
                    state["box_layer"]
                )
            predictions_combo.blockSignals(True)
            predictions_combo.clear()
            for name in state["available"]:
                predictions_combo.addItem(name)
            predictions_combo.blockSignals(False)
            has_any = bool(state["available"])
            copy_button.setEnabled(has_any)
            if not has_any:
                status_label.setText(
                    "⚠️ No predictions overlap the selected ROI."
                )
            else:
                status_label.setText("")

        def do_copy():
            name = predictions_combo.currentText()
            pred_layer = state["available"].get(name)
            box_layer_now = state["box_layer"]
            if pred_layer is None or box_layer_now is None:
                return
            if self._copy_predictions_into_labels(box_layer_now, pred_layer):
                print(f"✓ Copied {name} predictions to labels layer")
                dialog.accept()

        if source_combo is not None:
            source_combo.currentTextChanged.connect(
                lambda _t: refresh_predictions()
            )
        copy_button.clicked.connect(do_copy)
        cancel_button.clicked.connect(dialog.reject)

        refresh_predictions()
        dialog.exec_()

    def show_copy_predictions_dialog(self):
        """Public entry point used by child widgets (e.g. the label widget's
        'Copy predictions to labels' button) to open the copy-predictions
        dialog with a Source ROI chooser (2D Label box or 3D Bounding
        Boxes)."""
        if (
            not hasattr(self, "predictions_layers")
            or not self.predictions_layers
        ):
            from qtpy.QtWidgets import QMessageBox

            QMessageBox.information(
                self,
                "Copy Predictions",
                "No predictions layers available to copy from.",
            )
            return
        self._show_copy_predictions_dialog(allow_source_choice=True)
