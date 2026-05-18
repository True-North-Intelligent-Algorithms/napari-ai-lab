import napari
import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..models import ImageDataModel
from ..Segmenters.InteractiveSegmenters import InteractiveSegmenterBase
from ..utility import create_artifact_name, get_current_slice_indices
from .base_nd_app import BaseNDApp


class NDEasyLabel(BaseNDApp):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        image_data_model: "ImageDataModel" = None,
        embedded: bool = False,
        axes_to_collapse: str | list[str] | None = None,
    ):
        super().__init__(viewer, image_data_model)
        self.embedded = embedded
        self.axes_to_collapse = axes_to_collapse
        self.frameworks = InteractiveSegmenterBase.get_registered_frameworks()
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""

        self.setLayout(QVBoxLayout())

        # Add directory selection button (only in standalone mode)
        if not self.embedded:
            self.dir_btn = QPushButton("Open Image Directory")
            self.dir_btn.clicked.connect(self._on_open_directory)
            self.layout().addWidget(self.dir_btn)

        # Add Interactive Segmenter selection
        self.segmenter_label = QLabel("Interactive Segmenter:")
        self.layout().addWidget(self.segmenter_label)

        self.segmenter_combo = QComboBox()
        self.segmenter_combo.currentTextChanged.connect(
            self._on_segmenter_changed
        )
        self.layout().addWidget(self.segmenter_combo)

        # Parameter form widget for segmenter parameters (from base)
        self.layout().addWidget(self._create_segmenter_parameter_form())

        # Connect to axis changes to reinitialize segmenter
        self.segmenter_parameter_form.axis_changed.connect(
            self._on_axis_changed
        )

        # Populate segmenter combo with registered frameworks
        self._populate_segmenter_combo()

        # Single button saves annotations, boxes, and label patches
        self.save_project_btn = QPushButton("Save Project")
        self.save_project_btn.clicked.connect(self._on_save_project)
        self.layout().addWidget(self.save_project_btn)

        # Current label number spinbox
        label_num_row = QHBoxLayout()
        label_num_row.addWidget(QLabel("Current Label:"))
        self.label_num_spinbox = QSpinBox()
        self.label_num_spinbox.setMinimum(1)
        self.label_num_spinbox.setMaximum(65535)
        self.label_num_spinbox.setValue(self.current_label_num)
        self.label_num_spinbox.valueChanged.connect(self._on_label_num_changed)
        label_num_row.addWidget(self.label_num_spinbox)
        self.layout().addLayout(label_num_row)

        self.auto_increment_checkbox = QCheckBox("Auto-increment label")
        self.auto_increment_checkbox.setChecked(True)
        self.layout().addWidget(self.auto_increment_checkbox)

        # Combo: choose which layer drives live interactive segmentation.
        # Populated dynamically ("None" plus whatever box-like layers exist).
        interactive_layer_row = QHBoxLayout()
        interactive_layer_row.addWidget(QLabel("Interactive Layer:"))
        self.interactive_layer_combo = QComboBox()
        self.interactive_layer_combo.addItem("None")
        interactive_layer_row.addWidget(self.interactive_layer_combo)
        self.layout().addLayout(interactive_layer_row)

        # Button to add an interactive-label (ROI box) shapes layer
        self.add_interactive_layer_btn = QPushButton(
            "Add Interactive-Label Layer"
        )
        self.add_interactive_layer_btn.clicked.connect(
            self._on_add_interactive_label_layer
        )
        self.layout().addWidget(self.add_interactive_layer_btn)

        # Button to add an interactive 3D bounding-boxes layer
        self.add_interactive_3D_boxes_btn = QPushButton(
            "Add Interactive 3D Boxes Layer"
        )
        self.add_interactive_3D_boxes_btn.clicked.connect(
            self._on_add_interactive_3D_boxes_layer
        )
        self.layout().addWidget(self.add_interactive_3D_boxes_btn)

        # Holder for the interactive-label shapes layer (created on demand)
        self.interactive_labels_layer = None
        # Holder for the interactive 3D bounding-boxes layer (created on demand)
        self.interactive_3D_boxes_layer = None

        # Add stretch to push everything to the top (prevents button stretching)
        self.layout().addStretch()

    def _populate_segmenter_combo(self):
        """Populate the segmenter combo box with registered frameworks."""
        # Clear existing items
        self.segmenter_combo.clear()

        # Get registered frameworks
        frameworks = self.frameworks

        if frameworks:
            # Create custom ordering with Square2D first, then others alphabetically
            framework_names = list(frameworks.keys())
            if "Square2D" in framework_names:
                # Put Square2D first, then sort the rest
                framework_names.remove("Square2D")
                ordered_names = ["Square2D"] + sorted(framework_names)
            else:
                # Fallback to alphabetical if Square2D not found
                ordered_names = sorted(framework_names)

            # Add frameworks to combo box in custom order
            for name in ordered_names:
                self.segmenter_combo.addItem(name)

            # Select first item if available (which will be Square2D)
            if self.segmenter_combo.count() > 0:
                self.segmenter_combo.setCurrentIndex(0)
                self._on_segmenter_changed(self.segmenter_combo.currentText())
        else:
            # No frameworks registered
            self.segmenter_combo.addItem("No segmenters available")
            self.segmenter_combo.setEnabled(False)

    def _post_segmenter_selection(self):
        # Delegate the real work to the initializer
        self._initialize_segmenter()

    def _on_axis_changed(self, new_axis):
        """Handle axis selection changes - reinitialize segmenter for new axis."""
        print(f"Axis changed to: {new_axis}")
        # Reinitialize segmenter with new axis configuration
        self._initialize_segmenter()

    def _on_label_num_changed(self, value):
        """Sync current_label_num when the spinbox is edited directly."""
        self.current_label_num = value

    def _apply_segmenter_mask(self, mask, segmentation_indices):
        """Write a segmenter mask into the annotation layer.

        If the segmenter exposes ``last_roi_bbox`` (a tuple of slices marking
        the touched sub-region), the boolean-index write is restricted to
        that sub-region.  Otherwise the full mask is scanned.  Equivalent
        result either way — just much faster for ROI-style segmenters
        (Otsu2D / Otsu3D), especially in 3D.
        """
        target = self.annotation_layer.data[segmentation_indices]
        roi_bbox = getattr(self.segmenter, "last_roi_bbox", None)

        if roi_bbox is not None:
            sub_mask = mask[roi_bbox]
            sub_target = target[roi_bbox]
            nz = sub_mask != 0
            sub_target[nz] = sub_mask[nz] * self.current_label_num
            # Clear voxels in ROI that previously held the current label but
            # are now background in the new mask (e.g. tighter interactive
            # re-segmentation).
            clear = (sub_mask == 0) & (sub_target == self.current_label_num)
            sub_target[clear] = 0
        else:
            nz = mask != 0
            target[nz] = mask[nz] * self.current_label_num
            clear = (mask == 0) & (target == self.current_label_num)
            target[clear] = 0

    def _maybe_increment_label(self):
        """Auto-increment current_label_num and sync the spinbox if enabled."""
        if self.auto_increment_checkbox.isChecked():
            self.current_label_num += 1
            self.label_num_spinbox.blockSignals(True)
            self.label_num_spinbox.setValue(self.current_label_num)
            self.label_num_spinbox.blockSignals(False)

    def _initialize_segmenter(self):
        """Initialize predictor if an image is loaded and a segmenter exists."""
        if self.image_layer is None:
            # Nothing to initialize yet
            return

        try:
            parent_dir = self.image_data_model.get_parent_directory()

            image_paths = self.image_data_model.get_image_paths()
            image_name = image_paths[self.current_image_index].stem

            selected_axis = self.segmenter_parameter_form.get_selected_axis()

            indices = get_current_slice_indices(
                self.viewer.dims.current_step, selected_axis
            )

            image_data = self.image_layer.data[indices]

            # Get current step tuple
            step = self.viewer.dims.current_step

            # Create artifact name from non-spatial dims
            image_name = create_artifact_name(image_name, step, selected_axis)

            self.segmenter = (
                self.segmenter_parameter_form.sync_nd_operation_instance(
                    getattr(self, "segmenter", None)
                )
            )

            result = self.segmenter.initialize_predictor(
                image_data, str(parent_dir), image_name
            )
            # result may be None for implementations that don't return a value
            if result is not None and not result.get("success", True):
                QMessageBox.critical(
                    self,
                    result.get("error_type", "Error"),
                    result.get("message", "Unknown error"),
                )
        except (
            AttributeError,
            ValueError,
            TypeError,
            RuntimeError,
            IndexError,
            OSError,
        ) as e:
            # Keep behavior similar to previous implementation
            print(f"Error during segmenter initialization: {e}")

    def _on_points_changed(self, event):
        """Handle points layer data changes - creates segmentation around point using current segmenter."""
        points_layer = event.source
        if event.action == "added" and len(points_layer.data) > 0:
            # Get the most recently added point (last in the list)
            # TODO: send all points to segmenter for multi-point support
            latest_point = points_layer.data[-1]
            print(f"Point added at location: {latest_point}")

            # Print all points for reference
            print(f"Total points: {len(points_layer.data)}")
            for i, point in enumerate(points_layer.data):
                print(f"  Point {i+1}: {point}")

            # Get current image data
            if self.image_layer is None:
                print("No image layer available")
                return

            selected_axis = self.segmenter_parameter_form.get_selected_axis()

            if selected_axis == "YX" or selected_axis == "YXC":
                latest_point = latest_point[
                    -2:
                ]  # Use last two coordinates for 2D YX
            elif selected_axis == "ZYX" or selected_axis == "ZYXC":
                latest_point = latest_point[
                    -3:
                ]  # Use last three coordinates for 3D ZYX

            indices = get_current_slice_indices(
                self.viewer.dims.current_step, selected_axis
            )

            # if image has more dims than annotation, adjust segmentation indices
            if self.image_layer.data.ndim > self.annotation_layer.data.ndim:

                segmentation_indices = get_current_slice_indices(
                    self.viewer.dims.current_step,
                    selected_axis,
                    ignore_channel=True,
                )
            else:
                segmentation_indices = indices

            image_data = self.image_layer.data[indices]

            # Ensure segmenter is synced with current parameters before use
            self.segmenter = (
                self.segmenter_parameter_form.sync_nd_operation_instance(
                    self.segmenter
                )
            )

            # Call segmenter with the latest point
            try:
                mask = self.segmenter.segment(
                    image_data,
                    points=[latest_point],
                    shapes=None,
                )

                # Apply the mask to the labels layer (restricted to ROI bbox
                # if the segmenter advertises one).
                self._apply_segmenter_mask(mask, segmentation_indices)

                print(
                    f"Added segmentation with label {self.current_label_num}"
                )
                self._maybe_increment_label()

                self.annotation_layer.refresh()

            except (
                AttributeError,
                ValueError,
                TypeError,
                RuntimeError,
                IndexError,
            ) as e:
                print(f"Error during segmentation: {e}")
                import traceback

                traceback.print_exc()

    def _on_shapes_changed(self, event):
        """Handle shapes layer data changes - prints shape information."""
        shapes_layer = event.source
        if event.action == "added" and len(shapes_layer.data) > 0:
            # Get the most recently added shape (last in the list)
            latest_shape = shapes_layer.data[-1]
            print(
                f"Shape added: {latest_shape.shape} with {len(latest_shape)} points"
            )
            print(f"Shape coordinates: {latest_shape}")

            # Print all shapes for reference
            print(f"Total shapes: {len(shapes_layer.data)}")
            for i, shape in enumerate(shapes_layer.data):
                print(f"  Shape {i+1}: {shape.shape} with {len(shape)} points")

    def _on_boxes_changed(self, event):
        """Handle boxes layer data changes - just logs the new box; saving happens with annotations.

        If the Interactive Layer combo selects this "Label box" layer, also
        runs interactive 2D segmentation using the currently active box.
        """
        if event.action == "added":
            boxes_layer = event.source
            if len(boxes_layer.data) > 0:
                box = boxes_layer.data[-1]
                ystart = int(np.floor(np.min(box[:, -2])))
                yend = int(np.ceil(np.max(box[:, -2])))
                xstart = int(np.floor(np.min(box[:, -1])))
                xend = int(np.ceil(np.max(box[:, -1])))
                print(
                    f"New ROI added: y=[{ystart}, {yend}], x=[{xstart}, {xend}] "
                    f"(total boxes: {len(boxes_layer.data)})"
                )

        # Optional: drive interactive 2D segmentation from the active Label box.
        if event.action in (
            "added",
            "changed",
        ) and self._is_chosen_interactive_layer(event.source):
            if self.image_layer is None or self.annotation_layer is None:
                return
            box = self._get_active_box(event.source)
            if box is None:
                return
            self._process_2D_box(
                box, increment_label=(event.action == "added")
            )

    def _on_3D_boxes_changed(self, event):
        """Run 3D segmentation when a 3D bounding box is added or changed.

        Only triggers when the Interactive Layer combo selects this layer.
        For the "Interactive 3D Boxes" choice, also enforces one-box-at-a-time
        (the previous box is replaced by the new one).
        """
        if event.action not in ("added", "changed"):
            return

        layer = event.source
        if not self._is_chosen_interactive_layer(layer):
            return

        if len(layer.data) == 0:
            return
        if self.image_layer is None or self.annotation_layer is None:
            print("No image / annotation layer available")
            return

        box = self._get_active_box(layer)
        if box is None:
            return

        chosen = self.interactive_layer_combo.currentText()
        # Interactive variant: keep only the most recently added box.
        if (
            chosen == "Interactive 3D Boxes"
            and event.action == "added"
            and len(layer.data) > 1
        ):
            QTimer.singleShot(
                0,
                lambda b=box, lyr=layer: self._trim_layer_to_single_box(
                    lyr, b
                ),
            )

        self._process_3D_box(box, increment_label=(event.action == "added"))

    def _process_3D_box(self, box, increment_label):
        """Send a 3D bounding box to the current segmenter and apply the mask."""
        selected_axis = self.segmenter_parameter_form.get_selected_axis()

        # Keep only the spatial Z,Y,X columns (the box may carry extra leading dims).
        spatial_box = box[:, -3:]

        indices = get_current_slice_indices(
            self.viewer.dims.current_step, selected_axis
        )
        if self.image_layer.data.ndim > self.annotation_layer.data.ndim:
            segmentation_indices = get_current_slice_indices(
                self.viewer.dims.current_step,
                selected_axis,
                ignore_channel=True,
            )
        else:
            segmentation_indices = indices

        image_data = self.image_layer.data[indices]

        self.segmenter = (
            self.segmenter_parameter_form.sync_nd_operation_instance(
                self.segmenter
            )
        )

        try:
            mask = self.segmenter.segment(
                image_data,
                points=None,
                shapes=[spatial_box],
            )
            self._apply_segmenter_mask(mask, segmentation_indices)
            print(
                f"3D box segmentation applied with label {self.current_label_num}"
            )
            if increment_label:
                self._maybe_increment_label()
            self.annotation_layer.refresh()
        except (
            AttributeError,
            ValueError,
            TypeError,
            RuntimeError,
            IndexError,
        ) as e:
            print(f"Error during 3D box segmentation: {e}")
            import traceback

            traceback.print_exc()

    # === Interactive layer combo helpers ===
    def _interactive_layer_map(self):
        """Map combo entry name -> layer instance (None when not present)."""
        return {
            "Interactive 3D Boxes": getattr(
                self, "interactive_3D_boxes_layer", None
            ),
            "Interactive Boxes": getattr(
                self, "interactive_labels_layer", None
            ),
            "3D Bounding Boxes": getattr(self, "boxes_3D_layer", None),
            "Label box": getattr(self, "boxes_layer", None),
        }

    def _refresh_interactive_layer_combo(self):
        """Repopulate the Interactive Layer combo with the layers that exist."""
        combo = getattr(self, "interactive_layer_combo", None)
        if combo is None:
            return
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("None")
        for name, layer in self._interactive_layer_map().items():
            if layer is not None and layer in self.viewer.layers:
                combo.addItem(name)
        idx = combo.findText(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _is_chosen_interactive_layer(self, layer):
        """True if `layer` is the one selected in the Interactive Layer combo."""
        combo = getattr(self, "interactive_layer_combo", None)
        if combo is None:
            return False
        chosen = combo.currentText()
        if chosen == "None":
            return False
        return self._interactive_layer_map().get(chosen) is layer

    def _get_active_box(self, layer):
        """Return the active (selected) box on `layer`, or the last one if none selected."""
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

    def _trim_layer_to_single_box(self, layer, box):
        """Replace `layer.data` with a single box (one-at-a-time interactive layers)."""
        if layer not in self.viewer.layers:
            return
        try:
            layer.data = [box]
        except (ValueError, RuntimeError, TypeError) as e:
            print(f"Could not trim interactive layer to single box: {e}")

    def _process_2D_box(self, box, increment_label):
        """Segment using a 2D ROI box (active box from a multi-box layer)."""
        selected_axis = self.segmenter_parameter_form.get_selected_axis()
        box = np.asarray(box)

        if selected_axis in ("YX", "YXC"):
            spatial_box = box[:, -2:]
        elif selected_axis in ("ZYX", "ZYXC"):
            spatial_box = box[:, -3:]
        else:
            spatial_box = box

        indices = get_current_slice_indices(
            self.viewer.dims.current_step, selected_axis
        )
        if self.image_layer.data.ndim > self.annotation_layer.data.ndim:
            segmentation_indices = get_current_slice_indices(
                self.viewer.dims.current_step,
                selected_axis,
                ignore_channel=True,
            )
        else:
            segmentation_indices = indices

        image_data = self.image_layer.data[indices]

        self.segmenter = (
            self.segmenter_parameter_form.sync_nd_operation_instance(
                self.segmenter
            )
        )

        try:
            mask = self.segmenter.segment(
                image_data, points=None, shapes=[spatial_box]
            )
            self._apply_segmenter_mask(mask, segmentation_indices)
            print(
                f"2D box segmentation applied with label {self.current_label_num}"
            )
            if increment_label:
                self._maybe_increment_label()
            self.annotation_layer.refresh()
        except (
            AttributeError,
            ValueError,
            TypeError,
            RuntimeError,
            IndexError,
        ) as e:
            print(f"Error during 2D box segmentation: {e}")
            import traceback

            traceback.print_exc()

    def _on_add_interactive_label_layer(self):
        """Create a shapes layer used to drive box-based interactive segmentation."""
        if self.annotation_layer is None:
            QMessageBox.warning(
                self,
                "No image loaded",
                "Load an image before adding the interactive-label layer.",
            )
            return

        # If it already exists and is still in the viewer, just select it.
        if (
            self.interactive_labels_layer is not None
            and self.interactive_labels_layer in self.viewer.layers
        ):
            self.viewer.layers.selection.active = self.interactive_labels_layer
            return

        annotation_ndim = len(self.annotation_layer.data.shape)
        self.interactive_labels_layer = self.viewer.add_shapes(
            ndim=annotation_ndim,
            name="Interactive Labels",
            face_color="transparent",
            edge_color="yellow",
            edge_width=3,
            scale=self.image_data_model.get_scale(
                axes_to_collapse=self.axes_to_collapse
            )
            or None,
        )
        self.interactive_labels_layer.events.data.connect(
            self._on_interactive_roi_change
        )
        self.viewer.layers.selection.active = self.interactive_labels_layer
        print("✨ Added Interactive Labels shapes layer")
        self._refresh_interactive_layer_combo()
        idx = self.interactive_layer_combo.findText("Interactive Boxes")
        if idx >= 0:
            self.interactive_layer_combo.setCurrentIndex(idx)

    def _on_add_interactive_3D_boxes_layer(self):
        """Create an interactive 3D bounding-boxes layer that drives 3D segmentation."""
        if self.annotation_layer is None:
            QMessageBox.warning(
                self,
                "No image loaded",
                "Load an image before adding the interactive 3D boxes layer.",
            )
            return

        # Reuse if it already exists and is still in the viewer.
        if (
            self.interactive_3D_boxes_layer is not None
            and self.interactive_3D_boxes_layer in self.viewer.layers
        ):
            self.viewer.layers.selection.active = (
                self.interactive_3D_boxes_layer
            )
            return

        from napari_ai_lab.vendored.napari_bbox import BoundingBoxLayer

        annotation_ndim = len(self.annotation_layer.data.shape)
        scale = (
            self.image_data_model.get_scale(
                axes_to_collapse=self.axes_to_collapse
            )
            or None
        )
        self.interactive_3D_boxes_layer = BoundingBoxLayer(
            name="Interactive 3D Boxes",
            edge_color="yellow",
            face_color="transparent",
            edge_width=3,
            ndim=annotation_ndim,
            scale=scale,
        )
        self.viewer.add_layer(self.interactive_3D_boxes_layer)
        self.interactive_3D_boxes_layer.events.data.connect(
            self._on_3D_boxes_changed
        )
        self.viewer.layers.selection.active = self.interactive_3D_boxes_layer
        print("✨ Added Interactive 3D Boxes layer")
        self._refresh_interactive_layer_combo()
        idx = self.interactive_layer_combo.findText("Interactive 3D Boxes")
        if idx >= 0:
            self.interactive_layer_combo.setCurrentIndex(idx)

    def _on_interactive_roi_change(self, event):
        """Handle a new box on the interactive-labels layer: segment within the box."""
        if event.action != "added":
            return

        if not self._is_chosen_interactive_layer(event.source):
            return

        shapes_layer = event.source
        if len(shapes_layer.data) == 0:
            return

        if self.image_layer is None or self.annotation_layer is None:
            print("No image / annotation layer available")
            return

        # Re-entrancy / overlap guard: ignore new boxes while one is still
        # being processed (e.g. user draws a second box very quickly).
        if getattr(self, "_interactive_roi_busy", False):
            return
        self._interactive_roi_busy = True

        # Snapshot the latest box NOW (before deferring), so we don't depend
        # on what happens to shapes_layer.data in the meantime.
        box = np.asarray(shapes_layer.data[-1])

        # Defer the actual work until AFTER napari's add_rectangle generator
        # finishes.  Mutating shapes_layer.data inside this callback would
        # corrupt that generator and trigger an IndexError on mouse release.
        QTimer.singleShot(
            0, lambda: self._process_interactive_roi(shapes_layer, box)
        )

    def _process_interactive_roi(self, shapes_layer, box):
        """Deferred body of _on_interactive_roi_change (runs on next tick)."""
        try:
            if shapes_layer not in self.viewer.layers:
                return

            # Now it's safe to trim the shapes layer to just the latest box.
            try:
                shapes_layer.data = [box]
            except (ValueError, RuntimeError) as e:
                print(f"Could not trim interactive-labels layer: {e}")

            selected_axis = self.segmenter_parameter_form.get_selected_axis()

            # Strip leading non-spatial dims to match what the segmenter expects.
            if selected_axis in ("YX", "YXC"):
                spatial_box = box[:, -2:]
            elif selected_axis in ("ZYX", "ZYXC"):
                spatial_box = box[:, -3:]
            else:
                spatial_box = box

            indices = get_current_slice_indices(
                self.viewer.dims.current_step, selected_axis
            )

            # Adjust segmentation indices when image has extra dims (e.g. channel)
            if self.image_layer.data.ndim > self.annotation_layer.data.ndim:
                segmentation_indices = get_current_slice_indices(
                    self.viewer.dims.current_step,
                    selected_axis,
                    ignore_channel=True,
                )
            else:
                segmentation_indices = indices

            image_data = self.image_layer.data[indices]

            # Sync segmenter with current parameters
            self.segmenter = (
                self.segmenter_parameter_form.sync_nd_operation_instance(
                    self.segmenter
                )
            )

            try:
                mask = self.segmenter.segment(
                    image_data,
                    points=None,
                    shapes=[spatial_box],
                )

                self._apply_segmenter_mask(mask, segmentation_indices)

                print(
                    f"Added box-segmentation with label {self.current_label_num}"
                )
                self._maybe_increment_label()

                self.annotation_layer.refresh()

            except (
                AttributeError,
                ValueError,
                TypeError,
                RuntimeError,
                IndexError,
            ) as e:
                print(f"Error during box segmentation: {e}")
                import traceback

                traceback.print_exc()
        finally:
            self._interactive_roi_busy = False

    def _on_save_project(self):
        """Save annotations, boxes, and label patches in one go."""
        # Save annotations
        self._on_save_annotations()

        # Save label patches (only if boxes exist)
        boxes_layer = getattr(self, "boxes_layer", None)
        if boxes_layer is not None and len(boxes_layer.data) > 0:
            self.image_data_model.crop_and_save_label_patches(
                boxes_layer.data,
                self.image_layer.data,
                self.annotation_layer.data,
                self.current_image_index,
            )
            print("Label patches saved.")
        else:
            print("No boxes drawn — skipping label patch save.")

    def _save_label_patches_on_close(self):
        """Called by the close handler to save label patches silently (no dialog)."""
        boxes_layer = getattr(self, "boxes_layer", None)
        if (
            boxes_layer is None
            or len(boxes_layer.data) == 0
            or self.image_data_model is None
            or self.image_layer is None
            or self.annotation_layer is None
        ):
            return
        self.image_data_model.crop_and_save_label_patches(
            boxes_layer.data,
            self.image_layer.data,
            self.annotation_layer.data,
            self.current_image_index,
        )
        print("Label patches saved on close.")

    def _get_current_image_name(self) -> str | None:
        """Return the file name of the currently displayed image, or None."""
        if self.image_data_model is None:
            return None
        try:
            image_paths = self.image_data_model.get_image_paths()
            if 0 <= self.current_image_index < len(image_paths):
                return image_paths[self.current_image_index].name
        except (AttributeError, IndexError, OSError):
            pass
        return None

    def _load_existing_boxes(self):
        """Load boxes from CSV and populate the boxes_layer with saved ROIs.

        In normal mode the rectangle is placed at z=0 (or no leading axis).
        In stacked-sequence mode the rectangle is placed at the frame index
        stored in ``row["frame_index"]`` so that the box appears on the
        correct frame of the stack.
        """
        if self.boxes_layer is None or self.image_data_model is None:
            return

        try:
            all_boxes = self.image_data_model.load_existing_boxes()
        except (OSError, ValueError) as e:
            print(f"Failed to load existing boxes: {e}")
            return

        if not all_boxes:
            return

        shapes = []
        shape_types = []

        for row in all_boxes:
            y0, y1 = row["ystart"], row["yend"]
            x0, x1 = row["xstart"], row["xend"]
            middle = list(row.get("middle_positions", []))

            """
            if stacked:
                frame_idx = row.get("frame_index")
                if frame_idx is None:
                    # Image not found in current stack — skip
                    print(
                        f"⚠️  _load_existing_boxes: '{row['file_name']}' "
                        "not in current stack — skipping"
                    )
                    continue
                # Stacked: leading axis is the frame index, then any middle axes
                leading = [float(frame_idx), *middle]
            else:
                # Non-stacked: only the middle axes (e.g., Z, T) prefix YX
            """
            leading = [float(p) for p in middle]

            # Build a rectangle with the right number of leading dims
            rect = np.array(
                [
                    [*leading, y0, x0],
                    [*leading, y0, x1],
                    [*leading, y1, x1],
                    [*leading, y1, x0],
                ],
                dtype=float,
            )
            shapes.append(rect)
            shape_types.append("rectangle")

        if shapes:
            self.boxes_layer.add(shapes, shape_type=shape_types)
            print(f"📦 Loaded {len(shapes)} existing boxes into boxes_layer")

    # _on_open_directory and load_image_directory inherited from BaseNDApp
    def _set_image_layer(self, image_layer):
        """Set up all annotation layers based on the provided image layer."""
        try:
            # Store the image layer reference
            self.image_layer = image_layer

            # Get image data from the layer
            image_data = image_layer.data

            # Load existing labels or create empty ones (delegated to model)
            labels_data = self.image_data_model.load_existing_annotations(
                image_data.shape,
                self.current_image_index,
                axes_to_collapse=self.axes_to_collapse,
            )

            # Add labels layer and store reference
            annotation_scale = self.image_data_model.get_scale(
                axes_to_collapse=self.axes_to_collapse
            )
            self.annotation_layer = self.viewer.add_labels(
                labels_data,
                name="Labels (Persistent)",
                scale=annotation_scale or None,
            )

            # Add points layer for annotation with point type choices
            self._point_choices = ["positive", "negative"]
            LABEL_COLOR_CYCLE = ["red", "blue"]  # positive=red, negative=blue

            # For annotation layers, we want ndim to match the displayed dimensions
            # This prevents issues with 4D data slice comparisons
            annotation_ndim = len(self.annotation_layer.data.shape)

            # Add points layer and store reference
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
                scale=annotation_scale or None,
            )

            # Connect point event handler
            self.points_layer.events.data.connect(self._on_points_changed)

            # Add shapes layer for region annotation and store reference
            self.shapes_layer = self.viewer.add_shapes(
                name="Shapes Layer",
                edge_color="green",
                face_color="transparent",
                edge_width=2,
                ndim=annotation_ndim,
                scale=annotation_scale or None,
            )

            # Connect shapes event handler
            self.shapes_layer.events.data.connect(self._on_shapes_changed)

            self.boxes_layer = self.viewer.add_shapes(
                ndim=annotation_ndim,
                name="Label box",
                face_color="transparent",
                edge_color="blue",
                edge_width=5,
                text={"string": "{split_set}", "size": 15, "color": "green"},
                scale=annotation_scale or None,
            )

            # Connect boxes layer event handler
            self.boxes_layer.events.data.connect(self._on_boxes_changed)

            # Load any previously saved boxes from CSV
            self._load_existing_boxes()

            # Refresh interactive-layer combo now that boxes_layer exists.
            self._refresh_interactive_layer_combo()

            print(
                f"Successfully set up annotation layers for image layer: {image_layer.name}"
            )
            print("Added points layer for annotation. Click to add points!")
            print(
                "Added shapes layer for region annotation. Draw shapes to define regions!"
            )

            # Initialize or reinitialize segmenter for the new image layer
            self._initialize_segmenter()

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
                f"An error occurred while setting up annotation layers: {str(e)}",
            )
