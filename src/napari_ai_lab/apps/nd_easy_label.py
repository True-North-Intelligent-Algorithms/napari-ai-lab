import contextlib

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
    # Label index used in the working layer for all interactive segmenter
    # output.  Chosen because napari's default labels colormap renders 7 as
    # a bright green, making the "uncommitted" region easy to spot.
    WORKING_LABEL_INDEX = 6

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

        # "Active Annotation Layer" combo at the top of the widget.  All
        # downstream actions (segmenters, augmenters, commit/erase,
        # painting, save, ...) operate against ``self.annotation_layer``
        # which always points at the currently-selected entry of
        # :attr:`annotations_layers`.
        active_annot_row = QHBoxLayout()
        active_annot_row.addWidget(QLabel("Active Annotation Layer:"))
        self.active_annotation_combo = QComboBox()
        self.active_annotation_combo.currentTextChanged.connect(
            self._on_active_annotation_changed
        )
        active_annot_row.addWidget(self.active_annotation_combo)
        self.layout().addLayout(active_annot_row)

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

        # Commit / Erase row.  Interactive segmenters paint into the
        # "Labels (Working)" layer with WORKING_LABEL_INDEX; the user
        # then either commits (move to Persistent with current_label_num,
        # auto-incrementing if enabled) or erases (clear working layer).
        commit_row = QHBoxLayout()
        self.commit_working_btn = QPushButton("Commit")
        self.commit_working_btn.setToolTip(
            "Move Labels (Working) into Labels (Persistent) "
            "with the current label number"
        )
        self.commit_working_btn.clicked.connect(self._on_commit_working)
        commit_row.addWidget(self.commit_working_btn)

        self.erase_working_btn = QPushButton("Erase")
        self.erase_working_btn.setToolTip(
            "Clear the Labels (Working) layer (discard uncommitted "
            "interactive segmentation)"
        )
        self.erase_working_btn.clicked.connect(self._on_erase_working)
        commit_row.addWidget(self.erase_working_btn)
        self.layout().addLayout(commit_row)

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

        # Live readout of the active 2D / 3D box size (in pixels).
        self.active_box_size_label = QLabel("Active box size: —")
        self.layout().addWidget(self.active_box_size_label)

        # Button to open a secondary napari viewer showing just the current
        # image and labels (useful for side-by-side inspection).
        # Combo controls how the 2nd-viewer view is built: full volume vs
        # cropped to the active 3D bounding box (sliced views so live updates
        # in the primary viewer still propagate).
        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel("Label preview layer:"))
        self.label_preview_combo = QComboBox()
        self.label_preview_combo.addItems(["None", "3D Bounding Box"])
        preview_row.addWidget(self.label_preview_combo)
        self.layout().addLayout(preview_row)

        self.show_labels_in_second_viewer_btn = QPushButton(
            "Show labels in 2nd Napari"
        )
        self.show_labels_in_second_viewer_btn.clicked.connect(
            self._on_show_labels_in_second_viewer
        )
        self.layout().addWidget(self.show_labels_in_second_viewer_btn)

        # Launch the interactive local-ML plugin on the active 3D bbox crop.
        self.local_ml_btn = QPushButton("Local Machine Learning")
        self.local_ml_btn.setToolTip(
            "Open a new napari viewer on the active 3D bounding box crop "
            "with the interactive local machine-learning plugin loaded "
            "and ready to go."
        )
        self.local_ml_btn.clicked.connect(self._on_local_machine_learning)
        self.layout().addWidget(self.local_ml_btn)

        # Button to open the copy-predictions-to-labels dialog (handles both
        # 2D "Label box" and 3D "3D Bounding Boxes" source ROIs).  Only
        # functional when this widget is hosted inside NDAILab (which owns
        # the predictions layers and wires ``self.ai_lab``).
        self.copy_predictions_to_labels_btn = QPushButton(
            "Copy predictions to labels"
        )
        self.copy_predictions_to_labels_btn.clicked.connect(
            self._on_copy_predictions_to_labels
        )
        self.layout().addWidget(self.copy_predictions_to_labels_btn)

        # Holder for the secondary viewer (kept alive across button clicks).
        self._second_viewer = None

        # Holder for the interactive-label shapes layer (created on demand)
        self.interactive_labels_layer = None
        # Holder for the interactive 3D bounding-boxes layer (created on demand)
        self.interactive_3D_boxes_layer = None

        # Context of the most recent interactive segmentation, used by
        # segmenters that opt-in to live parameter updates (e.g. RegionGrow3D).
        # Populated by _on_points_changed; consumed by
        # _on_segmenter_parameters_changed -> _rerun_last_interactive_segmentation.
        self._last_interactive_segmentation = None

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
        """Write a segmenter mask into the **working** labels layer.

        All interactive segmenter output lands in ``self.working_layer`` with
        :data:`WORKING_LABEL_INDEX` (so the colour is consistent and the
        result is clearly "uncommitted").  The user later promotes it to
        ``self.annotation_layer`` with ``current_label_num`` via Commit.

        If the segmenter exposes ``last_roi_bbox`` (a tuple of slices marking
        the touched sub-region), the boolean-index write is restricted to
        that sub-region.  This also clears stale working-layer voxels inside
        the ROI that aren't covered by the new mask — important so that
        re-running a segmenter (e.g. RegionGrow3D live param tuning) shrinks
        the visible region instead of leaking previous results.
        """
        if self.working_layer is None:
            print(
                "_apply_segmenter_mask: no working layer; "
                "falling back to annotation layer"
            )
            target = self.annotation_layer.data[segmentation_indices]
            value = self.current_label_num
        else:
            target = self.working_layer.data[segmentation_indices]
            value = self.WORKING_LABEL_INDEX

        roi_bbox = getattr(self.segmenter, "last_roi_bbox", None)

        if roi_bbox is not None:
            sub_mask = mask[roi_bbox]
            sub_target = target[roi_bbox]
            nz = sub_mask != 0
            sub_target[nz] = value
            clear = (sub_mask == 0) & (sub_target == value)
            sub_target[clear] = 0
        else:
            nz = mask != 0
            target[nz] = value

        if self.working_layer is not None:
            self.working_layer.refresh()

    def _on_commit_working(self):
        """Promote the working layer into the persistent annotation layer.

        All voxels in ``working_layer`` equal to :data:`WORKING_LABEL_INDEX`
        are copied into ``annotation_layer`` with the user's current label
        number, then the working layer is cleared and the label number is
        auto-incremented (when the checkbox is on), matching the previous
        per-segmentation behaviour.
        """
        if self.working_layer is None or self.annotation_layer is None:
            return
        work = self.working_layer.data
        nz = work == self.WORKING_LABEL_INDEX
        if not nz.any():
            print("Commit: working layer is empty — nothing to commit.")
            return
        self.annotation_layer.data[nz] = self.current_label_num
        work[nz] = 0
        self.annotation_layer.refresh()
        self.working_layer.refresh()
        print(f"Committed working layer with label {self.current_label_num}")
        self._maybe_increment_label()
        # Invalidate any stashed live-rerun context: the seed has been
        # committed, so further parameter tweaks should not repaint it.
        self._last_interactive_segmentation = None

    def _on_erase_working(self):
        """Clear the working labels layer (discard uncommitted segmentation)."""
        if self.working_layer is None:
            return
        self.working_layer.data[...] = 0
        self.working_layer.refresh()
        print("Erased working layer")
        self._last_interactive_segmentation = None

    def _maybe_increment_label(self):
        """Auto-increment current_label_num and sync the spinbox if enabled."""
        if self.auto_increment_checkbox.isChecked():
            self.current_label_num += 1
            self.label_num_spinbox.blockSignals(True)
            self.label_num_spinbox.setValue(self.current_label_num)
            self.label_num_spinbox.blockSignals(False)

    def _compute_preview_crop_slices(self):
        """Return ``(image_slice, labels_slice)`` for the 2nd-viewer crop.

        Uses the active 3D bounding box on ``boxes_3D_layer`` to crop both
        the image and the annotation layers, via the shared
        ``_compute_crop_slice`` helper on :class:`BaseNDApp`.  Returns
        ``(None, None)`` if no usable box is available so the caller can
        fall back to the full-volume preview.
        """
        box_layer = getattr(self, "boxes_3D_layer", None)
        if (
            box_layer is None
            or self.annotation_layer is None
            or self.image_layer is None
        ):
            return None, None
        labels_slice = self._compute_crop_slice(
            box_layer, self.annotation_layer
        )
        image_slice = self._compute_crop_slice(box_layer, self.image_layer)
        if labels_slice is None:
            return None, None
        return image_slice, labels_slice

    def _on_show_labels_in_second_viewer(self):
        """Open (or refresh) a second napari viewer with image + both labels.

        Adds three layers — image, persistent labels, and the working labels
        layer — so the user can compare against the cluttered primary viewer
        and see live interactive segmentation feedback there too.

        The "Label preview layer" combo controls the view:
          - "None": full volumes, as before.
          - "3D Bounding Box": crops all three layers to the active 3D
            bounding box using numpy views, so live writes to the underlying
            buffers (paint, segmenter masks, commit/erase) still propagate.

        Re-pressing the button refreshes the layers in the existing
        secondary viewer if it is still open, otherwise a new one is created.
        """
        if self.image_layer is None or self.annotation_layer is None:
            QMessageBox.warning(
                self,
                "No image/labels loaded",
                "Load an image and create labels before opening a second viewer.",
            )
            return

        import napari

        # If the previous viewer was closed by the user, drop the stale ref.
        if self._second_viewer is not None:
            try:
                _ = self._second_viewer.window  # touch to detect closed state
            except (RuntimeError, AttributeError):
                self._second_viewer = None

        if self._second_viewer is None:
            self._second_viewer = napari.Viewer(title="Labels Preview")
        else:
            # Refresh: clear any existing layers so we only show the two.
            try:
                self._second_viewer.layers.clear()
            except (RuntimeError, AttributeError):
                self._second_viewer = napari.Viewer(title="Labels Preview")

        scale = None
        if self.image_data_model is not None:
            try:
                scale = (
                    self.image_data_model.get_scale(
                        axes_to_collapse=self.axes_to_collapse
                    )
                    or None
                )
            except (AttributeError, TypeError, ValueError):
                scale = None

        # Decide crop vs full view based on the combo.
        preview_mode = (
            self.label_preview_combo.currentText()
            if hasattr(self, "label_preview_combo")
            else "None"
        )
        image_slice = labels_slice = None
        if preview_mode == "3D Bounding Box":
            image_slice, labels_slice = self._compute_preview_crop_slices()
            if labels_slice is None:
                print(
                    "Label preview: no active 3D bounding box found — "
                    "falling back to full volume."
                )

        image_data = (
            self.image_layer.data[image_slice]
            if image_slice is not None
            else self.image_layer.data
        )
        labels_data = (
            self.annotation_layer.data[labels_slice]
            if labels_slice is not None
            else self.annotation_layer.data
        )

        self._second_viewer.add_image(
            image_data,
            name=self.image_layer.name,
            scale=scale,
        )
        mirror_labels = self._second_viewer.add_labels(
            labels_data,
            name=self.annotation_layer.name,
            scale=scale,
        )
        self._second_labels_layer = mirror_labels

        # Also mirror the working layer (interactive segmenter scratch),
        # so live RegionGrow3D / SAM3D feedback shows up in the 2nd viewer.
        mirror_working = None
        if self.working_layer is not None:
            working_data = (
                self.working_layer.data[labels_slice]
                if labels_slice is not None
                else self.working_layer.data
            )
            mirror_working = self._second_viewer.add_labels(
                working_data,
                name=self.working_layer.name,
                scale=scale,
            )
        self._second_working_layer = mirror_working

        # Forward refresh events from the primary layers to their mirrors so
        # paint / fill / segmenter writes in the first viewer immediately
        # repaint in the second.  Disconnect any previous forwarders first to
        # avoid stacking handlers across repeated button presses.
        primary_sources = [
            (
                self.annotation_layer,
                "_mirror_refresh_cb",
                "_second_labels_layer",
            ),
            (
                self.working_layer,
                "_mirror_working_refresh_cb",
                "_second_working_layer",
            ),
        ]

        for primary, cb_attr, mirror_attr in primary_sources:
            if primary is None:
                continue
            old_cb = getattr(self, cb_attr, None)
            if old_cb is not None:
                for evt_name in ("set_data", "paint", "refresh"):
                    with contextlib.suppress(
                        TypeError, ValueError, RuntimeError, AttributeError
                    ):
                        getattr(primary.events, evt_name).disconnect(old_cb)

            def _make_forwarder(mirror_attr_name):
                def _forward_refresh(_event=None):
                    mirror = getattr(self, mirror_attr_name, None)
                    if mirror is None:
                        return
                    try:
                        mirror.refresh()
                    except (RuntimeError, AttributeError):
                        setattr(self, mirror_attr_name, None)

                return _forward_refresh

            cb = _make_forwarder(mirror_attr)
            setattr(self, cb_attr, cb)
            # paint  -> brush / fill from napari's built-in tools
            # set_data -> bulk data assignment (mirror update for free)
            # refresh -> our segmenter-mask code calls .refresh() after
            #            writing into the underlying numpy buffer directly.
            for evt_name in ("paint", "set_data", "refresh"):
                with contextlib.suppress(AttributeError, TypeError):
                    getattr(primary.events, evt_name).connect(cb)

        # Remember the crop slice (or None) so the morphology "Apply" button
        # knows where in the original annotation_layer.data to write back.
        self._second_labels_slice = labels_slice
        self._second_labels_scale = scale

        # Add (or refresh) the label-morphology dock widget on the 2nd viewer.
        self._ensure_morphology_panel()

    # ------------------------------------------------------------------
    # Label morphology dock (lives on the 2nd napari viewer)
    # ------------------------------------------------------------------
    def _ensure_morphology_panel(self):
        """Add the morphology side panel to the 2nd viewer if missing.

        Re-uses the same panel widget if it's still attached; otherwise
        rebuilds it (e.g. when the 2nd viewer was closed and reopened).
        """
        if self._second_viewer is None:
            return
        panel = getattr(self, "_second_morph_panel", None)
        # If we have a panel but the viewer was recreated, the old dock is
        # stale.  A simple sentinel: track the viewer it belongs to.
        owning = getattr(self, "_second_morph_panel_viewer", None)
        if panel is not None and owning is self._second_viewer:
            # Already attached to this viewer; nothing to do.
            return
        # Build a fresh panel and dock it.
        panel = self._build_label_morphology_panel()
        try:
            self._second_viewer.window.add_dock_widget(
                panel, area="right", name="Label Morphology"
            )
        except (RuntimeError, AttributeError) as e:
            print(f"Could not add morphology dock: {e}")
            return
        self._second_morph_panel = panel
        self._second_morph_panel_viewer = self._second_viewer
        # Reset the preview-layer reference for the new viewer.
        self._second_morph_preview_layer = None

    def _build_label_morphology_panel(self):
        """Construct the morphology operations side panel."""
        from qtpy.QtWidgets import QGroupBox, QWidget

        panel = QWidget()
        outer = QVBoxLayout(panel)

        group = QGroupBox("Label Morphology")
        layout = QVBoxLayout(group)

        # Operation
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Operation:"))
        self._morph_op_combo = QComboBox()
        self._morph_op_combo.addItems(
            [
                "Dilate",
                "Erode",
                "Open",
                "Close",
                "Fill holes",
                "Remove small objects",
                "Remove small embedded labels",
            ]
        )
        op_row.addWidget(self._morph_op_combo)
        layout.addLayout(op_row)

        # Element / size
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Element size:"))
        self._morph_size_spin = QSpinBox()
        self._morph_size_spin.setMinimum(1)
        self._morph_size_spin.setMaximum(99)
        self._morph_size_spin.setValue(3)
        size_row.addWidget(self._morph_size_spin)
        layout.addLayout(size_row)

        # Element shape (only matters for dilate/erode/open/close)
        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("Element shape:"))
        self._morph_shape_combo = QComboBox()
        self._morph_shape_combo.addItems(["Ball / Disk", "Cube / Square"])
        shape_row.addWidget(self._morph_shape_combo)
        layout.addLayout(shape_row)

        # Mode: Binary / Per-label / Single label
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._morph_mode_combo = QComboBox()
        self._morph_mode_combo.addItems(
            [
                "Binary (all labels)",
                "Per-label (preserve IDs)",
                "Single label",
            ]
        )
        self._morph_mode_combo.setCurrentText("Per-label (preserve IDs)")
        self._morph_mode_combo.setToolTip(
            "Binary: collapse all labels to one mask; new pixels take the "
            "'Current Label' value.\n"
            "Per-label: operate on each label ID independently and merge.\n"
            "Single label: operate only on the chosen Target label; when "
            "the region grows or fills, it overwrites neighbouring labels "
            "(useful for cleaning small islands of label B sitting inside "
            "label A)."
        )
        mode_row.addWidget(self._morph_mode_combo)
        layout.addLayout(mode_row)

        # Target label (only used when mode == "Single label")
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target label:"))
        self._morph_target_label_spin = QSpinBox()
        self._morph_target_label_spin.setMinimum(1)
        self._morph_target_label_spin.setMaximum(65535)
        self._morph_target_label_spin.setValue(int(self.current_label_num))
        target_row.addWidget(self._morph_target_label_spin)
        layout.addLayout(target_row)

        # Buttons
        btn_row = QHBoxLayout()
        self._morph_preview_btn = QPushButton("Preview")
        self._morph_preview_btn.clicked.connect(self._on_morph_preview)
        btn_row.addWidget(self._morph_preview_btn)

        self._morph_clear_btn = QPushButton("Clear preview")
        self._morph_clear_btn.clicked.connect(self._on_morph_clear_preview)
        btn_row.addWidget(self._morph_clear_btn)
        layout.addLayout(btn_row)

        self._morph_apply_btn = QPushButton("Apply to labels")
        self._morph_apply_btn.clicked.connect(self._on_morph_apply)
        layout.addWidget(self._morph_apply_btn)

        outer.addWidget(group)
        outer.addStretch(1)
        return panel

    # ------------------------------------------------------------------
    # Morphology core
    # ------------------------------------------------------------------
    def _make_structuring_element(self, ndim, size, shape):
        """Build a (2*size+1)^ndim structuring element of the given shape."""
        import numpy as np

        radius = int(size)
        diameter = 2 * radius + 1
        if shape == "Cube / Square":
            return np.ones((diameter,) * ndim, dtype=bool)
        # Ball / Disk
        coords = np.indices((diameter,) * ndim) - radius
        dist2 = np.sum(coords * coords, axis=0)
        return dist2 <= (radius * radius)

    def _apply_morphology_to_array(
        self, data, op, size, shape, mode, target_label
    ):
        """Return a new array with the chosen morphology op applied.

        ``data`` is a labels array (integer dtype).  ``mode`` is one of
        ``"Binary (all labels)"``, ``"Per-label (preserve IDs)"``, or
        ``"Single label"``.  ``target_label`` is the label ID used when
        ``mode == "Single label"``.
        """
        import numpy as np
        from scipy import ndimage as ndi

        data = np.asarray(data)
        if data.size == 0:
            return data.copy()

        ndim = data.ndim
        selem = self._make_structuring_element(ndim, size, shape)
        single = mode == "Single label"
        target = int(target_label)

        if op == "Fill holes":
            if single:
                # Fill holes inside the chosen label only; new pixels
                # overwrite whatever was there (including other labels),
                # so a small embedded label B is replaced by A.
                mask = data == target
                filled = ndi.binary_fill_holes(mask)
                new_pixels = filled & ~mask
                out = data.copy()
                out[new_pixels] = target
                return out
            mask = data > 0
            filled = ndi.binary_fill_holes(mask)
            new_pixels = filled & ~mask
            out = data.copy()
            out[new_pixels] = int(self.current_label_num)
            return out

        if op == "Remove small objects":
            threshold = int(size) ** ndim
            if single:
                # Only drop small components of the chosen label.
                mask = data == target
                labeled, _ = ndi.label(mask, structure=np.ones((3,) * ndim))
                counts = np.bincount(labeled.ravel())
                too_small = np.zeros_like(counts, dtype=bool)
                too_small[1:] = counts[1:] < threshold
                drop_mask = too_small[labeled]
                out = data.copy()
                out[drop_mask] = 0
                return out
            mask = data > 0
            labeled, _ = ndi.label(mask, structure=np.ones((3,) * ndim))
            counts = np.bincount(labeled.ravel())
            too_small = np.zeros_like(counts, dtype=bool)
            too_small[1:] = counts[1:] < threshold
            drop_mask = too_small[labeled]
            out = data.copy()
            out[drop_mask] = 0
            return out

        if op == "Remove small embedded labels":
            # Per-label: find small connected components and replace them
            # with the majority non-zero, non-self neighbor label.  Useful
            # when a small island of label B is embedded inside a large
            # region of label A: the island gets recoloured as A.
            # Components with no labeled neighbour (floating in 0) are
            # left alone.
            out = data.copy()
            threshold = int(size) ** ndim
            structure = np.ones((3,) * ndim)
            labels = np.unique(data)
            labels = labels[labels != 0]
            for value in labels:
                mask = data == value
                cc, _ = ndi.label(mask, structure=structure)
                counts = np.bincount(cc.ravel())
                small_ids = np.where(
                    (counts < threshold) & (np.arange(len(counts)) != 0)
                )[0]
                for id_ in small_ids:
                    component = cc == id_
                    border = ndi.binary_dilation(
                        component, structure=structure
                    )
                    border &= ~component
                    neighbors = data[border]
                    neighbors = neighbors[neighbors != value]
                    neighbors = neighbors[neighbors != 0]
                    if len(neighbors) == 0:
                        continue
                    replacement = np.bincount(neighbors.astype(int)).argmax()
                    out[component] = replacement
            return out

        # Dilate / Erode / Open / Close
        op_map = {
            "Dilate": ndi.binary_dilation,
            "Erode": ndi.binary_erosion,
            "Open": ndi.binary_opening,
            "Close": ndi.binary_closing,
        }
        if op not in op_map:
            return data.copy()
        fn = op_map[op]

        if single:
            # Single-label mode: morph only the chosen label's mask.  When
            # the region grows (Dilate / Close), new pixels OVERWRITE
            # whatever was there (other labels included).  When it shrinks
            # (Erode / Open), removed pixels become background.
            mask = data == target
            new_mask = fn(mask, structure=selem)
            out = data.copy()
            removed = mask & ~new_mask
            out[removed] = 0
            new_pixels = new_mask & ~mask
            out[new_pixels] = target
            return out

        if mode == "Per-label (preserve IDs)":
            out = np.zeros_like(data)
            for lbl in np.unique(data):
                if lbl == 0:
                    continue
                mask = data == lbl
                new_mask = fn(mask, structure=selem)
                # Don't overwrite already-assigned (other-label) pixels.
                paint = new_mask & (out == 0)
                out[paint] = lbl
            return out

        # Binary path: collapse to mask, morph, then fill new pixels with
        # the current label number; keep existing labels where they were.
        mask = data > 0
        new_mask = fn(mask, structure=selem)
        out = np.where(mask, data, 0)
        new_pixels = new_mask & ~mask
        out[new_pixels] = int(self.current_label_num)
        # Erode/Open may shrink the mask: clear pixels that fell out.
        removed = mask & ~new_mask
        out[removed] = 0
        return out

    def _on_morph_preview(self):
        """Compute the chosen morphology op on the *displayed* labels and
        add (or refresh) a preview layer in the 2nd viewer."""
        if self._second_viewer is None:
            return
        mirror = getattr(self, "_second_labels_layer", None)
        if mirror is None:
            QMessageBox.warning(
                self,
                "No labels",
                "No labels are shown in the 2nd viewer.",
            )
            return
        op = self._morph_op_combo.currentText()
        size = self._morph_size_spin.value()
        shape = self._morph_shape_combo.currentText()
        mode = self._morph_mode_combo.currentText()
        target_label = self._morph_target_label_spin.value()

        try:
            preview = self._apply_morphology_to_array(
                mirror.data, op, size, shape, mode, target_label
            )
        except (ValueError, MemoryError, ImportError) as e:
            QMessageBox.warning(
                self, "Morphology failed", f"{type(e).__name__}: {e}"
            )
            return

        scale = getattr(self, "_second_labels_scale", None)
        name = f"Preview: {op} (size={size})"
        prev = getattr(self, "_second_morph_preview_layer", None)
        if prev is not None:
            with contextlib.suppress(KeyError, ValueError, RuntimeError):
                self._second_viewer.layers.remove(prev)
            self._second_morph_preview_layer = None

        try:
            self._second_morph_preview_layer = self._second_viewer.add_labels(
                preview, name=name, scale=scale
            )
        except (RuntimeError, ValueError) as e:
            print(f"Could not add preview layer: {e}")

    def _on_morph_clear_preview(self):
        prev = getattr(self, "_second_morph_preview_layer", None)
        if prev is None or self._second_viewer is None:
            return
        with contextlib.suppress(KeyError, ValueError, RuntimeError):
            self._second_viewer.layers.remove(prev)
        self._second_morph_preview_layer = None

    def _on_morph_apply(self):
        """Apply the chosen morphology op to the real annotation_layer
        data, restricted to the current crop slice (if any)."""
        if self.annotation_layer is None:
            QMessageBox.warning(self, "No labels", "No labels loaded.")
            return
        op = self._morph_op_combo.currentText()
        size = self._morph_size_spin.value()
        shape = self._morph_shape_combo.currentText()
        mode = self._morph_mode_combo.currentText()
        target_label = self._morph_target_label_spin.value()

        labels_slice = getattr(self, "_second_labels_slice", None)
        full = self.annotation_layer.data
        target_view = full if labels_slice is None else full[labels_slice]

        confirm = QMessageBox.question(
            self,
            "Apply morphology",
            f"Apply '{op}' (size={size}) to the labels"
            + (" in the current 3D ROI" if labels_slice is not None else "")
            + "?\n\nThis modifies the real labels layer.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        try:
            result = self._apply_morphology_to_array(
                target_view, op, size, shape, mode, target_label
            )
        except (ValueError, MemoryError, ImportError) as e:
            QMessageBox.warning(
                self, "Morphology failed", f"{type(e).__name__}: {e}"
            )
            return

        # Write back in place so views (and the mirror in the 2nd viewer)
        # see the change.
        if labels_slice is None:
            full[...] = result
        else:
            full[labels_slice] = result
        self.annotation_layer.refresh()
        # Also clear the preview so the user sees the applied result on the
        # mirror layer directly.
        self._on_morph_clear_preview()
        mirror = getattr(self, "_second_labels_layer", None)
        if mirror is not None:
            with contextlib.suppress(RuntimeError, AttributeError):
                mirror.refresh()

    def _on_local_machine_learning(self):
        """Launch the interactive local-ML plugin on the active 3D bbox.

        Crops both the image and the persistent annotation layer to the
        currently active 3D bounding box, opens a fresh napari viewer with
        the crop loaded, and wires the plugin's "Commit Prediction" button
        to write back into ``self.annotation_layer`` at the same slice.
        """
        if self.image_layer is None or self.annotation_layer is None:
            QMessageBox.warning(
                self,
                "No image/labels loaded",
                "Load an image and create labels before launching Local "
                "Machine Learning.",
            )
            return

        image_slice, labels_slice = self._compute_preview_crop_slices()
        if image_slice is None or labels_slice is None:
            QMessageBox.warning(
                self,
                "No 3D bounding box",
                "Draw / select a 3D bounding box first.  Use the "
                "'Add Interactive 3D Boxes Layer' button to create one.",
            )
            return

        image_crop = np.asarray(self.image_layer.data[image_slice])

        contrast_limits = None
        try:
            contrast_limits = list(self.image_layer.contrast_limits)
        except (AttributeError, TypeError):
            contrast_limits = None

        scale = None
        if self.image_data_model is not None:
            try:
                scale = (
                    self.image_data_model.get_scale(
                        axes_to_collapse=self.axes_to_collapse
                    )
                    or None
                )
            except (AttributeError, TypeError, ValueError):
                scale = None

        # Restrict scale to the spatial dims that survived the crop.
        if scale is not None:
            try:
                scale = tuple(scale)[-image_crop.ndim :]
            except (TypeError, ValueError):
                scale = None

        from .interactive_local_learning import (
            launch_interactive_local_ml,
        )

        # Ensure a "Sparse Labels" annotation layer exists.  It is
        # initialised from the active persistent annotation with all
        # non-zero voxels shifted by +1 (so values 1, 2, ... become 2, 3,
        # ...).  The +1 reserves label 1 for "background" — the user can
        # paint label 1 in the new viewer so the ML model learns
        # background vs. foreground.  Subsequent launches reuse the same
        # layer rather than re-initialising it (preserving prior strokes).
        sparse_name = "Sparse Labels"
        if sparse_name in self.annotations_layers:
            sparse_layer = self.annotations_layers[sparse_name]
        else:

            def _init_from_active(annot_data):
                sparse = np.zeros_like(annot_data)
                nz = annot_data > 0
                sparse[nz] = annot_data[nz] + 1
                return sparse

            sparse_layer = self._create_new_annotations_layer(
                sparse_name,
                init_from_active=_init_from_active,
            )

        # Take a *view* (not a copy) of the sparse-labels layer at the same
        # crop slice as the image, so paint strokes in the new viewer write
        # straight back to the layer in the main viewer.
        painting_view = sparse_layer.data[labels_slice]

        # Mirror the main viewer's working layer (interactive segmenter
        # scratch) into the local-ML viewer so segmenter strokes show up
        # there too.
        extra_mirrors = []
        if self.working_layer is not None:
            extra_mirrors.append((self.working_layer, labels_slice))

        # Build the {name: (layer, slice)} map of every annotation layer
        # the user can commit the prediction into.  All annotation layers
        # share the image grid, so the same ``labels_slice`` applies to
        # each one.
        commit_targets = {
            name: (layer, labels_slice)
            for name, layer in self.annotations_layers.items()
            if layer is not None
        }

        launch_interactive_local_ml(
            image_crop,
            scale=scale,
            contrast_limits=contrast_limits,
            target_labels_layer=self.annotation_layer,
            target_slice=labels_slice,
            commit_targets=commit_targets,
            source_image=self.image_layer.data,
            source_slice=image_slice,
            painting_data=painting_view,
            painting_name=f"{sparse_layer.name} (crop view)",
            extra_mirror_layers=extra_mirrors,
            on_commit=lambda _pred: sparse_layer.refresh(),
        )

    # ------------------------------------------------------------------
    # Annotation layer collection
    # ------------------------------------------------------------------
    def _load_existing_annotation_layers(self, image_shape, annotation_scale):
        """Populate :attr:`annotations_layers` from disk subdirectories.

        Mirrors :meth:`NDEasySegment._load_existing_prediction_layers`:
        each subdirectory under ``annotations/`` becomes a separate napari
        labels layer named after the subdirectory.  Falls back to creating
        a single empty ``"Labels (Persistent)"`` layer when no
        subdirectories exist yet.
        """
        self.annotations_layers = {}
        names = []
        with contextlib.suppress(AttributeError, OSError):
            names = self.image_data_model.list_annotation_subdirectories()

        if not names:
            names = ["Labels (Persistent)"]

        first_layer = None
        for name in names:
            try:
                data = self.image_data_model.load_existing_annotations(
                    image_shape,
                    self.current_image_index,
                    subdirectory=name,
                    axes_to_collapse=self.axes_to_collapse,
                )
            except (OSError, ValueError, RuntimeError) as e:
                print(f"Failed to load annotations '{name}': {e}")
                continue
            layer = self.viewer.add_labels(
                data, name=name, scale=annotation_scale or None
            )
            self.annotations_layers[name] = layer
            if first_layer is None:
                first_layer = layer

        self.annotation_layer = first_layer
        self._refresh_active_annotation_combo()

    def _refresh_active_annotation_combo(self):
        """Rebuild the active-annotation combo from :attr:`annotations_layers`."""
        combo = getattr(self, "active_annotation_combo", None)
        if combo is None:
            return
        combo.blockSignals(True)
        try:
            combo.clear()
            for name in self.annotations_layers:
                combo.addItem(name)
            if self.annotation_layer is not None:
                idx = combo.findText(self.annotation_layer.name)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(False)

    def _on_active_annotation_changed(self, name: str):
        """Switch the active annotation layer based on the combo selection."""
        if not name or name not in self.annotations_layers:
            return
        self.annotation_layer = self.annotations_layers[name]
        # Broadcast to sibling widgets when hosted inside NDAILab so
        # interactive segmenters / augmenters target the same active layer.
        ai_lab = getattr(self, "ai_lab", None)
        if ai_lab is not None:
            with contextlib.suppress(AttributeError):
                ai_lab.annotations_layer = self.annotation_layer
            for sibling_name in ("augment_widget", "segment_widget"):
                sibling = getattr(ai_lab, sibling_name, None)
                if sibling is not None and sibling is not self:
                    sibling.annotation_layer = self.annotation_layer
        print(f"Active annotation layer set to: {name}")

    def _create_new_annotations_layer(
        self,
        name: str,
        init_data=None,
        init_from_active=None,
    ):
        """Create (and register) a new annotation layer named ``name``.

        Parameters
        ----------
        name:
            Layer name; doubles as the on-disk subdirectory name when
            annotations are saved.
        init_data:
            Optional ndarray to use as the layer's initial data.  When
            omitted the layer is initialised by ``init_from_active`` or as
            zeros shaped like the current active annotation layer.
        init_from_active:
            Optional callable ``f(active_data) -> ndarray`` invoked when
            ``init_data`` is None.  Useful for derived layers (e.g. the
            "Sparse Labels" layer is built as ``active + 1`` for non-zero
            voxels).

        Existing layers with the same name are returned unchanged so this
        helper is safe to call multiple times.
        """
        if name in self.annotations_layers:
            return self.annotations_layers[name]

        if init_data is None:
            if self.annotation_layer is None:
                raise RuntimeError(
                    "Cannot create a new annotation layer without a "
                    "template — no active annotation layer is loaded."
                )
            active = self.annotation_layer.data
            init_data = (
                init_from_active(active)
                if init_from_active is not None
                else np.zeros_like(active)
            )

        scale = None
        if self.annotation_layer is not None:
            with contextlib.suppress(AttributeError):
                scale = self.annotation_layer.scale

        layer = self.viewer.add_labels(init_data, name=name, scale=scale)
        self.annotations_layers[name] = layer
        self._refresh_active_annotation_combo()
        return layer

    def _find_available_annotations(self, box_layer):
        """Return ``{name: annotation_layer}`` with non-zero data inside the active box.

        Mirrors :meth:`NDAILab._find_available_predictions` so callers can
        present the user with a choice of which annotation collection to
        act on for a given ROI.
        """
        available = {}
        if box_layer is None or not self.annotations_layers:
            return available
        for name, layer in self.annotations_layers.items():
            try:
                slc = self._compute_crop_slice(box_layer, layer)
            except (AttributeError, ValueError, TypeError):
                slc = None
            if slc is None:
                continue
            try:
                if np.any(layer.data[slc]):
                    available[name] = layer
            except (IndexError, TypeError, ValueError):
                continue
        return available

    def _on_copy_predictions_to_labels(self):
        """Open the copy-predictions-to-labels dialog on the parent NDAILab.

        Only works when this widget is hosted inside NDAILab (which owns
        the predictions layers and sets ``self.ai_lab``).
        """
        ai_lab = getattr(self, "ai_lab", None)
        if ai_lab is None or not hasattr(
            ai_lab, "show_copy_predictions_dialog"
        ):
            QMessageBox.information(
                self,
                "Copy Predictions",
                "Copy predictions is only available when running inside "
                "ND AI Lab.",
            )
            return
        ai_lab.show_copy_predictions_dialog()

    def _on_segmenter_parameters_changed(self, parameters):
        """Sync segmenter, then live-rerun the last interactive call if supported."""
        super()._on_segmenter_parameters_changed(parameters)
        # Re-run the last interactive segmentation with the current parameter
        # values so the user can see the effect of each tweak.  Only segmenters
        # that opt-in via ``supports_live_param_update`` are eligible.
        if getattr(self.segmenter, "supports_live_param_update", False):
            self._rerun_last_interactive_segmentation()

    def _rerun_last_interactive_segmentation(self):
        """Re-run the last interactive segment() call with the current segmenter.

        Used to give live visual feedback while the user tunes segmenter
        parameters (e.g. RegionGrow3D's tolerance / window_size).  Output
        always lands in the working layer with ``WORKING_LABEL_INDEX``, so
        no label-number bookkeeping is needed here.
        """
        ctx = getattr(self, "_last_interactive_segmentation", None)
        if not ctx:
            return
        if type(self.segmenter).__name__ != ctx.get("segmenter_name"):
            # Segmenter was swapped since the last seed click; don't reuse it.
            return
        if self.annotation_layer is None:
            return

        try:
            mask = self.segmenter.segment(
                ctx["image_data"],
                points=ctx.get("points"),
                shapes=ctx.get("shapes"),
            )
            self._apply_segmenter_mask(mask, ctx["segmentation_indices"])
            print("Live-updated working segmentation with new parameters")
        except (
            AttributeError,
            ValueError,
            TypeError,
            RuntimeError,
            IndexError,
        ) as e:
            print(f"Live re-segmentation failed: {e}")

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

                # Stash the call context so live parameter changes can
                # re-run the segmenter with the same seeds + label, giving
                # the user immediate feedback while tuning.
                if getattr(
                    self.segmenter, "supports_live_param_update", False
                ):
                    self._last_interactive_segmentation = {
                        "image_data": image_data,
                        "segmentation_indices": segmentation_indices,
                        "points": [latest_point],
                        "shapes": None,
                        "label_num": self.current_label_num,
                        "segmenter_name": type(self.segmenter).__name__,
                    }
                else:
                    self._last_interactive_segmentation = None

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

    def _update_active_box_size_label(self, box):
        """Update the live box-size readout with the size of ``box``.

        ``box`` is a (N, D) numpy array of corner coordinates (napari
        Shapes rectangle or vendored BoundingBox). The last 3 columns are
        treated as (Z, Y, X) when present, last 2 as (Y, X) otherwise.
        Sizes are reported in pixels (rounded to integers).
        """
        if not hasattr(self, "active_box_size_label"):
            return
        try:
            arr = np.asarray(box)
            if arr.ndim != 2 or arr.shape[0] == 0:
                return
            ncols = arr.shape[1]
            if ncols >= 3:
                z = arr[:, -3]
                y = arr[:, -2]
                x = arr[:, -1]
                dz = int(np.ceil(z.max() - z.min()))
                dy = int(np.ceil(y.max() - y.min()))
                dx = int(np.ceil(x.max() - x.min()))
                if dz > 0:
                    text = f"Active box size (Z x Y x X): {dz} x {dy} x {dx}"
                else:
                    text = f"Active box size (Y x X): {dy} x {dx}"
            else:
                y = arr[:, -2]
                x = arr[:, -1]
                dy = int(np.ceil(y.max() - y.min()))
                dx = int(np.ceil(x.max() - x.min()))
                text = f"Active box size (Y x X): {dy} x {dx}"
            self.active_box_size_label.setText(text)
        except (ValueError, TypeError, IndexError, AttributeError):
            pass

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
        # Update live box-size readout if there is any data on the layer.
        if len(shapes_layer.data) > 0:
            self._update_active_box_size_label(shapes_layer.data[-1])

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

        # Live box-size readout for any change on this layer.
        boxes_layer = event.source
        if len(boxes_layer.data) > 0:
            self._update_active_box_size_label(boxes_layer.data[-1])

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
        try:
            if event.action not in ("added", "changed"):
                return
        except AttributeError as e:
            print(f"Error checking event action: {e}")
            return

        layer = event.source

        selected_index = list(event.source.selected_data)[0]

        # Live box-size readout fires for any 3D-box change, regardless of
        # whether this layer is the chosen interactive driver.
        if len(layer.data) > 0:
            self._update_active_box_size_label(layer.data[selected_index])

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
                f"3D box segmentation applied (working layer; commit to assign label {self.current_label_num})"
            )
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
        """Backwards-compat shim. Defers to :meth:`BaseNDApp._get_active_box`."""
        return super()._get_active_box(layer)

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
                f"2D box segmentation applied (working layer; commit to assign label {self.current_label_num})"
            )
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
        # Live box-size readout for any change on this layer.
        layer = event.source
        if len(layer.data) > 0:
            self._update_active_box_size_label(layer.data[-1])

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
                    f"Added box-segmentation (working layer; commit to assign label {self.current_label_num})"
                )

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

        # Save 3D bounding boxes (if the shared 3D boxes layer exists)
        boxes_3D_layer = getattr(self, "boxes_3D_layer", None)
        if boxes_3D_layer is not None and self.image_data_model is not None:
            try:
                self.image_data_model.save_3D_boxes(
                    list(boxes_3D_layer.data),
                    self.current_image_index,
                )
            except (AttributeError, OSError, ValueError) as e:
                print(f"Failed to save 3D boxes: {e}")

            # Save 3D patch pairs (input0 / truth0 under labels3d/)
            if (
                len(boxes_3D_layer.data) > 0
                and self.image_layer is not None
                and self.annotation_layer is not None
            ):
                try:
                    self.image_data_model.crop_and_save_3D_label_patches(
                        list(boxes_3D_layer.data),
                        self.image_layer.data,
                        self.annotation_layer.data,
                        self.current_image_index,
                    )
                    print("3D label patches saved.")
                except (AttributeError, ValueError, OSError, IndexError) as e:
                    print(f"Failed to save 3D label patches: {e}")
            else:
                print("No 3D boxes drawn — skipping 3D label patch save.")

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

        # Also save 3D boxes + 3D patches on close (mirrors _on_save_project).
        boxes_3D_layer = getattr(self, "boxes_3D_layer", None)
        if boxes_3D_layer is not None and len(boxes_3D_layer.data) > 0:
            try:
                self.image_data_model.save_3D_boxes(
                    list(boxes_3D_layer.data),
                    self.current_image_index,
                )
                self.image_data_model.crop_and_save_3D_label_patches(
                    list(boxes_3D_layer.data),
                    self.image_layer.data,
                    self.annotation_layer.data,
                    self.current_image_index,
                )
                print("3D label patches saved on close.")
            except (AttributeError, ValueError, OSError, IndexError) as e:
                print(f"Failed to save 3D boxes/patches on close: {e}")

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

    def _load_existing_3D_boxes(self):
        """Load saved 3D bounding boxes from labels3d/boxes.csv into boxes_3D_layer."""
        boxes_3D_layer = getattr(self, "boxes_3D_layer", None)
        if boxes_3D_layer is None or self.image_data_model is None:
            return
        try:
            saved = self.image_data_model.load_3D_boxes()
        except (OSError, ValueError) as e:
            print(f"Failed to load existing 3D boxes: {e}")
            return
        if not saved:
            return
        try:
            boxes_3D_layer.add(saved)
        except (AttributeError, ValueError, TypeError) as e:
            # Fall back to direct data assignment if .add isn't supported.
            try:
                boxes_3D_layer.data = list(boxes_3D_layer.data) + list(saved)
            except (ValueError, TypeError) as e2:
                print(f"Failed to populate boxes_3D_layer: {e} / {e2}")
                return
        print(
            f"📦 Loaded {len(saved)} existing 3D box(es) into boxes_3D_layer"
        )

    # _on_open_directory and load_image_directory inherited from BaseNDApp
    def _set_image_layer(self, image_layer):
        """Set up all annotation layers based on the provided image layer."""
        try:
            # Store the image layer reference
            self.image_layer = image_layer

            # Get image data from the layer
            image_data = image_layer.data

            # Load existing labels or create empty ones (delegated to model).
            # Multiple annotation collections are supported: each
            # subdirectory of ``annotations/`` becomes a separate napari
            # labels layer named after the subdir.  If no subdirs exist
            # yet, a default "Labels (Persistent)" layer is created.
            annotation_scale = self.image_data_model.get_scale(
                axes_to_collapse=self.axes_to_collapse
            )
            self._load_existing_annotation_layers(
                image_data.shape, annotation_scale
            )

            # Working / scratch labels layer for interactive segmenter output.
            # Always uses WORKING_LABEL_INDEX (7) so its colour is consistent
            # (napari’s default colormap renders 7 as green-ish, easy to spot).
            working_data = np.zeros_like(self.annotation_layer.data)
            self.working_layer = self.viewer.add_labels(
                working_data,
                name="Labels (Working)",
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
