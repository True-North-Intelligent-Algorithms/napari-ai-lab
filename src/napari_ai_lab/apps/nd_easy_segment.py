"""
Unified ND Easy Segmentation Widget.

This module provides a unified interface for both interactive (point/shape-based)
and automatic (full plane/volume) segmentation workflows.
"""

import contextlib

import napari
import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from ..models import ImageDataModel
from ..utilities import QtProgressLogger
from ..utilities.slice_processor import (
    SliceProcessor,
    SliceProcessorThread,
    TrainingThread,
)
from ..utility import get_current_slice_indices
from ..widgets import NDOperationWidget
from ..widgets.train_dialog import TrainDialog
from .base_nd_app import BaseNDApp


class NDEasySegment(BaseNDApp):
    """
    Unified segmentation widget supporting both interactive and automatic modes.

    Modes:
    - Interactive Mode: Point/shape-based segmentation (like nd_easy_label)
    - Automatic Mode: Full plane/volume segmentation (like nd_easy_segment)
    """

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        image_data_model: ImageDataModel = None,
        embedded: bool = False,
        training_widget_mode: str = "embedded",
        axes_to_collapse: str | list[str] | None = None,
    ):
        """
        Initialize NDEasySegment widget.

        Args:
            viewer: The napari viewer instance
            image_data_model: Optional ImageDataModel instance
            embedded: Whether running in embedded mode (no directory button)
            training_widget_mode: How to handle training UI
                - "dialog": Show popup dialog for training parameters (classic)
                - "embedded": Use training parameters from embedded training form (default)
            axes_to_collapse: Axis names to collapse when loading/saving annotations/predictions
        """
        super().__init__(viewer, image_data_model)
        self.embedded = embedded
        self.training_widget_mode = training_widget_mode
        self.axes_to_collapse = axes_to_collapse

        # Create Qt progress logger for training tab
        self.progress_logger = QtProgressLogger()

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
        # Directory selection (only in standalone mode)
        if not self.embedded:
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

        self.save_slice_wise_cb = QCheckBox("Save slice-wise predictions")
        self.save_slice_wise_cb.setChecked(True)
        main_layout.addWidget(self.save_slice_wise_cb)

        # Parameter form widget (from base)
        main_layout.addWidget(self._create_segmenter_parameter_form())

        # === Mode-Specific Controls ===
        # Automatic mode controls
        self.auto_controls_group = QGroupBox("Automatic Segmentation")
        auto_layout = QVBoxLayout(self.auto_controls_group)

        self.segment_current_btn = QPushButton("Segment Current Image")
        self.segment_current_btn.clicked.connect(self._on_segment_current)
        auto_layout.addWidget(self.segment_current_btn)

        self.segment_all_btn = QPushButton("Segment Range")
        self.segment_all_btn.clicked.connect(self._on_segment_range)
        auto_layout.addWidget(self.segment_all_btn)

        # Start/end slice spinboxes for the range
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Start:"))
        self.start_slice_spin = QSpinBox()
        self.start_slice_spin.setMinimum(0)
        self.start_slice_spin.setMaximum(0)
        self.start_slice_spin.setValue(0)
        range_row.addWidget(self.start_slice_spin)
        range_row.addWidget(QLabel("End:"))
        self.end_slice_spin = QSpinBox()
        self.end_slice_spin.setMinimum(0)
        self.end_slice_spin.setMaximum(0)
        self.end_slice_spin.setValue(0)
        range_row.addWidget(self.end_slice_spin)
        auto_layout.addLayout(range_row)

        # === ROI segmentation ===
        # The ROI comes from one of the box layers shared with the rest of
        # the app (Label box / 3D Bounding Box) — created elsewhere
        # (NDEasyLabel or the parent NDAILab wires them in).
        self._segment_roi_viewer = None

        roi_source_row = QHBoxLayout()
        roi_source_row.addWidget(QLabel("ROI source:"))
        self.roi_source_combo = QComboBox()
        self.roi_source_combo.addItems(["Label Box", "3D Bounding Box"])
        roi_source_row.addWidget(self.roi_source_combo)
        auto_layout.addLayout(roi_source_row)

        self.show_roi_in_new_viewer_cb = QCheckBox(
            "Show ROI segmentation in new napari viewer"
        )
        auto_layout.addWidget(self.show_roi_in_new_viewer_cb)

        self.segment_roi_btn = QPushButton("Segment ROI")
        self.segment_roi_btn.clicked.connect(self._on_segment_roi)
        auto_layout.addWidget(self.segment_roi_btn)

        # Add progress logger widget for segmentation
        self.segment_progress_logger = QtProgressLogger()
        auto_layout.addWidget(self.segment_progress_logger.get_widget())

        main_layout.addWidget(self.auto_controls_group)

        main_layout.addWidget(self.save_annotations_btn)

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

        # Add stretch to push everything to the top (prevents button stretching)
        main_layout.addStretch()

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
        """Populate the segmenter combo box(es) with registered frameworks."""
        from ..Segmenters.GlobalSegmenters import GlobalSegmenterBase

        self.segmenter_combo.clear()

        frameworks = GlobalSegmenterBase.get_registered_frameworks()

        if frameworks:
            framework_names = list(frameworks.keys())
            for name in framework_names:
                self.segmenter_combo.addItem(name)

            if self.segmenter_combo.count() > 0:
                self.segmenter_combo.setCurrentIndex(0)
                # Don't trigger change yet - will sync after populating training combo
        else:
            self.segmenter_combo.addItem("No segmenters available")
            self.segmenter_combo.setEnabled(False)

        # Also populate training combo if it exists
        if hasattr(self, "training_segmenter_combo"):
            self.training_segmenter_combo.clear()
            if frameworks:
                for name in framework_names:
                    self.training_segmenter_combo.addItem(name)
                if self.training_segmenter_combo.count() > 0:
                    self.training_segmenter_combo.setCurrentIndex(0)
            else:
                self.training_segmenter_combo.addItem(
                    "No segmenters available"
                )
                self.training_segmenter_combo.setEnabled(False)

        # Now trigger the change handler if we have items
        if frameworks and self.segmenter_combo.count() > 0:
            self._on_segmenter_changed(self.segmenter_combo.currentText())

    def _on_segmenter_changed(self, segmenter_name):
        """
        Handle segmenter selection changes.

        Overrides base class to keep both combo boxes in sync.
        """
        # Sync both combo boxes to the same selection (block signals to avoid recursion)
        sender = self.sender()

        # Only sync if training combo exists and sender is one of our combos
        if hasattr(self, "training_segmenter_combo"):
            if sender == self.segmenter_combo:
                self.training_segmenter_combo.blockSignals(True)
                self.training_segmenter_combo.setCurrentText(segmenter_name)
                self.training_segmenter_combo.blockSignals(False)
            elif sender == self.training_segmenter_combo:
                self.segmenter_combo.blockSignals(True)
                self.segmenter_combo.setCurrentText(segmenter_name)
                self.segmenter_combo.blockSignals(False)

        # Call base class implementation to actually change the segmenter
        super()._on_segmenter_changed(segmenter_name)

    def _create_training_parameter_form(self):
        """Create the training parameter form widget."""
        self.training_parameter_form = NDOperationWidget(
            param_type_to_parse="training"
        )
        # Connect to the same handler as segmenter params (updates will sync)
        self.training_parameter_form.parameters_changed.connect(
            self._on_segmenter_parameters_changed
        )
        return self.training_parameter_form

    def get_training_widget(self):
        """
        Get a widget showing training-specific controls.

        Returns a widget with:
        - Segmenter combo (separate instance, synced with main segment view)
        - Training parameters form
        - Train button

        This allows the same NDEasySegment instance to show different views
        in different tabs without duplication.
        """
        from qtpy.QtWidgets import QLabel, QLineEdit, QVBoxLayout, QWidget

        training_widget = QWidget()
        layout = QVBoxLayout(training_widget)

        # Segmenter selection (create separate combo that stays in sync)
        layout.addWidget(QLabel("Segmenter:"))

        # Create a second combo box for the training tab
        self.training_segmenter_combo = QComboBox()
        self.training_segmenter_combo.currentTextChanged.connect(
            self._on_segmenter_changed
        )
        layout.addWidget(self.training_segmenter_combo)

        # Populate it with the same items as the main combo
        self._populate_segmenter_combo()

        # Training parameters form
        layout.addWidget(self._create_training_parameter_form())

        # Model name input
        layout.addWidget(QLabel("Model name:"))
        self.model_name_edit = QLineEdit("my_model")
        layout.addWidget(self.model_name_edit)

        # Train button (same as in automatic mode controls)
        train_btn = QPushButton("Train Model")
        train_btn.clicked.connect(self._on_train)
        layout.addWidget(train_btn)

        # Add stretch before progress widget to push it to bottom
        layout.addStretch()

        # Add progress logger widget at bottom (stretch factor 1 to fill available space)
        layout.addWidget(self.progress_logger.get_widget(), 1)

        return training_widget

    def _update_segmenter_parameter_form(self, segmenter):
        """Update both segmenter and training parameter forms, cross-wire combos."""
        self.segmenter_parameter_form.set_nd_operation(segmenter)

        if hasattr(self, "training_parameter_form"):
            self.training_parameter_form.set_nd_operation(segmenter)
            self._cross_connect_combos()

        # Update range bounds and connect axis-change signal (idempotent)
        with contextlib.suppress(TypeError, RuntimeError):
            self.segmenter_parameter_form.axis_changed.disconnect(
                self._update_range_bounds
            )
        with contextlib.suppress(AttributeError, RuntimeError):
            self.segmenter_parameter_form.axis_changed.connect(
                self._update_range_bounds
            )
        self._update_range_bounds()

    def _update_range_bounds(self, *_):
        """Recompute Segment-Range spinbox bounds from current image+axis."""
        if not (
            hasattr(self, "start_slice_spin")
            and hasattr(self, "end_slice_spin")
        ):
            return
        if self.image_layer is None:
            return
        try:
            selected_axis = self.segmenter_parameter_form.get_selected_axis()
        except AttributeError:
            selected_axis = None
        if not selected_axis:
            return
        try:
            processor = SliceProcessor(
                self.image_layer.data.shape,
                selected_axis,
                self.axes_to_collapse,
            )
        except (ValueError, AttributeError):
            return
        max_idx = max(0, processor.total_slices - 1)
        # Preserve current selection where possible.
        old_start = self.start_slice_spin.value()
        old_end = self.end_slice_spin.value()
        self.start_slice_spin.setMaximum(max_idx)
        self.end_slice_spin.setMaximum(max_idx)
        # If bounds previously degenerate (0..0), reset to full range.
        if old_end == 0 and old_start == 0:
            self.end_slice_spin.setValue(max_idx)
        else:
            self.start_slice_spin.setValue(min(old_start, max_idx))
            self.end_slice_spin.setValue(min(old_end, max_idx))

    def _cross_connect_combos(self):
        """Cross-wire model and axis combos so changing one updates the other."""
        seg = self.segmenter_parameter_form
        trn = self.training_parameter_form

        # Model combo sync
        if hasattr(seg, "_model_combo") and hasattr(trn, "_model_combo"):
            seg._model_combo.currentTextChanged.connect(trn.set_model_combo)
            trn._model_combo.currentTextChanged.connect(seg.set_model_combo)

        # Axis combo sync
        if (
            hasattr(seg, "_axis_combo")
            and seg._axis_combo
            and hasattr(trn, "_axis_combo")
            and trn._axis_combo
        ):
            seg._axis_combo.currentTextChanged.connect(trn.set_axis_combo)
            trn._axis_combo.currentTextChanged.connect(seg.set_axis_combo)

    # === Interactive Mode Methods ===
    def _on_3D_boxes_changed(self, event):
        """Handle changes to the 3D bounding boxes layer."""
        print(f"3D boxes event action: {event.action}")
        print(f"3D boxes event source: {event.source}")
        print()

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
                self.segmenter = (
                    self.segmenter_parameter_form.sync_nd_operation_instance(
                        self.segmenter
                    )
                )

            try:
                mask = self.image_data_model.segment(
                    self.segmenter,
                    image_data,
                    points=[latest_point],
                    shapes=None,
                )

                self.annotation_layer.data[mask] = self.current_label_num
                print(
                    f"Added segmentation with label {self.current_label_num}"
                )
                self.current_label_num += 1
                self.annotation_layer.refresh()

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

        # Clear and update progress logger
        self.segment_progress_logger.clear()
        self.segment_progress_logger.log_info("Segmenting current image...")

        print("Segmenting current image...")

        # Ensure segmenter is synced with current parameters
        self.segmenter = (
            self.segmenter_parameter_form.sync_nd_operation_instance(
                self.segmenter
            )
        )

        # Use SliceProcessor for single-slice processing
        selected_axis = self.segmenter_parameter_form.get_selected_axis()
        self._setup_segment_context()

        processor = SliceProcessor(
            self.image_layer.data.shape,
            selected_axis,
            self.axes_to_collapse,
        )

        self.segment_progress_logger.update_progress(1, 2, "Processing...")
        processor.process_slice(
            self.viewer.dims.current_step,
            self._do_segment_slice,
            self._on_segment_slice_done,
        )

        # If the segmenter produced a 3D "stacked labels" preview (e.g.
        # MicroSamYoloSegmenter), show it as a separate labels layer.
        # Only done in single-slice mode (not during _on_segment_range).
        stacked = getattr(self.segmenter, "last_stacked_labels", None)
        if stacked is not None:
            self._show_stacked_labels_layer(stacked)

        self.segment_progress_logger.update_progress(2, 2, "\u2705 Complete")
        self.segment_progress_logger.log_info("\u2705 Segmentation complete")

    def _on_segment_range(self):
        """Segment a range of slices in the ND data (threaded)."""
        if self.image_layer is None:
            QMessageBox.warning(self, "Warning", "No images loaded")
            return

        if not hasattr(self, "segmenter") or self.segmenter is None:
            QMessageBox.warning(self, "Warning", "No segmenter selected")
            return

        # Clear progress logger
        self.segment_progress_logger.clear()
        self.segment_progress_logger.log_info(
            "Starting segmentation of slice range..."
        )

        print("Segmenting slice range...")

        # Ensure segmenter is synced with current parameters
        self.segmenter = (
            self.segmenter_parameter_form.sync_nd_operation_instance(
                self.segmenter
            )
        )

        selected_axis = self.segmenter_parameter_form.get_selected_axis()
        image_shape = self.image_layer.data.shape

        print(f"Selected axis: {selected_axis}")
        print(f"Image shape: {image_shape}")

        self.segment_progress_logger.log_info(
            f"Selected axis: {selected_axis}"
        )
        self.segment_progress_logger.log_info(f"Image shape: {image_shape}")

        # Setup shared context for the operation and callback
        self._setup_segment_context()

        processor = SliceProcessor(
            image_shape, selected_axis, self.axes_to_collapse
        )

        # Clamp spinbox bounds to the processor's slice count and read range.
        max_idx = max(0, processor.total_slices - 1)
        self.start_slice_spin.setMaximum(max_idx)
        self.end_slice_spin.setMaximum(max_idx)
        start_idx = min(self.start_slice_spin.value(), max_idx)
        end_idx = min(self.end_slice_spin.value(), max_idx)
        if end_idx < start_idx:
            end_idx = start_idx
            self.end_slice_spin.setValue(end_idx)
        num_to_run = end_idx - start_idx + 1

        print(
            f"Total slices: {processor.total_slices}, "
            f"running {num_to_run} ({start_idx}..{end_idx})"
        )
        self.segment_progress_logger.log_info(
            f"Total slices: {processor.total_slices}, "
            f"running {num_to_run} ({start_idx}..{end_idx})"
        )

        # Disable button while running
        self.segment_all_btn.setEnabled(False)

        # Launch threaded processing
        self._segment_thread = SliceProcessorThread(
            processor,
            self._do_segment_slice,
            start_index=start_idx,
            end_index=end_idx,
        )
        self._segment_thread.progress.connect(self._on_segment_all_progress)
        self._segment_thread.slice_done.connect(self._on_segment_slice_done)
        self._segment_thread.finished.connect(self._on_segment_all_finished)
        self._segment_thread.error.connect(self._on_segment_all_error)
        self._segment_thread.start()

    def _on_segment_all_progress(self, current, total):
        """Handle progress updates from the worker thread (runs on main thread)."""
        self.segment_progress_logger.update_progress(
            current, total, f"Processing slice {current}/{total}"
        )
        print(f"Processing slice {current}/{total}")

    def _on_segment_all_finished(self):
        """Handle completion of threaded segment-all (runs on main thread)."""
        self.segment_all_btn.setEnabled(True)
        total = getattr(self, "_segment_thread", None)
        total_str = (
            str(total.worker.processor.total_slices) if total else "all"
        )
        print(f"✅ Completed segmentation of {total_str} slices")
        self.segment_progress_logger.log_info(
            f"✅ Completed segmentation of {total_str} slices"
        )
        self._segment_thread = None

    def _on_segment_all_error(self, error_msg):
        """Handle errors from the worker thread (runs on main thread)."""
        self.segment_all_btn.setEnabled(True)
        print(f"Error during segment all: {error_msg}")
        QMessageBox.critical(
            self, "Error", f"Segmentation failed: {error_msg}"
        )
        self._segment_thread = None

    def _setup_segment_context(self):
        """Store shared context needed by _do_segment_slice and _on_segment_slice_done."""
        self._seg_selected_axis = (
            self.segmenter_parameter_form.get_selected_axis()
        )
        self._seg_segmenter_name = self.segmenter.__class__.__name__
        self.image_data_model.set_current_segmenter_name(
            self._seg_segmenter_name
        )
        self._seg_segmentation_axis = self.segmenter.get_segmentation_axis(
            self._seg_selected_axis
        )

    def _do_segment_slice(self, current_step):
        """Extract a slice and run segmentation via the model.

        Args:
            current_step: Tuple of indices identifying the slice position.

        Returns:
            numpy.ndarray: The segmentation mask for this slice.
        """
        return self.image_data_model.segment_slice(
            self.segmenter, current_step, self._seg_selected_axis
        )

    def _on_segment_slice_done(self, current_step, mask):
        """Save predictions and update the viewer layer after segmenting a slice.

        Args:
            current_step: Tuple of indices identifying the slice position.
            mask: The segmentation mask returned by _do_segment_slice.
        """
        try:
            segmentation_axis = self._seg_segmentation_axis
            segmenter_name = self._seg_segmenter_name

            # Save predictions via model
            if self.save_slice_wise_cb.isChecked():
                self.image_data_model.save_predictions(
                    mask,
                    self.current_image_index,
                    current_step=current_step,
                    selected_axis=segmentation_axis,
                    axes_to_collapse=self.axes_to_collapse,
                )

            segmentation_indices = get_current_slice_indices(
                current_step,
                segmentation_axis,
                ignore_channel=True,
                shape=mask.shape,
            )
            # Get or create the predictions layer for this segmenter
            predictions_layer = self._get_or_create_predictions_layer(
                segmenter_name, self.annotation_layer.data.shape
            )

            # Update the layer with new predictions
            predictions_layer.data[segmentation_indices] = mask

            predictions_layer.refresh()
            print(
                f"Automatic segmentation completed - updated layer: {segmenter_name}"
            )

            # If the segmenter produced predicted object boxes, add them to a
            # dedicated shapes layer at the correct slice position.
            boxes = getattr(self.segmenter, "last_napari_boxes", None)
            if boxes is not None and len(boxes) > 0:
                self._add_predicted_boxes(
                    boxes, current_step, segmentation_axis
                )

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

    # === ROI segmentation ===
    def _get_roi_box_layer(self, roi_source):
        """Return the shared box layer matching the ``roi_source`` combo entry."""
        if roi_source == "3D Bounding Box":
            return getattr(self, "boxes_3D_layer", None)
        if roi_source == "Label Box":
            return getattr(self, "boxes_layer", None)
        return None

    def _on_segment_roi(self):
        """Segment a single ROI selected via the ROI-source combo.

        The combo picks between the shared ``Label Box`` (2D Shapes) and
        ``3D Bounding Box`` layers.  After segmentation, the resulting mask
        is pasted into the corresponding predictions layer at the same
        crop; optionally a second napari viewer is opened with the cropped
        image and prediction for inspection.
        """
        if self.image_layer is None:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return
        if not hasattr(self, "segmenter") or self.segmenter is None:
            QMessageBox.warning(self, "Warning", "No segmenter selected")
            return

        roi_source = self.roi_source_combo.currentText()
        box_layer = self._get_roi_box_layer(roi_source)
        if box_layer is None or len(getattr(box_layer, "data", [])) == 0:
            QMessageBox.warning(
                self,
                "Warning",
                f"No '{roi_source}' layer with boxes found. "
                "Draw a box in the corresponding layer first.",
            )
            return

        image_slice = self._compute_crop_slice(box_layer, self.image_layer)
        if image_slice is None:
            QMessageBox.warning(
                self,
                "Warning",
                f"Could not compute ROI crop from {roi_source}.",
            )
            return

        sub = self.image_layer.data[image_slice]
        print(
            f"Segment ROI ({roi_source}): "
            f"slice={image_slice}, sub.shape={sub.shape}"
        )

        # Sync params from the form to the segmenter
        self.segmenter = (
            self.segmenter_parameter_form.sync_nd_operation_instance(
                self.segmenter
            )
        )

        self.segment_progress_logger.clear()
        self.segment_progress_logger.log_info(
            f"Segmenting ROI from {roi_source}..."
        )

        try:
            mask = self.image_data_model.segment(self.segmenter, sub)
        except (
            AttributeError,
            ValueError,
            TypeError,
            RuntimeError,
            IndexError,
        ) as e:
            print(f"ROI segmentation failed: {e}")
            QMessageBox.critical(
                self, "Error", f"ROI segmentation failed: {e}"
            )
            return

        if mask is None:
            self.segment_progress_logger.log_info("Segmenter returned no mask")
            return

        # Paste mask into the predictions layer using its own aligned crop.
        segmenter_name = self.segmenter.__class__.__name__
        predictions_layer = self._get_or_create_predictions_layer(
            segmenter_name, self.annotation_layer.data.shape
        )
        preds_slice = self._compute_crop_slice(box_layer, predictions_layer)
        if preds_slice is None:
            QMessageBox.critical(
                self,
                "Error",
                "Could not align mask to predictions layer.",
            )
            return

        try:
            mask_to_paste = np.asarray(mask)
            target_region = predictions_layer.data[preds_slice]
            if mask_to_paste.shape != target_region.shape:
                mask_to_paste = np.squeeze(mask_to_paste)
            if mask_to_paste.shape != target_region.shape:
                mask_to_paste = mask_to_paste.reshape(target_region.shape)
            predictions_layer.data[preds_slice] = mask_to_paste
            predictions_layer.refresh()
            self.segment_progress_logger.log_info(
                f"\u2705 ROI segmentation written to '{segmenter_name}'"
            )
        except (ValueError, TypeError, IndexError) as e:
            print(
                f"Failed to paste ROI mask (shape {np.asarray(mask).shape} "
                f"vs target {predictions_layer.data[preds_slice].shape}): {e}"
            )
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to write ROI mask into predictions layer: {e}",
            )
            return

        # Optional preview viewer with cropped image + cropped prediction.
        if self.show_roi_in_new_viewer_cb.isChecked():
            self._show_roi_in_new_viewer(
                box_layer, predictions_layer, image_slice, preds_slice
            )

    def _show_roi_in_new_viewer(
        self, box_layer, predictions_layer, image_slice, preds_slice
    ):
        """Open (or refresh) a second napari viewer with the cropped ROI."""
        if image_slice is None or preds_slice is None:
            return

        # Drop a stale viewer reference if it has been closed.
        if self._segment_roi_viewer is not None:
            try:
                _ = self._segment_roi_viewer.window  # touch
            except (RuntimeError, AttributeError):
                self._segment_roi_viewer = None

        if self._segment_roi_viewer is None:
            self._segment_roi_viewer = napari.Viewer(
                title="Segment ROI Preview"
            )
        else:
            try:
                self._segment_roi_viewer.layers.clear()
            except (RuntimeError, AttributeError):
                self._segment_roi_viewer = napari.Viewer(
                    title="Segment ROI Preview"
                )

        scale = None
        if self.image_data_model is not None:
            with contextlib.suppress(Exception):
                scale = (
                    self.image_data_model.get_scale(
                        axes_to_collapse=self.axes_to_collapse
                    )
                    or None
                )

        self._segment_roi_viewer.add_image(
            self.image_layer.data[image_slice],
            name=f"{self.image_layer.name} (ROI)",
            scale=scale,
        )
        self._segment_roi_viewer.add_labels(
            predictions_layer.data[preds_slice],
            name=f"{predictions_layer.name} (ROI)",
            scale=scale,
        )

    def _on_train(self):
        """Handle training - either via dialog or embedded form based on training_widget_mode."""
        if not hasattr(self, "segmenter") or self.segmenter is None:
            QMessageBox.warning(self, "Warning", "No segmenter selected")
            return

        # Check if segmenter supports training
        if not hasattr(self.segmenter, "train") or not callable(
            getattr(self.segmenter, "train", None)
        ):
            QMessageBox.warning(
                self,
                "Warning",
                "Selected segmenter does not support training",
            )
            return

        # Route to appropriate training method based on mode
        if self.training_widget_mode == "dialog":
            self._train_with_dialog()
        else:  # embedded mode (default)
            self._train_with_embedded_form()

    def _train_with_dialog(self):
        """Train using popup dialog (classic mode)."""
        # Create and show training dialog
        dialog = TrainDialog(self.segmenter, parent=self)
        result = dialog.exec_()

        if result == TrainDialog.Accepted:
            # Get training parameters from dialog
            training_params = dialog.get_training_parameters()
            print(f"Training parameters accepted: {training_params}")

            # Run training with dialog parameters (no progress updater in dialog mode)
            self._run_training(training_params, use_progress_logger=False)
        else:
            print("Training cancelled")

    def _train_with_embedded_form(self):
        """Train using embedded training parameter form (no dialog)."""
        # Get training parameters directly from the training_parameter_form
        if not hasattr(self, "training_parameter_form"):
            QMessageBox.warning(
                self,
                "Warning",
                "Training parameter form not available",
            )
            return

        # Sync current form values to segmenter
        self.segmenter = (
            self.training_parameter_form.sync_nd_operation_instance(
                self.segmenter
            )
        )

        # Get training parameters from segmenter (already synced from form)
        training_params = self.training_parameter_form.parameter_values

        print(f"Training with embedded parameters: {training_params}")

        # Clear progress logger before training
        self.progress_logger.clear()

        # Run training with embedded parameters (WITH progress updater)
        self._run_training(training_params, use_progress_logger=True)

    def _run_training(
        self, training_params: dict, use_progress_logger: bool = False
    ):
        """
        Execute training with given parameters.

        Args:
            training_params: Dictionary of training parameter names and values
            use_progress_logger: If True, pass progress_logger to train method (embedded mode)
        """
        # Get the selected axis to find the correct patches directory
        selected_axis = self.segmenter_parameter_form.get_selected_axis()
        if selected_axis:
            # Convert axis to lowercase for directory name (e.g., "YX" -> "yx")
            axis_lower = selected_axis.lower()
            print(f"Using axis: {axis_lower} for patches directory")
        else:
            axis_lower = None
            print("No axis selected, using default patches directory")

        # Set patch path, model save dir, and model name directly on segmenter
        self.segmenter.patch_path = str(
            self.image_data_model.get_patches_directory(axis=axis_lower)
        )
        self.segmenter.model_save_dir = str(
            self.image_data_model.get_models_directory()
        )
        self.segmenter.training_model_name = (
            self.model_name_edit.text().strip() or "model"
        )
        print(f"patch_path:    {self.segmenter.patch_path}")
        print(f"model_save_dir:{self.segmenter.model_save_dir}")
        print(f"training_model_name:    {self.segmenter.training_model_name}")

        # Sync hyper-parameters from the training form to segmenter
        for param_name, param_value in training_params.items():
            setattr(self.segmenter, param_name, param_value)

        # Stash flag so the finished handler knows whether to talk to the logger
        self._training_use_progress_logger = use_progress_logger

        print("Starting training...")
        if use_progress_logger:
            self.progress_logger.log_info("Starting training...")
            self.progress_logger.log_info(
                f"Model: {self.segmenter.training_model_name}"
            )
            self.progress_logger.log_info(
                f"Patches: {self.segmenter.patch_path}"
            )

        # Disable train button while training runs (if present)
        if hasattr(self, "train_btn"):
            self.train_btn.setEnabled(False)

        # Set to False to run training synchronously on the GUI thread
        # (useful when stepping through the training code in a debugger).
        thread_training = True

        if not thread_training:
            # Synchronous path — blocks the GUI but is debugger-friendly.
            updater = (
                self.progress_logger.update_progress
                if use_progress_logger
                else None
            )
            result = None
            try:
                result = self.segmenter.train(updater=updater)
            except (
                RuntimeError,
                ValueError,
                TypeError,
                OSError,
                IndexError,
                AttributeError,
            ) as e:
                self._on_training_error(str(e))
            self._on_training_finished(result)
            return

        # Run training in a worker thread so the GUI stays responsive.
        # We must keep a reference to the thread wrapper until it finishes.
        self._training_thread = TrainingThread(self.segmenter.train)

        if use_progress_logger:
            self._training_thread.progress.connect(
                self.progress_logger.update_progress
            )
        self._training_thread.error.connect(self._on_training_error)
        self._training_thread.finished.connect(self._on_training_finished)

        self._training_thread.start()

    def _on_training_error(self, error_msg: str):
        """Handle an exception raised inside the training worker thread."""
        full_msg = f"Training failed with error:\n{error_msg}"
        if getattr(self, "_training_use_progress_logger", False):
            self.progress_logger.log_error(full_msg)
        QMessageBox.critical(self, "Training Error", full_msg)

    def _on_training_finished(self, result):
        """Handle training completion (called on the main/GUI thread)."""
        # Re-enable the train button
        if hasattr(self, "train_btn"):
            self.train_btn.setEnabled(True)

        use_progress_logger = getattr(
            self, "_training_use_progress_logger", False
        )

        # If the worker emitted error(), result will be None and the error
        # handler already informed the user; nothing more to do here.
        if result is None:
            return

        if result.get("success"):
            if use_progress_logger:
                self.progress_logger.log_info(
                    f"✅ {result.get('message', 'Training complete!')}"
                )
            QMessageBox.information(
                self,
                "Training Complete",
                f"Training completed successfully!\n\n{result.get('message', '')}",
            )
        else:
            if use_progress_logger:
                self.progress_logger.log_warning(
                    f"Training status: {result.get('message', 'Unknown')}"
                )
            QMessageBox.warning(
                self,
                "Training Status",
                f"Training status:\n{result.get('message', 'Unknown status')}\n{result.get('error', '')}",
            )

        # Refresh model combos in both forms to include the newly trained model
        trained_name = self.segmenter.inference_model_name
        self.segmenter_parameter_form.refresh_model_combo(
            select_name=trained_name
        )
        if (
            hasattr(self.segmenter, "model_file_path")
            and self.segmenter.model_file_path
        ):
            self.segmenter_parameter_form.set_parameter(
                "model_file_path", self.segmenter.model_file_path
            )
        if hasattr(self, "training_parameter_form"):
            self.training_parameter_form.refresh_model_combo(
                select_name=trained_name
            )

    # === Common Methods (from original nd_easy_label) ===
    # _on_open_directory and load_image_directory inherited from BaseNDApp

    def _set_image_layer(self, image_layer):
        """Set up annotation layers based on the provided image layer."""
        self.image_layer = image_layer
        image_data = image_layer.data

        # Refresh the Segment-Range spinbox bounds based on the new image.
        self._update_range_bounds()

        # Load existing labels or create empty ones
        labels_data = self.image_data_model.load_existing_annotations(
            image_data.shape,
            self.current_image_index,
            axes_to_collapse=self.axes_to_collapse,
        )

        annotation_scale = self.image_data_model.get_scale(
            axes_to_collapse=self.axes_to_collapse
        )
        self.annotation_layer = self.viewer.add_labels(
            labels_data,
            name="Labels (Persistent)",
            scale=annotation_scale or None,
        )

        # Dictionary to hold prediction layers for different segmenters
        # Key: segmenter name, Value: napari labels layer
        self.predictions_layers = {}

        # Load any existing predictions from subdirectories (different segmenter methods)
        self._load_existing_prediction_layers(image_data.shape)

        # Only create interactive layers if in interactive mode
        if self.is_interactive_mode():
            self._setup_interactive_layers(image_data)

        print(f"Successfully set up layers for image: {image_layer.name}")

        # move image layer to bottom
        # self.viewer.layers.move(self.image_layer, len(self.viewer.layers)-1)

    def _load_existing_prediction_layers(self, image_shape):
        """Load predictions from all segmenter subdirectories as separate layers."""

        predictions_dir = (
            self.image_data_model.parent_directory / "predictions"
        )

        if not predictions_dir.exists():
            print(
                "No predictions directory found - no prediction layers created"
            )
            return

        # Find all subdirectories (each is a segmenter method)
        # Skip individual files in predictions/ - only process subdirectories
        subdirs = [d for d in predictions_dir.iterdir() if d.is_dir()]

        if not subdirs:
            print(
                "No prediction subdirectories found - no prediction layers created"
            )
            return

        print(f"Found {len(subdirs)} prediction subdirectories")

        for method_dir in subdirs:
            method_name = method_dir.name
            print(f"Checking predictions for: {method_name}")

            # Set current segmenter name for loading
            self.image_data_model.set_current_segmenter_name(method_name)

            try:
                # Load predictions for this method
                predictions = self.image_data_model.load_existing_predictions(
                    image_shape=image_shape,
                    image_index=self.current_image_index,
                    axes_to_collapse=self.axes_to_collapse,
                )

                # Only create layer if there's actual prediction data (not just empty array)
                if (
                    predictions is not None
                    and predictions.size > 0
                    and predictions.max() > 0
                ):
                    # Add as labels layer with method name
                    layer = self.viewer.add_labels(
                        predictions,
                        name=method_name,
                        scale=self.image_data_model.get_scale(
                            axes_to_collapse=self.axes_to_collapse
                        )
                        or None,
                    )
                    self.predictions_layers[method_name] = layer
                    print(
                        f"  ✓ Created layer '{method_name}' with predictions"
                    )
                else:
                    print(
                        f"  ✗ Skipped '{method_name}' - no prediction data found"
                    )
            except (ValueError, OSError, RuntimeError, AttributeError) as e:
                print(f"  ✗ Error loading {method_name}: {e}")

    def _get_or_create_predictions_layer(
        self, segmenter_name, predictions_shape
    ):
        """Get existing prediction layer for segmenter or create new one."""
        # Check if we already have a layer for this segmenter in the dictionary
        if segmenter_name in self.predictions_layers:
            layer = self.predictions_layers[segmenter_name]
            # Verify the layer still exists in the viewer
            if layer in self.viewer.layers:
                return layer
            else:
                # Layer was removed, delete from dictionary
                del self.predictions_layers[segmenter_name]

        # Create new empty predictions layer for this segmenter

        from ..utility import create_empty_instance_image

        empty_predictions = create_empty_instance_image(
            predictions_shape, dtype=np.uint16
        )
        new_layer = self.viewer.add_labels(
            empty_predictions,
            name=segmenter_name,
            scale=self.image_data_model.get_scale(
                axes_to_collapse=self.axes_to_collapse
            )
            or None,
        )
        self.predictions_layers[segmenter_name] = new_layer
        print(f"Created new predictions layer: {segmenter_name}")
        return new_layer

    # === Optional extra outputs from segmenters (e.g. MicroSamYoloSegmenter) ===
    def _show_stacked_labels_layer(self, stacked_labels):
        """Replace/create a 3D 'Stacked Labels (preview)' labels layer.

        Used for segmenters that produce a 3D stack representing overlapping
        2D labels. Not saved — purely a visualization aid.
        """
        layer_name = "Stacked Labels (preview)"
        # Remove existing layer if present so we always show the latest preview
        if layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)
        self.viewer.add_labels(stacked_labels, name=layer_name)

    def _add_predicted_boxes(self, boxes, current_step, segmentation_axis):
        """Add predicted object boxes to a 'predicted object boxes' shapes layer.

        Boxes from the segmenter are 2D (YX). For ND images, leading
        non-spatial step indices are prepended to each vertex so the boxes
        appear on the correct slice.
        """
        layer_name = "predicted object boxes"
        image_ndim = len(self.image_layer.data.shape)

        # Determine number of leading (non-spatial) dims for the segmentation axis.
        axis = segmentation_axis.replace("C", "")
        spatial_ndim = len(axis)  # "YX" -> 2, "ZYX" -> 3
        leading = list(current_step[: image_ndim - spatial_ndim])

        # Prepend leading indices to every vertex of every box
        nd_boxes = []
        for box in boxes:
            arr = np.asarray(box)
            if leading:
                pad = np.tile(
                    np.array(leading, dtype=arr.dtype), (arr.shape[0], 1)
                )
                arr = np.concatenate([pad, arr], axis=1)
            nd_boxes.append(arr)

        # Get or create the shapes layer
        if layer_name in self.viewer.layers:
            shapes_layer = self.viewer.layers[layer_name]
            shapes_layer.add_rectangles(nd_boxes)
        else:
            self.viewer.add_shapes(
                nd_boxes,
                name=layer_name,
                shape_type="rectangle",
                edge_color="green",
                face_color="transparent",
                edge_width=2,
                ndim=image_ndim,
            )

    def _setup_interactive_layers(self, image_data):
        """Setup interactive annotation layers (points and shapes)."""
        # Points layer
        self._point_choices = ["positive", "negative"]
        LABEL_COLOR_CYCLE = ["red", "blue"]

        annotation_ndim = len(self.annotation_layer.data.shape)

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
            scale=self.image_data_model.get_scale(
                axes_to_collapse=self.axes_to_collapse
            )
            or None,
        )
        self.points_layer.events.data.connect(self._on_points_changed)

        # Shapes layer
        annotation_ndim = min(len(image_data.shape), 3)
        shapes_scale = self.image_data_model.get_scale(
            axes_to_collapse=self.axes_to_collapse
        )
        # Trim to match the (possibly clamped) annotation_ndim by taking
        # the trailing axes (Y/X[/Z]).
        if shapes_scale and len(shapes_scale) >= annotation_ndim:
            shapes_scale = shapes_scale[-annotation_ndim:]
        else:
            shapes_scale = None
        self.shapes_layer = self.viewer.add_shapes(
            name="Shapes Layer",
            edge_color="green",
            face_color="transparent",
            edge_width=2,
            ndim=annotation_ndim,
            scale=shapes_scale,
        )
