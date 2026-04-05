"""
Unified ND Easy Segmentation Widget.

This module provides a unified interface for both interactive (point/shape-based)
and automatic (full plane/volume) segmentation workflows.
"""

import itertools

import napari
import numpy as np
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

from ..models import ImageDataModel
from ..utilities import QtProgressLogger
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

        # Parameter form widget (from base)
        main_layout.addWidget(self._create_segmenter_parameter_form())

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

        self.train_btn = QPushButton("Train")
        self.train_btn.clicked.connect(self._on_train)
        auto_layout.addWidget(self.train_btn)

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
        """Update both segmenter and training parameter forms."""
        # Update the main segmenter parameter form (inference params)
        self.segmenter_parameter_form.set_nd_operation(segmenter)

        # Also update training parameter form if it exists
        if hasattr(self, "training_parameter_form"):
            self.training_parameter_form.set_nd_operation(segmenter)

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

        self.segment_progress_logger.update_progress(1, 2, "Processing...")
        self._segment_nd_slice(current_step=self.viewer.dims.current_step)
        self.segment_progress_logger.update_progress(2, 2, "✅ Complete")
        self.segment_progress_logger.log_info("✅ Segmentation complete")

    def _on_segment_all(self):
        """Segment all slices in the ND data automatically."""
        if self.image_layer is None:
            QMessageBox.warning(self, "Warning", "No images loaded")
            return

        if not hasattr(self, "segmenter") or self.segmenter is None:
            QMessageBox.warning(self, "Warning", "No segmenter selected")
            return

        # Clear progress logger
        self.segment_progress_logger.clear()
        self.segment_progress_logger.log_info(
            "Starting segmentation of all slices..."
        )

        print("Segmenting all slices...")

        # Ensure segmenter is synced with current parameters
        self.segmenter = (
            self.segmenter_parameter_form.sync_nd_operation_instance(
                self.segmenter
            )
        )

        # Get selected axis and dataset axis types
        selected_axis = self.segmenter_parameter_form.get_selected_axis()
        dataset_axis_types = self.image_data_model.axis_types
        image_shape = self.image_layer.data.shape

        print(f"Selected axis: {selected_axis}")
        print(f"Dataset axis types: {dataset_axis_types}")
        print(f"Image shape: {image_shape}")

        self.segment_progress_logger.log_info(
            f"Selected axis: {selected_axis}"
        )
        self.segment_progress_logger.log_info(f"Image shape: {image_shape}")

        # Determine number of spatial dimensions
        if selected_axis.endswith("ZYX"):
            num_spatial = 3
        elif selected_axis.endswith("YX"):
            num_spatial = 2
        else:
            num_spatial = 2

        # Calculate number of non-spatial dimensions
        num_non_spatial = len(image_shape) - num_spatial

        num_collapsed = (
            len(self.axes_to_collapse) if self.axes_to_collapse else 0
        )

        num_for_loop = (
            num_non_spatial - num_collapsed
            if self.axes_to_collapse
            else num_non_spatial
        )

        # Here we make a naive assumption.
        # 1.  Dimensions to loop through are first
        # 2. Spatial dimensions are next
        # 3. Collapsed dimensions are last (if any)

        # this will work for say NZYXC loop through N, ignore C and segment ZYX
        # TODO: make general to handle any ordering of dimensions and collapsing (e.g., NZCYX with C collapsed should still segment ZYX correctly)

        # Get shape of non-spatial dimensions
        non_spatial_shape = image_shape[:num_for_loop]

        print(f"Non-spatial dimensions: {num_non_spatial}")
        print(f"Number of collapsed axes: {num_collapsed}")
        print(f"num for loop: {num_for_loop}")
        print(f"Non-spatial shape: {non_spatial_shape}")

        # Calculate total number of slices
        total_slices = (
            int(np.prod(non_spatial_shape)) if non_spatial_shape else 1
        )

        print(f"Total slices to segment: {total_slices}")
        self.segment_progress_logger.log_info(
            f"Total slices to segment: {total_slices}"
        )

        # Iterate through all combinations of non-spatial indices
        for idx, non_spatial_indices in enumerate(
            itertools.product(*[range(dim) for dim in non_spatial_shape])
        ):
            # Build current_step tuple
            current_step = non_spatial_indices + (0,) * num_spatial

            # Update progress
            self.segment_progress_logger.update_progress(
                idx + 1,
                total_slices,
                f"Processing slice {idx + 1}/{total_slices}",
            )

            print(
                f"Processing slice {idx + 1}/{total_slices}: step={current_step}"
            )

            # Segment the slice
            self._segment_nd_slice(current_step=current_step)

        print(f"✅ Completed segmentation of all {total_slices} slices")
        self.segment_progress_logger.log_info(
            f"✅ Completed segmentation of all {total_slices} slices"
        )

        QMessageBox.information(
            self,
            "Success",
            f"Successfully segmented all {total_slices} slices",
        )

    def _segment_nd_slice(self, current_step: tuple):
        """Perform automatic segmentation on image data.

        Args:
            image_data: The MD slice to segment
            input_axis: The axis mode (e.g., "YX", "ZYX", "YXC")
            current_step: The current step/position in the ND data
        """

        try:

            if not hasattr(self, "segmenter") or self.segmenter is None:
                QMessageBox.warning(self, "Warning", "No segmenter selected")
                return

            # Set segmenter name for organizing predictions
            segmenter_name = self.segmenter.__class__.__name__
            self.image_data_model.set_current_segmenter_name(segmenter_name)

            # Print the axis mode the user chose
            selected_axis = self.segmenter_parameter_form.get_selected_axis()
            print(f"User selected axis mode: {selected_axis}")

            # Extract current slice based on selected axis mode
            indices = get_current_slice_indices(current_step, selected_axis)

            current_yx_slice = self.image_layer.data[indices]

            # Call segmenter through model (provides parent_directory automatically)
            mask = self.image_data_model.segment(
                self.segmenter,
                current_yx_slice,
                points=None,
                shapes=None,
            )

            # Get the segmentation axis from the segmenter
            # This handles cases where the segmenter transforms the axis
            # (e.g., YXC -> YX when channel dimension is collapsed)
            segmentation_axis = self.segmenter.get_segmentation_axis(
                selected_axis
            )

            # save predictions via model
            # Use the current segmenter name as subdirectory to organize predictions by method
            subdirectory = (
                self.image_data_model._current_segmenter_name or "default"
            )
            self.image_data_model.save_predictions(
                mask,
                self.current_image_index,
                subdirectory=subdirectory,
                current_step=current_step,
                selected_axis=segmentation_axis,
                axes_to_collapse=self.axes_to_collapse,
            )

            segmentation_indices = get_current_slice_indices(
                current_step, segmentation_axis
            )

            # Get or create the predictions layer for this segmenter
            segmenter_name = (
                self.image_data_model._current_segmenter_name or "default"
            )
            predictions_layer = self._get_or_create_predictions_layer(
                segmenter_name, self.annotation_layer.data.shape
            )

            # Update the layer with new predictions
            predictions_layer.data[segmentation_indices] = mask
            predictions_layer.refresh()
            print(
                f"Automatic segmentation completed - updated layer: {segmenter_name}"
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
        self.segmenter.model_name = (
            self.model_name_edit.text().strip() or "model"
        )
        print(f"patch_path:    {self.segmenter.patch_path}")
        print(f"model_save_dir:{self.segmenter.model_save_dir}")
        print(f"model_name:    {self.segmenter.model_name}")

        # Sync hyper-parameters from the training form to segmenter
        for param_name, param_value in training_params.items():
            setattr(self.segmenter, param_name, param_value)

        # Call the train method
        try:
            print("Starting training...")

            # Log to progress widget if in embedded mode
            if use_progress_logger:
                self.progress_logger.log_info("Starting training...")
                self.progress_logger.log_info(
                    f"Model: {self.segmenter.model_name}"
                )
                self.progress_logger.log_info(
                    f"Patches: {self.segmenter.patch_path}"
                )

            # Call train with or without updater based on mode
            if use_progress_logger:
                result = self.segmenter.train(
                    updater=self.progress_logger.update_progress
                )
            else:
                # No updater (dialog mode)
                result = self.segmenter.train()

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
            trained_name = self.segmenter.model_preset
            self.segmenter_parameter_form.refresh_model_combo(
                select_name=trained_name
            )
            if hasattr(self, "training_parameter_form"):
                self.training_parameter_form.refresh_model_combo(
                    select_name=trained_name
                )

        except (RuntimeError, ValueError, TypeError, OSError) as e:
            error_msg = f"Training failed with error:\n{str(e)}"

            if use_progress_logger:
                self.progress_logger.log_error(error_msg)

            QMessageBox.critical(
                self,
                "Training Error",
                error_msg,
            )

    # === Common Methods (from original nd_easy_label) ===
    # _on_open_directory and load_image_directory inherited from BaseNDApp

    def _set_image_layer(self, image_layer):
        """Set up annotation layers based on the provided image layer."""
        self.image_layer = image_layer
        image_data = image_layer.data

        # Load existing labels or create empty ones
        labels_data = self.image_data_model.load_existing_annotations(
            image_data.shape,
            self.current_image_index,
            axes_to_collapse=self.axes_to_collapse,
        )

        self.annotation_layer = self.viewer.add_labels(
            labels_data, name="Labels (Persistent)"
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
                        predictions, name=method_name
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
        import numpy as np

        from ..utility import create_empty_instance_image

        empty_predictions = create_empty_instance_image(
            predictions_shape, dtype=np.uint16
        )
        new_layer = self.viewer.add_labels(
            empty_predictions, name=segmenter_name
        )
        self.predictions_layers[segmenter_name] = new_layer
        print(f"Created new predictions layer: {segmenter_name}")
        return new_layer

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
