"""
ND Easy Augment - Augmentation interface with dynamic parameter controls.

This module provides an interface for data augmentation with support for
different augmentation frameworks (SimpleAugmenter, AlbumentationsAugmenter, etc.).
Uses nd_operation_widget for dynamic parameter controls based on selected augmenter.
"""

import napari
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..Augmenters import AugmenterBase
from ..models import ImageDataModel
from ..utilities import QtProgressLogger, SliceProcessor, SliceProcessorThread
from ..utility import get_supported_axes_from_shape
from ..widgets.nd_operation_widget import NDOperationWidget
from .base_nd_app import BaseNDApp


class NDEasyAugment(BaseNDApp):
    """
    Augmentation app with dynamic parameter controls.

    Provides combo box selection of different augmenters and displays
    their parameters using nd_operation_widget.
    """

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
        self.augmenter_cache = {}  # Cache for augmenter instances

        # Create Qt progress logger
        self.progress_logger = QtProgressLogger()

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        self.setLayout(QVBoxLayout())

        # Patch mode selection (first — determines how patch positions are sampled)
        self.patch_mode_label = QLabel("Patch Mode:")
        self.layout().addWidget(self.patch_mode_label)

        self.patch_mode_combo = QComboBox()
        self.patch_mode_combo.addItems(ImageDataModel.get_patch_modes())
        self.layout().addWidget(self.patch_mode_combo)

        # Add Augmenter selection
        self.augmenter_label = QLabel("Augmenter:")
        self.layout().addWidget(self.augmenter_label)

        self.augmenter_combo = QComboBox()
        self.augmenter_combo.currentTextChanged.connect(
            self._on_augmenter_changed
        )
        self.layout().addWidget(self.augmenter_combo)

        # Create augmentation parameter form (separate from base parameter_form)
        # This parses "augmentation" param_type instead of "inference"
        self.augmentation_form = NDOperationWidget(
            param_type_to_parse="augmentation"
        )
        self.layout().addWidget(self.augmentation_form)

        # Add patch size control
        patch_size_layout = QHBoxLayout()
        patch_size_layout.addWidget(QLabel("Patch Size (XY):"))
        self.patch_size_xy_spinbox = QSpinBox()
        self.patch_size_xy_spinbox.setMinimum(16)
        self.patch_size_xy_spinbox.setMaximum(2048)
        self.patch_size_xy_spinbox.setValue(128)
        self.patch_size_xy_spinbox.setSingleStep(16)
        patch_size_layout.addWidget(self.patch_size_xy_spinbox)

        self.layout().addLayout(patch_size_layout)

        # Add Z patch size control
        patch_size_z_layout = QHBoxLayout()
        patch_size_z_layout.addWidget(QLabel("Patch Size (Z):"))
        self.patch_size_z_spinbox = QSpinBox()
        self.patch_size_z_spinbox.setMinimum(1)
        self.patch_size_z_spinbox.setMaximum(512)
        self.patch_size_z_spinbox.setValue(32)
        self.patch_size_z_spinbox.setSingleStep(4)
        patch_size_z_layout.addWidget(self.patch_size_z_spinbox)
        self.layout().addLayout(patch_size_z_layout)

        # Add number of patches control
        num_patches_layout = QHBoxLayout()
        num_patches_layout.addWidget(QLabel("Number of Patches:"))
        self.num_patches_spinbox = QSpinBox()
        self.num_patches_spinbox.setMinimum(1)
        self.num_patches_spinbox.setMaximum(10000)
        self.num_patches_spinbox.setValue(200)
        self.num_patches_spinbox.setSingleStep(10)
        num_patches_layout.addWidget(self.num_patches_spinbox)
        self.layout().addLayout(num_patches_layout)

        # Add buttons for augmentation operations
        buttons_layout = QHBoxLayout()

        self.perform_augmentation_button = QPushButton("Perform Augmentations")
        self.perform_augmentation_button.clicked.connect(
            self._on_perform_augmentations
        )
        buttons_layout.addWidget(self.perform_augmentation_button)

        self.delete_augmentations_button = QPushButton("Delete Augmentations")
        self.delete_augmentations_button.clicked.connect(
            self._on_delete_augmentations
        )
        buttons_layout.addWidget(self.delete_augmentations_button)

        self.show_patches_button = QPushButton("Show Patches")
        self.show_patches_button.clicked.connect(self._on_show_patches)
        buttons_layout.addWidget(self.show_patches_button)

        self.layout().addLayout(buttons_layout)

        # Add stretch before progress widget to push it to bottom
        self.layout().addStretch()

        # Add progress logger widget at bottom (stretch factor 1 to fill available space)
        self.layout().addWidget(self.progress_logger.get_widget(), 1)

        # Populate augmenter combo with registered frameworks
        self._populate_augmenter_combo()

    def _populate_augmenter_combo(self):
        """Populate the augmenter combo box with registered frameworks."""
        # Clear existing items
        self.augmenter_combo.clear()

        # Get registered frameworks
        frameworks = AugmenterBase.get_registered_frameworks()

        if frameworks:
            # Sort augmenter names alphabetically
            framework_names = sorted(frameworks.keys())
            self.augmenter_combo.addItems(framework_names)

            # Trigger selection of first augmenter to populate the form
            if self.augmenter_combo.count() > 0:
                self.augmenter_combo.setCurrentIndex(0)
                self._on_augmenter_changed(self.augmenter_combo.currentText())
        else:
            self.augmenter_combo.addItem("No augmenters available")

    def _on_augmenter_changed(self, augmenter_name: str):
        """Handle augmenter selection change."""
        if augmenter_name == "No augmenters available":
            return

        print(f"Selected augmenter: {augmenter_name}")

        # Check if we already have this augmenter in cache
        if augmenter_name in self.augmenter_cache:
            augmenter = self.augmenter_cache[augmenter_name]
            print(f"Using cached augmenter: {augmenter_name}")
        else:
            # Get the augmenter class from registry
            augmenter_class = AugmenterBase.get_framework(augmenter_name)

            if augmenter_class is None:
                print(f"Warning: Could not find augmenter: {augmenter_name}")
                return

            # Create new instance and cache it
            augmenter = augmenter_class()
            self.augmenter_cache[augmenter_name] = augmenter
            print(f"Created new augmenter instance: {augmenter_name}")
            print(f"Augmenter type: {type(augmenter)}")
            print(
                f"Augmenter has get_parameters_dict: {hasattr(augmenter, 'get_parameters_dict')}"
            )
            if hasattr(augmenter, "get_parameters_dict"):
                print(
                    f"Augmenter parameters: {augmenter.get_parameters_dict()}"
                )

        # Store current augmenter
        self.augmenter = augmenter

        # Filter axes based on current image shape
        if self.image_layer is not None and hasattr(
            self.augmenter, "_potential_axes"
        ):
            filtered = get_supported_axes_from_shape(
                self.image_layer.data.shape,
                self.augmenter._potential_axes,
                getattr(self.image_data_model, "axis_types", "U"),
            )
            self.augmenter.supported_axes = filtered
            if (
                hasattr(self.augmenter, "selected_axis")
                and self.augmenter.selected_axis not in filtered
                and filtered
            ):
                self.augmenter.selected_axis = filtered[0]

        print("About to set augmenter in augmentation form...")
        # Set the new augmenter in the form - this rebuilds the UI with new parameters
        self.augmentation_form.set_nd_operation(self.augmenter)
        print("Augmentation form updated successfully")

    def _set_image_layer(self, image_layer):
        """
        Set up annotation layer based on the provided image layer.

        For augmentation, we only need the annotations layer (not predictions).
        Augmentation works with annotated data to create training patches.
        """
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

        print(
            f"Successfully set up annotation layer for image: {image_layer.name}"
        )

    def get_augmentation_parameters(self):
        """Get current augmentation parameters from the selected augmenter."""
        if hasattr(self, "augmenter") and self.augmenter is not None:
            return self.augmenter.get_parameters_dict()
        return {}

    def _on_perform_augmentations(self):
        """Perform augmentations by generating patches."""
        # Validate that we have an augmenter
        if not hasattr(self, "augmenter") or self.augmenter is None:
            print("⚠️ No augmenter selected!")
            return

        # Validate that we have image and annotations
        if not hasattr(self, "image_layer") or self.image_layer is None:
            print("⚠️ No image loaded!")
            return

        if (
            not hasattr(self, "annotation_layer")
            or self.annotation_layer is None
        ):
            print("⚠️ No annotations layer!")
            return

        try:
            patch_mode = self.patch_mode_combo.currentText()

            if patch_mode == "from_label_boxes":
                self.progress_logger.clear()
                self.augmenter = (
                    self.augmentation_form.sync_nd_operation_instance(
                        self.augmenter
                    )
                )
                patch_size_xy = self.patch_size_xy_spinbox.value()
                num_patches = self.num_patches_spinbox.value()
                selected_axis = self.augmentation_form.get_selected_axis()
                if "Z" in selected_axis:
                    patch_size_z = self.patch_size_z_spinbox.value()
                    patch_size = (patch_size_z, patch_size_xy, patch_size_xy)
                else:
                    patch_size = (patch_size_xy, patch_size_xy)
                self.image_data_model.set_augmenter(self.augmenter)
                self.image_data_model.set_patch_size(patch_size)
                self.image_data_model.set_num_patches(num_patches)
                self.image_data_model.generate_patches_from_labels(
                    progress_logger=self.progress_logger,
                )
                return

            # Get current values from UI
            patch_size_xy = self.patch_size_xy_spinbox.value()
            num_patches = self.num_patches_spinbox.value()

            # Sync augmenter parameters from form back to augmenter instance
            self.augmenter = self.augmentation_form.sync_nd_operation_instance(
                self.augmenter
            )

            annotations = self.annotation_layer.data

            selected_axis = self.augmentation_form.get_selected_axis()

            if "Z" in selected_axis:
                patch_size_z = self.patch_size_z_spinbox.value()
                patch_size = (patch_size_z, patch_size_xy, patch_size_xy)
            else:
                patch_size = (patch_size_xy, patch_size_xy)

            # Configure the model
            self.image_data_model.set_augmenter(self.augmenter)
            self.image_data_model.set_patch_size(patch_size)
            self.image_data_model.set_num_patches(num_patches)
            image_shape = self.image_layer.data.shape

            print("\n🔧 Setting up augmentation...")
            print(f"  Image shape: {image_shape}")
            print(f"  Selected axis: {selected_axis}")
            print(f"  Patch size: {patch_size}")
            print(f"  Number of patches: {num_patches}")

            self.progress_logger.clear()

            processor = SliceProcessor(
                image_shape, selected_axis, self.axes_to_collapse
            )

            print(f"Total slices to augment: {processor.total_slices}")
            self.progress_logger.log_info(
                f"Total slices to augment: {processor.total_slices}"
            )

            # Disable button while running
            self.perform_augmentation_button.setEnabled(False)

            def do_augment_slice(current_step):
                return self.image_data_model.augment_slice(
                    annotations,
                    current_step,
                    selected_axis,
                    patch_mode=patch_mode,
                    progress_logger=self.progress_logger,
                )

            use_threading = False

            if use_threading:
                self._augment_thread = SliceProcessorThread(
                    processor, do_augment_slice
                )
                self._augment_thread.progress.connect(
                    self._on_augment_all_progress
                )
                self._augment_thread.finished.connect(
                    self._on_augment_all_finished
                )
                self._augment_thread.error.connect(self._on_augment_all_error)
                self._augment_thread.start()
            else:
                processor.process_all(
                    do_augment_slice,
                    on_progress=lambda cur, tot: print(
                        f"Augmenting slice {cur}/{tot}"
                    ),
                )
                self._on_augment_all_finished()

        except (ValueError, RuntimeError, OSError, AttributeError) as e:
            error_msg = f"Error during augmentation: {e}"
            self.progress_logger.log_error(error_msg)

            import traceback

            traceback.print_exc()

    def _on_augment_all_progress(self, current, total):
        """Handle progress updates from the augmentation worker thread."""
        self.progress_logger.update_progress(
            current, total, f"Processing slice {current}/{total}"
        )
        print(f"Augmenting slice {current}/{total}")

    def _on_augment_all_finished(self):
        """Handle completion of threaded augment-all."""
        self.perform_augmentation_button.setEnabled(True)
        total = getattr(self, "_augment_thread", None)
        total_str = (
            str(total.worker.processor.total_slices) if total else "all"
        )
        print(f"✅ Completed augmentation of {total_str} slices")
        self.progress_logger.log_info(
            f"✅ Completed augmentation of {total_str} slices"
        )
        self._augment_thread = None

    def _on_augment_all_error(self, error_msg):
        """Handle errors from the augmentation worker thread."""
        self.perform_augmentation_button.setEnabled(True)
        print(f"Error during augment all: {error_msg}")
        self.progress_logger.log_error(f"Augmentation failed: {error_msg}")
        self._augment_thread = None

    def _on_delete_augmentations(self):
        """Delete augmentation patches by removing the patches directory."""
        try:
            print("\n🗑️  Deleting augmentation patches...")

            # Delete patches (axis=None means delete base patches/ directory)
            self.image_data_model.delete_patches(axis=None)

            print("✅ Augmentation patches deleted successfully!")

        except (OSError, AttributeError) as e:
            print(f"\n❌ Error deleting patches: {e}")
            import traceback

            traceback.print_exc()

    def _on_show_patches(self):
        """Open a second napari viewer showing input/truth patch stacks."""

        import numpy as np
        from skimage.io import imread

        patches_dir = self.image_data_model.get_patches_directory(axis="yx")
        input_dir = patches_dir / "input0"
        truth_dir = patches_dir / "ground_truth0"

        if not input_dir.exists() or not truth_dir.exists():
            QMessageBox.warning(
                self, "No Patches", "No patches found. Run augmentation first."
            )
            return

        input_files = sorted(input_dir.glob("*.tif"))
        truth_files = sorted(truth_dir.glob("*.tif"))

        if len(input_files) == 0:
            QMessageBox.warning(
                self, "No Patches", "No .tif files found in patches."
            )
            return

        inputs_stack = np.stack([imread(str(f)) for f in input_files])
        truths_stack = np.stack([imread(str(f)) for f in truth_files])

        viewer = napari.Viewer(title="Patch Viewer")
        viewer.add_image(inputs_stack, name="input")
        viewer.add_labels(truths_stack, name="truth")
