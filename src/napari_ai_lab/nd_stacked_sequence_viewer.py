import contextlib

import napari
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QWidget,
)

from .artifact_io import StackedSequenceArtifactIO
from .models import ImageDataModel


class NDStackedSequenceViewer(QWidget):
    """
    A Napari plugin for browsing image series as a single stacked layer.

    This widget loads all images into one stacked layer instead of switching
    between individual image layers. Compatible with NDSequenceViewer interface.
    """

    # Signal emitted when the current image changes
    # Emits (image_layer, image_index)
    image_changed = Signal(object, int)

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self._loading_new_image = False
        # Initialize image model and current image layer
        self.image_data_model = None
        self.current_index = 0
        self.current_image_layer = None
        self.image_names = []  # List of image names in stack order

        # Set up the UI with horizontal layout
        self.setLayout(QHBoxLayout())

        # Add directory selection button (narrower)
        self.dir_btn = QPushButton("Open Dir")
        self.dir_btn.clicked.connect(self._on_open_directory)
        self.dir_btn.setMaximumWidth(80)
        self.layout().addWidget(self.dir_btn)

        # Add label to show current image info
        self.image_info_label = QLabel("No directory selected")
        self.image_info_label.setMinimumWidth(200)
        self.layout().addWidget(self.image_info_label)

        print("NDStackedSequenceViewer initialized")

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
            self.set_image_data_model(ImageDataModel(directory))
        else:
            print("No directory selected")
            return

    def set_image_data_model(self, model: ImageDataModel):
        """Set the image data model and load all images into a stack."""
        try:
            self.image_data_model = model

            # Get all image paths from the model
            image_paths = model.get_image_paths()

            if image_paths:
                print(f"Found {len(image_paths)} image files")
                print(
                    "Loading all images into stack using StackedSequenceArtifactIO..."
                )

                # Use StackedSequenceArtifactIO to load directory as stack with normalization
                # Note that the stacked_images are not loaded into the model itself
                # because they are just a view of the data, the model still has file paths
                # which can be used to access individual images.
                stacked_sequence_io = StackedSequenceArtifactIO()
                stacked_images = stacked_sequence_io.load_full_stack(
                    str(model.parent_directory), normalize=True
                )

                # Store image names from stacked_sequence_io
                self.image_names = (
                    stacked_sequence_io._image_names
                    if stacked_sequence_io._image_names
                    else []
                )

                if stacked_images.size == 0:
                    QMessageBox.warning(
                        self,
                        "Load Error",
                        "Could not load any images from directory.",
                    )
                    self._reset_display()
                    return

                print(f"Created stack with shape: {stacked_images.shape}")

                # Remove old layer if exists
                if self.current_image_layer is not None:
                    self.viewer.layers.remove(self.current_image_layer)
                    self.current_image_layer = None

                # Add stacked image to viewer
                self.current_image_layer = self.viewer.add_image(
                    stacked_images, name="Stacked Image Series"
                )

                # Subscribe to dimension change events to track current image
                # self.viewer.dims.events.current_step.connect(self._on_dims_changed)

                # Reset current index
                self.current_index = 0

                # Update display
                self._update_image_info()

                # Set viewer to display first slice
                self.viewer.dims.set_point(0, 0)

                # Emit signal for first image
                self.image_changed.emit(self.current_image_layer, 0)

            else:
                QMessageBox.information(
                    self,
                    "No Images Found",
                    "No image files found in the selected directory.",
                )
                self._reset_display()

        except (OSError, PermissionError, ValueError, FileNotFoundError) as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while setting image model: {str(e)}",
            )
            print(f"Error setting image model: {e}")
            self._reset_display()

    def _on_dims_changed(self, event):
        """Handle viewer dimension changes to track current image."""
        if self.image_names and len(event.value) > 0:
            idx = int(event.value[0])
            if 0 <= idx < len(self.image_names):
                self.current_index = idx
                print(
                    f"Image {idx + 1}/{len(self.image_names)}: {self.image_names[idx]}"
                )
                self.image_changed.emit(self.current_image_layer, idx)

    def _update_image_info(self):
        """Update the image info label with current image details."""
        if self.image_data_model:
            image_paths = self.image_data_model.get_image_paths()
            info_text = f"Loaded {len(image_paths)} images as stack"
            self.image_info_label.setText(info_text)
        else:
            self.image_info_label.setText("No images available")

    def _reset_display(self):
        """Reset the display when no images are available."""
        # Remove current image layer
        if self.current_image_layer is not None:
            with contextlib.suppress(ValueError, KeyError):
                self.viewer.layers.remove(self.current_image_layer)
            self.current_image_layer = None

        self.image_data_model = None
        self.current_index = 0
        self.image_names = []
        self.image_info_label.setText("No directory selected")

    def get_current_image_path(self):
        """Get the path of the currently selected image."""
        if (
            self.image_data_model
            and 0
            <= self.current_index
            < self.image_data_model.get_image_count()
        ):
            image_paths = self.image_data_model.get_image_paths()
            return str(image_paths[self.current_index])
        return None

    def get_image_count(self):
        """Get the total number of images in the series."""
        return (
            self.image_data_model.get_image_count()
            if self.image_data_model
            else 0
        )

    def get_all_image_paths(self):
        """Get all image paths in the series."""
        if self.image_data_model:
            return [str(f) for f in self.image_data_model.get_image_paths()]
        return []
