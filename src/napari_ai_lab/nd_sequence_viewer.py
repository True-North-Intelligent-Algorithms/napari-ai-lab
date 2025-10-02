import contextlib
from pathlib import Path

import napari
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollBar,
    QWidget,
)
from skimage import io


class NDSequenceViewer(QWidget):
    """
    A simple Napari plugin for browsing image series with a scroll bar.

    This widget allows users to select an image directory and scroll through
    the list of images without opening them yet.
    """

    # Signal emitted when the current image changes
    # Emits (image_layer, image_path, parent_directory)
    image_changed = Signal(object, str, str)

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self._loading_new_image = False
        # Initialize image list and current image layer
        self.image_files = []
        self.current_index = 0
        self.current_image_layer = None
        self.parent_directory = None

        # Set up the UI with horizontal layout
        self.setLayout(QHBoxLayout())

        # Add directory selection button (narrower)
        self.dir_btn = QPushButton("Open Dir")
        self.dir_btn.clicked.connect(self._on_open_directory)
        self.dir_btn.setMaximumWidth(80)  # Make button narrower
        self.layout().addWidget(self.dir_btn)

        # Add label to show current image info
        self.image_info_label = QLabel("No directory selected")
        self.image_info_label.setMinimumWidth(
            200
        )  # Ensure minimum space for info
        self.layout().addWidget(self.image_info_label)

        # Add scroll bar for browsing images (wider)
        self.image_scrollbar = QScrollBar(Qt.Horizontal)
        self.image_scrollbar.setMinimum(0)
        self.image_scrollbar.setMaximum(0)
        self.image_scrollbar.setValue(0)
        self.image_scrollbar.setEnabled(False)
        self.image_scrollbar.valueChanged.connect(self._on_scroll_changed)
        # Add with stretch factor to make scrollbar take up more space
        self.layout().addWidget(self.image_scrollbar, 1)

        print("NDSeriesViewer initialized")

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
            self._load_image_list(directory)
        else:
            print("No directory selected")
            return

    def _load_image_list(self, directory):
        """Load list of image files from the selected directory."""
        try:
            # Store parent directory for zarr persistence
            self.parent_directory = str(Path(directory).resolve())

            # Common image extensions
            image_extensions = {
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff",
                ".bmp",
                ".gif",
            }

            # Get all files in directory
            directory_path = Path(directory)
            all_files = list(directory_path.iterdir())

            # Filter for image files
            self.image_files = [
                f
                for f in all_files
                if f.is_file() and f.suffix.lower() in image_extensions
            ]

            # Sort files by name
            self.image_files.sort(key=lambda x: x.name.lower())

            if self.image_files:
                print(f"Found {len(self.image_files)} image files")

                # Update scroll bar
                self.image_scrollbar.setMaximum(len(self.image_files) - 1)
                self.image_scrollbar.setEnabled(True)
                self.image_scrollbar.setValue(0)

                # Reset current index
                self.current_index = 0

                # Update display and load first image
                self._update_image_info()
                self._load_current_image()

            else:
                QMessageBox.information(
                    self,
                    "No Images Found",
                    "No image files found in the selected directory.",
                )
                self._reset_display()

        except (OSError, PermissionError, ValueError) as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while loading image list: {str(e)}",
            )
            print(f"Error loading image list: {e}")
            self._reset_display()

    def _on_scroll_changed(self, value):
        """Handle scroll bar value changes - load new image."""

        print(f"Scroll bar changed to {value}")

        if self._loading_new_image:
            print("Already loading a new image, ignoring scroll event")
            return  # Already loading, ignore this event

        self._loading_new_image = True

        try:
            if 0 <= value < len(self.image_files):
                self.current_index = value
                self._update_image_info()
                self._load_current_image()
                print(
                    f"Scrolled to image {value + 1}/{len(self.image_files)}: {self.image_files[value].name}\n\n"
                )
        finally:
            self._loading_new_image = False

    def _update_image_info(self):
        """Update the image info label with current image details."""
        if self.image_files and 0 <= self.current_index < len(
            self.image_files
        ):
            current_file = self.image_files[self.current_index]
            info_text = f"Image {self.current_index + 1}/{len(self.image_files)}: {current_file.name}"
            self.image_info_label.setText(info_text)
        else:
            self.image_info_label.setText("No images available")

    def _load_current_image(self):
        """Load the current image into the napari viewer."""
        if not self.image_files or not (
            0 <= self.current_index < len(self.image_files)
        ):
            return

        try:
            # Remove old image layer if it exists
            if self.current_image_layer is not None:
                self.viewer.layers.remove(self.current_image_layer)
                self.current_image_layer = None

            # Load new image
            image_path = self.image_files[self.current_index]
            print(f"Loading image: {image_path}")

            # Read image using skimage
            image_data = io.imread(str(image_path))

            # Add image to viewer
            self.current_image_layer = self.viewer.add_image(
                image_data, name=f"Series Image: {image_path.name}"
            )

            print(f"Loaded image shape: {image_data.shape}")

            # Emit signal that image has changed with enhanced information

            self.image_changed.emit(
                self.current_image_layer,
                str(image_path),
                self.parent_directory,
            )

        except (OSError, ValueError, TypeError, RuntimeError) as e:
            print(f"Error loading image {image_path}: {e}")
            QMessageBox.warning(
                self,
                "Image Load Error",
                f"Could not load image: {image_path.name}\nError: {str(e)}",
            )

    def _reset_display(self):
        """Reset the display when no images are available."""
        # Remove current image layer
        if self.current_image_layer is not None:
            with contextlib.suppress(ValueError, KeyError):
                self.viewer.layers.remove(self.current_image_layer)
            self.current_image_layer = None

        self.image_files = []
        self.current_index = 0
        self.image_scrollbar.setMaximum(0)
        self.image_scrollbar.setValue(0)
        self.image_scrollbar.setEnabled(False)
        self.image_info_label.setText("No directory selected")

    def get_current_image_path(self):
        """Get the path of the currently selected image."""
        if self.image_files and 0 <= self.current_index < len(
            self.image_files
        ):
            return str(self.image_files[self.current_index])
        return None

    def get_image_count(self):
        """Get the total number of images in the series."""
        return len(self.image_files)

    def get_all_image_paths(self):
        """Get all image paths in the series."""
        return [str(f) for f in self.image_files]
