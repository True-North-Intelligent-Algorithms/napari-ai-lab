import contextlib

import napari
import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import ensure_main_thread

from .InteractiveSegmenters import InteractiveSegmenterBase
from .utility import load_images_from_directory, pad_to_largest
from .widgets import ParameterFormWidget, ParameterSlider


class NDEasyLabel(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        # Initialize layer references
        self.image_layer = None
        self.label_layer = None
        self.points_layer = None
        self.shapes_layer = None

        # Initialize label counter
        self.current_label_num = 1

        # Track current image context so we can load labels and other info
        self.current_image_path = None
        self.current_parent_directory = None

        # Signal processing state protection
        self._processing_image_change = False

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        # Add directory selection button
        self.dir_btn = QPushButton("Open Image Directory")
        self.dir_btn.clicked.connect(self._on_open_directory)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(btn)
        self.layout().addWidget(self.dir_btn)

        # Add Interactive Segmenter selection
        self.segmenter_label = QLabel("Interactive Segmenter:")
        self.layout().addWidget(self.segmenter_label)

        self.segmenter_combo = QComboBox()
        self.segmenter_combo.currentTextChanged.connect(
            self._on_segmenter_changed
        )
        self.layout().addWidget(self.segmenter_combo)

        # Parameter form widget for segmenter parameters
        self.parameter_form = ParameterFormWidget()
        self.parameter_form.parameters_changed.connect(
            self._on_parameters_changed
        )
        self.layout().addWidget(self.parameter_form)

        # Populate segmenter combo with registered frameworks
        self._populate_segmenter_combo()

        # Add parameter controls
        self.iou_threshold_slider = ParameterSlider(
            "IoU Threshold", 0.0, 1.0, 0.7
        )
        self.layout().addWidget(self.iou_threshold_slider)

        self.box_extension_slider = ParameterSlider(
            "Box Extension", 0.0, 2.0, 0.5
        )
        self.layout().addWidget(self.box_extension_slider)

        # Add instructions
        instructions_text = """
Instructions:
1. Activate Points layer
2. Draw points on objects
3. SAM creates 3D segmentation
4. Press 'C' to commit current label
5. Press 'X' to erase current label
6. Press 'V' to toggle positive/negative points
        """
        self.instructions_label = QLabel(instructions_text)
        self.instructions_label.setStyleSheet(
            "QLabel { font-size: 14px; color: #357; }"
        )
        self.instructions_label.setWordWrap(True)
        self.layout().addWidget(self.instructions_label)

    def _on_click(self):
        print("Welcome to NDEasyLabel! Let's Go!")

    def _populate_segmenter_combo(self):
        """Populate the segmenter combo box with registered frameworks."""
        # Clear existing items
        self.segmenter_combo.clear()

        # Get registered frameworks
        frameworks = InteractiveSegmenterBase.get_registered_frameworks()

        if frameworks:
            # Add frameworks to combo box
            for name in sorted(frameworks.keys()):
                self.segmenter_combo.addItem(name)

            # Select first item if available
            if self.segmenter_combo.count() > 0:
                self.segmenter_combo.setCurrentIndex(0)
                self._on_segmenter_changed(self.segmenter_combo.currentText())
        else:
            # No frameworks registered
            self.segmenter_combo.addItem("No segmenters available")
            self.segmenter_combo.setEnabled(False)

    def _on_segmenter_changed(self, segmenter_name):
        """Handle changes to the segmenter selection."""
        if not segmenter_name or segmenter_name == "No segmenters available":
            self.parameter_form.clear_form()
            return

        # Get the selected segmenter class
        segmenter_class = InteractiveSegmenterBase.get_framework(
            segmenter_name
        )
        if segmenter_class:
            # Update parameter form with new segmenter class
            self.parameter_form.set_segmenter_class(segmenter_class)
            print(f"Selected segmenter: {segmenter_name}")
            print(f"Supported axes: {segmenter_class().supported_axes}")
        else:
            print(
                f"Warning: Segmenter '{segmenter_name}' not found in registry"
            )
            self.parameter_form.clear_form()

    def _on_parameters_changed(self, parameters):
        """Handle changes to segmenter parameters."""
        segmenter_name = self.segmenter_combo.currentText()
        print(f"Parameters changed for {segmenter_name}: {parameters}")
        # Here you could store parameters or trigger updates as needed

    def get_current_segmenter(self):
        """Get the currently selected segmenter with current parameter values."""
        segmenter_name = self.segmenter_combo.currentText()
        if not segmenter_name or segmenter_name == "No segmenters available":
            return None

        try:
            # Create segmenter instance with current parameters
            return self.parameter_form.create_segmenter_instance()
        except (AttributeError, ValueError, TypeError, RuntimeError) as e:
            print(f"Error creating segmenter instance: {e}")
            return None

    def _on_points_changed(self, event):
        """Handle points layer data changes - creates 2D ROI around point."""
        points_layer = event.source
        if event.action == "added" and len(points_layer.data) > 0:
            # Get the most recently added point (last in the list)
            latest_point = points_layer.data[-1]
            print(f"Point added at location: {latest_point}")

            # Create 2D ROI around point and fill with label 354
            # self._create_roi_at_point(latest_point)

            # Print all points for reference
            print(f"Total points: {len(points_layer.data)}")
            for i, point in enumerate(points_layer.data):
                print(f"  Point {i+1}: {point}")

            # Use current label number and increment for next use
            self.set_square(
                self.label_layer.data,
                latest_point,
                size=20,
                value=self.current_label_num,
            )

            print(f"Added square with label {self.current_label_num}")
            self.current_label_num += 1

            self.label_layer.refresh()

    def set_square(self, arr, point, size=20, value=80):
        """
        Set a square region of given size around (y, x) at the slice defined by
        the leading coordinates of `point`.

        Parameters
        ----------
        arr : np.ndarray
            N-dimensional array.
        point : tuple or list
            Indices into arr, length = arr.ndim. Last two are (y, x).
        size : int
            Side length of the square (default 20).
        value : scalar
            Value to assign inside the square.
        """
        *leading, y, x = point

        # Convert coordinates to integers (napari points can be floats)
        leading = [int(coord) for coord in leading]
        y, x = int(y), int(x)

        half = size // 2

        # bounds (ensure integers)
        y0 = int(max(0, y - half))
        y1 = int(min(arr.shape[-2], y + half))
        x0 = int(max(0, x - half))
        x1 = int(min(arr.shape[-1], x + half))

        # build slice object for all dims
        index = tuple(leading) + (slice(y0, y1), slice(x0, x1))

        arr[index] = value

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
            images, axis_infos, image_paths = load_images_from_directory(
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
                images, axis_infos, force8bit=True, normalize_per_channel=False
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

    def _cleanup_existing_layers(self):
        """Remove existing annotation layers from viewer safely."""
        print("Starting layer cleanup...")

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
            ("label", self.label_layer),
            ("points", self.points_layer),
            ("shapes", self.shapes_layer),
        ]

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
        self.label_layer = None
        self.points_layer = None
        self.shapes_layer = None

        print("Completed layer cleanup")

    def _set_image_layer(self, image_layer):
        """Set up all annotation layers based on the provided image layer."""
        try:
            # Store the image layer reference
            self.image_layer = image_layer

            # Get image data from the layer
            image_data = image_layer.data

            # Add labels layer and store reference
            self.label_layer = self.viewer.add_labels(
                np.zeros_like(image_data, dtype=np.uint16), name="Labels"
            )

            # Add points layer for annotation with point type choices
            self._point_choices = ["positive", "negative"]
            LABEL_COLOR_CYCLE = ["red", "blue"]  # positive=red, negative=blue

            # For annotation layers, we want ndim to match the displayed dimensions
            # This prevents issues with 4D data slice comparisons
            annotation_ndim = min(
                len(image_data.shape), 3
            )  # Cap at 3D for annotation layers

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
                ndim=len(image_data.shape),
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
            )

            # Connect shapes event handler
            self.shapes_layer.events.data.connect(self._on_shapes_changed)

            print(
                f"Successfully set up annotation layers for image layer: {image_layer.name}"
            )
            print("Added points layer for annotation. Click to add points!")
            print(
                "Added shapes layer for region annotation. Draw shapes to define regions!"
            )

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

    def connect_sequence_viewer(self, sequence_viewer):
        """Connect to sequence viewer for automatic layer updates."""
        sequence_viewer.image_changed.connect(self._on_sequence_image_changed)
        print("Connected to sequence viewer for automatic layer updates")

    @ensure_main_thread
    def _on_sequence_image_changed(
        self, image_layer, image_path, parent_directory
    ):
        """Handle sequence viewer image changes with simple processing lock to prevent crashes."""
        # If we're already processing a signal, ignore this one to prevent conflicts
        if self._processing_image_change:
            print(
                "Signal received while processing - ignoring to prevent conflicts"
            )
            return

        self._processing_image_change = True

        # Process the image change immediately
        self._process_image_change(image_layer, image_path, parent_directory)

    def _process_image_change(self, image_layer, image_path, parent_directory):
        """Process the image change with simple processing lock."""
        # Set processing flag to prevent re-entrant calls

        try:
            print(f"Processing image change: {image_path}")

            print("Switching images")

            if image_layer and image_path and parent_directory:
                print(f"Setting up new image: {image_layer.name}")
                print(f"Image path: {image_path}")
                print(f"Parent directory: {parent_directory}")

                # Update current context
                self.current_image_path = image_path
                self.current_parent_directory = parent_directory

                # Clean up existing layers
                self._cleanup_existing_layers()

                # Not sure if this helps, but try a small delay to let
                # Napari properly release layer resources
                # This prevents race conditions between layer removal and creation
                import time

                time.sleep(0.05)  # 50ms delay

                print("Creating new layers...")
                self._set_image_layer(image_layer)
            else:
                print("Received invalid image data from sequence viewer")
                self._cleanup_existing_layers()
                # Clear current context
                self.current_image_path = None
                self.current_parent_directory = None

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
