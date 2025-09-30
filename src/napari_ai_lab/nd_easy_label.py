import napari
from qtpy.QtWidgets import (
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from tnia.widgets import ParameterSlider

from .utility import load_images_from_directory, pad_to_largest


class NDEasyLabel(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        # Add directory selection button
        self.dir_btn = QPushButton("Open Image Directory")
        self.dir_btn.clicked.connect(self._on_open_directory)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(btn)
        self.layout().addWidget(self.dir_btn)

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

    def _on_points_changed(self, event):
        """Handle points layer data changes - prints point locations."""
        points_layer = event.source
        if event.action == "added" and len(points_layer.data) > 0:
            # Get the most recently added point (last in the list)
            latest_point = points_layer.data[-1]
            print(f"Point added at location: {latest_point}")

            # Print all points for reference
            print(f"Total points: {len(points_layer.data)}")
            for i, point in enumerate(points_layer.data):
                print(f"  Point {i+1}: {point}")

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

            # Add the processed image stack to napari viewer
            self.viewer.add_image(
                padded_images, name=f"Image Stack ({len(images)} images)"
            )

            # Add points layer for annotation with point type choices
            self._point_choices = ["positive", "negative"]
            LABEL_COLOR_CYCLE = ["red", "blue"]  # positive=red, negative=blue

            # For annotation layers, we want ndim to match the displayed dimensions
            # This prevents issues with 4D data slice comparisons
            annotation_ndim = min(
                len(padded_images.shape), 3
            )  # Cap at 3D for annotation layers

            points_layer = self.viewer.add_points(
                name="Point Layer",
                property_choices={"label": self._point_choices},
                border_color="label",
                border_color_cycle=LABEL_COLOR_CYCLE,
                symbol="o",
                face_color="transparent",
                border_width=0.5,
                size=1,
                ndim=len(padded_images.shape),
            )

            # Connect point event handler
            points_layer.events.data.connect(self._on_points_changed)

            # Add shapes layer for region annotation
            shapes_layer = self.viewer.add_shapes(
                name="Shapes Layer",
                edge_color="green",
                face_color="transparent",
                edge_width=2,
                ndim=annotation_ndim,
            )

            # Connect shapes event handler
            shapes_layer.events.data.connect(self._on_shapes_changed)

            print(
                f"Successfully loaded and processed {len(images)} images into napari as a stack."
            )
            print("Added points layer for annotation. Click to add points!")
            print(
                "Added shapes layer for region annotation. Draw shapes to define regions!"
            )

        except (OSError, ValueError, ImportError, RuntimeError) as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while loading images: {str(e)}",
            )
            print(f"Error loading images: {e}")
