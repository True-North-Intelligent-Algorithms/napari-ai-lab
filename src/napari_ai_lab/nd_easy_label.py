import napari
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from .base_nd_easy_widget import BaseNDEasyWidget
from .models import ImageDataModel
from .Segmenters.InteractiveSegmenters import InteractiveSegmenterBase
from .widgets import SegmenterWidget


class NDEasyLabel(BaseNDEasyWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        image_data_model: "ImageDataModel",
    ):
        super().__init__(viewer, image_data_model)
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""

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
        self.parameter_form = SegmenterWidget()
        self.parameter_form.parameters_changed.connect(
            self._on_parameters_changed
        )
        self.layout().addWidget(self.parameter_form)

        # Populate segmenter combo with registered frameworks
        self._populate_segmenter_combo()

    def _on_click(self):
        print("Welcome to NDEasyLabel! Let's Go!")

    def _populate_segmenter_combo(self):
        """Populate the segmenter combo box with registered frameworks."""
        # Clear existing items
        self.segmenter_combo.clear()

        # Get registered frameworks
        frameworks = InteractiveSegmenterBase.get_registered_frameworks()

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

        self.segmenter = InteractiveSegmenterBase.get_framework(
            segmenter_name
        )()

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

            image_data = self.image_layer.data

            # Ensure segmenter is synced with current parameters before use
            if hasattr(self, "segmenter") and self.segmenter is not None:
                self.segmenter = self.parameter_form.sync_segmenter_instance(
                    self.segmenter
                )

            # Call segmenter with the latest point
            try:

                mask = self.segmenter.segment(
                    image_data,
                    points=[latest_point],
                    shapes=None,
                    parent_directory=self.image_data_model.parent_directory,
                )

                # Apply the mask to the labels layer
                self.label_layer.data[mask] = self.current_label_num

                print(
                    f"Added segmentation with label {self.current_label_num}"
                )
                self.current_label_num += 1

                self.label_layer.refresh()

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

    # _on_open_directory and load_image_directory inherited from BaseNDEasyWidget
    def _set_image_layer(self, image_layer):
        """Set up all annotation layers based on the provided image layer."""
        try:
            # Store the image layer reference
            self.image_layer = image_layer

            # Get image data from the layer
            image_data = image_layer.data

            # Load existing labels or create empty ones
            labels_data = self._load_existing_labels(image_data.shape)

            # Add labels layer and store reference
            self.label_layer = self.viewer.add_labels(
                labels_data, name="Labels (Persistent)"
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
