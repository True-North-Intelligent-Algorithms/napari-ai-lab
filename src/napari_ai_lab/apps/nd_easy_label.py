import napari
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QMessageBox,
    QPushButton,
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
        image_data_model: "ImageDataModel",
    ):
        super().__init__(viewer, image_data_model)
        self.frameworks = InteractiveSegmenterBase.get_registered_frameworks()
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

        # Parameter form widget for segmenter parameters (from base)
        self.layout().addWidget(self._create_parameter_form())

        # Connect to axis changes to reinitialize segmenter
        self.parameter_form.axis_changed.connect(self._on_axis_changed)

        # Populate segmenter combo with registered frameworks
        self._populate_segmenter_combo()

    def _on_click(self):
        print("Welcome to NDEasyLabel! Let's Go!")

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

    def _initialize_segmenter(self):
        """Initialize predictor if an image is loaded and a segmenter exists."""
        if self.image_layer is None:
            # Nothing to initialize yet
            return

        try:
            parent_dir = self.image_data_model.get_parent_directory()

            image_paths = self.image_data_model.get_image_paths()
            image_name = image_paths[self.current_image_index].stem

            selected_axis = self.parameter_form.get_selected_axis()

            indices = get_current_slice_indices(
                self.viewer.dims.current_step, selected_axis
            )

            image_data = self.image_layer.data[indices]

            # Get current step tuple
            step = self.viewer.dims.current_step

            # Create artifact name from non-spatial dims
            image_name = create_artifact_name(image_name, step, selected_axis)

            self.segmenter = self.parameter_form.sync_nd_operation_instance(
                getattr(self, "segmenter", None)
            )

            self.segmenter.initialize_predictor(
                image_data, str(parent_dir), image_name
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

            selected_axis = self.parameter_form.get_selected_axis()

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
            self.segmenter = self.parameter_form.sync_nd_operation_instance(
                self.segmenter
            )

            # Call segmenter with the latest point
            try:
                mask = self.segmenter.segment(
                    image_data,
                    points=[latest_point],
                    shapes=None,
                )

                # self.predictions_layer.data[indices] = (
                #    mask  # self.current_label_num
                # )

                # Apply the mask to the labels layer
                self.annotation_layer.data[segmentation_indices][mask != 0] = (
                    mask[mask != 0] * self.current_label_num
                )

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
                image_data.shape, self.current_image_index
            )

            # Add labels layer and store reference
            self.annotation_layer = self.viewer.add_labels(
                labels_data, name="Labels (Persistent)"
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
