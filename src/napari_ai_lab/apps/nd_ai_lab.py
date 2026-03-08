"""
ND AI Lab - Combined plugin with tabbed interface.

This module combines NDEasyLabel, NDEasyAugment, and NDEasySegment
into a single tabbed interface with shared model and viewer.
"""

from pathlib import Path

import napari
from qtpy.QtWidgets import (
    QFileDialog,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread

from ..models import ImageDataModel
from .nd_easy_augment import NDEasyAugment
from .nd_easy_label import NDEasyLabel
from .nd_easy_segment import NDEasySegment


class NDAILab(QWidget):
    """
    Combined AI Lab widget with tabbed interface.

    Provides Label, Augment, and Segment functionality in separate tabs,
    all sharing the same napari viewer and ImageDataModel.
    """

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        image_data_model: ImageDataModel = None,
    ):
        """
        Initialize the combined AI Lab widget.

        Args:
            viewer: The napari viewer instance.
            image_data_model: Optional ImageDataModel. If not provided, can be set later.
        """
        super().__init__()
        self.viewer = viewer
        self.image_data_model = image_data_model

        # Tracking for sequence viewer changes
        self._processing_image_change = False
        self.current_image_index = 0

        # Create sub-apps in EMBEDDED mode (no individual directory buttons)
        # Model can be set later via set_image_data_model()
        self.label_widget = NDEasyLabel(
            viewer, image_data_model, embedded=True
        )
        self.augment_widget = NDEasyAugment(
            viewer, image_data_model, embedded=True
        )
        self.segment_widget = NDEasySegment(
            viewer,
            image_data_model,
            embedded=True,
            training_widget_mode="embedded",  # Use embedded training form, not dialog
        )

        # If model provided, set it now
        if image_data_model is not None:
            self.set_image_data_model(image_data_model)

        # Setup UI
        self._setup_ui()

    def set_image_data_model(self, image_data_model: ImageDataModel):
        """
        Set the shared image data model and propagate to all sub-apps.

        Args:
            image_data_model: The ImageDataModel instance to share.
        """
        self.image_data_model = image_data_model

        # Propagate to all sub-apps
        self.label_widget.set_image_data_model(image_data_model)
        self.augment_widget.set_image_data_model(image_data_model)
        self.segment_widget.set_image_data_model(image_data_model)

        print("✅ Shared image data model set for all tabs")

    def _set_image_layer(self, image_layer):
        """
        Central layer management - creates ALL layers needed by all sub-apps.

        This prevents duplicate layer creation and ensures consistency.
        Layers are created once here and distributed to sub-apps via direct assignment.

        Args:
            image_layer: The napari image layer to set up annotations for.
        """
        # Store image layer reference
        self.image_layer = image_layer

        # Get current image index (default to 0 for now)
        current_image_index = 0

        # Load existing data or create empty
        labels_data = self.image_data_model.load_existing_annotations(
            image_layer.data.shape, current_image_index
        )
        predictions_data = self.image_data_model.load_existing_predictions(
            image_layer.data.shape, current_image_index
        )

        # Create shared layers ONCE
        self.labels_layer = self.viewer.add_labels(
            labels_data, name="Labels (Persistent)"
        )
        self.predictions_layer = self.viewer.add_labels(
            predictions_data, name="Predictions (Persistent)"
        )

        # Create points layer for interactive segmentation
        self._point_choices = ["positive", "negative"]
        LABEL_COLOR_CYCLE = ["red", "blue"]
        annotation_ndim = len(self.labels_layer.data.shape)

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

        # Create shapes layer for segment widget (if in interactive mode)
        self.shapes_layer = self.viewer.add_shapes(
            name="Shapes Layer",
            edge_color="green",
            face_color="transparent",
            edge_width=2,
            ndim=annotation_ndim,
        )

        # Distribute layers to sub-apps (direct assignment, not calling their _set_image_layer)
        self._distribute_layers_to_sub_apps()

        print(f"✅ Central layer setup complete for image: {image_layer.name}")

    def _distribute_layers_to_sub_apps(self):
        """
        Distribute the centrally-created layers to each sub-app.

        Uses direct attribute assignment instead of calling sub-apps' _set_image_layer()
        to avoid duplicate layer creation.
        """
        # Label widget needs: image, labels, points
        self.label_widget.image_layer = self.image_layer
        self.label_widget.annotation_layer = self.labels_layer
        self.label_widget.points_layer = self.points_layer

        # Connect points layer events for label widget
        if self.points_layer and hasattr(
            self.label_widget, "_on_points_changed"
        ):
            self.points_layer.events.data.connect(
                self.label_widget._on_points_changed
            )

        # Augment widget needs: image, labels
        self.augment_widget.image_layer = self.image_layer
        self.augment_widget.annotation_layer = self.labels_layer

        # Segment widget needs: image, labels, predictions, points, shapes
        self.segment_widget.image_layer = self.image_layer
        self.segment_widget.annotation_layer = self.labels_layer
        self.segment_widget.predictions_layer = self.predictions_layer
        self.segment_widget.points_layer = self.points_layer
        self.segment_widget.shapes_layer = self.shapes_layer

        # Connect interactive layers events for segment widget (if in interactive mode)
        if self.segment_widget.is_interactive_mode():
            if hasattr(self.segment_widget, "_on_points_changed"):
                self.points_layer.events.data.connect(
                    self.segment_widget._on_points_changed
                )
            if hasattr(self.segment_widget, "_on_shapes_changed"):
                self.shapes_layer.events.data.connect(
                    self.segment_widget._on_shapes_changed
                )

        print("   → Layers distributed to all sub-apps")

    def _setup_ui(self):
        """Setup the tabbed user interface."""
        layout = QVBoxLayout(self)

        # Top-level controls (shared across all tabs)
        self.dir_btn = QPushButton("Open Image Directory")
        self.dir_btn.clicked.connect(self._on_open_directory)
        layout.addWidget(self.dir_btn)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.addTab(self.label_widget, "Label")
        self.tabs.addTab(self.augment_widget, "Augment")
        self.tabs.addTab(self.segment_widget, "Segment")

        # Add training tab - shows training view of segment widget
        # Same widget, different controls (segmenter combo + training params only)
        self.tabs.addTab(self.segment_widget.get_training_widget(), "Train")

        layout.addWidget(self.tabs)

        # TODO Phase 3: Add central layer management
        # TODO Phase 4: Add sequence viewer connection

    def _on_open_directory(self):
        """Open directory and create/set model for all sub-apps."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Image Directory",
            "...",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )

        if directory:
            print(f"📁 Loading directory: {directory}")

            # Create model from directory
            parent_dir = Path(directory)
            self.image_data_model = ImageDataModel(parent_dir)

            # Propagate to all tabs
            self.set_image_data_model(self.image_data_model)

            print("✅ Model created and shared across all tabs")
            # TODO Phase 3: Load images and create layers

    def connect_sequence_viewer(self, sequence_viewer):
        """Connect to sequence viewer for automatic layer updates."""
        sequence_viewer.image_changed.connect(self._on_sequence_image_changed)
        print("Connected to sequence viewer for automatic layer updates")

    @ensure_main_thread
    def _on_sequence_image_changed(self, image_layer, image_index):
        """Handle sequence viewer image changes with simple processing lock to prevent crashes."""
        # If we're already processing a signal, ignore this one to prevent conflicts
        if self._processing_image_change:
            print(
                "Signal received while processing - ignoring to prevent conflicts"
            )
            return

        self._processing_image_change = True

        # Process the image change immediately
        self._process_image_change(image_layer, image_index)

    def _process_image_change(self, image_layer, image_index):
        """
        Process image change from sequence viewer.

        Handles cleanup and recreation of layers centrally for all sub-apps.
        """
        try:
            print(
                f"🔄 nd_ai_lab: Processing image change to index {image_index}"
            )

            # Save current annotations before switching (delegate to sub-apps)
            # Only save from the currently active tab to avoid duplicate saves
            active_widget_name = self.tabs.tabText(self.tabs.currentIndex())

            if (
                active_widget_name == "Label"
                and hasattr(self, "labels_layer")
                and self.labels_layer
                and self.image_data_model.parent_directory
            ):
                try:
                    self.image_data_model.save_annotations(
                        self.labels_layer.data,
                        self.current_image_index,
                        current_step=self.viewer.dims.current_step,
                    )
                    print("   Saved annotations from Label tab")
                except (OSError, ValueError, RuntimeError) as e:
                    print(f"   Failed to save annotations: {e}")

            # Update current image index
            self.current_image_index = image_index

            if image_layer:
                print(f"   Setting up new image: {image_layer.name}")

                # Cleanup existing layers centrally
                self._cleanup_layers()

                # Create new layers centrally (this also distributes to sub-apps)
                self._set_image_layer(image_layer)

                print("✅ nd_ai_lab: Image change complete")
            else:
                print("⚠️  Received invalid image data from sequence viewer")
                self._cleanup_layers()

        except (
            OSError,
            ValueError,
            RuntimeError,
            AttributeError,
            KeyError,
        ) as e:
            print(f"❌ Error processing image change: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Always clear the processing flag
            self._processing_image_change = False

    def _cleanup_layers(self):
        """
        Cleanup existing layers before switching images.

        Central cleanup for combined app - removes all layers from viewer.
        """
        print("   🧹 Cleaning up existing layers...")

        layers_to_remove = []

        # Collect layers to remove
        if hasattr(self, "labels_layer") and self.labels_layer:
            layers_to_remove.append(("Labels", self.labels_layer))
        if hasattr(self, "predictions_layer") and self.predictions_layer:
            layers_to_remove.append(("Predictions", self.predictions_layer))
        if hasattr(self, "points_layer") and self.points_layer:
            layers_to_remove.append(("Points", self.points_layer))
        if hasattr(self, "shapes_layer") and self.shapes_layer:
            layers_to_remove.append(("Shapes", self.shapes_layer))

        # Remove layers from viewer
        for layer_name, layer in layers_to_remove:
            try:
                if layer in self.viewer.layers:
                    self.viewer.layers.remove(layer)
                    print(f"      Removed {layer_name} layer")
            except (ValueError, KeyError, RuntimeError) as e:
                print(f"      Error removing {layer_name}: {e}")

        # Clear references
        if hasattr(self, "labels_layer"):
            self.labels_layer = None
        if hasattr(self, "predictions_layer"):
            self.predictions_layer = None
        if hasattr(self, "points_layer"):
            self.points_layer = None
        if hasattr(self, "shapes_layer"):
            self.shapes_layer = None

        print("   ✅ Layer cleanup complete")
