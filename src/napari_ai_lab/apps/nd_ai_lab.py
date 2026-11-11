"""
ND AI Lab - Combined plugin with tabbed interface.

This module combines NDEasyLabel, NDEasyAugment, and NDEasySegment
into a single tabbed interface with shared model and viewer.
"""

from pathlib import Path

import napari
import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread

from ..models import ImageDataModel
from ..utilities.image_util import compute_collapsed_shape
from .nd_easy_augment import NDEasyAugment
from .nd_easy_label import NDEasyLabel
from .nd_easy_segment import NDEasySegment
from .profiles import BOX_ANNOTATIONS, get_profile


def _box_bounds(box):
    """(y0, x0, y1, x1) around a napari shape's last two axes."""
    yx = np.asarray(box)[:, -2:]
    (y0, x0), (y1, x1) = yx.min(axis=0), yx.max(axis=0)
    return y0, x0, y1, x1


def _iou(a, b):
    """Intersection over union of two (y0, x0, y1, x1) boxes."""
    h = min(a[2], b[2]) - max(a[0], b[0])
    w = min(a[3], b[3]) - max(a[1], b[1])
    if h <= 0 or w <= 0:
        return 0.0
    inter = h * w
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return inter / (union - inter)


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
        axes_to_collapse: str | list[str] | None = None,
        profile=None,
    ):
        """
        Initialize the combined AI Lab widget.

        Args:
            viewer: The napari viewer instance.
            image_data_model: Optional ImageDataModel. If not provided, can be set later.
            axes_to_collapse: Axis names to collapse when creating annotations/predictions
                            (e.g., "C" for channels). Passed to model load/save methods.
        """
        super().__init__()
        self.viewer = viewer
        self.image_data_model = image_data_model
        self.axes_to_collapse = axes_to_collapse

        # Tracking for sequence viewer changes
        self._processing_image_change = False
        self.current_image_index = 0

        # What this session offers. See apps/profiles.py.
        self.profile = get_profile(profile)

        # Create sub-apps in EMBEDDED mode (no individual directory buttons)
        # Model can be set later via set_image_data_model()
        self.label_widget = NDEasyLabel(
            viewer,
            image_data_model,
            embedded=True,
            axes_to_collapse=axes_to_collapse,
        )
        self.augment_widget = NDEasyAugment(
            viewer,
            image_data_model,
            embedded=True,
            axes_to_collapse=axes_to_collapse,
        )
        self.segment_widget = NDEasySegment(
            viewer,
            image_data_model,
            embedded=True,
            training_widget_mode="embedded",  # Use embedded training form, not dialog
            axes_to_collapse=axes_to_collapse,
        )

        # The profile's GUI groups depend on nothing but the profile, so
        # settle them now. apply_mode() runs only from the launcher and from
        # "Open Project" -- built from the napari menu, this widget would
        # otherwise show every optional button until a project was opened.
        if hasattr(self.label_widget, "apply_profile_features"):
            self.label_widget.apply_profile_features(self.profile)

        # Back-reference so child widgets can call into NDAILab
        # (used by "Open Project" button, "Copy predictions to labels", ...).
        # Set here (not in _distribute_layers_to_sub_apps) so it is available
        # even before the first image is loaded.
        self.label_widget.ai_lab = self
        self.augment_widget.ai_lab = self
        self.segment_widget.ai_lab = self

        # If model provided, set it now
        if image_data_model is not None:
            self.set_image_data_model(image_data_model)

        # Setup UI
        self._setup_ui()

        # The sub-apps' close handler saves only when there is a labels
        # layer, which a detect profile does not have; box annotations are
        # saved here, ahead of it, without asking.
        try:
            qt_window = viewer.window._qt_window
            original_close_event = qt_window.closeEvent

            def close_event(event):
                self._save_box_annotations()
                original_close_event(event)

            qt_window.closeEvent = close_event
        except (AttributeError, RuntimeError) as e:
            print(f"Could not connect to viewer closing: {e}")

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

        # Per-axis scale for annotation layers (collapsed axes dropped)
        annotation_scale = (
            self.image_data_model.get_scale(
                axes_to_collapse=self.axes_to_collapse
            )
            or None
        )

        # Multiple named annotation collections are supported: each
        # subdirectory under ``annotations/`` becomes a separate labels
        # layer.  Falls back to a single ``"Labels (Persistent)"`` layer
        # when no subdirectories exist yet.
        # A box-annotation profile has none of them: no annotations, no
        # working layer. See Profile.show_labels.
        annotation_names = (
            self.image_data_model.list_annotation_subdirectories()
            or ["Labels (Persistent)"]
        )
        if not self.profile.show_labels:
            annotation_names = []
        self.annotations_layers = {}
        for ann_name in annotation_names:
            data = self.image_data_model.load_existing_annotations(
                self.current_image_index,
                image_layer.data.shape,
                subdirectory=ann_name,
                axes_to_collapse=self.axes_to_collapse,
            )
            self.annotations_layers[ann_name] = self.viewer.add_labels(
                data,
                name=ann_name,
                scale=annotation_scale,
            )
        # Pick the first as the active layer (backwards-compatible ref).
        self.annotations_layer = next(
            iter(self.annotations_layers.values()), None
        )

        # Working / scratch labels layer for interactive segmenter output.
        # Always uses WORKING_LABEL_INDEX (7) so its colour is consistent.
        import numpy as _np

        self.working_layer = None
        if self.profile.show_labels:
            working_data = _np.zeros_like(self.annotations_layer.data)
            self.working_layer = self.viewer.add_labels(
                working_data,
                name="Labels (Working)",
                scale=annotation_scale,
            )

        # Dictionary to hold prediction layers for different segmenters
        # Key: segmenter name (e.g., "CellposeSegmenter", "StarDist")
        # Value: napari labels layer
        # Populated on demand when segmentation runs or existing predictions are loaded
        # The index has to be pushed *before* this call, not with the rest of
        # the distribution below: _load_existing_prediction_layers reads
        # current_image_index off the segment widget, so loading here with the
        # previous image's index finds nothing and creates no layer -- which
        # looks like a saved segmentation having vanished.
        self.segment_widget.current_image_index = self.current_image_index
        if self.profile.show_labels:
            self.segment_widget._load_existing_prediction_layers(
                self.image_layer.data.shape
            )
        self.predictions_layers = self.segment_widget.predictions_layers

        # Create points layer for interactive segmentation
        self._point_choices = ["positive", "negative"]
        LABEL_COLOR_CYCLE = ["red", "blue"]
        annotation_ndim = len(self._annotation_shape(image_layer.data.shape))

        self.points_layer = self.viewer.add_points(
            name="Point prompts",
            property_choices={"label": self._point_choices},
            border_color="label",
            border_color_cycle=LABEL_COLOR_CYCLE,
            symbol="o",
            face_color="transparent",
            border_width=0.5,
            size=1,
            ndim=annotation_ndim,
            scale=annotation_scale,
        )

        # Create shapes layer for segment widget (if in interactive mode).
        # Without labels it holds the annotations themselves, one class per
        # box, drawn with its name from the project's classes.yaml.
        # None when the profile drops shape prompts (show_shapes=False).
        self.shapes_layer = None
        if self.profile.show_labels and self._wants_shapes():
            self.shapes_layer = self.viewer.add_shapes(
                name="Box prompts",
                edge_color="green",
                face_color="transparent",
                edge_width=2,
                ndim=annotation_ndim,
                scale=annotation_scale,
            )
        elif not self.profile.show_labels:
            # One layer per folder under box_annotations/ -- runners, bibs
            # -- each on every image, empty where nothing is saved yet.
            from ..utilities.box_annotations import box_folders

            self.box_layers = {}
            self._box_layer_ndim = annotation_ndim
            self._box_layer_scale = annotation_scale
            project = self.image_data_model.parent_directory
            for folder in box_folders(project):
                self._add_box_layer(folder)
            self.shapes_layer = next(iter(self.box_layers.values()), None)
            self._load_box_annotations(image_layer.data.shape)

        # Create boxes layer for label widget (bounding-box annotations)
        self.boxes_layer = self.viewer.add_shapes(
            ndim=annotation_ndim,
            name="Label box",
            face_color="transparent",
            edge_color="blue",
            edge_width=5,
            text={"string": "{split_set}", "size": 15, "color": "green"},
            scale=annotation_scale,
        )

        from napari_ai_lab.vendored.napari_bbox import BoundingBoxLayer

        self.boxes_3D_layer = BoundingBoxLayer(
            name="3D Bounding Boxes",
            edge_color="magenta",
            face_color="transparent",
            edge_width=5,
            ndim=annotation_ndim,
            scale=annotation_scale,
            text={"string": "{split_set}", "size": 15, "color": "green"},
        )

        # self.boxes_3D_layer.feature_defaults["split_set"] = ""

        # Only for ND data. apply_mode decides; this is the other place the
        # layer would otherwise come back, once per image change.
        model = self.image_data_model
        if model is None or model.mode != "2d":
            self.viewer.add_layer(self.boxes_3D_layer)

        # Distribute layers to sub-apps (direct assignment, not calling their _set_image_layer)
        self._distribute_layers_to_sub_apps()
        self._load_box_predictions(image_layer.data.shape)

        print(f"✅ Central layer setup complete for image: {image_layer.name}")

        # sync augmenter parameters with current image shape
        self.augment_widget.sync_augmenter_parameters()

        # guard against missing augmenter
        if (
            not hasattr(self.augment_widget, "augmenter")
            or self.augment_widget.augmenter is None
        ):
            return

        # Todo: this logic could fail if the new image is part of a sequence and we want to remember the last selected axis.
        # For now, just reset to the first supported axis.
        self.augment_widget.augmenter.selected_axis = (
            self.augment_widget.augmenter.supported_axes[0]
        )

        self.augment_widget.augmentation_form.set_axis_combo(
            self.augment_widget.augmenter.selected_axis
        )

    def _update_layers_for_image(self, image_layer) -> bool:
        """Point the existing layers at a new image, instead of rebuilding.

        Returns False when the layer set cannot be reused, in which case the
        caller falls back to the destroy-and-recreate path. Reuse needs every
        layer still present in the viewer, the same annotation subdirectories,
        and the same number of dimensions -- a different *size* is fine, since
        assigning ``.data`` handles that.

        Prediction layers are updated but never removed, and a segmenter with
        nothing saved for this image gets an all-zero layer rather than
        disappearing. That keeps the layer list stable while scrolling -- no
        selection, ordering or colormap resets -- and matches what the stacked
        viewer shows, where an unsegmented slice is simply empty. Nothing is
        written to disk for an empty layer.
        """
        import numpy as _np

        # Nothing to reuse on the first image of a session, when these
        # attributes do not exist yet.
        # Without labels there are none to find, so the points layer stands
        # in as the sign that a first image has been set up.
        show_labels = self.profile.show_labels
        annotations_layers = getattr(self, "annotations_layers", None)
        if show_labels and not annotations_layers:
            return False
        if getattr(self, "points_layer", None) is None:
            return False

        layers = [
            getattr(self, "points_layer", None),
            getattr(self, "boxes_layer", None),
            *annotations_layers.values(),
        ]
        # Box annotation layers are not required: one the user deleted stays
        # deleted, and _load_box_annotations forgets it.
        if self._wants_shapes() and show_labels:
            layers.append(getattr(self, "shapes_layer", None))
        if show_labels:
            layers.append(getattr(self, "working_layer", None))

        # The 3D box layer is only in the viewer for an ND project -- for a 2D
        # one apply_mode removes it, using this same condition. Requiring it
        # unconditionally made the check below fail on every 2D project, so
        # reuse never ran and every switch took the rebuild path.
        if self._wants_3d():
            layers.append(getattr(self, "boxes_3D_layer", None))
        if any(layer is None for layer in layers):
            return False
        if any(layer not in self.viewer.layers for layer in layers):
            return False

        # A new annotation subdirectory needs a new layer; rebuilding is the
        # simple answer for a case that happens once, not once per switch.
        annotation_names = (
            self.image_data_model.list_annotation_subdirectories()
            or ["Labels (Persistent)"]
        )
        if show_labels and set(annotation_names) != set(
            self.annotations_layers
        ):
            return False

        # The auxiliary layers -- points, shapes, boxes -- were constructed
        # with a fixed ndim, so a change in dimensionality needs new ones. A
        # change in *size* does not: assigning .data handles that.
        image_shape = image_layer.data.shape
        first = next(iter(annotations_layers.values()), None)
        annotation_ndim = self.points_layer.ndim
        if len(image_shape) != annotation_ndim + len(
            self.axes_to_collapse or ""
        ):
            return False

        self.predictions_layers = (
            getattr(self, "predictions_layers", None) or {}
        )
        self.image_layer = image_layer

        # The sequence viewer adds its new image layer on top of the list.
        # Rebuilding used to put it underneath implicitly, by creating every
        # label layer after it; reuse does not, so the image would cover the
        # labels. Move it rather than remove and re-add -- removal is the cost
        # this whole path exists to avoid.
        if image_layer in self.viewer.layers:
            index = self.viewer.layers.index(image_layer)
            if index != 0:
                self.viewer.layers.move(index, 0)

        # NB: every assignment below is followed by refresh(). Since napari
        # 0.8 the data setter (ScalarFieldBase.data) updates the array, the
        # thumbnail and the dims, but does not reslice -- so with an unchanged
        # shape the canvas keeps showing the previous image's labels while the
        # thumbnail and pixel readout are already correct. Rebuilding layers
        # hid this, because a new layer slices on creation.
        for ann_name, layer in self.annotations_layers.items():
            layer.data = self.image_data_model.load_existing_annotations(
                self.current_image_index,
                image_shape,
                subdirectory=ann_name,
                axes_to_collapse=self.axes_to_collapse,
            )
            layer.refresh()
        self.annotations_layer = first

        if self.working_layer is not None:
            self.working_layer.data = _np.zeros_like(
                self.annotations_layer.data
            )
            self.working_layer.refresh()

        # Per-image annotations that are not persisted here start empty, as
        # they did when these layers were recreated.
        self.points_layer.data = _np.empty((0, annotation_ndim))
        self.boxes_layer.data = []
        self.boxes_3D_layer.data = []
        for layer in (
            self.points_layer,
            self.boxes_layer,
            self.boxes_3D_layer,
        ):
            layer.refresh()
        if show_labels and self.shapes_layer is not None:
            self.shapes_layer.data = []
            self.shapes_layer.refresh()
        if not show_labels:
            self._load_box_annotations(image_shape)

        if show_labels:
            self._update_prediction_layers(image_shape)

        self._distribute_layers_to_sub_apps()
        # After distributing: the segment widget makes these layers, and
        # needs the new image layer to size them.
        self._load_box_predictions(image_shape)
        self.augment_widget.sync_augmenter_parameters()
        return True

    @property
    def hidden_box_sources(self) -> set:
        """Box prediction layers the user deleted, by layer name.

        Not brought back by an image change; the segment widget takes one
        out again when its segmenter runs.
        """
        hidden = getattr(self, "_hidden_box_sources", None)
        if hidden is None:
            hidden = self._hidden_box_sources = set()
        return hidden

    def _load_box_predictions(self, image_shape):
        """Show each source's saved boxes for this image (box_predictions/).

        Every source with a folder on disk gets its "Box predictions" layer,
        holding this image's boxes, or none where the source has not run on
        it. A layer the user deleted stays deleted, until its source runs
        again. 2D images only, as box annotations.
        """
        from ..utilities.box_annotations import (
            prediction_file,
            prediction_sources,
            read_yolo_predictions,
        )
        from .profiles import box_predictions_name

        segment = self.segment_widget
        known = segment.box_predictions_layers
        hidden = self.hidden_box_sources
        for name, layer in list(known.items()):
            if layer not in self.viewer.layers:
                hidden.add(name)
                del known[name]
            else:
                # The last image's boxes go, whether or not this one has any.
                layer.data = []
                layer.refresh()

        project = self.image_data_model.parent_directory
        shape = self._annotation_shape(image_shape)
        if len(shape) != 2:
            return
        stem = self.image_data_model.get_image_paths()[
            self.current_image_index
        ].stem
        for source in prediction_sources(project):
            if box_predictions_name(source) in hidden:
                continue
            boxes, _, confidences = read_yolo_predictions(
                prediction_file(project, source, stem), *shape
            )
            segment.show_box_predictions(source, boxes, confidences)

    def _update_prediction_layers(self, image_shape):
        """Refresh every prediction layer for the current image.

        A segmenter with no saved prediction for this image gets zeros rather
        than losing its layer. Subdirectories that appeared since the last
        switch gain a layer; none is ever removed, because removal is the
        expensive half.
        """
        import numpy as _np

        subdirs = self.image_data_model.get_prediction_subdirectories() or []
        for method_dir in subdirs:
            method_name = method_dir.name
            try:
                # Named directly rather than by setting the model's current
                # segmenter first: that value is also the default target for
                # save_predictions, so steering a read with it would leave the
                # model pointing at whichever directory happened to be last.
                data = self.image_data_model.load_existing_predictions(
                    image_index=self.current_image_index,
                    image_shape=image_shape,
                    subdirectory=method_name,
                    axes_to_collapse=self.axes_to_collapse,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                print(
                    f"   Could not load predictions for {method_name}: {exc}"
                )
                continue

            layer = self.predictions_layers.get(method_name)
            if layer is not None and layer in self.viewer.layers:
                layer.data = data
                layer.refresh()  # see the note in _update_layers_for_image
            else:
                self.predictions_layers[method_name] = self.viewer.add_labels(
                    data,
                    name=method_name,
                    scale=self.image_data_model.get_scale(
                        axes_to_collapse=self.axes_to_collapse
                    )
                    or None,
                )

        # A layer whose subdirectory no longer exists keeps its place in the
        # list, emptied rather than removed.
        known = {d.name for d in subdirs}
        for name, layer in self.predictions_layers.items():
            if name not in known and layer in self.viewer.layers:
                layer.data = _np.zeros_like(layer.data)
                layer.refresh()

    def _distribute_layers_to_sub_apps(self):
        """
        Distribute the centrally-created layers to each sub-app.

        Uses direct attribute assignment instead of calling sub-apps' _set_image_layer()
        to avoid duplicate layer creation.
        """
        # Every sub-widget reads current_image_index when it saves or loads
        # anything for the current image, but each keeps its own copy,
        # inherited from BaseNDApp and set only in the standalone image-change
        # handler that this app does not call. So they all stayed at 0 for a
        # whole session: annotations and predictions were loaded from image 0,
        # and -- worse -- saved to image 0, so working on a second image
        # overwrote the first one's files.
        #
        # Pushing it is the small fix. The real one is for sub-widgets to stop
        # keeping their own copy of state that belongs to the model; see
        # docs/spec/0005-ai-lab-cleanup.md.
        for widget in (
            self.label_widget,
            self.segment_widget,
            self.augment_widget,
        ):
            widget.current_image_index = self.current_image_index

        # Label widget needs: image, labels, points, boxes
        self.label_widget.image_layer = self.image_layer
        self.label_widget.annotation_layer = self.annotations_layer
        self.label_widget.annotations_layers = self.annotations_layers
        self.label_widget.working_layer = self.working_layer
        self.label_widget.points_layer = self.points_layer
        self.label_widget.boxes_layer = self.boxes_layer
        # Sync the label widget's Active Annotation combo with the
        # centrally-loaded collection.
        if hasattr(self.label_widget, "_refresh_active_annotation_combo"):
            self.label_widget._refresh_active_annotation_combo()

        # Connect points layer events for label widget
        if self.points_layer and hasattr(
            self.label_widget, "_on_points_changed"
        ):
            self.points_layer.events.data.connect(
                self.label_widget._on_points_changed
            )

        # Connect boxes layer events for label widget
        if self.boxes_layer and hasattr(
            self.label_widget, "_on_boxes_changed"
        ):
            self.boxes_layer.events.data.connect(
                self.label_widget._on_boxes_changed
            )
            self.boxes_layer.selected_data.events.items_changed.connect(
                self.label_widget._on_box_selection_changed
            )

        # Loading boxes from CSV emits the same "added" event as drawing one,
        # so the handlers are silenced while it happens -- otherwise returning
        # to an image with a saved box asks whether to copy predictions into a
        # box the user did not just draw. Connecting afterwards was enough when
        # every layer was rebuilt per image; with layers reused the connection
        # already exists, so it has to be blocked explicitly.
        if hasattr(self.label_widget, "_load_existing_boxes"):
            if self.boxes_layer is not None:
                sel = self.boxes_layer.selected_data.events.items_changed
                with self.boxes_layer.events.data.blocker(), sel.blocked():
                    self.label_widget._load_existing_boxes()
            else:
                self.label_widget._load_existing_boxes()

        # Connect boxes layer events for nd_ai_lab (for prediction copying)
        if self.boxes_layer and hasattr(self, "_on_boxes_changed"):
            self.boxes_layer.events.data.connect(self._on_boxes_changed)

        # Distribute the 3D boxes layer to label and segment widgets.
        # Note: this is the annotation/labeling 3D boxes layer.  Interactive
        # segmentation is driven by each widget's own "Interactive 3D Boxes"
        # layer (created on demand from the widget UI), so we do NOT wire
        # this layer's events to _on_3D_boxes_changed here.
        self.label_widget.boxes_3D_layer = self.boxes_3D_layer
        self.segment_widget.boxes_3D_layer = self.boxes_3D_layer
        # Back-reference so child widgets (e.g. label widget's
        # "Copy predictions to labels" button) can call into NDAILab.
        self.label_widget.ai_lab = self
        self.segment_widget.ai_lab = self
        # Segment widget also needs the 2D "Label box" layer for its
        # ROI-source combo ("Label Box" option).
        self.segment_widget.boxes_layer = self.boxes_layer

        # Load any previously saved 3D boxes from labels3d/boxes.csv BEFORE
        # connecting the interactive handler, so populating the layer doesn't
        # trigger segmentation on every saved box.
        if hasattr(self.label_widget, "_load_existing_3D_boxes"):
            self.label_widget._load_existing_3D_boxes()

        # Wire shared 3D boxes layer to the label widget's 3D handler so it
        # can be chosen from the Interactive Layer combo ("3D Bounding Boxes").
        # The handler itself ignores the event unless the combo selects it.
        if hasattr(self.label_widget, "_on_3D_boxes_changed"):
            self.boxes_3D_layer.events.data.connect(
                self.label_widget._on_3D_boxes_changed
            )

        # Refresh the label widget's interactive-layer combo so it lists the
        # newly-distributed Label box / 3D Bounding Boxes layers.
        if hasattr(self.label_widget, "_refresh_interactive_layer_combo"):
            self.label_widget._refresh_interactive_layer_combo()

        # Augment widget needs: image, labels
        self.augment_widget.image_layer = self.image_layer
        self.augment_widget.annotation_layer = self.annotations_layer
        self.augment_widget.annotations_layers = self.annotations_layers
        self.augment_widget.boxes_layer = (
            self.boxes_layer
        )  # Provide boxes layer for augmentation if needed

        # Segment widget needs: image, labels, predictions, points, shapes
        self.segment_widget.image_layer = self.image_layer
        self.segment_widget.annotation_layer = self.annotations_layer
        self.segment_widget.annotations_layers = self.annotations_layers
        self.segment_widget.predictions_layers = self.predictions_layers
        self.segment_widget.points_layer = self.points_layer
        self.segment_widget.shapes_layer = self.shapes_layer

        # Connect interactive layers events for segment widget (if in interactive mode)
        if self.segment_widget.is_interactive_mode():
            if hasattr(self.segment_widget, "_on_points_changed"):
                self.points_layer.events.data.connect(
                    self.segment_widget._on_points_changed
                )
            if self.shapes_layer is not None and hasattr(
                self.segment_widget, "_on_shapes_changed"
            ):
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

        # Open/Save Project live on the label widget; shown here, above the
        # tabs, so they are reachable whichever tab is first.
        project_row = QHBoxLayout()
        project_row.addWidget(self.label_widget.open_project_btn)
        project_row.addWidget(self.label_widget.save_project_btn)
        layout.addLayout(project_row)

        # Class of the next box drawn, or of the selected boxes. Type a name
        # to add a class; it is added on Enter or when you click away.
        # Only for box annotations.
        class_row = QWidget()
        class_layout = QHBoxLayout(class_row)
        class_layout.setContentsMargins(0, 0, 0, 0)
        class_layout.addWidget(QLabel("Box class:"))
        self.class_combo = QComboBox()
        self.class_combo.setEditable(True)
        # Not Qt's own insert, which only happens on Enter.
        self.class_combo.setInsertPolicy(QComboBox.NoInsert)
        self.class_combo.lineEdit().setPlaceholderText("type a class name")
        self.class_combo.setToolTip(
            "Class of the next box. Type a new name to add a class."
        )
        self.class_combo.lineEdit().editingFinished.connect(
            self._on_class_typed
        )
        self.class_combo.currentIndexChanged.connect(self._on_class_chosen)
        class_layout.addWidget(self.class_combo, 1)
        class_row.setVisible(not self.profile.show_labels)
        layout.addWidget(class_row)

        # A box layer per kind of thing boxed -- runners, bibs -- each with
        # its own folder under box_annotations/ and its own classes. The
        # class combo above follows whichever box layer is selected.
        box_layer_row = QWidget()
        box_layer_layout = QHBoxLayout(box_layer_row)
        box_layer_layout.setContentsMargins(0, 0, 0, 0)
        box_layer_layout.addWidget(QLabel("New box layer:"))
        self.new_box_layer_edit = QLineEdit()
        self.new_box_layer_edit.setPlaceholderText("e.g. runners")
        self.new_box_layer_edit.returnPressed.connect(self._on_add_box_layer)
        box_layer_layout.addWidget(self.new_box_layer_edit, 1)
        add_box_layer_btn = QPushButton("Add")
        add_box_layer_btn.setToolTip(
            "Add a box annotation layer, saved in box_annotations/<name>/"
        )
        add_box_layer_btn.clicked.connect(self._on_add_box_layer)
        box_layer_layout.addWidget(add_box_layer_btn)
        box_layer_row.setVisible(not self.profile.show_labels)
        layout.addWidget(box_layer_row)
        self.viewer.layers.selection.events.active.connect(
            self._on_active_layer_changed
        )

        # Checking a detector's work: copy its boxes on this image into a box
        # annotation layer, then fix them there.
        copy_row = QWidget()
        copy_layout = QHBoxLayout(copy_row)
        copy_layout.setContentsMargins(0, 0, 0, 0)
        copy_layout.addWidget(QLabel("Copy"))
        self.copy_from_combo = QComboBox()
        self.copy_from_combo.setToolTip(
            "Box predictions to copy, on this image. Only the selected boxes, "
            "if some are selected."
        )
        copy_layout.addWidget(self.copy_from_combo, 1)
        copy_layout.addWidget(QLabel("to"))
        self.copy_to_combo = QComboBox()
        self.copy_to_combo.setToolTip(
            "The box annotation layer to add them to"
        )
        copy_layout.addWidget(self.copy_to_combo, 1)
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(self._on_copy_box_predictions)
        copy_layout.addWidget(copy_btn)
        copy_row.setVisible(not self.profile.show_labels)
        layout.addWidget(copy_row)
        self.viewer.layers.events.inserted.connect(self._refresh_copy_combos)
        self.viewer.layers.events.removed.connect(self._refresh_copy_combos)

        def _make_scroll(widget):
            scroll = QScrollArea()
            scroll.setWidget(widget)
            scroll.setWidgetResizable(True)
            return scroll

        # Tab widget
        self.tabs = QTabWidget()
        # Segment first: if a ready-made model works, nothing else is needed.
        self.tabs.addTab(_make_scroll(self.segment_widget), "Segment")
        self.tabs.addTab(_make_scroll(self.label_widget), "Label")
        self.tabs.addTab(_make_scroll(self.augment_widget), "Augment")

        # Add training tab - shows training view of segment widget
        # Same widget, different controls (segmenter combo + training params only)
        self.tabs.addTab(
            _make_scroll(self.segment_widget.get_training_widget()), "Train"
        )

        layout.addWidget(self.tabs)
        # Expanding, or napari pads the dock with a stretch and the tabs get
        # only their size hint -- half the dock on some tabs.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Sync patch_size_xy between Augment and Train tabs
        self._cross_connect_patch_size()

    def _cross_connect_patch_size(self):
        """Sync patch_size_xy spinbox (Augment) with train_patch_size_xy (Train)."""
        aug_spin = self.augment_widget.patch_size_xy_spinbox
        trn_form = self.segment_widget.training_parameter_form

        def _aug_to_train(val):
            trn_form.set_parameter("train_patch_size_xy", val)

        def _train_to_aug(params):
            if "train_patch_size_xy" in params:
                aug_spin.blockSignals(True)
                aug_spin.setValue(params["train_patch_size_xy"])
                aug_spin.blockSignals(False)

        aug_spin.valueChanged.connect(_aug_to_train)
        trn_form.parameters_changed.connect(_train_to_aug)

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

    def _annotation_shape(self, image_shape):
        """Shape of an annotation for an image of *image_shape*.

        The labels layer's shape when there is one; otherwise worked out the
        way an empty annotation would be, so a profile without labels still
        gets points and shapes of the right dimensionality.
        """
        if self.annotations_layer is not None:
            return self.annotations_layer.data.shape
        # Colour is never a box axis, whatever "Axes to collapse" says.
        return compute_collapsed_shape(
            image_shape,
            self.image_data_model.axis_types,
            [*(self.axes_to_collapse or ""), "C"],
        )

    #: Edge colours for box annotation layers, one per layer in turn. Cyan is
    #: left to box predictions and magenta to the 3D boxes.
    BOX_LAYER_COLORS = ("yellow", "orange", "lime", "deepskyblue", "violet")

    def _add_box_layer(self, folder):
        """Add the box annotation layer for *folder* (see box_annotations).

        Named after the folder -- "runners", "bibs" -- or "Box annotations"
        for the older single-folder layout. Returns the layer.
        """
        colour = self.BOX_LAYER_COLORS[
            len(self.box_layers) % len(self.BOX_LAYER_COLORS)
        ]
        layer = napari.layers.Shapes(
            name=folder or BOX_ANNOTATIONS,
            edge_color=colour,
            face_color="transparent",
            edge_width=3,
            ndim=self._box_layer_ndim,
            scale=self._box_layer_scale,
            features={
                "class": np.zeros(0, dtype=int),
                "name": np.zeros(0, dtype=object),
            },
            feature_defaults={"class": 0, "name": "0"},
            text={"string": "{name}", "size": 12, "color": colour},
            metadata={"box_folder": folder},
        )
        layer.events.highlight.connect(self._on_box_selection_changed)
        # Known as a box layer before it is added: adding makes it the active
        # layer, and _on_active_layer_changed looks it up then.
        self.box_layers[folder] = layer
        self.viewer.add_layer(layer)
        return layer

    def _live_box_layers(self):
        """The box layers still in the viewer; ones the user deleted go.

        Their files stay on disk. The active one (shapes_layer) moves to the
        first left, or None.
        """
        layers = getattr(self, "box_layers", None)
        if layers is None:
            layers = self.box_layers = {}
        gone = [
            f for f, lay in layers.items() if lay not in self.viewer.layers
        ]
        for folder in gone:
            del layers[folder]
        if getattr(self, "shapes_layer", None) not in layers.values():
            self.shapes_layer = next(iter(layers.values()), None)
        return layers

    def _box_folder(self, layer=None) -> str:
        """The folder of *layer*, by default the active box layer."""
        layer = layer if layer is not None else self.shapes_layer
        return (layer.metadata or {}).get("box_folder", "") if layer else ""

    def _load_box_annotations(self, image_shape, folders=None):
        """Fill the box annotation layers from this image's YOLO txt files.

        Each box gets its class id and the name its layer's classes.yaml
        gives it. A layer with no file for this image is empty. *folders*
        limits it to those layers; by default, all. 2D only for now: a YOLO
        txt file has no axis beyond y and x.
        """
        from ..utilities.box_annotations import (
            box_file,
            read_class_names,
            read_yolo_boxes,
        )

        project = self.image_data_model.parent_directory
        shape = self._annotation_shape(image_shape)
        image_path = self.image_data_model.get_image_paths()[
            self.current_image_index
        ]
        for folder, layer in self._live_box_layers().items():
            if folders is not None and folder not in folders:
                continue
            names = read_class_names(project, folder)
            boxes, classes = [], np.zeros(0, dtype=int)
            if len(shape) == 2:
                boxes, classes = read_yolo_boxes(
                    box_file(project, folder, image_path.stem), *shape
                )
            layer.data = []
            if boxes:
                layer.add_rectangles(boxes)
            layer.features = {
                "class": classes,
                "name": np.array(
                    [names.get(c, str(c)) for c in classes], dtype=object
                ),
            }
            layer.refresh()
        # After the features: setting them resets the next box's class.
        if self.shapes_layer is not None:
            self._fill_class_combo(
                read_class_names(project, self._box_folder())
            )

    def _on_active_layer_changed(self, event=None):
        """A box annotation layer was selected: the class combo follows it."""
        from ..utilities.box_annotations import read_class_names

        layer = self.viewer.layers.selection.active
        if layer is None or layer not in self._live_box_layers().values():
            return
        if layer is self.shapes_layer and self.class_combo.count():
            return
        self.shapes_layer = layer
        self.segment_widget.shapes_layer = layer
        project = self.image_data_model.parent_directory
        self._fill_class_combo(read_class_names(project, self._box_folder()))

    def _on_add_box_layer(self):
        """Add a box layer named as typed, with a folder and a first class.

        Class 0 is named after the layer, so its first box has a name; type
        more classes into the class combo. An existing name selects that
        layer instead.
        """
        from ..utilities.box_annotations import add_box_folder

        if (
            getattr(self, "image_layer", None) is None
            or not hasattr(self, "_box_layer_ndim")
            or self.image_data_model is None
        ):
            QMessageBox.warning(
                self, "Box layer", "Open a project first, in a box profile."
            )
            return
        name = self.new_box_layer_edit.text()
        try:
            folder = add_box_folder(
                self.image_data_model.parent_directory, name, first_class=name
            )
        except ValueError as e:
            QMessageBox.warning(self, "Box layer name", str(e))
            return
        layer = self._live_box_layers().get(folder)
        if layer is None:
            layer = self._add_box_layer(folder)
            self._load_box_annotations(
                self.image_layer.data.shape, folders=[folder]
            )
        self.new_box_layer_edit.clear()
        self.viewer.layers.selection.active = layer

    #: Two boxes overlapping this much (IoU) are the same box: copying
    #: predictions skips one its target layer already has.
    SAME_BOX_IOU = 0.9

    def _refresh_copy_combos(self, event=None):
        """List the box prediction and annotation layers in the copy row.

        From the viewer's own list, so a layer shows up the moment it is
        added, whoever added it. Keeps each combo's choice where it can.
        """
        from .profiles import box_predictions_name

        prefix = box_predictions_name("")[:-1]  # "Box predictions ("
        sources = [
            lay.name
            for lay in self.viewer.layers
            if lay.name.startswith(prefix) and lay.name.endswith(")")
        ]
        targets = [lay.name for lay in self._live_box_layers().values()]
        for combo, names in (
            (self.copy_from_combo, sources),
            (self.copy_to_combo, targets),
        ):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            if current in names:
                combo.setCurrentText(current)
            combo.blockSignals(False)

    def _on_copy_box_predictions(self):
        """Add the chosen predictions' boxes, on this image, to a box layer.

        Only the selected boxes when some are selected, else all of them.
        Added to what the layer has, skipping any box it already has (see
        SAME_BOX_IOU), so copying twice does not double them. They take the
        class showing in the class combo when the target is the active box
        layer, else the target's class 0. Saved with the layer, as any edit.
        """
        from napari.utils.notifications import show_info

        from ..utilities.box_annotations import read_class_names

        source_name = self.copy_from_combo.currentText()
        source = None
        if source_name in self.viewer.layers:
            source = self.viewer.layers[source_name]
        target = next(
            (
                lay
                for lay in self._live_box_layers().values()
                if lay.name == self.copy_to_combo.currentText()
            ),
            None,
        )
        if source is None or target is None:
            QMessageBox.warning(
                self,
                "Copy boxes",
                "Choose box predictions to copy and a box layer to copy to.",
            )
            return
        if source.ndim != 2 or target.ndim != 2:
            QMessageBox.warning(
                self, "Copy boxes", "Boxes are copied on 2D images only."
            )
            return

        rows = sorted(source.selected_data) or range(len(source.data))
        have = [_box_bounds(b) for b in target.data]
        new = []
        for row in rows:
            box = np.asarray(source.data[row])[:, -2:]
            bounds = _box_bounds(box)
            if all(_iou(bounds, h) < self.SAME_BOX_IOU for h in have):
                new.append(box)
                have.append(bounds)

        chosen = self.class_combo.currentData()
        if target is self.shapes_layer and chosen is not None:
            class_id = chosen
            name = self.class_combo.currentText()
        else:
            project = self.image_data_model.parent_directory
            class_id = 0
            name = read_class_names(project, self._box_folder(target)).get(
                0, "0"
            )
        if new:
            target.feature_defaults = {"class": class_id, "name": name}
            target.add_rectangles(new)
            target.refresh()

        skipped = len(rows) - len(new)
        message = f"Copied {len(new)} boxes to {target.name} as {name}"
        if skipped:
            message += f"; {skipped} already there"
        print(message)
        show_info(message)
        self.viewer.layers.selection.active = target

    def _fill_class_combo(self, names):
        """List *names* in the class combo, keeping the current choice.

        Each item carries its class id, since classes.yaml may skip ids.
        """
        combo = self.class_combo
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for class_id, name in sorted(names.items()):
            combo.addItem(name, class_id)
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)
        self._set_next_box_class()

    def _set_next_box_class(self):
        """Give the next box drawn the class showing in the combo."""
        combo = self.class_combo
        class_id = combo.currentData()
        if class_id is None:
            class_id, name = 0, "0"
        else:
            name = combo.currentText()
        if self.shapes_layer is not None:
            self.shapes_layer.feature_defaults = {
                "class": class_id,
                "name": name,
            }

    def _class_names(self):
        """Class id -> name, as the combo lists them."""
        combo = self.class_combo
        return {
            combo.itemData(i): combo.itemText(i)
            for i in range(combo.count())
            if combo.itemData(i) is not None
        }

    def _on_class_typed(self):
        """Choose the class typed into the combo, adding it if it is new.

        Runs on Enter and when the combo loses focus, so a name typed and
        then left behind by a click on the image still counts.
        """
        combo = self.class_combo
        name = combo.lineEdit().text().strip()
        if not name:
            # Nothing typed: show the current class again.
            combo.lineEdit().setText(combo.itemText(combo.currentIndex()))
            return
        index = combo.findText(name)
        if index < 0:
            combo.addItem(name)  # no id yet; _on_class_chosen gives it one
            index = combo.count() - 1
        if index != combo.currentIndex():
            combo.setCurrentIndex(index)

    def _on_class_chosen(self, index):
        """Make the chosen class the next box's, and the selected boxes'.

        A name typed into the combo is a new class: it gets the next free id
        and is written to classes.yaml.
        """
        from ..utilities.box_annotations import write_class_names

        combo = self.class_combo
        if index < 0:
            class_id, name = 0, "0"
        else:
            name = combo.itemText(index)
            class_id = combo.itemData(index)
            if class_id is None:
                names = self._class_names()
                class_id = max(names, default=-1) + 1
                combo.setItemData(index, class_id)
                names[class_id] = name
                project = self.image_data_model.parent_directory
                if project:
                    write_class_names(project, names, self._box_folder())

        layer = getattr(self, "shapes_layer", None)
        if layer is None or self.profile.show_labels:
            return
        selected = list(layer.selected_data)
        if selected:
            features = layer.features.copy()
            features.loc[selected, "class"] = class_id
            features.loc[selected, "name"] = name
            layer.features = features
        # Last: setting the features resets the next box's class.
        layer.feature_defaults = {"class": class_id, "name": name}

    def _on_box_selection_changed(self, event=None):
        """Show the class of a single selected box in the combo."""
        layer = event.source if event is not None else self.shapes_layer
        if layer is not self.shapes_layer or len(layer.selected_data) != 1:
            return
        (row,) = layer.selected_data
        index = self.class_combo.findData(int(layer.features["class"][row]))
        if index >= 0 and index != self.class_combo.currentIndex():
            self.class_combo.blockSignals(True)
            self.class_combo.setCurrentIndex(index)
            self.class_combo.blockSignals(False)
            # The next box drawn takes this class too.
            self._set_next_box_class()

    def _save_box_annotations(self):
        """Write every box annotation layer to this image's YOLO txt files.

        An image that never had boxes in a layer gets no file there; one
        whose boxes were all deleted gets an empty file, so they stay
        deleted -- and the image counts as checked, holding none.
        """
        from ..utilities.box_annotations import box_file, write_yolo_boxes

        project = (
            self.image_data_model and self.image_data_model.parent_directory
        )
        if self.profile.show_labels or not project:
            return
        layers = self._live_box_layers()
        if not layers:
            return

        shape = self._annotation_shape(self.image_layer.data.shape)
        if len(shape) != 2:
            return
        image_path = self.image_data_model.get_image_paths()[
            self.current_image_index
        ]
        for folder, layer in layers.items():
            txt = box_file(project, folder, image_path.stem)
            if len(layer.data) == 0 and not txt.exists():
                continue
            try:
                write_yolo_boxes(
                    txt, layer.data, layer.features["class"], *shape
                )
                print(f"   Saved {len(layer.data)} {layer.name} boxes")
            except OSError as e:
                print(f"   Failed to save boxes to {txt}: {e}")

    def _wants_shapes(self):
        """Whether the Shapes layer belongs in this session.

        Profile only -- no project shape makes shape prompts inapplicable.
        Asked here rather than inline so layer creation and the reuse check
        cannot drift apart, which is what made every 2D switch rebuild.
        """
        return self.profile.show_shapes

    def _wants_3d(self):
        """Whether the 3D box layer and its button belong in this session.

        Both the project and the profile get a veto: an ND project shows them,
        unless the profile says not to.  One answer, used by apply_mode to
        decide what to put in the viewer and by _update_layers_for_image to
        decide what to expect there -- those two disagreeing is what made
        every 2D switch take the slow path.
        """
        model = self.image_data_model
        nd = model is None or model.mode != "2d"
        return nd and self.profile.show_3d

    def apply_mode(self):
        """Show what this project needs, once one is loaded.

        Everything that depends on 2D-versus-ND, or on there being a
        sequence, is decided here and nowhere else.
        """
        model = self.image_data_model
        nd = self._wants_3d()
        if hasattr(self, "boxes_3D_layer"):
            if nd and self.boxes_3D_layer not in self.viewer.layers:
                self.viewer.add_layer(self.boxes_3D_layer)
            elif not nd and self.boxes_3D_layer in self.viewer.layers:
                self.viewer.layers.remove(self.boxes_3D_layer)
        self.label_widget.add_interactive_3D_boxes_btn.setVisible(nd)

        # The profile's own say over the optional GUI groups. Here, with
        # everything else that decides what is visible.
        if hasattr(self.label_widget, "apply_profile_features"):
            self.label_widget.apply_profile_features(self.profile)

        sequence = model is not None and model.viewer_type == "sequence"
        self.segment_widget.segment_sequence_btn.setVisible(sequence)
        self.segment_widget.segment_all_btn.setVisible(not sequence)

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

        # finally, because the flag is a lock: an exception that escapes would
        # leave it True and every later image change would be silently dropped.
        try:
            self._process_image_change(image_layer, image_index)
        finally:
            self._processing_image_change = False

    def _process_image_change(self, image_layer, image_index):
        """
        Process image change from sequence viewer.

        Handles cleanup and recreation of layers centrally for all sub-apps.
        """
        print(f"🔄 nd_ai_lab: Processing image change to index {image_index}")

        # Save the current image's work before switching away from it. This
        # used to be gated on the Label tab being visible and on a
        # "labels_layer" attribute that does not exist, so it never ran. See
        # docs/spec/0005-ai-lab-cleanup.md.
        #
        # getattr: on the first image of a session there are no layers yet.
        # Not keyed on annotations_layers, which is empty without labels and
        # would skip the box save below with it.
        if (
            getattr(self, "points_layer", None) is not None
            and self.image_data_model.parent_directory
        ):
            # Every layer, under its own name. Annotations are a collection --
            # one subdirectory per layer -- so saving only the active one drops
            # the rest, and omitting the name falls back to save_annotations'
            # "class_0" default and invents a subdirectory nothing asked for.
            for name, layer in self.annotations_layers.items():
                try:
                    self.image_data_model.save_annotations(
                        layer.data,
                        self.current_image_index,
                        subdirectory=name,
                        current_step=self.viewer.dims.current_step,
                        axes_to_collapse=self.axes_to_collapse,
                    )
                    print(f"   Saved annotations: {name}")
                except (OSError, ValueError, RuntimeError) as e:
                    print(f"   Failed to save annotations '{name}': {e}")

            # Save boxes at the same time as annotations
            self._save_boxes()
            self._save_box_annotations()
            self._save_label_patches_on_close()

        # Update current image index
        self.current_image_index = image_index

        if image_layer:
            print(f"   Setting up new image: {image_layer.name}")

            # Prefer pointing the existing layers at the new image. Removing a
            # layer costs a full gc.collect inside napari -- a deliberate
            # workaround for a Windows OpenGL driver bug, see
            # napari/_vispy/canvas.py -- and there are seven of them per
            # switch, which benchmarks/bench_sequence_switch.py measures as
            # most of the switch time. Nothing about these layers is
            # image-specific except their data.
            if self._update_layers_for_image(image_layer):
                print("✅ nd_ai_lab: Image change complete (layers reused)")
            else:
                self._cleanup_layers()
                self._set_image_layer(image_layer)
                print("✅ nd_ai_lab: Image change complete (layers rebuilt)")
        else:
            print("⚠️  Received invalid image data from sequence viewer")
            self._cleanup_layers()

        # The lock is released by the caller's finally, not here: one owner.

    def _save_boxes(self):
        """Delegate box saving to label_widget, which inherits it from BaseNDApp."""
        if hasattr(self.label_widget, "_save_boxes"):
            self.label_widget._save_boxes()

    def _save_label_patches_on_close(self):
        """Delegate label patch saving to label_widget (called by base close handler)."""
        if hasattr(self, "label_widget"):
            self.label_widget._save_label_patches_on_close()

    def _cleanup_layers(self):
        """
        Cleanup existing layers before switching images.

        Central cleanup for combined app - removes all layers from viewer.
        """
        print("   🧹 Cleaning up existing layers...")

        layers_to_remove = []

        # Named annotation layers (one per annotations/ subdirectory).
        if hasattr(self, "annotations_layers") and self.annotations_layers:
            for ann_name, layer in self.annotations_layers.items():
                layers_to_remove.append((f"Annotation ({ann_name})", layer))
        elif getattr(self, "annotations_layer", None) is not None:
            # Fallback for the single-layer case. Was guarded on a
            # "labels_layer" attribute that does not exist, so it never ran.
            layers_to_remove.append(("Labels", self.annotations_layer))

        # Handle predictions_layers dictionary (multiple prediction layers)
        if hasattr(self, "predictions_layers") and self.predictions_layers:
            for segmenter_name, layer in self.predictions_layers.items():
                layers_to_remove.append(
                    (f"Predictions ({segmenter_name})", layer)
                )

        if hasattr(self, "points_layer") and self.points_layer:
            layers_to_remove.append(("Points", self.points_layer))
        if hasattr(self, "shapes_layer") and self.shapes_layer:
            layers_to_remove.append(("Shapes", self.shapes_layer))
        for folder, layer in (getattr(self, "box_layers", None) or {}).items():
            if layer is not self.shapes_layer:
                layers_to_remove.append((folder or "Box annotations", layer))
        self.box_layers = {}
        if hasattr(self, "boxes_layer") and self.boxes_layer:
            layers_to_remove.append(("Boxes", self.boxes_layer))
        if hasattr(self, "working_layer") and self.working_layer:
            layers_to_remove.append(("Labels (Working)", self.working_layer))
        if hasattr(self, "boxes_3D_layer") and self.boxes_3D_layer:
            layers_to_remove.append(("3D Bounding Boxes", self.boxes_3D_layer))
        for name, layer in self.segment_widget.box_predictions_layers.items():
            layers_to_remove.append((name, layer))
        self.segment_widget.box_predictions_layers.clear()

        # Remove layers from viewer
        for layer_name, layer in layers_to_remove:
            try:
                if layer in self.viewer.layers:
                    self.viewer.layers.remove(layer)
                    print(f"      Removed {layer_name} layer")
            except (ValueError, KeyError, RuntimeError) as e:
                print(f"      Error removing {layer_name}: {e}")

        # Clear references
        if hasattr(self, "annotations_layers"):
            self.annotations_layers = {}
        self.annotations_layer = None
        if hasattr(self, "predictions_layers"):
            self.predictions_layers = {}
        if hasattr(self, "points_layer"):
            self.points_layer = None
        if hasattr(self, "shapes_layer"):
            self.shapes_layer = None
        if hasattr(self, "boxes_layer"):
            self.boxes_layer = None
        if hasattr(self, "working_layer"):
            self.working_layer = None
        if hasattr(self, "boxes_3D_layer"):
            self.boxes_3D_layer = None
        if hasattr(self, "boxes_layer"):
            self.boxes_layer = None

        print("   ✅ Layer cleanup complete")

    def _on_boxes_changed(self, event):
        """Handle 2D boxes layer data changes - show dialog to copy predictions to new ROI."""

        # Only respond to 'added' events (same check as in nd_easy_label)
        if event.action != "added":
            return

        boxes_layer = event.source
        if len(boxes_layer.data) == 0:
            return

        if (
            not hasattr(self, "predictions_layers")
            or not self.predictions_layers
        ):
            print("No predictions layers available to copy from")
            return

        available = self._find_available_predictions(boxes_layer)
        if not available:
            print("⚠️ No predictions exist at this location")
            return

        self._show_copy_predictions_dialog(
            box_layer=boxes_layer,
            available_predictions=available,
            allow_source_choice=False,
        )

    # ------------------------------------------------------------------
    # Shared helpers for "copy predictions to labels" (2D and 3D paths)
    # ------------------------------------------------------------------
    def _find_available_predictions(self, box_layer):
        """Return ``{name: pred_layer}`` for predictions with non-zero data
        inside the active box on ``box_layer``.

        Uses the shared ``_compute_crop_slice`` helper on
        :class:`BaseNDApp` (inherited by ``label_widget``) so this works
        uniformly for 2D Shapes and 3D BoundingBoxLayer sources.
        """
        available = {}
        if box_layer is None:
            return available
        if (
            not hasattr(self, "predictions_layers")
            or not self.predictions_layers
        ):
            return available
        # _compute_crop_slice lives on BaseNDApp; reuse via label_widget
        compute = getattr(self.label_widget, "_compute_crop_slice", None)
        if compute is None:
            return available
        for name, pred_layer in self.predictions_layers.items():
            try:
                slc = compute(box_layer, pred_layer)
                if slc is None:
                    print(f"✗ {name}: no slice for ROI")
                    continue
                pred_roi = pred_layer.data[slc]
                if np.any(pred_roi):
                    available[name] = pred_layer
                    print(f"✓ {name}: predictions found in ROI")
                else:
                    print(f"✗ {name}: no non-zero predictions in ROI")
            except (IndexError, ValueError, TypeError) as e:
                print(f"✗ {name}: no predictions at this location ({e})")
        return available

    def _copy_predictions_into_labels(self, box_layer, pred_layer):
        """Copy ``pred_layer`` ROI into ``annotations_layer`` ROI using
        per-target slices from ``_compute_crop_slice``."""
        compute = getattr(self.label_widget, "_compute_crop_slice", None)
        if compute is None or self.annotations_layer is None:
            print("✗ Cannot copy: missing helper or annotations layer")
            return False
        preds_slice = compute(box_layer, pred_layer)
        labels_slice = compute(box_layer, self.annotations_layer)
        if preds_slice is None or labels_slice is None:
            print("✗ Cannot copy: failed to compute crop slice")
            return False
        try:
            self.annotations_layer.data[labels_slice] = pred_layer.data[
                preds_slice
            ]
            self.annotations_layer.refresh()
            return True
        except (IndexError, ValueError) as e:
            print(f"✗ Copy failed: {e}")
            return False

    def _show_copy_predictions_dialog(
        self,
        box_layer=None,
        available_predictions=None,
        allow_source_choice=False,
    ):
        """Show the copy-predictions dialog.

        - When ``allow_source_choice`` is False (the 2D auto-trigger path):
          ``box_layer`` and ``available_predictions`` must be supplied and
          only the predictions combo is shown.
        - When ``allow_source_choice`` is True (the 3D button path): a
          Source ROI combo is also shown (Label box / 3D Bounding Boxes),
          and available predictions are recomputed when the source ROI
          changes.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Copy Predictions to ROI")
        dialog.setMinimumWidth(420)

        layout = QVBoxLayout(dialog)
        layout.addWidget(
            QLabel(
                "Choose predictions layer and press 'Copy' to copy "
                "predictions into the labels layer."
            )
        )

        # Optional source ROI combo
        source_combo = None
        source_layers = {}
        if allow_source_choice:
            if (
                hasattr(self, "boxes_3D_layer")
                and self.boxes_3D_layer is not None
            ):
                source_layers["3D Bounding Boxes"] = self.boxes_3D_layer
            if hasattr(self, "boxes_layer") and self.boxes_layer is not None:
                source_layers["Label box"] = self.boxes_layer
            if not source_layers:
                print("⚠️ No ROI source layers available")
                return
            row = QHBoxLayout()
            row.addWidget(QLabel("Source ROI:"))
            source_combo = QComboBox()
            for name in source_layers:
                source_combo.addItem(name)
            row.addWidget(source_combo)
            layout.addLayout(row)

        # Predictions combo
        row = QHBoxLayout()
        row.addWidget(QLabel("Predictions Layer:"))
        predictions_combo = QComboBox()
        row.addWidget(predictions_combo)
        layout.addLayout(row)

        status_label = QLabel("")
        layout.addWidget(status_label)

        # Buttons
        button_row = QHBoxLayout()
        copy_button = QPushButton("Copy")
        cancel_button = QPushButton("Cancel")
        button_row.addWidget(copy_button)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)

        # Mutable state shared with refresh()
        state = {
            "box_layer": box_layer,
            "available": dict(available_predictions or {}),
        }

        def refresh_predictions():
            if allow_source_choice and source_combo is not None:
                state["box_layer"] = source_layers.get(
                    source_combo.currentText()
                )
                state["available"] = self._find_available_predictions(
                    state["box_layer"]
                )
            predictions_combo.blockSignals(True)
            predictions_combo.clear()
            for name in state["available"]:
                predictions_combo.addItem(name)
            predictions_combo.blockSignals(False)
            has_any = bool(state["available"])
            copy_button.setEnabled(has_any)
            if not has_any:
                status_label.setText(
                    "⚠️ No predictions overlap the selected ROI."
                )
            else:
                status_label.setText("")

        def do_copy():
            name = predictions_combo.currentText()
            pred_layer = state["available"].get(name)
            box_layer_now = state["box_layer"]
            if pred_layer is None or box_layer_now is None:
                return
            if self._copy_predictions_into_labels(box_layer_now, pred_layer):
                print(f"✓ Copied {name} predictions to labels layer")
                dialog.accept()

        if source_combo is not None:
            source_combo.currentTextChanged.connect(
                lambda _t: refresh_predictions()
            )
        copy_button.clicked.connect(do_copy)
        cancel_button.clicked.connect(dialog.reject)

        refresh_predictions()
        dialog.exec_()

    def show_copy_predictions_dialog(self):
        """Public entry point used by child widgets (e.g. the label widget's
        'Copy predictions to labels' button) to open the copy-predictions
        dialog with a Source ROI chooser (2D Label box or 3D Bounding
        Boxes)."""
        if (
            not hasattr(self, "predictions_layers")
            or not self.predictions_layers
        ):
            from qtpy.QtWidgets import QMessageBox

            QMessageBox.information(
                self,
                "Copy Predictions",
                "No predictions layers available to copy from.",
            )
            return
        self._show_copy_predictions_dialog(allow_source_choice=True)
