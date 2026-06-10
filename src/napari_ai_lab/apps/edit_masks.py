"""Edit-Masks workflow: build a stack of (image, mask) pairs from points
drawn on the original viewer, edit the masks in a second viewer, and save
the stack under ``<project>/masks/<name>/`` for downstream training (e.g.
SAM-style fine-tuning).

The widget lives in the *second* napari viewer.  Workflow:

1. User draws points in the *original* viewer (a points layer named
   ``"Mask Points (<name>)"`` is created on demand the first time the
   "Add Masks From Points" button is pressed).
2. Clicking "Add Masks From Points" extracts a square ``XY × XY`` patch
   centred on each point from the original image, appends the patches
   onto the running image / mask stacks, and refreshes the two layers in
   the second viewer.
3. The user edits the labels in the second viewer (paint / fill / etc.).
4. "Save Masks" persists the stacks under ``masks/<name>/`` via
   :meth:`ImageDataModel.save_masks`.
"""

from __future__ import annotations

import contextlib

import napari
import numpy as np
from qtpy.QtWidgets import (
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


def _extract_patch_around_point(
    image_data: np.ndarray,
    point_coord,
    has_trailing_channel: bool,
    mask_size: int,
) -> tuple[np.ndarray, tuple]:
    """Extract a ``mask_size × mask_size`` patch centred on a point.

    Always operates on the last two spatial axes:

    - ``has_trailing_channel = False`` → spatial Y/X are axes ``-2, -1``.
    - ``has_trailing_channel = True``  → spatial Y/X are axes ``-3, -2``
      and the trailing axis is taken in full (RGB / RGBA-like).

    Any leading axes (Z, T, …) are indexed with the corresponding leading
    entries of ``point_coord`` so the returned patch is always a 2D
    grayscale or 2D RGB tile (no extra leading dims).

    The patch is zero-padded when the requested window runs off the
    image edges so every returned image patch has shape
    ``(mask_size, mask_size[, C])`` and the matching ``mask_shape`` is
    always ``(mask_size, mask_size)``.
    """
    image_data = np.asarray(image_data)
    point_coord = np.asarray(point_coord, dtype=float)
    half = mask_size // 2

    if has_trailing_channel:
        y_axis = image_data.ndim - 3
        x_axis = image_data.ndim - 2
    else:
        y_axis = image_data.ndim - 2
        x_axis = image_data.ndim - 1

    if y_axis < 0:
        raise ValueError(
            f"Image with shape {image_data.shape} doesn't have two spatial "
            "axes."
        )

    # Length of point coords should match napari's "displayed" ndim:
    # image.ndim - (1 if trailing channel else 0).  Last 2 entries are Y, X.
    if len(point_coord) < 2:
        raise ValueError(
            f"Point {point_coord} has too few coords for a 2D crop."
        )

    cy = int(round(float(point_coord[-2])))
    cx = int(round(float(point_coord[-1])))

    H = image_data.shape[y_axis]
    W = image_data.shape[x_axis]

    y_start = cy - half
    y_end = y_start + mask_size
    sy_start = max(0, y_start)
    sy_end = min(H, y_end)

    x_start = cx - half
    x_end = x_start + mask_size
    sx_start = max(0, x_start)
    sx_end = min(W, x_end)

    if sy_end <= sy_start or sx_end <= sx_start:
        raise ValueError(
            f"Point {(cy, cx)} falls outside the image extent {(H, W)}."
        )

    # Index any leading non-spatial axes via the leading point coords.
    n_leading = y_axis  # axes before Y
    leading_pt = point_coord[:-2]
    slices: list = []
    for i in range(n_leading):
        idx = int(round(float(leading_pt[i]))) if i < len(leading_pt) else 0
        idx = max(0, min(image_data.shape[i] - 1, idx))
        slices.append(idx)

    slices.append(slice(sy_start, sy_end))
    slices.append(slice(sx_start, sx_end))
    if has_trailing_channel:
        slices.append(slice(None))

    image_patch = image_data[tuple(slices)]

    # Pad Y and X up to mask_size; pad C with 0.
    pad_y = (sy_start - y_start, y_end - sy_end)
    pad_x = (sx_start - x_start, x_end - sx_end)
    pad_width: list[tuple[int, int]] = [pad_y, pad_x]
    if has_trailing_channel:
        pad_width.append((0, 0))

    if any(p != (0, 0) for p in pad_width):
        image_patch = np.pad(image_patch, pad_width, mode="constant")

    mask_shape = (mask_size, mask_size)
    return image_patch, mask_shape


class EditMasksWidget(QWidget):
    """QWidget docked in the second viewer driving the Edit-Masks workflow."""

    POINTS_LAYER_PREFIX = "Mask Points"

    def __init__(
        self,
        source_viewer: napari.viewer.Viewer,
        second_viewer: napari.viewer.Viewer,
        image_data_model,
        image_layer,
        axes_to_collapse=None,
        default_mask_size: int = 256,
    ):
        super().__init__()
        self.source_viewer = source_viewer
        self.second_viewer = second_viewer
        self.image_data_model = image_data_model
        self.source_image_layer = image_layer
        self.axes_to_collapse = axes_to_collapse

        # State
        self.mask_name: str | None = None
        self.image_stack: np.ndarray | None = None
        self.mask_stack: np.ndarray | None = None
        self.points_layer = None
        self.image_layer_2nd = None
        self.mask_layer_2nd = None

        self._setup_ui(default_mask_size)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _setup_ui(self, default_mask_size: int):
        layout = QVBoxLayout()
        self.setLayout(layout)

        instructions = QLabel(
            "Draw a point on the original viewer (only the most recent "
            "point is kept), then press 'Add Masks From Points' to add a "
            "new image / mask pair to this viewer for editing.\n\n"
            "Edit the masks here, then 'Save Masks' to persist the stack "
            "under <project>/masks/<name>/."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        size_label = QLabel("Mask XY size:")
        layout.addWidget(size_label)
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setMinimum(8)
        self.size_spinbox.setMaximum(4096)
        self.size_spinbox.setSingleStep(16)
        self.size_spinbox.setValue(int(default_mask_size))
        layout.addWidget(self.size_spinbox)

        self.add_btn = QPushButton("Add Masks From Points")
        self.add_btn.clicked.connect(self._on_add_masks)
        layout.addWidget(self.add_btn)

        self.save_btn = QPushButton("Save Masks")
        self.save_btn.clicked.connect(self._on_save_masks)
        layout.addWidget(self.save_btn)

        self.status_label = QLabel("No masks yet.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_points_layer(self):
        """Return (and create when missing) the shared points layer in the
        source viewer.
        """
        layer_name = f"{self.POINTS_LAYER_PREFIX} ({self.mask_name})"

        # Reuse existing if still alive
        if self.points_layer is not None:
            with contextlib.suppress(KeyError, ValueError, RuntimeError):
                if self.points_layer in self.source_viewer.layers:
                    return self.points_layer
            self.points_layer = None

        if layer_name in self.source_viewer.layers:
            self.points_layer = self.source_viewer.layers[layer_name]
            return self.points_layer

        # Build a new points layer with ndim matching the source image
        # display ndim (handles RGB-like images where C is hidden).
        try:
            ndim = int(self.source_image_layer.ndim)
        except (AttributeError, TypeError, ValueError):
            ndim = self.source_image_layer.data.ndim

        empty = np.empty((0, ndim))
        self.points_layer = self.source_viewer.add_points(
            empty,
            name=layer_name,
            size=10,
            face_color="lime",
            border_color="black",
        )

        # Enforce "only one point at a time": whenever the user adds a
        # point, drop everything except the most recently added one.
        # ``self._enforcing_single_point`` guards against recursion when
        # we mutate ``layer.data`` from inside the data event.
        self._enforcing_single_point = False

        def _enforce_single(event=None):
            if self._enforcing_single_point:
                return
            layer = self.points_layer
            if layer is None:
                return
            data = np.asarray(layer.data)
            if len(data) <= 1:
                return
            self._enforcing_single_point = True
            try:
                # Keep the last (most recently added) row.
                layer.data = data[-1:].copy()
            finally:
                self._enforcing_single_point = False

        with contextlib.suppress(AttributeError, TypeError):
            self.points_layer.events.data.connect(_enforce_single)

        return self.points_layer

    def _prompt_for_name(self) -> str | None:
        name, ok = QInputDialog.getText(
            self,
            "Mask collection name",
            "Name for this mask collection:",
            text="masks",
        )
        if not ok:
            return None
        name = name.strip()
        if not name:
            QMessageBox.warning(
                self, "Invalid name", "Please enter a non-empty name."
            )
            return None
        return name

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def _on_add_masks(self):
        if self.source_image_layer is None:
            QMessageBox.warning(
                self, "No image", "Original image layer is not available."
            )
            return

        # First time: ask for collection name and create the points layer.
        if self.mask_name is None:
            name = self._prompt_for_name()
            if name is None:
                return
            self.mask_name = name
            self._ensure_points_layer()
            QMessageBox.information(
                self,
                "Draw a point",
                "Draw a single point on the original viewer to mark the "
                "mask centre, then press 'Add Masks From Points' again. "
                "Only the most recent point is kept.",
            )
            return

        points_layer = self._ensure_points_layer()
        coords = np.asarray(points_layer.data)
        if len(coords) == 0:
            QMessageBox.information(
                self,
                "No point",
                "Add a point on the original viewer first.",
            )
            return

        # Single-point mode: use the (only) point.  The points layer
        # auto-trims to a single row in _ensure_points_layer, but if more
        # than one slipped through, take the most recent.
        coord = coords[-1]

        mask_size = int(self.size_spinbox.value())
        image_data = np.asarray(self.source_image_layer.data)

        # Detect a trailing channel axis the napari-way: napari hides RGB-
        # like trailing axes from its "displayed" ndim, so when the layer's
        # data has one extra axis vs. its layer.ndim, that's the channel.
        try:
            display_ndim = int(self.source_image_layer.ndim)
        except (AttributeError, TypeError, ValueError):
            display_ndim = image_data.ndim
        has_trailing_channel = image_data.ndim == display_ndim + 1

        try:
            image_patch, mask_shape = _extract_patch_around_point(
                image_data, coord, has_trailing_channel, mask_size
            )
        except (ValueError, IndexError) as e:
            QMessageBox.warning(
                self,
                "No patch",
                f"Could not extract a patch at point {coord.tolist()}: {e}\n\n"
                f"image shape: {image_data.shape}, "
                f"layer ndim: {display_ndim}, "
                f"trailing channel: {has_trailing_channel}",
            )
            return

        new_image_patches = [image_patch]
        new_mask_patches = [np.zeros(mask_shape, dtype=np.uint16)]

        # Validate consistent shapes against any prior stack.
        first_image_shape = new_image_patches[0].shape
        if (
            self.image_stack is not None
            and self.image_stack.shape[1:] != first_image_shape
        ):
            QMessageBox.warning(
                self,
                "Shape mismatch",
                "New patches have a different shape than the existing "
                f"stack ({first_image_shape} vs "
                f"{self.image_stack.shape[1:]}). Save and start a new "
                "collection if you want to change the mask size.",
            )
            return

        new_image_arr = np.stack(new_image_patches, axis=0)
        new_mask_arr = np.stack(new_mask_patches, axis=0)

        # Pull the latest user edits from the labels layer (if it exists)
        # before concatenating, so painted strokes survive the rebuild.
        if self.mask_layer_2nd is not None:
            with contextlib.suppress(AttributeError, RuntimeError):
                self.mask_stack = np.asarray(self.mask_layer_2nd.data)

        if self.image_stack is None:
            self.image_stack = new_image_arr
            self.mask_stack = new_mask_arr
        else:
            self.image_stack = np.concatenate(
                [self.image_stack, new_image_arr], axis=0
            )
            self.mask_stack = np.concatenate(
                [self.mask_stack, new_mask_arr], axis=0
            )

        self._refresh_second_viewer_layers()

        # Clear the points so the next batch starts fresh.
        with contextlib.suppress(AttributeError, RuntimeError, ValueError):
            points_layer.data = np.empty((0, points_layer.data.shape[-1]))

        n = self.image_stack.shape[0]
        self.status_label.setText(
            f"Collection '{self.mask_name}': {n} mask(s) of size "
            f"{mask_size}×{mask_size}"
        )

    def _refresh_second_viewer_layers(self):
        """Add or update image / labels layers in the second viewer."""
        image_name = f"{self.mask_name} images"
        mask_name = f"{self.mask_name} masks"

        # Image layer — look up by name so we don't accumulate stale refs.
        if image_name in self.second_viewer.layers:
            self.image_layer_2nd = self.second_viewer.layers[image_name]
            self.image_layer_2nd.data = self.image_stack
        else:
            self.image_layer_2nd = self.second_viewer.add_image(
                self.image_stack, name=image_name
            )

        # Labels layer
        if mask_name in self.second_viewer.layers:
            self.mask_layer_2nd = self.second_viewer.layers[mask_name]
            self.mask_layer_2nd.data = self.mask_stack
        else:
            self.mask_layer_2nd = self.second_viewer.add_labels(
                self.mask_stack, name=mask_name
            )

        # Kick napari to rebuild its dim slider for the new stack length.
        # Assigning ``layer.data`` to a longer array does not by itself
        # propagate the new range to the slider widget.
        n = int(self.image_stack.shape[0]) - 1
        with contextlib.suppress(AttributeError, IndexError, ValueError):
            self.second_viewer.dims.set_range(0, (0, n, 1))
            self.second_viewer.dims.set_current_step(0, n)
        self.image_layer_2nd.refresh()
        self.mask_layer_2nd.refresh()

    def _on_save_masks(self):
        if (
            self.mask_name is None
            or self.image_stack is None
            or self.mask_stack is None
        ):
            QMessageBox.information(
                self, "Nothing to save", "Add some masks first."
            )
            return
        if self.image_data_model is None:
            QMessageBox.warning(
                self,
                "No project",
                "No image_data_model is attached — cannot determine where "
                "to save.",
            )
            return

        # Pull the latest edits from the labels layer.
        if self.mask_layer_2nd is not None:
            with contextlib.suppress(AttributeError, RuntimeError):
                self.mask_stack = np.asarray(self.mask_layer_2nd.data)

        try:
            out_dir = self.image_data_model.save_masks(
                self.mask_name, self.image_stack, self.mask_stack
            )
        except (OSError, ValueError, RuntimeError) as e:
            QMessageBox.warning(
                self, "Save failed", f"{type(e).__name__}: {e}"
            )
            return

        QMessageBox.information(
            self,
            "Masks saved",
            f"Saved {self.image_stack.shape[0]} masks to:\n{out_dir}",
        )


def launch_edit_masks_viewer(
    source_viewer: napari.viewer.Viewer,
    image_data_model,
    image_layer,
    axes_to_collapse=None,
    default_mask_size: int = 256,
) -> napari.viewer.Viewer:
    """Open a second napari viewer with the Edit-Masks widget docked.

    Parameters
    ----------
    source_viewer
        The original napari viewer (where the user draws points).
    image_data_model
        The :class:`ImageDataModel` driving the project (used for save).
    image_layer
        The active image layer in ``source_viewer`` to crop patches from.
    axes_to_collapse
        Optional axis letters that the active app collapses for labels;
        retained for parity with other 2nd-viewer launchers.
    default_mask_size
        Initial value of the XY-size spinbox.

    Returns
    -------
    The new :class:`napari.viewer.Viewer` (kept alive by the caller).
    """
    second_viewer = napari.Viewer(title="Edit Masks")

    widget = EditMasksWidget(
        source_viewer=source_viewer,
        second_viewer=second_viewer,
        image_data_model=image_data_model,
        image_layer=image_layer,
        axes_to_collapse=axes_to_collapse,
        default_mask_size=default_mask_size,
    )
    second_viewer.window.add_dock_widget(widget, name="Edit Masks")
    return second_viewer
