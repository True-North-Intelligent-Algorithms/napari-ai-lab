"""Interactive local (pixel-classifier) machine learning plugin.

Adapted from the "halfway to I2K 2023 America" DIY-segmentation example
(scikit-learn Random Forest on `multiscale_basic_features`).

Two notable differences vs the original example:

1.  The painting layer is **offset by 1** so that the user *can* draw
    background:

        painting value 0  -> "untouched" (ignored by the model)
        painting value 1  -> background class  (class 0 for the model)
        painting value 2  -> foreground class 1 (class 1 for the model)
        painting value k  -> class k-1 for the model

2.  The prediction layer stores the **raw model class index** (0 = background,
    invisible because napari labels render 0 as transparent).  So the
    prediction layer is effectively ``painting - 1``.

A ``Commit Prediction`` button writes the current prediction back into a
user-supplied target labels layer at the slice the crop was taken from, so
the plugin can be used as an ROI-preview / commit workflow on top of
:class:`NDEasyLabel`.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import sys
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path

import napari
import numpy as np
import toolz as tz
from psygnal import debounced
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from skimage import future
from skimage.feature import multiscale_basic_features
from sklearn.ensemble import RandomForestClassifier
from superqt import ensure_main_thread

from napari_ai_lab.utilities import normalize_percentile

# MONAI / torch are optional — only required for the "MONAI UNet
# (local overfit)" mode of the widget.
try:
    import torch
    import torch.nn as nn
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import UNet

    _IS_MONAI_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    _IS_MONAI_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    sliding_window_inference = None  # type: ignore[assignment]
    UNet = None  # type: ignore[assignment]

LOGGER = logging.getLogger("interactive_local_learning")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    LOGGER.addHandler(_h)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(image, feature_params):
    features_func = partial(
        multiscale_basic_features,
        intensity=feature_params["intensity"],
        edges=feature_params["edges"],
        texture=feature_params["texture"],
        sigma_min=feature_params["sigma_min"],
        sigma_max=feature_params["sigma_max"],
        channel_axis=None,
    )
    return features_func(np.squeeze(image))


def update_model(labels, features, model_type):
    """Fit a classifier on the labelled voxels.

    ``labels`` here are already in the model space (0 = background, 1.. =
    foreground classes).  Voxels with no label (``labels < 0``) are not
    passed in — the caller is responsible for masking them out.
    """
    if model_type == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    LOGGER.info(
        "fitting %s with labels=%s features=%s unique=%s",
        model_type,
        labels.shape,
        features.shape,
        np.unique(labels),
    )
    clf.fit(features, labels)
    return clf


def predict(model, features):
    pred = future.predict_segmenter(
        features.reshape(-1, features.shape[-1]), model
    ).reshape(features.shape[:-1])
    return pred


# ---------------------------------------------------------------------------
# MONAI local-overfit helpers
# ---------------------------------------------------------------------------
# Number of strided down-sampling stages in the UNet — fixed at 4, so any
# patch dim must be a multiple of 2**4 = 16.
_MONAI_DEPTH = 4
_MONAI_DIVISOR = 2**_MONAI_DEPTH


def _pick_patch_size(
    crop_shape: tuple[int, ...],
    source_shape: tuple[int, ...] | None,
    *,
    lateral: int,
    depth: int,
) -> tuple[int, ...]:
    """Pick a per-axis patch size.

    Convention: for 3D data axis 0 is the depth axis (Z), the rest are
    lateral (Y, X).  For 2D data both axes are lateral.

    Each axis is rounded *down* to the nearest multiple of
    :data:`_MONAI_DIVISOR` (so the UNet can downsample 4 times) and
    capped at the corresponding source dim (or crop dim if no source is
    available) so a patch always fits inside real image data with no
    artificial padding.
    """

    def _round(n: int) -> int:
        return max(_MONAI_DIVISOR, (n // _MONAI_DIVISOR) * _MONAI_DIVISOR)

    if len(crop_shape) == 2:
        hints = (lateral, lateral)
    elif len(crop_shape) == 3:
        hints = (depth, lateral, lateral)
    else:
        raise ValueError(f"Unsupported crop ndim={len(crop_shape)}")

    out = []
    for ax, hint in enumerate(hints):
        cap = source_shape[ax] if source_shape is not None else crop_shape[ax]
        # Patch can be larger than the crop (will read from surrounding
        # source data) but must fit inside the available source.
        size = min(_round(hint), _round(cap))
        out.append(max(_MONAI_DIVISOR, size))
    return tuple(out)


def _compute_starts(
    crop_dim: int, patch: int, source_dim: int, crop_offset: int
) -> list[int]:
    """Per-axis tile start positions in *source* coordinates.

    Patches are sized exactly ``patch`` (no padding).  When the crop is
    larger than one patch, the n tile starts are evenly distributed
    across ``[crop_offset, crop_offset + crop_dim - patch]`` so adjacent
    tiles overlap a bit instead of running off the edge.  When the crop
    is smaller than one patch, a single patch is centred on the crop and
    clipped into ``[0, source_dim - patch]`` (so the patch reads
    surrounding source pixels rather than zeros).
    """
    if patch >= source_dim:
        # Patch larger than source axis: clamp.  Caller should already
        # have rounded the patch down via _pick_patch_size, so this is a
        # rare safety net.
        return [0]
    if crop_dim <= patch:
        center = crop_offset + crop_dim // 2
        start = max(0, min(center - patch // 2, source_dim - patch))
        return [start]
    n = (crop_dim + patch - 1) // patch  # ceil(crop / patch)
    if n == 1:
        return [max(0, min(crop_offset, source_dim - patch))]
    span = crop_dim - patch
    starts = [crop_offset + int(round(i * span / (n - 1))) for i in range(n)]
    starts = [max(0, min(s, source_dim - patch)) for s in starts]
    # De-duplicate while preserving order (could happen on tiny spans).
    seen: list[int] = []
    for s in starts:
        if s not in seen:
            seen.append(s)
    return seen


def _read_source_patch(
    source: np.ndarray | None,
    fallback: np.ndarray,
    fallback_offset: tuple[int, ...],
    src_slice: tuple[slice, ...],
) -> np.ndarray:
    """Read an image patch from ``source`` if available; otherwise fall
    back to ``fallback`` (the crop) using ``edge`` replication for any
    out-of-crop region — never zero-fill, so patch normalization stays
    representative."""
    if source is not None:
        return np.asarray(source[src_slice])
    # Translate src_slice (source coords) into fallback (crop) coords.
    crop_idx = []
    pads = []
    for sl, off, fdim in zip(
        src_slice, fallback_offset, fallback.shape, strict=False
    ):
        lo = sl.start - off
        hi = sl.stop - off
        pad_lo = max(0, -lo)
        pad_hi = max(0, hi - fdim)
        crop_idx.append(slice(max(0, lo), min(fdim, hi)))
        pads.append((pad_lo, pad_hi))
    inner = fallback[tuple(crop_idx)]
    if any(p != (0, 0) for p in pads):
        return np.pad(inner, pads, mode="edge")
    return inner


def _extract_patch_set(
    crop_image: np.ndarray,
    crop_labels: np.ndarray,
    *,
    source_image: np.ndarray | None,
    crop_offset: tuple[int, ...],
    source_shape: tuple[int, ...],
    patch_size: tuple[int, ...],
    temp_dir: Path,
    normalize: bool = True,
) -> tuple[list[tuple[Path, Path]], list[tuple[int, ...]], tuple[int, ...]]:
    """Build a patch set covering the crop, reading image data from the
    source image (so patches never contain artificial zero borders).

    Labels are placed only where the patch overlaps the crop region;
    voxels outside the crop are set to ``-1`` (ignored by the loss).

    Returns ``(saved_paths, patch_starts, union_shape)`` where
    ``union_shape`` is the bounding-box size of all patch starts taken
    together (used as the inference-image size).
    """
    temp_dir.mkdir(parents=True, exist_ok=True)

    starts_per_axis = [
        _compute_starts(
            crop_image.shape[ax],
            patch_size[ax],
            source_shape[ax],
            crop_offset[ax],
        )
        for ax in range(crop_image.ndim)
    ]

    # Cartesian product of per-axis starts.
    def _product(axes_idx: int, acc: tuple[int, ...]):
        if axes_idx == crop_image.ndim:
            yield acc
            return
        for s in starts_per_axis[axes_idx]:
            yield from _product(axes_idx + 1, acc + (s,))

    saved: list[tuple[Path, Path]] = []
    starts_list: list[tuple[int, ...]] = []

    for starts in _product(0, ()):
        src_sl = tuple(
            slice(starts[ax], starts[ax] + patch_size[ax])
            for ax in range(crop_image.ndim)
        )
        img_patch = _read_source_patch(
            source_image, crop_image, crop_offset, src_sl
        ).astype(np.float32)
        if normalize:
            img_patch = normalize_percentile(img_patch)

        # Build the label patch: -1 everywhere except where the patch
        # overlaps the crop region (where we copy from crop_labels).
        lab_patch = np.full(patch_size, -1, dtype=np.int64)
        patch_local: list[slice] = []
        crop_local: list[slice] = []
        skip = False
        for ax in range(crop_image.ndim):
            ps = starts[ax]
            pe = ps + patch_size[ax]
            cs = crop_offset[ax]
            ce = cs + crop_image.shape[ax]
            ov_lo = max(ps, cs)
            ov_hi = min(pe, ce)
            if ov_hi <= ov_lo:
                skip = True
                break
            patch_local.append(slice(ov_lo - ps, ov_hi - ps))
            crop_local.append(slice(ov_lo - cs, ov_hi - cs))
        if not skip:
            lab_patch[tuple(patch_local)] = crop_labels[tuple(crop_local)]

        n = len(saved)
        img_p = temp_dir / f"img_{n:04d}.npy"
        lab_p = temp_dir / f"lab_{n:04d}.npy"
        np.save(img_p, img_patch)
        np.save(lab_p, lab_patch)
        saved.append((img_p, lab_p))
        starts_list.append(starts)

    # Union bounding box of all patches in source coords.
    if not starts_list:
        union_shape = patch_size
    else:
        mins = [
            min(starts[ax] for starts in starts_list)
            for ax in range(crop_image.ndim)
        ]
        maxs = [
            max(starts[ax] + patch_size[ax] for starts in starts_list)
            for ax in range(crop_image.ndim)
        ]
        union_shape = tuple(
            maxs[ax] - mins[ax] for ax in range(crop_image.ndim)
        )

    return saved, starts_list, union_shape


def _train_monai_local(
    image_crop: np.ndarray,
    painting: np.ndarray,
    *,
    epochs: int,
    patch_lateral: int,
    patch_depth: int,
    temp_dir: Path,
    source_image: np.ndarray | None = None,
    crop_offset: tuple[int, ...] | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Quickly overfit a tiny MONAI UNet to ``painting`` on ``image_crop``
    and return a per-voxel argmax prediction the same shape as
    ``image_crop`` (after squeezing singleton dims).

    Patches are read from ``source_image`` (the un-cropped image the
    crop was taken from) when supplied, so they never contain artificial
    zero borders — which matters for normalization.

    ``painting`` follows the local-ML convention:
        0 = untouched (ignored by the loss)
        1 = background class (-> 0 in model space)
        k = class k-1
    """
    if not _IS_MONAI_AVAILABLE:
        raise RuntimeError(
            "MONAI / torch not available — install monai + torch to use "
            "the local-overfit mode."
        )

    def _log(msg: str):
        LOGGER.info(msg)
        if progress_cb is not None:
            with contextlib.suppress(Exception):
                progress_cb(msg)

    img = np.squeeze(np.asarray(image_crop))
    paint = np.squeeze(np.asarray(painting))
    if img.shape != paint.shape:
        raise ValueError(
            f"image shape {img.shape} != painting shape {paint.shape}"
        )

    spatial_dims = img.ndim
    if spatial_dims not in (2, 3):
        raise ValueError(
            "Local MONAI overfit needs 2D or 3D data (after squeeze); "
            f"got {spatial_dims}D shape={img.shape}"
        )

    # Map painting -> model labels: untouched=-1, bg=0, class k -> k-1
    labels = paint.astype(np.int64) - 1  # 0->-1, 1->0, 2->1, ...
    num_classes = int(max(labels.max(), 0)) + 1
    if num_classes < 2:
        raise ValueError(
            "Need at least two distinct painted classes (e.g. background "
            "+ one foreground). Paint with the background brush first."
        )

    # Squeeze the source image to match the spatial dims of ``img``.
    src = None
    if source_image is not None:
        src = np.squeeze(np.asarray(source_image))
        if src.ndim != spatial_dims:
            _log(
                f"MONAI: source ndim {src.ndim} != spatial_dims "
                f"{spatial_dims}; ignoring source image."
            )
            src = None

    source_shape = src.shape if src is not None else img.shape

    # Crop offset (in source coords) for every spatial axis.
    if crop_offset is None or len(crop_offset) != spatial_dims:
        offset = (0,) * spatial_dims
    else:
        offset = tuple(int(o) for o in crop_offset)

    patch_size = _pick_patch_size(
        img.shape,
        source_shape if src is not None else None,
        lateral=patch_lateral,
        depth=patch_depth,
    )
    _log(
        f"MONAI: spatial_dims={spatial_dims} crop={img.shape} "
        f"crop_offset={offset} source={source_shape if src is not None else 'N/A'} "
        f"patch_size={patch_size} num_classes={num_classes}"
    )

    patches, starts_list, _union = _extract_patch_set(
        img,
        labels,
        source_image=src,
        crop_offset=offset,
        source_shape=source_shape,
        patch_size=patch_size,
        temp_dir=temp_dir,
        normalize=True,
    )
    _log(f"MONAI: {len(patches)} patches written to {temp_dir}")

    # Build a tiny UNet (depth 4).
    channels = (16, 32, 64, 128, 256)
    strides = (2, 2, 2, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=1,
        out_channels=num_classes,
        channels=channels,
        strides=strides,
        num_res_units=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    # Pre-load patches into tensors (the set is small by design).
    img_tensors = []
    lab_tensors = []
    for img_p, lab_p in patches:
        ip = np.load(img_p)[None, None, ...]  # (1, 1, *spatial)
        lp = np.load(lab_p)[None, ...]  # (1, *spatial)
        img_tensors.append(torch.from_numpy(ip).float().to(device))
        lab_tensors.append(torch.from_numpy(lp).long().to(device))

    model.train()
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        order = rng.permutation(len(patches))
        epoch_loss = 0.0
        n_steps = 0
        for i in order:
            lab_t = lab_tensors[i]
            # Skip patches with no labelled voxels at all.
            if (lab_t >= 0).sum().item() == 0:
                continue
            optimizer.zero_grad()
            logits = model(img_tensors[i])
            loss = loss_fn(logits, lab_t)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_steps += 1
        if (epoch + 1) % max(1, epochs // 10) == 0 and n_steps > 0:
            _log(
                f"MONAI: epoch {epoch + 1}/{epochs}  "
                f"loss={epoch_loss / n_steps:.4f}"
            )

    # ---- Inference ----
    # Build an inference image covering the union of all patch starts
    # (same source-only reads, edge-replicate fallback if no source).
    if starts_list:
        inf_lo = tuple(
            min(s[ax] for s in starts_list) for ax in range(spatial_dims)
        )
        inf_hi = tuple(
            max(s[ax] + patch_size[ax] for s in starts_list)
            for ax in range(spatial_dims)
        )
    else:
        inf_lo = offset
        inf_hi = tuple(
            offset[ax] + img.shape[ax] for ax in range(spatial_dims)
        )

    inf_sl = tuple(slice(inf_lo[ax], inf_hi[ax]) for ax in range(spatial_dims))
    inf_image = _read_source_patch(src, img, offset, inf_sl).astype(np.float32)
    inf_image = normalize_percentile(inf_image)

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(inf_image[None, None, ...]).float().to(device)
        logits = sliding_window_inference(
            x, roi_size=patch_size, sw_batch_size=1, predictor=model
        )
        pred_full = logits.argmax(dim=1).cpu().numpy()[0]

    # Extract the crop region from the inference output.
    crop_in_inf = tuple(
        slice(offset[ax] - inf_lo[ax], offset[ax] - inf_lo[ax] + img.shape[ax])
        for ax in range(spatial_dims)
    )
    pred = pred_full[crop_in_inf]

    # Restore singleton dims so the caller can write straight into the
    # prediction layer (which still has the original ``image_crop`` shape).
    return pred.reshape(np.asarray(image_crop).shape)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class NapariMLWidget(QWidget):
    """Settings + commit panel for the interactive local ML workflow."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._initUI()

    def _initUI(self):
        layout = QVBoxLayout()

        # Mode selector — switches between feature-based RF and a local
        # MONAI UNet overfit. The two control groups below are shown /
        # hidden based on the selection.
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        self.mode_dropdown = QComboBox()
        self.mode_dropdown.addItems(
            [
                "Random Forest (features)",
                "MONAI UNet (local overfit)",
            ]
        )
        mode_row.addWidget(self.mode_dropdown)
        layout.addLayout(mode_row)

        # ---- Random Forest group ----
        self.rf_group = QGroupBox("Random Forest")
        rf_layout = QVBoxLayout()

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Select Model"))
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["Random Forest"])
        model_row.addWidget(self.model_dropdown)
        rf_layout.addLayout(model_row)

        # Sigma range
        self.sigma_start_spinbox = QDoubleSpinBox()
        self.sigma_start_spinbox.setRange(0, 256)
        self.sigma_start_spinbox.setValue(1)
        self.sigma_end_spinbox = QDoubleSpinBox()
        self.sigma_end_spinbox.setRange(0, 256)
        self.sigma_end_spinbox.setValue(5)
        sigma_row = QHBoxLayout()
        sigma_row.addWidget(QLabel("Sigma Range: From"))
        sigma_row.addWidget(self.sigma_start_spinbox)
        sigma_row.addWidget(QLabel("To"))
        sigma_row.addWidget(self.sigma_end_spinbox)
        rf_layout.addLayout(sigma_row)

        # Feature toggles
        self.intensity_checkbox = QCheckBox("Intensity")
        self.intensity_checkbox.setChecked(True)
        self.edges_checkbox = QCheckBox("Edges")
        self.texture_checkbox = QCheckBox("Texture")
        self.texture_checkbox.setChecked(True)
        feats_group = QGroupBox("Features")
        feats_layout = QVBoxLayout()
        feats_layout.addWidget(self.intensity_checkbox)
        feats_layout.addWidget(self.edges_checkbox)
        feats_layout.addWidget(self.texture_checkbox)
        feats_group.setLayout(feats_layout)
        rf_layout.addWidget(feats_group)

        # Data choice
        data_row = QHBoxLayout()
        data_row.addWidget(QLabel("Select Data for Model Fitting"))
        self.data_dropdown = QComboBox()
        self.data_dropdown.addItems(
            ["Current Displayed Region", "Whole Image"]
        )
        self.data_dropdown.setCurrentText("Whole Image")
        data_row.addWidget(self.data_dropdown)
        rf_layout.addLayout(data_row)

        # Live toggles (OFF by default — features are cached once and
        # prediction is triggered manually via the Predict button).
        self.live_fit_checkbox = QCheckBox("Live Model Fitting")
        self.live_fit_checkbox.setChecked(False)
        rf_layout.addWidget(self.live_fit_checkbox)

        self.live_pred_checkbox = QCheckBox("Live Prediction")
        self.live_pred_checkbox.setChecked(False)
        rf_layout.addWidget(self.live_pred_checkbox)

        # Predict button: fit + predict on demand using cached features.
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.setToolTip(
            "Fit the model on the currently painted voxels and predict "
            "the whole crop using pre-computed features."
        )
        rf_layout.addWidget(self.predict_btn)

        # Recompute features (e.g. after changing sigma / feature toggles).
        self.recompute_features_btn = QPushButton("Recompute Features")
        self.recompute_features_btn.setToolTip(
            "Re-extract image features for the whole crop using the "
            "current sigma range and feature toggles."
        )
        rf_layout.addWidget(self.recompute_features_btn)

        self.rf_group.setLayout(rf_layout)
        layout.addWidget(self.rf_group)

        # ---- MONAI group ----
        self.monai_group = QGroupBox("MONAI UNet (local overfit)")
        monai_layout = QVBoxLayout()

        patch_row = QHBoxLayout()
        patch_row.addWidget(QLabel("Patch lateral (Y/X)"))
        self.monai_patch_lat_spinbox = QSpinBox()
        self.monai_patch_lat_spinbox.setRange(_MONAI_DIVISOR, 4096)
        self.monai_patch_lat_spinbox.setSingleStep(_MONAI_DIVISOR)
        self.monai_patch_lat_spinbox.setValue(256)
        patch_row.addWidget(self.monai_patch_lat_spinbox)
        patch_row.addWidget(QLabel("Depth (Z)"))
        self.monai_patch_z_spinbox = QSpinBox()
        self.monai_patch_z_spinbox.setRange(_MONAI_DIVISOR, 4096)
        self.monai_patch_z_spinbox.setSingleStep(_MONAI_DIVISOR)
        self.monai_patch_z_spinbox.setValue(64)
        patch_row.addWidget(self.monai_patch_z_spinbox)
        monai_layout.addLayout(patch_row)

        epoch_row = QHBoxLayout()
        epoch_row.addWidget(QLabel("Epochs"))
        self.monai_epochs_spinbox = QSpinBox()
        self.monai_epochs_spinbox.setRange(1, 10000)
        self.monai_epochs_spinbox.setValue(100)
        epoch_row.addWidget(self.monai_epochs_spinbox)
        monai_layout.addLayout(epoch_row)

        self.monai_apply_btn = QPushButton("Train + Apply")
        self.monai_apply_btn.setToolTip(
            "Build a tiny temporary patch set covering the ROI, train a "
            "small MONAI UNet on the painted voxels (sparse loss), and "
            "predict the whole crop."
        )
        monai_layout.addWidget(self.monai_apply_btn)

        if not _IS_MONAI_AVAILABLE:
            warn = QLabel(
                "MONAI / torch not available — install them to enable "
                "this mode."
            )
            warn.setStyleSheet("color: #c00;")
            monai_layout.addWidget(warn)
            self.monai_apply_btn.setEnabled(False)

        self.monai_group.setLayout(monai_layout)
        layout.addWidget(self.monai_group)

        # ---- Shared commit ----
        commit_row = QHBoxLayout()
        commit_row.addWidget(QLabel("Commit to"))
        self.commit_target_dropdown = QComboBox()
        commit_row.addWidget(self.commit_target_dropdown)
        layout.addLayout(commit_row)

        self.commit_prediction_btn = QPushButton("Commit Prediction")
        self.commit_prediction_btn.setToolTip(
            "Replace the contents of the selected labels layer at the "
            "crop location with the current prediction."
        )
        layout.addWidget(self.commit_prediction_btn)

        layout.addStretch(1)
        self.setLayout(layout)

        # Hook up mode switch + initial visibility.
        self.mode_dropdown.currentTextChanged.connect(self._on_mode_changed)
        self._on_mode_changed(self.mode_dropdown.currentText())

    def _on_mode_changed(self, mode: str):
        is_rf = mode.startswith("Random Forest")
        self.rf_group.setVisible(is_rf)
        self.monai_group.setVisible(not is_rf)

    def feature_params(self):
        return {
            "sigma_min": self.sigma_start_spinbox.value(),
            "sigma_max": self.sigma_end_spinbox.value(),
            "intensity": self.intensity_checkbox.isChecked(),
            "edges": self.edges_checkbox.isChecked(),
            "texture": self.texture_checkbox.isChecked(),
        }


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------
def launch_interactive_local_ml(
    image_crop: np.ndarray,
    *,
    scale=None,
    contrast_limits=None,
    target_labels_layer: napari.layers.Labels | None = None,
    target_slice: tuple | None = None,
    commit_targets: dict | None = None,
    source_image: np.ndarray | None = None,
    source_slice: tuple | None = None,
    painting_data: np.ndarray | None = None,
    painting_name: str = "Painting",
    extra_mirror_layers: list[tuple] | None = None,
    title: str = "Interactive Local Machine Learning",
    on_commit: Callable[[np.ndarray], None] | None = None,
):
    """Open a new napari viewer with the local-ML plugin loaded on ``image_crop``.

    Parameters
    ----------
    image_crop:
        The (typically 3D) image to work on.  The crop is loaded into the
        new viewer; painting and prediction layers are created in-memory.
    scale, contrast_limits:
        Forwarded to ``viewer.add_image``.
    target_labels_layer:
        Labels layer (in the *original* viewer) the prediction should be
        written into when the user presses "Commit Prediction".  A sliced
        *view* of this layer at ``target_slice`` is added to the new
        viewer and refreshed whenever the source layer changes.
    target_slice:
        Indexing tuple identifying *where in* ``target_labels_layer.data``
        the crop was taken from.  Must match the shape of ``image_crop``.
    commit_targets:
        Optional ``dict`` of ``name -> (labels_layer, slice)`` describing
        every annotation layer the user can commit the prediction into.
        Populates the "Commit to" dropdown in the widget; the entry whose
        layer is ``target_labels_layer`` is selected by default.  If
        omitted, only ``target_labels_layer`` is offered.
    source_image, source_slice:
        Optional un-cropped image data the crop was taken from and the
        indexing tuple identifying the crop location in source coords.
        Used by the MONAI local-overfit mode so border patches read real
        surrounding image data instead of zero-filled padding (which
        would distort percentile normalization).
    painting_data:
        Optional pre-built array to use as the painting layer's data
        (typically a numpy *view* into a "sparse labels" layer on the
        original viewer, so edits propagate back).  If ``None``, a fresh
        zero array of ``image_crop.shape`` is created.
    painting_name:
        Name for the painting layer in the new viewer.
    extra_mirror_layers:
        Optional list of ``(source_layer, slice)`` tuples — each
        ``source_layer.data[slice]`` is added to the new viewer as a
        sliced view layer and refreshed whenever the source layer
        changes.  Used e.g. to mirror the main viewer's "Labels (Working)"
        scratch layer so segmenter strokes in the original viewer
        propagate live into the crop view.
    on_commit:
        Optional extra callback invoked with the prediction array after a
        successful commit (e.g. to refresh a mirror layer in another viewer).
    """
    image_crop = np.asarray(image_crop)

    # Compute the crop offset (in source coords) for the MONAI patch
    # extractor.  Only axes that survive the crop (i.e. indexed by a
    # ``slice``) are kept; integer indices drop their axis from the crop
    # and so should be skipped in the offset tuple.
    source_offset: tuple[int, ...] | None = None
    reduced_source: np.ndarray | None = None
    if source_slice is not None:
        offsets = []
        for s in source_slice:
            if isinstance(s, slice):
                offsets.append(int(s.start) if s.start is not None else 0)
            # ints / Ellipsis / None: axis is dropped from the crop
        source_offset = tuple(offsets)
        # Build a source view aligned with the crop's surviving spatial
        # axes: replace each slice with a full ``slice(None)`` (so we
        # span the entire source on cropped axes) but keep int indices
        # so the same axes get dropped.
        if source_image is not None:
            reduced_idx = tuple(
                slice(None) if isinstance(s, slice) else s
                for s in source_slice
            )
            with contextlib.suppress(TypeError, ValueError, IndexError):
                reduced_source = np.asarray(source_image)[reduced_idx]

    viewer = napari.Viewer(title=title)

    data_layer = viewer.add_image(
        image_crop, scale=scale, contrast_limits=contrast_limits
    )

    # Sliced view (not a copy) of the persistent labels at the crop location,
    # so commits show up here *and* in the original viewer automatically.
    persistent_layer = None
    if target_labels_layer is not None and target_slice is not None:
        try:
            persistent_view = target_labels_layer.data[target_slice]
            if persistent_view.shape == image_crop.shape:
                persistent_layer = viewer.add_labels(
                    persistent_view,
                    name=f"{target_labels_layer.name} (crop view)",
                    scale=data_layer.scale,
                )
            else:
                LOGGER.warning(
                    "Persistent labels slice shape %s != image shape %s; "
                    "skipping crop-view layer.",
                    persistent_view.shape,
                    image_crop.shape,
                )
        except (TypeError, ValueError, IndexError) as e:
            LOGGER.warning(
                "Could not create persistent-labels crop view: %s", e
            )

    # Painting layer: offset by 1 so 1 is drawable background.
    # If the caller passed in a pre-built array (e.g. a numpy view into a
    # "sparse labels" layer on the original viewer), use that directly so
    # paint strokes propagate back; otherwise start from zeros.
    if painting_data is None:
        painting_data = np.zeros(image_crop.shape, dtype=np.int32)
    elif painting_data.shape != image_crop.shape:
        LOGGER.warning(
            "painting_data shape %s != image_crop shape %s; falling back "
            "to a fresh zero array.",
            painting_data.shape,
            image_crop.shape,
        )
        painting_data = np.zeros(image_crop.shape, dtype=np.int32)
    painting_layer = viewer.add_labels(
        painting_data, name=painting_name, scale=data_layer.scale
    )

    # Prediction layer: 0 = background (transparent in napari).
    prediction_data = np.zeros(image_crop.shape, dtype=np.int32)
    prediction_layer = viewer.add_labels(
        prediction_data, name="Prediction", scale=data_layer.scale
    )

    # Add any extra mirror layers (e.g. the main viewer's working layer).
    # Stored as (source_layer, mirror_layer) pairs so we can refresh
    # mirrors when the source emits paint/set_data/refresh events.
    mirror_pairs: list[tuple] = []
    if persistent_layer is not None:
        mirror_pairs.append((target_labels_layer, persistent_layer))
    if extra_mirror_layers:
        for src_layer, src_slice in extra_mirror_layers:
            if src_layer is None or src_slice is None:
                continue
            try:
                mirror_view = src_layer.data[src_slice]
            except (TypeError, ValueError, IndexError) as e:
                LOGGER.warning(
                    "Could not slice %s for mirror view: %s",
                    getattr(src_layer, "name", "?"),
                    e,
                )
                continue
            if mirror_view.shape != image_crop.shape:
                LOGGER.warning(
                    "Mirror layer %s slice shape %s != image shape %s; "
                    "skipping.",
                    getattr(src_layer, "name", "?"),
                    mirror_view.shape,
                    image_crop.shape,
                )
                continue
            mirror = viewer.add_labels(
                mirror_view,
                name=f"{src_layer.name} (crop view)",
                scale=data_layer.scale,
            )
            mirror_pairs.append((src_layer, mirror))

    # Forward refresh events from each source layer to its mirror so any
    # changes made in the original viewer (paint, fill, segmenter writes,
    # bulk set_data, .refresh() after a direct buffer write) show up in
    # the local-ML viewer immediately.
    def _make_forwarder(mirror_layer):
        def _forward(_event=None):
            with contextlib.suppress(RuntimeError, AttributeError):
                mirror_layer.refresh()

        return _forward

    for src_layer, mirror in mirror_pairs:
        cb = _make_forwarder(mirror)
        for evt_name in ("paint", "set_data", "refresh"):
            with contextlib.suppress(AttributeError, TypeError):
                getattr(src_layer.events, evt_name).connect(cb)

    widget = NapariMLWidget()
    viewer.window.add_dock_widget(widget, name="Local ML")

    # ------- commit-target dropdown -------
    # Build the {name: (layer, slice)} map shown in the "Commit to"
    # combo.  Falls back to the (target_labels_layer, target_slice)
    # passed in directly if no explicit map was provided.
    commit_map: dict[str, tuple] = {}
    if commit_targets:
        for name, ls in commit_targets.items():
            if ls is None:
                continue
            layer, sl = ls
            if layer is None or sl is None:
                continue
            commit_map[name] = (layer, sl)
    if (
        not commit_map
        and target_labels_layer is not None
        and target_slice is not None
    ):
        commit_map[target_labels_layer.name] = (
            target_labels_layer,
            target_slice,
        )

    for name in commit_map:
        widget.commit_target_dropdown.addItem(name)
    if (
        target_labels_layer is not None
        and target_labels_layer.name in commit_map
    ):
        widget.commit_target_dropdown.setCurrentText(target_labels_layer.name)

    # ------- cached features for the whole crop -------
    # Features are *expensive*; compute once up-front and re-use them for
    # every fit/predict.  Recomputed on demand when the user changes the
    # sigma range / feature toggles and clicks "Recompute Features".
    state = {"model": None, "features": None}

    def _compute_features_now():
        params = widget.feature_params()
        LOGGER.info(
            "Computing features for crop shape=%s params=%s",
            image_crop.shape,
            params,
        )
        try:
            state["features"] = extract_features(image_crop, params)
            LOGGER.info("Features shape: %s", state["features"].shape)
        except (ValueError, RuntimeError) as e:
            LOGGER.warning("Feature extraction failed: %s", e)
            state["features"] = None

    # Initial feature extraction (synchronous; user pressed the launch
    # button so they're expecting a short wait here).
    _compute_features_now()

    def _fit_and_predict():
        features = state["features"]
        if features is None:
            LOGGER.warning("No cached features; press 'Recompute Features'.")
            return
        paint = np.squeeze(painting_layer.data)
        labelled = paint > 0
        if not labelled.any():
            LOGGER.info("No painted voxels — nothing to fit.")
            return
        train_labels = paint[labelled].astype(np.int32) - 1
        train_features = features[labelled, :]
        try:
            state["model"] = update_model(
                train_labels,
                train_features,
                widget.model_dropdown.currentText(),
            )
        except (ValueError, RuntimeError) as e:
            LOGGER.warning("Model fit failed: %s", e)
            return
        try:
            pred = predict(state["model"], features)
        except (ValueError, RuntimeError) as e:
            LOGGER.warning("Prediction failed: %s", e)
            return
        # Reshape pred back into the layer's full shape if needed
        # (extract_features squeezes singleton dims).
        out = prediction_layer.data
        if pred.shape == out.shape:
            out[...] = pred
        else:
            out[...] = pred.reshape(out.shape)
        prediction_layer.refresh()

    widget.predict_btn.clicked.connect(_fit_and_predict)
    widget.recompute_features_btn.clicked.connect(_compute_features_now)

    # ------- MONAI local-overfit Apply -------
    # Each launcher session gets its own temp directory for the patch set
    # and trained model snapshot; cleaned up when the viewer window closes.
    monai_temp_dir = Path(
        tempfile.mkdtemp(prefix="napari_ai_lab_local_monai_")
    )
    LOGGER.info("MONAI temp dir: %s", monai_temp_dir)

    def _cleanup_temp():
        with contextlib.suppress(Exception):
            shutil.rmtree(monai_temp_dir, ignore_errors=True)
            LOGGER.info("MONAI temp dir cleaned: %s", monai_temp_dir)

    with contextlib.suppress(AttributeError, RuntimeError):
        viewer.window._qt_window.destroyed.connect(lambda *_: _cleanup_temp())

    def _train_and_apply_monai():
        if not _IS_MONAI_AVAILABLE:
            LOGGER.warning("MONAI / torch not available.")
            return
        # Use a fresh sub-dir per run so old patches don't accumulate.
        run_dir = monai_temp_dir / f"run_{np.random.randint(1_000_000):06d}"
        try:
            pred = _train_monai_local(
                image_crop,
                np.asarray(painting_layer.data),
                epochs=int(widget.monai_epochs_spinbox.value()),
                patch_lateral=int(widget.monai_patch_lat_spinbox.value()),
                patch_depth=int(widget.monai_patch_z_spinbox.value()),
                temp_dir=run_dir,
                source_image=reduced_source,
                crop_offset=source_offset,
            )
        except (ValueError, RuntimeError) as e:
            LOGGER.warning("MONAI train+apply failed: %s", e)
            return
        out = prediction_layer.data
        if pred.shape == out.shape:
            out[...] = pred
        else:
            out[...] = pred.reshape(out.shape)
        prediction_layer.refresh()
        LOGGER.info(
            "MONAI prediction written to layer %s", prediction_layer.name
        )

    widget.monai_apply_btn.clicked.connect(_train_and_apply_monai)

    # Live fit/predict on paint, only if the user opts in.  No camera/dims
    # listeners — those caused expensive refits on every rotate/zoom.
    @tz.curry
    def _on_paint(event, widget=None):
        if not (
            widget.live_fit_checkbox.isChecked()
            or widget.live_pred_checkbox.isChecked()
        ):
            return
        _fit_and_predict()

    painting_layer.events.paint.connect(
        debounced(
            ensure_main_thread(_on_paint(widget=widget)),
            timeout=500,
        )
    )

    # ------- commit -------
    def _commit_prediction():
        pred = np.asarray(prediction_layer.data)
        if pred.size == 0:
            LOGGER.info("Commit: prediction layer is empty.")
            return
        chosen = widget.commit_target_dropdown.currentText()
        entry = commit_map.get(chosen)
        if entry is None:
            LOGGER.warning(
                "Commit: no commit target selected (chosen=%r).", chosen
            )
            return
        commit_layer, commit_slice = entry
        try:
            target = commit_layer.data
            view = target[commit_slice]
            if view.shape != pred.shape:
                LOGGER.warning(
                    "Commit: target slice shape %s != prediction shape "
                    "%s; aborting.",
                    view.shape,
                    pred.shape,
                )
                return
            # Replace the whole crop region (including zeros) so the
            # committed labels match the prediction exactly.
            view[...] = pred
            commit_layer.refresh()
            # Refresh any mirror views of this layer in the local viewer.
            for src_layer, mirror in mirror_pairs:
                if src_layer is commit_layer:
                    with contextlib.suppress(RuntimeError, AttributeError):
                        mirror.refresh()
            LOGGER.info(
                "Committed prediction into %s at %s",
                commit_layer.name,
                commit_slice,
            )
        except (TypeError, ValueError, IndexError) as e:
            LOGGER.warning("Commit failed: %s", e)
            return
        if on_commit is not None:
            try:
                on_commit(pred)
            except (TypeError, ValueError, RuntimeError) as e:
                LOGGER.warning("on_commit callback failed: %s", e)

    widget.commit_prediction_btn.clicked.connect(_commit_prediction)

    return viewer, widget
