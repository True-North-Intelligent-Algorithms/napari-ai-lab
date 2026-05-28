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

import logging
import sys
from collections.abc import Callable
from functools import partial

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
    QVBoxLayout,
    QWidget,
)
from skimage import future
from skimage.feature import multiscale_basic_features
from sklearn.ensemble import RandomForestClassifier
from superqt import ensure_main_thread

LOGGER = logging.getLogger("interactive_local_machine_learning")
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
# Widget
# ---------------------------------------------------------------------------
class NapariMLWidget(QWidget):
    """Settings + commit panel for the interactive local ML workflow."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._initUI()

    def _initUI(self):
        layout = QVBoxLayout()

        # Model
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Select Model"))
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["Random Forest"])
        model_row.addWidget(self.model_dropdown)
        layout.addLayout(model_row)

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
        layout.addLayout(sigma_row)

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
        layout.addWidget(feats_group)

        # Data choice
        data_row = QHBoxLayout()
        data_row.addWidget(QLabel("Select Data for Model Fitting"))
        self.data_dropdown = QComboBox()
        self.data_dropdown.addItems(
            ["Current Displayed Region", "Whole Image"]
        )
        self.data_dropdown.setCurrentText("Whole Image")
        data_row.addWidget(self.data_dropdown)
        layout.addLayout(data_row)

        # Live toggles (OFF by default — features are cached once and
        # prediction is triggered manually via the Predict button).
        self.live_fit_checkbox = QCheckBox("Live Model Fitting")
        self.live_fit_checkbox.setChecked(False)
        layout.addWidget(self.live_fit_checkbox)

        self.live_pred_checkbox = QCheckBox("Live Prediction")
        self.live_pred_checkbox.setChecked(False)
        layout.addWidget(self.live_pred_checkbox)

        # Predict button: fit + predict on demand using cached features.
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.setToolTip(
            "Fit the model on the currently painted voxels and predict "
            "the whole crop using pre-computed features."
        )
        layout.addWidget(self.predict_btn)

        # Recompute features (e.g. after changing sigma / feature toggles).
        self.recompute_features_btn = QPushButton("Recompute Features")
        self.recompute_features_btn.setToolTip(
            "Re-extract image features for the whole crop using the "
            "current sigma range and feature toggles."
        )
        layout.addWidget(self.recompute_features_btn)

        # Commit
        self.commit_prediction_btn = QPushButton("Commit Prediction")
        self.commit_prediction_btn.setToolTip(
            "Write the current prediction back into the source labels "
            "layer at the location the crop was taken from."
        )
        layout.addWidget(self.commit_prediction_btn)

        layout.addStretch(1)
        self.setLayout(layout)

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
        written into when the user presses "Commit Prediction".
    target_slice:
        Indexing tuple identifying *where in* ``target_labels_layer.data``
        the crop was taken from.  Must match the shape of ``image_crop``.
    on_commit:
        Optional extra callback invoked with the prediction array after a
        successful commit (e.g. to refresh a mirror layer in another viewer).
    """
    image_crop = np.asarray(image_crop)

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
    painting_data = np.zeros(image_crop.shape, dtype=np.int32)
    painting_layer = viewer.add_labels(
        painting_data, name="Painting", scale=data_layer.scale
    )

    # Prediction layer: 0 = background (transparent in napari).
    prediction_data = np.zeros(image_crop.shape, dtype=np.int32)
    prediction_layer = viewer.add_labels(
        prediction_data, name="Prediction", scale=data_layer.scale
    )

    widget = NapariMLWidget()
    viewer.window.add_dock_widget(widget, name="Local ML")

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
        nz = pred > 0
        if not nz.any():
            LOGGER.info("Commit: prediction layer is empty.")
            return
        if target_labels_layer is not None and target_slice is not None:
            try:
                target = target_labels_layer.data
                view = target[target_slice]
                if view.shape != pred.shape:
                    LOGGER.warning(
                        "Commit: target slice shape %s != prediction "
                        "shape %s; aborting.",
                        view.shape,
                        pred.shape,
                    )
                    return
                view[nz] = pred[nz]
                target_labels_layer.refresh()
                if persistent_layer is not None:
                    persistent_layer.refresh()
                LOGGER.info(
                    "Committed prediction into %s at %s",
                    target_labels_layer.name,
                    target_slice,
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
