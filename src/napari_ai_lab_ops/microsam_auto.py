"""micro_sam automatic instance segmentation, and finetuning it.

The global (whole-image) counterpart of ``microsam_prompt``: no clicks,
every object at once. With a model that has micro_sam's instance decoder
(``vit_b_lm``, ``vit_l_lm``, ``vit_b_em_organelles`` and anything trained
here) that is AIS, fast; plain ``vit_b`` has none and falls back to SAM's
automatic mask generator, which is slow.

Training follows scikit-ops' training ops (design 0011): paths in, a path
out. What comes back is micro_sam's own layout,
``<model_dir>/checkpoints/<name>/best.pt``, a **file inside a directory
named for the model** -- neither Cellpose's bare file nor StarDist's
directory of weights. Beside it this writes ``skop_microsam.json``
recording which micro_sam model it was finetuned from, because loading the
checkpoint needs that ``model_type`` and the checkpoint does not say.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
from opspec.op import op
from opspec.runtime import cancel_requested, progress
from opspec.types import ImageOf, LabelsOf

#: The file, beside a trained best.pt, saying what it was finetuned from.
META = "skop_microsam.json"

#: One loaded model: (model_type, checkpoint, mtime, tiled) -> pair.
_loaded: dict = {}


def _ndim(shape) -> int:
    """2 for YX or YXC, 3 for ZYX -- ai-lab's get_ndim rule."""
    if len(shape) == 3 and shape[-1] < 5:
        return 2
    return len(shape)


def checkpoint_model_type(checkpoint: str, fallback: str) -> str:
    """The micro_sam model a trained checkpoint was finetuned from."""
    meta = Path(checkpoint).parent / META
    try:
        return json.loads(meta.read_text())["model_type"]
    except (OSError, ValueError, KeyError):
        return fallback


def _predictor_and_segmenter(model: str, checkpoint: str, tiled: bool):
    checkpoint = checkpoint or None
    mtime = Path(checkpoint).stat().st_mtime if checkpoint else None
    key = (model, checkpoint, mtime, tiled)
    if key not in _loaded:
        from micro_sam.automatic_segmentation import (
            get_predictor_and_segmenter,
        )

        _loaded.clear()
        progress(f"Loading micro_sam {checkpoint or model}")
        _loaded[key] = get_predictor_and_segmenter(
            model_type=model, checkpoint=checkpoint, is_tiled=tiled
        )
    return _loaded[key]


@op(env="pytorch")
def microsam_segment(
    image: ImageOf[np.ndarray],
    model: str = "vit_b_lm",
    checkpoint: str = "",
    tile: bool = False,
    tile_size: Annotated[
        int, {"widget_type": "SpinBox", "min": 64, "max": 4096, "step": 32}
    ] = 384,
    halo: Annotated[
        int, {"widget_type": "SpinBox", "min": 0, "max": 512}
    ] = 64,
    foreground_threshold: float = 0.5,
    center_distance_threshold: float = 0.5,
    boundary_distance_threshold: float = 0.5,
) -> LabelsOf[np.ndarray]:
    """Find every object in *image* with micro_sam.

    Args:
        image: YX, YXC or ZYX.
        model: micro_sam model, e.g. ``vit_b_lm``. With *checkpoint*, the one
            it was finetuned from, unless the checkpoint's ``skop_microsam.json``
            says otherwise.
        checkpoint: A ``best.pt`` from ``train_microsam``, or empty for the
            built-in *model*.
        tile: Embed in tiles, for images too large for one pass.
        tile_size, halo: Tile edge and overlap, in pixels.
        foreground_threshold, center_distance_threshold,
            boundary_distance_threshold: The instance decoder's settings.
            Ignored by models without one.

    Returns:
        A label image shaped like the image's spatial axes.
    """
    if checkpoint:
        model = checkpoint_model_type(checkpoint, model)
    predictor, segmenter = _predictor_and_segmenter(model, checkpoint, tile)

    from micro_sam.automatic_segmentation import (
        automatic_instance_segmentation,
    )

    progress("Segmenting")
    labels = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=np.asarray(image),
        ndim=_ndim(image.shape),
        tile_shape=(tile_size, tile_size) if tile else None,
        halo=(halo, halo) if tile else None,
        verbose=False,
        foreground_threshold=foreground_threshold,
        center_distance_threshold=center_distance_threshold,
        boundary_distance_threshold=boundary_distance_threshold,
    )
    labels = np.asarray(labels)
    progress(f"Found {len(np.unique(labels)) - 1} objects")
    return labels.astype(np.uint32)


def _to_8bit(image: np.ndarray) -> np.ndarray:
    """micro_sam trains on uint8. Grey only: a trailing RGB axis is averaged."""
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3].mean(axis=-1)
    if image.dtype == np.uint8:
        return image
    low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    if image.dtype.kind == "f" and low >= 0 and high <= 1:
        return (image * 255).astype(np.uint8)
    return (255 * (image - low) / (high - low)).astype(np.uint8)


class _Progress:
    """micro_sam's ``pbar_signals``, as opspec progress, and cancel.

    micro_sam drives a napari progress bar through three Qt signals; any
    object with ``emit`` stands in. Raising from ``update`` is how Cancel
    stops training: best.pt holds the best epoch so far.
    """

    class _Signal:
        def __init__(self, fn):
            self.emit = fn

    def __init__(self, name: str):
        self.name = name
        self.total = None
        self.done = 0
        self.shown = -1
        self.pbar_total = self._Signal(self._set_total)
        self.pbar_update = self._Signal(self._update)
        self.pbar_description = self._Signal(lambda text: None)

    def _set_total(self, total):
        self.total = int(total) if total else None

    def _update(self, steps):
        self.done += int(steps)
        if cancel_requested():
            raise RuntimeError("Training cancelled")
        if not self.total:
            return
        percent = 5 * (20 * self.done // self.total)  # in steps of 5%
        if percent != self.shown:
            self.shown = percent
            progress(
                f"Training {self.name}: {percent}%", self.done, self.total
            )


@op(env="pytorch")
def train_microsam(
    images: list[str],
    labels: list[str],
    model_dir: str,
    name: str = "microsam_model",
    model: str = "vit_b_lm",
    initial_checkpoint: str = "",
    epochs: Annotated[
        int, {"widget_type": "SpinBox", "min": 1, "max": 5000}
    ] = 100,
    n_objects_per_batch: Annotated[
        int, {"widget_type": "SpinBox", "min": 1, "max": 50}
    ] = 5,
    batch_size: Annotated[
        int, {"widget_type": "SpinBox", "min": 1, "max": 64}
    ] = 1,
    learning_rate: float = 1e-5,
    val_size: Annotated[
        int, {"widget_type": "SpinBox", "min": 0, "max": 100}
    ] = 2,
    early_stopping: Annotated[
        int, {"widget_type": "SpinBox", "min": 0, "max": 1000}
    ] = 10,
) -> str:
    """Finetune micro_sam, with its instance decoder, on your patches.

    Args:
        images: Paths to the input patches. Grey or RGB (averaged to grey),
            any dtype: they are scaled to uint8, which micro_sam trains on.
        labels: Label patches; ``labels[k]`` is the truth for ``images[k]``.
        model_dir: Where micro_sam writes ``checkpoints/<name>/`` and
            ``logs/``.
        name: The model's name, and its directory under ``checkpoints``.
        model: micro_sam model to finetune. Ignored for weights when
            *initial_checkpoint* is set, but still its architecture.
        initial_checkpoint: A ``best.pt`` to continue from, or empty.
        epochs: Passes over the training patches.
        n_objects_per_batch: Objects sampled per patch for prompt training.
            Lower uses less GPU memory.
        batch_size: Patches per batch.
        learning_rate: micro_sam's default for finetuning.
        val_size: Pairs held out from the end for validation. With too few
            pairs to spare, the training pairs validate as well.
        early_stopping: Stop after this many epochs without improvement;
            0 never stops early.

    Returns:
        The path of the trained ``best.pt``.
    """
    import shutil
    import tempfile

    import imageio.v3 as iio
    import torch
    from micro_sam import training as sam_training
    from torch_em.data import MinInstanceSampler

    if len(images) != len(labels) or not images:
        raise ValueError(
            f"{len(images)} images and {len(labels)} labels: need equal, nonzero"
        )
    if initial_checkpoint:
        model = checkpoint_model_type(initial_checkpoint, model)

    work = Path(tempfile.mkdtemp(prefix="microsam_train_"))
    try:
        held = val_size if 0 < val_size < len(images) else 0
        split = len(images) - held
        sets = {"train": range(split), "val": range(split, len(images))}
        if not held:
            sets["val"] = sets["train"]

        shapes = []
        for part, indices in sets.items():
            (work / part / "raw").mkdir(parents=True)
            (work / part / "label").mkdir(parents=True)
            for k in indices:
                raw = _to_8bit(iio.imread(images[k]))
                truth = np.asarray(iio.imread(labels[k])).astype(np.uint32)
                if raw.shape != truth.shape:
                    raise ValueError(
                        f"{images[k]} is {raw.shape} but its labels are "
                        f"{truth.shape}"
                    )
                shapes.append(raw.shape)
                iio.imwrite(work / part / "raw" / f"{k:05d}.tif", raw)
                iio.imwrite(work / part / "label" / f"{k:05d}.tif", truth)
        patch_shape = tuple(int(s) for s in np.min(shapes, axis=0))

        def loader(part):
            return sam_training.default_sam_loader(
                raw_paths=str(work / part / "raw"),
                raw_key="*.tif",
                label_paths=str(work / part / "label"),
                label_key="*.tif",
                with_segmentation_decoder=True,
                patch_shape=patch_shape,
                batch_size=batch_size,
                shuffle=True,
                raw_transform=sam_training.identity,
                sampler=MinInstanceSampler(min_size=25),
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        progress(
            f"Training {name} from {initial_checkpoint or model}: "
            f"{split} train, {held} val, {epochs} epochs, on {device}",
            0,
            epochs,
        )
        sam_training.train_sam(
            name=name,
            model_type=model,
            train_loader=loader("train"),
            val_loader=loader("val"),
            n_epochs=epochs,
            early_stopping=early_stopping or None,
            n_objects_per_batch=n_objects_per_batch,
            checkpoint_path=initial_checkpoint or None,
            with_segmentation_decoder=True,
            device=device,
            lr=learning_rate,
            save_root=model_dir,
            pbar_signals=_Progress(name),
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    best = Path(model_dir) / "checkpoints" / name / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"micro_sam finished but wrote no {best}")
    (best.parent / META).write_text(json.dumps({"model_type": model}))
    progress(f"Saved {best}")
    return str(best)
