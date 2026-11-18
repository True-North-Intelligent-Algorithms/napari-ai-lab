"""Random-zoom crops around boxes: a training set for a YOLO detector.

YOLO shrinks a whole photo to ``imgsz`` before it does anything else, so in a
4000 px photo a 60 px bib becomes 10-20 px -- too few to learn from -- and
ultralytics' own zoom only zooms the already-shrunk image. So the crops are
cut from the full-size photo, each zoomed so its object comes out a random
height between ``min_object`` and ``max_object`` pixels (spread on a log
scale, so small and large are as common). The same object is seen small and
large; the size is the main augmentation. Each photo is also kept whole,
longest side ``crop_size``: objects at the size a whole-photo detector sees
them.

The rest is mild: a little perspective and stretch, light, blur and JPEG.
No flips -- numbers on a bib would be mirrored -- and no rotation unless
asked for. Train with ultralytics' ``fliplr`` at 0 for the same reason.

Validation is whole photos held out, spread evenly through the list, with a
few unaugmented crops each -- never crops of a photo that is also trained
on, or the scores flatter the model.

Written out as::

    <out>/images/<stem>_<k>.jpg
    <out>/labels/<stem>_<k>.txt     YOLO, the source boxes' classes
    <out>/pairs.json                images and labels (relative to <out>),
                                    training first, val_size, class_names,
                                    imgsz, settings

which is what scikit-ops' ``train_yolo`` takes. Ported from runphoto-kit's
``bib_patches``, which writes the same layout for bibs alone.

No napari or Qt here.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: Grey past the photo's edge, as ultralytics pads; a reflection would mirror
#: any object near the edge, unlabelled.
PAD_GREY = 114


@dataclass
class LabelledPhoto:
    """One photo to cut crops from.

    ``load`` returns its pixels -- the ones its boxes were drawn on.
    ``rows`` are its boxes, YOLO-normalised: ``(class, xc, yc, w, h)``.
    No rows is a photo complete and empty: still a training image.
    ``regions``, if any, are label boxes ``(l, t, r, b)`` in pixels: only
    inside them is complete, and the rest is greyed out.
    """

    stem: str
    load: Callable[[], np.ndarray]
    rows: list
    regions: list | None = None


def read_rows(path) -> list[tuple[int, float, float, float, float]]:
    """``(class, xc, yc, w, h)`` from a YOLO txt file; a 6th column is dropped."""
    rows = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 5:
                rows.append((int(float(p[0])), *(float(v) for v in p[1:5])))
    return rows


def labelled_photos(
    project, folder, image_paths, load_image, label_boxes=None
) -> list[LabelledPhoto]:
    """The photos box layer *folder* can train on, in project order.

    A photo marked complete trains whole. One that is not, but has label
    boxes -- *label_boxes*, stem -> ``[(l, t, r, b), ...]`` in pixels --
    trains inside them. Any other photo is not complete and is left out,
    whatever boxes it has saved.

    *load_image(i)* gives photo *i*'s pixels, as the app shows them -- the
    pixels the boxes were drawn on.
    """
    from functools import partial

    from .box_annotations import box_file, read_complete

    complete = read_complete(project, folder)
    label_boxes = label_boxes or {}
    photos = []
    for i, path in enumerate(image_paths):
        regions = None if path.stem in complete else label_boxes.get(path.stem)
        if path.stem not in complete and not regions:
            continue
        txt = box_file(project, folder, path.stem)
        rows = read_rows(txt) if txt.exists() else []
        photos.append(
            LabelledPhoto(path.stem, partial(load_image, i), rows, regions)
        )
    return photos


def _inside_regions(rgb, boxes, regions):
    """*rgb* greyed outside *regions*, and the boxes centred inside one,
    clipped to it. Outside the label boxes nothing was boxed, so it must
    not be learnt as background."""
    h, w = rgb.shape[:2]
    keep = np.zeros((h, w), bool)
    for left, t, r, b in regions:
        keep[
            max(0, int(t)) : min(h, math.ceil(b)),
            max(0, int(left)) : min(w, math.ceil(r)),
        ] = True
    rgb = rgb.copy()
    rgb[~keep] = PAD_GREY
    kept = []
    for cls, left, t, r, b in boxes:
        cx, cy = (left + r) / 2, (t + b) / 2
        for rl, rt, rr, rb in regions:
            if rl <= cx <= rr and rt <= cy <= rb:
                kept.append(
                    (cls, max(left, rl), max(t, rt), min(r, rr), min(b, rb))
                )
                break
    return rgb, kept


def as_rgb8(img: np.ndarray) -> np.ndarray:
    """Any 2D image as uint8 RGB: grey made colour, alpha dropped, wider
    types scaled to their range."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        lo, hi = float(np.min(img)), float(np.max(img))
        scale = 255.0 / (hi - lo) if hi > lo else 0.0
        img = ((img.astype(np.float32) - lo) * scale).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim != 3 or img.shape[-1] not in (3, 4):
        raise ValueError(
            f"Box crops are for 2D photos (YX or YXC); got shape {img.shape}"
        )
    return np.ascontiguousarray(img[..., :3])


def val_indices(n: int, val_photos: int) -> set[int]:
    """*val_photos* of *n* photos, spread evenly through the list."""
    if val_photos <= 0:
        return set()
    return {
        min(n - 1, int((j + 0.5) * n / val_photos)) for j in range(val_photos)
    }


class _Pyramid:
    """The photo at 1, 1/2, 1/4, ... so a crop shrunk a lot is resampled
    from a smaller copy, not aliased from the full-size one."""

    def __init__(self, img):
        import cv2

        self.levels = [img]
        while min(self.levels[-1].shape[:2]) > 600:
            self.levels.append(cv2.pyrDown(self.levels[-1]))

    def for_zoom(self, zoom):
        """(image, factor): the smallest level the warp shrinks by no more
        than 2x; factor (1, 1/2, ...) maps full-size pixels onto it."""
        level = 0
        while (
            level + 1 < len(self.levels) and zoom / 0.5 ** (level + 1) <= 1.0
        ):
            level += 1
        return self.levels[level], 0.5**level


def _warp(pyramid, boxes, center, zoom, size, rng, deform, max_rotate):
    """One ``size`` x ``size`` crop centred near ``center`` (full-size
    pixels), magnified by ``zoom``, a little deformed.

    ``boxes`` are ``(class, l, t, r, b)`` in full-size pixels. Returns the
    crop and its rows, ``(class, xc, yc, w, h)`` normalised.
    """
    import cv2

    img, f = pyramid.for_zoom(zoom)
    side = size / zoom  # the crop's side, in full-size pixels
    cx, cy = center
    # Stretch: objects vary in aspect with the angle they are seen at.
    sx = side * math.exp(rng.uniform(-0.1, 0.1))
    sy = side * side / sx
    angle = math.radians(rng.uniform(-max_rotate, max_rotate))
    ca, sa = math.cos(angle), math.sin(angle)
    quad = []
    for ux, uy in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
        x = ux * sx + rng.uniform(-deform, deform) * side
        y = uy * sy + rng.uniform(-deform, deform) * side
        quad.append([cx + ca * x - sa * y, cy + sa * x + ca * y])
    quad = np.float32(quad)
    dst = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    m_full = cv2.getPerspectiveTransform(quad, dst)
    m_level = cv2.getPerspectiveTransform(quad * f, dst)
    out = cv2.warpPerspective(
        img,
        m_level,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(PAD_GREY,) * 3,
    )

    rows = []
    for cls, left, t, r, b in boxes:
        corners = np.float32([[[left, t], [r, t], [r, b], [left, b]]])
        p = cv2.perspectiveTransform(corners, m_full)[0]
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        area = (x1 - x0) * (y1 - y0)
        cx0, cy0 = max(x0, 0), max(y0, 0)
        cx1, cy1 = min(x1, size), min(y1, size)
        if area <= 0 or cx1 - cx0 < 4 or cy1 - cy0 < 4:
            continue
        if (cx1 - cx0) * (cy1 - cy0) / area < 0.5:
            # Mostly cut off: not one to learn from, and not background
            # either, so blank it.
            out[int(cy0) : math.ceil(cy1), int(cx0) : math.ceil(cx1)] = (
                out.mean(axis=(0, 1))
            )
            continue
        rows.append(
            (
                cls,
                (cx0 + cx1) / 2 / size,
                (cy0 + cy1) / 2 / size,
                (cx1 - cx0) / size,
                (cy1 - cy0) / size,
            )
        )
    return out, rows


def _photometric(img, rng):
    """Light, blur, noise and colour, each now and then, as cameras vary."""
    import cv2

    out = img.astype(np.float32)
    out = out * rng.uniform(0.75, 1.25) + rng.uniform(
        -20, 20
    )  # contrast, brightness
    out = 255 * (np.clip(out, 0, 255) / 255) ** rng.uniform(0.8, 1.25)  # gamma
    out = out.astype(np.uint8)
    if rng.random() < 0.3:
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= rng.uniform(0.7, 1.3)
        out = cv2.cvtColor(
            np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB
        )
    r = rng.random()
    if r < 0.2:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 1.5))
    elif r < 0.35:  # the subject moving, or the camera
        k = int(rng.integers(5, 12))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1 / k
        if rng.random() < 0.5:
            kernel = kernel.T
        out = cv2.filter2D(out, -1, kernel)
    if rng.random() < 0.2:
        noise = rng.normal(0, rng.uniform(2, 8), out.shape)
        out = np.clip(out + noise, 0, 255).astype(np.uint8)
    return out


def _save(path, rgb, quality):
    import cv2

    cv2.imwrite(
        path,
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )


def _write_label(path, rows):
    with open(path, "w") as fh:
        for cls, xc, yc, w, h in rows:
            fh.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def write_box_crops(
    photos: list[LabelledPhoto],
    out_dir,
    class_names: list[str] | None = None,
    crop_size: int = 1024,
    per_photo: int = 16,
    val_photos: int = 0,
    min_object: int = 16,
    max_object: int = 160,
    deform: float = 0.04,
    max_rotate: float = 0.0,
    seed: int = 0,
    source: str = "",
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Write the crop set to *out_dir*, replacing what is there.

    per_photo: crops per training photo, besides the photo whole. A photo
        with no boxes gives only the whole photo.
    crop_size: side of each crop, in pixels; train at this ``imgsz``.
    val_photos: whole photos held out for validation; 0 takes a tenth.
    min_object, max_object: range of object heights a crop is zoomed to, in
        crop pixels.
    deform: perspective jitter, as a fraction of the crop's side.
    max_rotate: +/- degrees.
    source: what the boxes are, for the record (e.g. the box layer).

    Returns the summary written to pairs.json, plus ``train`` and ``val``
    counts.
    """
    import cv2

    if not photos:
        raise ValueError(
            "No complete photos: tick Complete on the Boxes tab for each "
            "photo whose boxes are all done (or draw label boxes)."
        )
    if val_photos <= 0:
        val_photos = max(1, round(len(photos) / 10)) if len(photos) > 1 else 0
    if len(photos) <= val_photos:
        raise ValueError(
            f"{len(photos)} complete photos: too few to hold {val_photos} "
            "back for validation and still train on the rest."
        )
    held_out = val_indices(len(photos), val_photos)

    out_dir = os.fspath(out_dir)
    # Last time's crops would be trained on too.
    shutil.rmtree(out_dir, ignore_errors=True)
    image_dir = os.path.join(out_dir, "images")
    label_dir = os.path.join(out_dir, "labels")
    os.makedirs(image_dir)
    os.makedirs(label_dir)

    rng = np.random.default_rng(seed)
    train, val = [], []
    top_class, n_boxes = -1, 0
    for i, photo in enumerate(photos):
        is_val = i in held_out
        rgb = as_rgb8(photo.load())
        h, w = rgb.shape[:2]
        # (class, l, t, r, b) in full-size pixels.
        boxes = [
            (
                c,
                (xc - bw / 2) * w,
                (yc - bh / 2) * h,
                (xc + bw / 2) * w,
                (yc + bh / 2) * h,
            )
            for c, xc, yc, bw, bh in photo.rows
        ]
        if photo.regions:
            rgb, boxes = _inside_regions(rgb, boxes, photo.regions)
        n_boxes += len(boxes)
        if boxes:
            top_class = max(top_class, max(b[0] for b in boxes))
        samples = []

        # The whole photo, longest side crop_size: as a whole-photo detector
        # sees it.
        zoom = crop_size / max(w, h)
        whole = cv2.resize(
            rgb,
            (max(1, round(w * zoom)), max(1, round(h * zoom))),
            interpolation=cv2.INTER_AREA,
        )
        whole_rows = [
            (
                c,
                (left + r) / 2 / w,
                (t + b) / 2 / h,
                (r - left) / w,
                (b - t) / h,
            )
            for c, left, t, r, b in boxes
        ]
        samples.append((whole, whole_rows, 95))

        pyramid = _Pyramid(rgb)
        n_crops = 3 if is_val else per_photo
        for _ in range(n_crops if boxes else 0):
            _, left, t, r, b = boxes[int(rng.integers(len(boxes)))]
            target = math.exp(
                rng.uniform(math.log(min_object), math.log(max_object))
            )
            z = target / max(b - t, 1)
            z = max(
                z, crop_size / min(w, h)
            )  # never a crop bigger than the photo
            side = crop_size / z
            # The object anywhere in the crop, not always in the middle.
            cx = (left + r) / 2 + rng.uniform(-0.35, 0.35) * side
            cy = (t + b) / 2 + rng.uniform(-0.35, 0.35) * side
            cx = min(max(cx, side / 2), w - side / 2) if side < w else w / 2
            cy = min(max(cy, side / 2), h - side / 2) if side < h else h / 2
            crop, rows = _warp(
                pyramid,
                boxes,
                (cx, cy),
                z,
                crop_size,
                rng,
                0 if is_val else deform,
                0 if is_val else max_rotate,
            )
            if not is_val:
                crop = _photometric(crop, rng)
            samples.append(
                (crop, rows, 95 if is_val else rng.integers(60, 96))
            )

        for k, (img, rows, quality) in enumerate(samples):
            image = f"images/{photo.stem}_{k:02d}.jpg"
            label = f"labels/{photo.stem}_{k:02d}.txt"
            _save(os.path.join(out_dir, image), img, quality)
            _write_label(os.path.join(out_dir, label), rows)
            (val if is_val else train).append((image, label))
        if progress is not None:
            progress(i + 1, len(photos), photo.stem)

    if not class_names:
        class_names = (
            ["object"]
            if top_class <= 0
            else [f"class{c}" for c in range(top_class + 1)]
        )
    pairs = train + val
    summary = {
        "made": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "images": [p[0] for p in pairs],
        "labels": [p[1] for p in pairs],
        "val_size": len(val),
        "class_names": list(class_names),
        "imgsz": crop_size,
        "val_names": [photos[i].stem for i in sorted(held_out)],
        "photos": len(photos),
        "val_photos": val_photos,
        "boxes": n_boxes,
        "settings": {
            "per_photo": per_photo,
            "crop_size": crop_size,
            "min_object": min_object,
            "max_object": max_object,
            "deform": deform,
            "max_rotate": max_rotate,
            "seed": seed,
        },
    }
    with open(os.path.join(out_dir, "pairs.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return {**summary, "train": len(train), "val": len(val)}


def read_pairs(crop_dir) -> dict:
    """pairs.json from a crop set, with images and labels as full paths."""
    crop_dir = os.fspath(crop_dir)
    with open(os.path.join(crop_dir, "pairs.json")) as fh:
        pairs = json.load(fh)
    for key in ("images", "labels"):
        pairs[key] = [
            p if os.path.isabs(p) else os.path.join(crop_dir, p)
            for p in pairs[key]
        ]
    return pairs
