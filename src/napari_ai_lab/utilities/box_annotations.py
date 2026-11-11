"""Box annotations with class names, kept in YOLO form.

A project may hold several box layers, each a folder of its own under
``box_annotations/``, named for what is boxed in it -- ``runners``, ``bibs``:

- ``box_annotations/<layer>/<image stem>.txt``: one YOLO line per box,
  ``class x_center y_center width height``, normalized to the image size.
- ``box_annotations/<layer>/classes.yaml``: that layer's class names, the
  ``names:`` block of a YOLO data.yaml, e.g. ``0: runner``. Each layer
  numbers its own classes from 0, so each folder trains as it stands.

Boxes a detector or program found, not yet checked by anyone, are kept
apart, one folder per source -- the detector, or e.g. ``photo-enhance-bibs``:

- ``box_predictions/<source>/<image stem>.txt``: YOLO lines with a
  confidence after them, ``class x_center y_center width height conf``,
  as ultralytics saves its own; the confidence is left off when the source
  gives none. A file exists only for an image the source ran on.

The older layout, a single layer, is still read: txt files directly in
``box_annotations/``, with the project's own ``classes.yaml``. Its folder is
``LEGACY`` (""), and ai-lab shows it as the "Box annotations" layer.

Every file is optional. An image without a txt file has not been
annotated; an empty file means it was, and holds nothing. Without
``classes.yaml`` a box shows its class number.

No napari or Qt here: photo-enhance and other apps read and write the same
files.
"""

import re
from pathlib import Path

import numpy as np

CLASSES_FILE = "classes.yaml"
BOX_ANNOTATIONS_DIR = "box_annotations"
BOX_PREDICTIONS_DIR = "box_predictions"

#: The folder of the older single-layer layout: box_annotations/ itself.
LEGACY = ""

#: What a layer may be called: it is a folder name, on every platform.
_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]*$")


def box_folders(project) -> list[str]:
    """The project's box layers, as folder names, LEGACY first if present.

    Each subfolder of box_annotations/ is one. LEGACY is listed when txt
    files sit directly in box_annotations/, or when there is nothing else,
    so a project always has at least one box layer.
    """
    root = Path(project) / BOX_ANNOTATIONS_DIR
    named = []
    legacy = False
    if root.is_dir():
        named = sorted(d.name for d in root.iterdir() if d.is_dir())
        legacy = any(root.glob("*.txt"))
    return ([LEGACY] if legacy or not named else []) + named


def check_name(name: str) -> str:
    """*name* stripped, if it can be a box layer's folder; else ValueError."""
    name = name.strip()
    if not _VALID_NAME.match(name) or name.endswith((" ", ".")):
        raise ValueError(
            f"{name!r} cannot be a box layer name: use letters, digits, "
            "spaces, '_', '-' or '.', starting with a letter or digit"
        )
    return name


def folder_path(project, folder: str = LEGACY) -> Path:
    """Where *folder*'s txt files live."""
    root = Path(project) / BOX_ANNOTATIONS_DIR
    return root / folder if folder else root


def box_file(project, folder: str, image_stem: str) -> Path:
    """The YOLO txt file holding *folder*'s boxes for one image."""
    return folder_path(project, folder) / f"{image_stem}.txt"


def _classes_path(project, folder: str) -> Path:
    # The older layout keeps its classes at the project root.
    if folder:
        return folder_path(project, folder) / CLASSES_FILE
    return Path(project) / CLASSES_FILE


def add_box_folder(project, folder: str, first_class: str | None = None):
    """Create a box layer's folder, and its classes.yaml if it has none.

    *first_class* names class 0, so the first box drawn has a name.
    Returns the folder name, checked. An existing folder is left as it is.
    """
    folder = check_name(folder)
    folder_path(project, folder).mkdir(parents=True, exist_ok=True)
    if first_class and not _classes_path(project, folder).exists():
        write_class_names(project, {0: first_class}, folder)
    return folder


def read_class_names(project, folder: str = LEGACY) -> dict[int, str]:
    """Class id -> name from *folder*'s classes.yaml, or {} if none."""
    path = _classes_path(project, folder)
    if not path.exists():
        return {}

    import yaml

    with open(path) as f:
        names = (yaml.safe_load(f) or {}).get("names", {})
    # YOLO allows either a list or an id -> name mapping.
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(k): str(v) for k, v in names.items()}


def write_class_names(project, names, folder: str = LEGACY):
    """Write class id -> name to *folder*'s classes.yaml."""
    import yaml

    text = yaml.safe_dump({"names": dict(sorted(names.items()))})
    header = "# Class names for box annotations, in YOLO data.yaml form.\n"
    path = _classes_path(project, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + text)


def prediction_sources(project) -> list[str]:
    """The sources with box predictions saved: subfolders of box_predictions/."""
    root = Path(project) / BOX_PREDICTIONS_DIR
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def prediction_file(project, source: str, image_stem: str) -> Path:
    """The YOLO txt file holding *source*'s predicted boxes for one image."""
    return Path(project) / BOX_PREDICTIONS_DIR / source / f"{image_stem}.txt"


def read_yolo_predictions(path, height, width):
    """Read a YOLO txt file as rectangles, class ids and confidences.

    As read_yolo_boxes, plus each line's sixth column, the confidence; NaN
    on a line without one.
    """
    boxes, classes, confidences = [], [], []
    path = Path(path)
    if not path.exists():
        return boxes, np.zeros(0, dtype=int), np.zeros(0)

    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        xc, yc, w, h = (float(p) for p in parts[1:5])
        y1, y2 = (yc - h / 2) * height, (yc + h / 2) * height
        x1, x2 = (xc - w / 2) * width, (xc + w / 2) * width
        boxes.append(np.array([[y1, x1], [y1, x2], [y2, x2], [y2, x1]]))
        classes.append(int(parts[0]))
        confidences.append(float(parts[5]) if len(parts) > 5 else np.nan)
    return (
        boxes,
        np.array(classes, dtype=int),
        np.array(confidences, dtype=float),
    )


def read_yolo_boxes(path, height, width):
    """Read a YOLO txt file as napari rectangles and their class ids.

    Rectangles are four (y, x) corners in pixels. A missing file means no
    boxes.
    """
    boxes, classes, _ = read_yolo_predictions(path, height, width)
    return boxes, classes


def write_yolo_boxes(path, boxes, classes, height, width, confidences=None):
    """Write napari shapes and their class ids as a YOLO txt file.

    Each shape is saved as its bounding box, so a rotated or edited
    rectangle becomes the axis-aligned box around it. *confidences*, one
    per box, adds the sixth column a prediction file has.
    """
    if confidences is None:
        confidences = [None] * len(boxes)
    lines = []
    for box, cls, conf in zip(boxes, classes, confidences, strict=True):
        box = np.asarray(box)[:, -2:]
        (y1, x1), (y2, x2) = box.min(axis=0), box.max(axis=0)
        xc, yc = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        line = f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        if conf is not None and not np.isnan(conf):
            line += f" {float(conf):.4f}"
        lines.append(line + "\n")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))
