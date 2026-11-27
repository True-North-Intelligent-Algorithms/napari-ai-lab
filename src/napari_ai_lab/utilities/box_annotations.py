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
- ``box_predictions/<source>/classes.yaml``: the names of the source's
  classes, when it says what each box is -- a YOLO model trained on named
  classes. A source that finds every object, as FastSAM, has none, and its
  class column means nothing.

The older layout, a single layer, is still read: txt files directly in
``box_annotations/``, with the project's own ``classes.yaml``. Its folder is
``LEGACY`` (""), and ai-lab shows it as the "Box annotations" layer.

- ``box_annotations/<layer>/complete.list``: the image stems marked
  complete in that layer, one per line -- every object boxed, or none
  there. Only these train whole. Saved boxes alone do not make an image
  complete: it may be half done. (Images with label boxes, in
  ``labels/boxes.csv``, train inside those, complete or not.)

Every file is optional. Without ``classes.yaml`` a box shows its class
number.

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
    so a project always has at least one box layer. Folders starting with
    "_" (``_deleted``) are not layers; check_name keeps layers from doing so.
    """
    root = Path(project) / BOX_ANNOTATIONS_DIR
    named = []
    legacy = False
    if root.is_dir():
        named = sorted(
            d.name
            for d in root.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )
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


#: The photos marked complete in a layer, one stem per line. Not .txt: a
#: txt in a layer folder is an image's boxes.
COMPLETE_FILE = "complete.list"


def complete_path(project, folder: str) -> Path:
    """*folder*'s list of complete photos."""
    return folder_path(project, folder) / COMPLETE_FILE


def read_complete(project, folder: str) -> set[str]:
    """Stems of the photos marked complete in *folder*: every object boxed,
    or none there. Saved boxes alone do not make a photo complete."""
    path = complete_path(project, folder)
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def write_complete(project, folder: str, stems):
    """Replace *folder*'s list of complete photos with *stems*."""
    path = complete_path(project, folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{s}\n" for s in sorted(stems)))


def set_complete(project, folder: str, stem: str, complete: bool = True):
    """Mark one photo complete in *folder*, or take it off the list."""
    stems = read_complete(project, folder)
    if complete:
        stems.add(stem)
    else:
        stems.discard(stem)
    write_complete(project, folder, stems)


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


def rename_box_folder(project, old: str, new: str) -> str:
    """Rename box layer *old* to *new*, files and all; returns *new*, checked.

    Its boxes, classes.yaml and complete.list move with it. The older
    layout (*old* LEGACY) moves out of box_annotations/ into a folder of its
    own; the project's classes.yaml is copied, not moved, or class 0 is
    named *new* if there is none. *new* must not already hold files.
    Crop sets made from it (patches/yolo_<old>) keep their old name.
    """
    import shutil

    new = check_name(new)
    if new == old:
        return new
    src, dst = folder_path(project, old), folder_path(project, new)
    if old and new.lower() == old.lower():
        # Only the case changes: one folder to Windows, so by a detour.
        detour = src.with_name(src.name + ".renaming")
        src.rename(detour)
        detour.rename(dst)
        return new
    if dst.exists():
        if any(dst.iterdir()):
            raise ValueError(f"There is already a box layer called {new!r}.")
        dst.rmdir()
    if old:
        src.rename(dst)
        return new

    dst.mkdir(parents=True)
    for f in [*src.glob("*.txt"), complete_path(project, LEGACY)]:
        if f.is_file():
            f.rename(dst / f.name)
    if _classes_path(project, LEGACY).exists():
        shutil.copy2(
            _classes_path(project, LEGACY), _classes_path(project, new)
        )
    else:
        write_class_names(project, {0: new}, new)
    return new


#: Where deleted box layers go, under box_annotations/, to be got back.
DELETED_DIR = "_deleted"


def delete_box_folder(project, folder: str) -> Path:
    """Delete box layer *folder*, by moving it to box_annotations/_deleted/.

    Into a folder named for it and the time, so deleting twice keeps both.
    The older layout (LEGACY) moves its txt files and complete.list; the
    project's classes.yaml stays. Returns where it went: move it back
    to undo.
    """
    import time

    root = Path(project) / BOX_ANNOTATIONS_DIR
    dst = (
        root
        / DELETED_DIR
        / (
            f"{folder or 'Box annotations'} {time.strftime('%Y-%m-%d %H-%M-%S')}"
        )
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    if folder:
        folder_path(project, folder).rename(dst)
        return dst
    dst.mkdir()
    for f in [*root.glob("*.txt"), complete_path(project, LEGACY)]:
        if f.is_file():
            f.rename(dst / f.name)
    return dst


def read_class_names(project, folder: str = LEGACY) -> dict[int, str]:
    """Class id -> name from *folder*'s classes.yaml, or {} if none."""
    return read_names(_classes_path(project, folder))


def read_names(path) -> dict[int, str]:
    """Class id -> name from the ``names:`` of a YOLO yaml, or {} if none.

    A classes.yaml, or a YOLO dataset's data.yaml.
    """
    path = Path(path)
    if not path.is_file():
        return {}

    import yaml

    with open(path, encoding="utf-8") as f:
        names = (yaml.safe_load(f) or {}).get("names") or {}
    # YOLO allows either a list or an id -> name mapping.
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(k): str(v) for k, v in names.items()}


def write_class_names(project, names, folder: str = LEGACY):
    """Write class id -> name to *folder*'s classes.yaml."""
    _write_names(_classes_path(project, folder), names)


def _write_names(path, names):
    import yaml

    text = yaml.safe_dump({"names": dict(sorted(names.items()))})
    header = "# Class names for box annotations, in YOLO data.yaml form.\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + text, encoding="utf-8")


def merge_class_names(names, into):
    """Match the classes *names* names to the ids *into* has for them.

    For copying a detector's boxes into a box layer: a class keeps its name,
    whatever id the layer has for it. A name *into* lacks is added, under
    its own id if that is free, else the next free one. Returns
    ``(id -> id, the names *into* ends up with)``.
    """
    merged = dict(into)
    ids = {name: i for i, name in merged.items()}
    mapping = {}
    for class_id, name in sorted(names.items()):
        if name not in ids:
            free = class_id not in merged
            ids[name] = class_id if free else max(merged) + 1
            merged[ids[name]] = name
        mapping[class_id] = ids[name]
    return mapping, merged


def prediction_sources(project) -> list[str]:
    """The sources with box predictions saved: subfolders of box_predictions/."""
    root = Path(project) / BOX_PREDICTIONS_DIR
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def prediction_file(project, source: str, image_stem: str) -> Path:
    """The YOLO txt file holding *source*'s predicted boxes for one image."""
    return Path(project) / BOX_PREDICTIONS_DIR / source / f"{image_stem}.txt"


def read_prediction_class_names(project, source: str) -> dict[int, str]:
    """The names of *source*'s classes, or {} when it does not name them."""
    return read_names(
        Path(project) / BOX_PREDICTIONS_DIR / source / CLASSES_FILE
    )


def write_prediction_class_names(project, source: str, names):
    """Record the names of *source*'s classes; None or {} removes them."""
    path = Path(project) / BOX_PREDICTIONS_DIR / source / CLASSES_FILE
    if names:
        _write_names(path, names)
    else:
        path.unlink(missing_ok=True)


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
