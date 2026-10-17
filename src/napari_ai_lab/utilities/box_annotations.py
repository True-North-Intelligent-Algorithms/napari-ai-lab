"""Box annotations with class names, kept in YOLO form.

A project may hold

- ``classes.yaml``: the ``names:`` block of a YOLO data.yaml, class id to
  name, e.g. ``0: buffalo``.
- ``box_annotations/<image stem>.txt``: one YOLO line per box,
  ``class x_center y_center width height``, normalized to the image size.

Both are optional. Without ``classes.yaml`` a box shows its class number.
"""

from pathlib import Path

import numpy as np

CLASSES_FILE = "classes.yaml"
BOX_ANNOTATIONS_DIR = "box_annotations"


def read_class_names(project) -> dict[int, str]:
    """Class id -> name from the project's classes.yaml, or {} if none."""
    path = Path(project) / CLASSES_FILE
    if not path.exists():
        return {}

    import yaml

    with open(path) as f:
        names = (yaml.safe_load(f) or {}).get("names", {})
    # YOLO allows either a list or an id -> name mapping.
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(k): str(v) for k, v in names.items()}


def read_yolo_boxes(path, height, width):
    """Read a YOLO txt file as napari rectangles and their class ids.

    Rectangles are four (y, x) corners in pixels. A missing file means no
    boxes.
    """
    boxes, classes = [], []
    path = Path(path)
    if not path.exists():
        return boxes, np.zeros(0, dtype=int)

    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        xc, yc, w, h = (float(p) for p in parts[1:5])
        y1, y2 = (yc - h / 2) * height, (yc + h / 2) * height
        x1, x2 = (xc - w / 2) * width, (xc + w / 2) * width
        boxes.append(np.array([[y1, x1], [y1, x2], [y2, x2], [y2, x1]]))
        classes.append(int(parts[0]))
    return boxes, np.array(classes, dtype=int)


def write_yolo_boxes(path, boxes, classes, height, width):
    """Write napari shapes and their class ids as a YOLO txt file.

    Each shape is saved as its bounding box, so a rotated or edited
    rectangle becomes the axis-aligned box around it.
    """
    lines = []
    for box, cls in zip(boxes, classes, strict=True):
        box = np.asarray(box)[:, -2:]
        (y1, x1), (y2, x2) = box.min(axis=0), box.max(axis=0)
        xc, yc = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


def write_class_names(project, names):
    """Write class id -> name to the project's classes.yaml."""
    import yaml

    text = yaml.safe_dump({"names": dict(sorted(names.items()))})
    header = "# Class names for box annotations, in YOLO data.yaml form.\n"
    (Path(project) / CLASSES_FILE).write_text(header + text)
