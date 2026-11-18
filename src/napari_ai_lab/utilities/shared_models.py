"""Models kept once per user, not per project.

A project's own ``models/`` folder holds what was trained for it. Stock
models -- a COCO YOLO that finds people, say -- belong to no project, so they
live in one folder per user: ``~/.napari_ai_lab/models``, or wherever
``NAPARI_AI_LAB_MODELS`` points. The YOLO model combo lists them next to the
project's, and a segmenter with a fixed model (People) fetches its model
there on first use.
"""

import os
import urllib.request
from pathlib import Path

ENV_VAR = "NAPARI_AI_LAB_MODELS"


def shared_models_dir() -> Path:
    """The per-user models folder. Not created until something is put in it."""
    custom = os.environ.get(ENV_VAR)
    if custom:
        return Path(custom).expanduser()
    return Path.home() / ".napari_ai_lab" / "models"


def shared_model(name: str, url: str) -> Path:
    """The shared model *name*, downloaded from *url* the first time.

    The download goes to a temporary name and is renamed when complete, so an
    interrupted one is not mistaken for the model next time.
    """
    path = shared_models_dir() / name
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    print(f"Downloading {name} to {path.parent} ...")
    urllib.request.urlretrieve(url, partial)
    partial.replace(path)
    return path
