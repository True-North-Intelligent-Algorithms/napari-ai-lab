"""Step 1 of docs/spec/0004: run StarDist through scikit-ops, headless.

No napari, no napari-ai-lab. Just: numpy array in, label image out, with the
op executing in scikit-ops' own environment rather than this one. If this does
not work, nothing downstream can fix it -- so it is worth proving alone.

It also answers two questions the spec leaves open:

  * what the first-run environment build actually reports, and
  * where the built environment is written to, which the UI has to tell the
    user before it starts building.

Run it from the host environment:

    cd pixi/pytorch_napari
    pixi run poc

The first run builds the stardist-tf environment -- minutes and gigabytes, with
progress reported as it goes. Later runs reuse it and take seconds, and print
no build output at all because there is no build.

Flags:

    --verbose   the raw build output and error streams. These carry pixi's
                entire -vv debug log, because subscribing to progress is what
                turns the monitor on (scikit-ops design 0004), so they are off
                by default.
    --show      open the image and labels in napari afterwards. Off by default:
                the point of this script is that it needs no GUI.

Confirmed on this host: neither tensorflow nor stardist is importable here, and
the op still runs. That is the whole argument of docs/spec/0001.
"""

import sys
import time
from pathlib import Path

import numpy as np
from skimage.io import imread
from skop.ops.segment.stardist2d import stardist2d_fluo
from skop.runner import Runner

PROJECT_ROOT = Path(__file__).parent.parent
IMAGE = (
    PROJECT_ROOT
    / "tests"
    / "test_images"
    / "neurips blood cells"
    / "cell_00147.png"
)

VERBOSE = "--verbose" in sys.argv

#: Pass --show to open the image and labels in napari afterwards.
SHOW = "--show" in sys.argv


def on_progress(title: str, current: int, maximum: int) -> None:
    """Build phases: Solving, Installing conda packages, Done, and friends.

    Each carries a real denominator, so a GUI can show a determinate bar
    rather than a spinner.
    """
    print(f"  [build] {title}: {current}/{maximum}")


def on_text(text: str) -> None:
    """Raw stdout/stderr from pixi.

    NB: the "error" channel is just stderr. Pixi writes ordinary status there,
    success message included, so this must not be treated as failure. A real
    failure raises out of run() instead.
    """
    if VERBOSE:
        print(f"  [build:text] {text.rstrip()}")


def main() -> None:
    if not IMAGE.is_file():
        raise SystemExit(f"Test image not found: {IMAGE}")

    image = imread(IMAGE)
    print(f"image: {IMAGE.name}  shape={image.shape}  dtype={image.dtype}")

    runner = Runner()
    runner.subscribe_build_progress(on_progress)
    runner.subscribe_build_output(on_text)
    runner.subscribe_build_error(on_text)

    print("\nRunning stardist2d_fluo through skop.")
    print(
        "First run builds the stardist-tf environment -- minutes, and GBs.\n"
    )

    start = time.perf_counter()
    labels = runner.run(stardist2d_fluo, image=image)
    elapsed = time.perf_counter() - start

    # Asked after the run so the environment is definitely built. This path is
    # what the UI needs to show the user *before* building, so that they can
    # find it, inspect it, or delete it later.
    try:
        base = runner.environment("stardist-tf").base
        # A property on some appose Environment types, a method on the object
        # the pixi builder returns.
        env_path = base() if callable(base) else base
    except Exception as exc:  # noqa: BLE001 - informational only
        env_path = f"(could not determine: {exc})"

    print("\n" + "=" * 56)
    print(f"elapsed         : {elapsed:.1f} s")
    print(f"labels shape    : {labels.shape}")
    print(f"labels dtype    : {labels.dtype}")
    print(f"objects found   : {len(np.unique(labels)) - 1}")
    print(f"environment at  : {env_path}")
    print("=" * 56)

    if labels.shape[:2] != image.shape[:2]:
        print(
            "\nNOTE: label shape does not match the image's YX extent. "
            "Worth understanding before wiring this into the app, since "
            "ImageDataModel expects a mask it can save against the image."
        )

    # Off by default: the point of this script is that it needs no GUI. But
    # an object count alone does not tell you whether the segmentation is any
    # good, so --show opens the result for a look.
    if SHOW:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image, name=IMAGE.name)
        viewer.add_labels(labels, name="stardist2d_fluo (skop)")
        napari.run()


if __name__ == "__main__":
    main()
