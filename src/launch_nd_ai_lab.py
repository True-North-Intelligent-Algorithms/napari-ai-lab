"""
Launch script for ND AI Lab - Combined tabbed interface.

This script launches the combined AI Lab plugin with Label, Augment,
and Segment tabs in a single interface.
"""

import argparse
from pathlib import Path

import napari

from napari_ai_lab.apps.nd_ai_lab_launcher import launch_nd_ai_lab
from napari_ai_lab.Augmenters import (
    AlbumentationsAugmenter,
    SimpleAugmenter,
)

# Register all segmenters and augmenters
from napari_ai_lab.Segmenters.GlobalSegmenters import (
    CellCastStardistSegmenter,
    CellposeSegmenter,
    MicroSamSegmenter,
    MicroSamYoloSegmenter,
    MonaiUNetSegmenter,
    MonaiUNetSegmenter3D,
    SkImageWatershedSegmenter,
    StardistSegmenter,
    ThresholdSegmenter,
)
from napari_ai_lab.Segmenters.InteractiveSegmenters import (
    SAM3D,
    AnisotropicSphereFit3D,
    FeatureRegionGrow3D,
    HoughSphereFit3D,
    Otsu2D,
    Otsu3D,
    RegionGrow3D,
    SAMSphere3D,
    Square2D,
)

# Flag to control viewer type
viewer_type = "none"  # Options: "none", "stacked", "sequence"

# Each test set below picks its own viewer type.  Pass --stacked or --sequence
# to override it, so the same images can be run through both paths and the
# results compared -- N of YXC against a single NYXC dataset.
parser = argparse.ArgumentParser(description="Launch ND AI Lab")
viewer_group = parser.add_mutually_exclusive_group()
viewer_group.add_argument(
    "--stacked",
    dest="viewer_type_override",
    action="store_const",
    const="stacked",
    help="Force stacked mode (one N-dimensional dataset)",
)
viewer_group.add_argument(
    "--sequence",
    dest="viewer_type_override",
    action="store_const",
    const="sequence",
    help="Force sequence mode (one image at a time)",
)
args = parser.parse_args()

# Register global segmenters (only if successfully imported)
if CellposeSegmenter is not None:
    CellposeSegmenter.register()
if StardistSegmenter is not None:
    StardistSegmenter.register()
if CellCastStardistSegmenter is not None:
    CellCastStardistSegmenter.register()
if ThresholdSegmenter is not None:
    ThresholdSegmenter.register()
if MicroSamSegmenter is not None:
    MicroSamSegmenter.register()
if MonaiUNetSegmenter is not None:
    MonaiUNetSegmenter.register()
if MonaiUNetSegmenter3D is not None:
    MonaiUNetSegmenter3D.register()
if MicroSamYoloSegmenter is not None:
    MicroSamYoloSegmenter.register()
if SkImageWatershedSegmenter is not None:
    SkImageWatershedSegmenter.register()

# Register one scikit-ops op as a global segmenter. It runs in scikit-ops' own
# environment, so this works even though StarDist and TensorFlow are not
# installed here -- which is what lets this host stay on a current napari.
# Explicit rather than discovered: see docs/spec/0004.
try:
    from napari_ai_lab.Segmenters.GlobalSegmenters.StardistSkopSegmenter import (
        StardistSkopSegmenter,
    )

    # One row, with a combo choosing the model -- matching StardistSegmenter,
    # and leaving room for models trained into the project. Fluo and H&E stay
    # two ops in scikit-ops, because they do not accept the same thing: fluo
    # declares Axes("y", "x", "c?") and averages a trailing RGB axis away, he
    # declares Axes("y", "x", "c") and needs the colour. Which op runs is the
    # segmenter's business, not the user's.
    StardistSkopSegmenter.register()
except ImportError as exc:
    print(f"ℹ️  scikit-ops segmenter not registered: {exc}")

# Register interactive segmenters
Square2D.register()
Otsu2D.register()
Otsu3D.register()
SAM3D.register()
SAMSphere3D.register()
RegionGrow3D.register()
FeatureRegionGrow3D.register()
AnisotropicSphereFit3D.register()
HoughSphereFit3D.register()

# Register augmenters
SimpleAugmenter.register()
AlbumentationsAugmenter.register()

# Create viewer
viewer = napari.Viewer()

# Detect napari-ai-lab project root (works on both Windows and Linux)
# This script is in: napari-ai-lab/src/launch_nd_ai_lab.py
# So we go up one level to get to napari-ai-lab root
project_root = Path(__file__).parent.parent
test_images_dir = project_root / "tests" / "test_images"

print(f"📁 Detected project root: {project_root}")
print(f"📁 Test images directory: {test_images_dir}")

test_sets = [
    "vessels",  # 0
    "neurips blood cells",  # 1
    "fluorescent blobs",  # 2
    "czi cells",  # 3
    "cells cropped",  # 4
    "Stardist_3D",  # 5
    "nuclei",  # 6
    "spheres",  # 7
    "overlapping",  # 8
    "tough cellpose",  # 9
    "new",  # 10
]

test_set = test_sets[1]

annotations_viewer_type = "none"

if test_set == "vessels":
    # Load test data (vessels_project)
    # parent_dir = test_images_dir / "vessels_project"
    parent_dir = test_images_dir / "vessels_ds2"
    viewer_type = "none"
    annotations_viewer_type = "stacked"
    axes_to_collapse = None
elif test_set == "neurips blood cells":
    parent_dir = test_images_dir / "neurips blood cells"
    viewer_type = "sequence"
    axes_to_collapse = "C"
    axis_types = "NYXC"
elif test_set == "fluorescent blobs":
    parent_dir = test_images_dir / "fluorescent blobs"
    viewer_type = "stacked"
    axes_to_collapse = None
elif test_set == "czi cells":
    parent_dir = test_images_dir / "czi_cells"
    viewer_type = "none"
    axes_to_collapse = None
elif test_set == "cells cropped":
    parent_dir = test_images_dir / "cells_cropped"
    viewer_type = "none"
    axes_to_collapse = None
elif test_set == "Stardist_3D":
    parent_dir = test_images_dir / "Stardist_3D"
    viewer_type = "sequence"
    axes_to_collapse = None
elif test_set == "nuclei":
    parent_dir = test_images_dir / "nuclei"
    viewer_type = "stacked"
    axes_to_collapse = None
elif test_set == "spheres":
    parent_dir = r"D:\deep-learning\labels\For_AI_lab"
    viewer_type = "stacked"
    axes_to_collapse = None
    axis_types = "NZYX"  # Manually set axis types for testing purposes
elif test_set == "overlapping":
    parent_dir = test_images_dir / "overlapping"
    viewer_type = "stacked"
    axes_to_collapse = "C"
elif test_set == "tough cellpose":
    parent_dir = test_images_dir / "tough_cellpose"
    viewer_type = "stacked"
    axes_to_collapse = "C"
    axis_types = "NYX"
elif test_set == "new":
    parent_dir = (
        r"/home/bnorthan/images/tnia-python-images/imagesc/2026_07_07_bubbles"
    )
    viewer_type = ("none",)
    axes_to_collapse = ("C",)
    axis_types = "YXC"

if args.viewer_type_override is not None:
    print(
        f"🔀 Viewer type overridden on the command line: "
        f"{viewer_type} -> {args.viewer_type_override}"
    )
    viewer_type = args.viewer_type_override

# Create model
# Launch using helper; keep the test-set selection code above unchanged so this
# script remains a clear example for third-party projects.
nd_ai_lab_widget, nd_sequence_viewer_widget, model = launch_nd_ai_lab(
    viewer,
    parent_dir,
    viewer_type=viewer_type,
    axes_to_collapse=axes_to_collapse,
    axis_types=axis_types if "axis_types" in locals() else None,
)

print("✨ ND AI Lab launched successfully!")
print(f"   Loaded: {parent_dir}")
print(f"   Viewer Type: {viewer_type}")
print("   Tabs: Label | Augment | Segment | Train")
print("   Phase 2: Embedded mode with shared model ✅")
print("   Phase 3: Central layer management ✅")
if viewer_type in ["stacked", "sequence"]:
    print(f"   Sequence Viewer: {viewer_type} mode ✅")

napari.run()
