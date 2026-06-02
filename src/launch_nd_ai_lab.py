"""
Launch script for ND AI Lab - Combined tabbed interface.

This script launches the combined AI Lab plugin with Label, Augment,
and Segment tabs in a single interface.
"""

from pathlib import Path

import napari

from napari_ai_lab.apps.nd_ai_lab import NDAILab
from napari_ai_lab.Augmenters import (
    AlbumentationsAugmenter,
    SimpleAugmenter,
)
from napari_ai_lab.models import ImageDataModel
from napari_ai_lab.nd_sequence_viewer import NDSequenceViewer
from napari_ai_lab.nd_stacked_sequence_viewer import NDStackedSequenceViewer

# Register all segmenters and augmenters
from napari_ai_lab.Segmenters.GlobalSegmenters import (
    CellCastStardistSegmenter,
    CellposeSegmenter,
    MicroSamSegmenter,
    MicroSamYoloSegmenter,
    MonaiUNetSegmenter,
    MonaiUNetSegmenter3D,
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
]
test_set = test_sets[7]

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
    viewer_type = "stacked"
    axes_to_collapse = "C"
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
elif test_set == "overlapping":
    parent_dir = test_images_dir / "overlapping"
    viewer_type = None
    axes_to_collapse = None

# Create model
model = ImageDataModel(parent_dir)

##### HACK
model.axis_types = "NZYX"  # Manually set axis types for testing purposes

# Configure annotation and prediction writer types based on viewer_type
if viewer_type == "stacked" or annotations_viewer_type == "stacked":
    model.set_annotation_io_type(
        "stacked_sequence", axes_to_collapse=axes_to_collapse
    )
    model.set_prediction_io_type(
        "stacked_sequence", axes_to_collapse=axes_to_collapse
    )
    # Set save granularity for testing
    model.set_annotation_save_granularity("YX")
    model.set_prediction_save_granularity("YX")

# Create combined widget WITH model
nd_ai_lab_widget = NDAILab(viewer, model, axes_to_collapse=axes_to_collapse)
viewer.window.add_dock_widget(nd_ai_lab_widget, area="right", name="AI Lab")

# Add the appropriate sequence viewer widget based on viewer_type
if viewer_type == "stacked":
    nd_sequence_viewer_widget = NDStackedSequenceViewer(viewer)
elif viewer_type == "sequence":
    nd_sequence_viewer_widget = NDSequenceViewer(viewer)

if viewer_type in ["stacked", "sequence"]:
    viewer.window.add_dock_widget(
        nd_sequence_viewer_widget, name="Sequence Viewer", area="bottom"
    )

    # Connect sequence viewer to segment widget for automatic layer updates
    nd_ai_lab_widget.connect_sequence_viewer(nd_sequence_viewer_widget)

    # Automatically load images from the parent directory into sequence viewer
    nd_sequence_viewer_widget.set_image_data_model(model)
else:
    # Load first image (viewer_type = "none" logic)
    image_data = model.load_image(0)
    image_layer = viewer.add_image(
        image_data, name="Image", scale=model.get_scale()
    )

    # Phase 3: Central layer setup - call nd_ai_lab's _set_image_layer
    # This creates all layers once and distributes to sub-apps
    nd_ai_lab_widget._set_image_layer(image_layer)

nd_ai_lab_widget.segment_widget.automatic_mode_btn.setChecked(True)

print("✨ ND AI Lab launched successfully!")
print(f"   Loaded: {parent_dir}")
print(f"   Viewer Type: {viewer_type}")
print("   Tabs: Label | Augment | Segment | Train")
print("   Phase 2: Embedded mode with shared model ✅")
print("   Phase 3: Central layer management ✅")
if viewer_type in ["stacked", "sequence"]:
    print(f"   Sequence Viewer: {viewer_type} mode ✅")

napari.run()
