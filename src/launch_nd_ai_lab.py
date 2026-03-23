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
    MonaiUNetSegmenter,
    StardistSegmenter,
    ThresholdSegmenter,
)
from napari_ai_lab.Segmenters.InteractiveSegmenters import (
    SAM3D,
    Otsu2D,
    Otsu3D,
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

# Register interactive segmenters
Square2D.register()
Otsu2D.register()
Otsu3D.register()
SAM3D.register()

# Register augmenters
SimpleAugmenter.register()
AlbumentationsAugmenter.register()

# Create viewer
viewer = napari.Viewer()

test_sets = ["vessels", "neurips blood cells", "fluorescent blobs"]
test_set = test_sets[2]

if test_set == "vessels":
    # Load test data (vessels_project)
    parent_dir = Path(
        "/home/bnorthan/code/i2k/tnia/napari-ai-lab/tests/test_images/vessels_project"
    )
    viewer_type = "none"
    axis_to_collapse = None
elif test_set == "neurips blood cells":
    parent_dir = Path(
        "/home/bnorthan/code/i2k/tnia/napari-ai-lab/tests/test_images/neurips blood cells"
    )
    viewer_type = "stacked"
    axis_to_collapse = "C"
elif test_set == "fluorescent blobs":
    parent_dir = Path(
        "/home/bnorthan/code/i2k/tnia/napari-ai-lab/tests/test_images/fluorescent blobs"
    )
    viewer_type = "stacked"
    axis_to_collapse = None

# Create model
model = ImageDataModel(parent_dir)

##### HACK
model.axis_types = "NYXC"  # Manually set axis types for testing purposes

# Configure annotation and prediction writer types based on viewer_type
if viewer_type == "stacked":
    model.set_annotation_io_type("stacked_sequence")
    model.set_prediction_io_type("stacked_sequence")

# Create combined widget WITH model
nd_ai_lab_widget = NDAILab(viewer, model, axes_to_collapse=axis_to_collapse)
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
    image_layer = viewer.add_image(image_data, name="Image")

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
