"""
Launch script for ND Easy Augment app.

Simple launcher to explore the augmentation widget interface.
"""

import napari

from napari_ai_lab.apps.nd_easy_augment import NDEasyAugment
from napari_ai_lab.models import ImageDataModel
from napari_ai_lab.nd_sequence_viewer import NDSequenceViewer

# Flag to control viewer type
viewer_type = None  # Options: None, "sequence"

viewer = napari.Viewer()

# Use a test directory with images
parent_dir = r"/home/bnorthan/code/i2k/tnia/napari-ai-lab/tests/test_images/vessels_project"

model = ImageDataModel(parent_dir)

# Add the NDEasyAugment widget to the viewer
nd_easy_augment_widget = NDEasyAugment(viewer, model)
viewer.window.add_dock_widget(
    nd_easy_augment_widget, name="ND Easy Augment", area="right"
)

# Add the appropriate sequence viewer widget based on viewer_type
if viewer_type == "sequence":
    # Add the NDSequenceViewer widget to the viewer
    nd_sequence_viewer_widget = NDSequenceViewer(viewer)
    viewer.window.add_dock_widget(
        nd_sequence_viewer_widget, name="Sequence Viewer", area="bottom"
    )

    # Connect sequence viewer to augment widget for automatic layer updates
    nd_easy_augment_widget.connect_sequence_viewer(nd_sequence_viewer_widget)

    # Automatically load images from the parent directory into sequence viewer
    nd_sequence_viewer_widget.set_image_data_model(model)
else:
    # No sequence viewer - load image directly
    image_data = model.load_image(0)
    image_layer = viewer.add_image(image_data, name="Image")
    nd_easy_augment_widget._set_image_layer(image_layer)

napari.run()
