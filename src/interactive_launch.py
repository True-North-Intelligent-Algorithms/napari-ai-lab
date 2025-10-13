import napari

from napari_ai_lab.nd_easy_label import NDEasyLabel
from napari_ai_lab.nd_sequence_viewer import NDSequenceViewer
from napari_ai_lab.Segmenters.InteractiveSegmenters import (
    SAM3D,
    Otsu2D,
    Otsu3D,
    Square2D,
)

# Register all interactive segmenters
Square2D.register()
Otsu2D.register()
Otsu3D.register()
SAM3D.register()

viewer = napari.Viewer()

parent_dir = (
    r"D:\images\tnia-python-images\imagesc\2025_09_29_gray_scale_3d_test_set"
)

# Add the NDSequenceViewer widget to the viewer
nd_sequence_viewer_widget = NDSequenceViewer(viewer)
viewer.window.add_dock_widget(
    nd_sequence_viewer_widget, name="Sequence Viewer", area="bottom"
)

# Add the NDEasyLabel widget to the viewer
nd_easy_label_widget = NDEasyLabel(viewer)
viewer.window.add_dock_widget(
    nd_easy_label_widget, name="ND Easy Label", area="right"
)

# Connect sequence viewer to easy label for automatic layer updates
nd_easy_label_widget.connect_sequence_viewer(nd_sequence_viewer_widget)

# Automatically load images from the parent directory into sequence viewer
nd_sequence_viewer_widget._load_image_list(parent_dir)

napari.run()
