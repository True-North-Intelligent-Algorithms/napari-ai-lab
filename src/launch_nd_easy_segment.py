import napari

from napari_ai_lab.models import ImageDataModel
from napari_ai_lab.nd_easy_segment import NDEasySegment
from napari_ai_lab.nd_sequence_viewer import NDSequenceViewer
from napari_ai_lab.Segmenters.GlobalSegmenters import (
    CellposeSegmenter,
    StardistSegmenter,
    ThresholdSegmenter,
)

# Register all global segmenters
CellposeSegmenter.register()
StardistSegmenter.register()
ThresholdSegmenter.register()

viewer = napari.Viewer()
parent_dir = (
    # r"D:\images\tnia-python-images\imagesc\2025_09_29_gray_scale_3d_test_set"
    r"D:\images\tnia-python-images\imagesc\2025_10_16_grayscale_subset2"
)

model = ImageDataModel(parent_dir)

# Add the NDEasySegment widget to the viewer
nd_easy_segment_widget = NDEasySegment(viewer, model)
viewer.window.add_dock_widget(nd_easy_segment_widget)

# nd_easy_segment_widget.load_image_directory(parent_dir)

# Add the NDSequenceViewer widget to the viewer
nd_sequence_viewer_widget = NDSequenceViewer(viewer)
viewer.window.add_dock_widget(
    nd_sequence_viewer_widget, name="Sequence Viewer", area="bottom"
)

# Connect sequence viewer to easy segment for automatic layer updates
nd_easy_segment_widget.connect_sequence_viewer(nd_sequence_viewer_widget)

# Automatically load images from the parent directory into sequence viewer
nd_sequence_viewer_widget.set_image_data_model(model)

napari.run()
