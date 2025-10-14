import napari

from napari_ai_lab.nd_easy_segment import NDEasySegment
from napari_ai_lab.Segmenters.GlobalSegmenters import (
    CellposeSegmenter,
    ThresholdSegmenter,
)

# Register all global segmenters
CellposeSegmenter.register()
ThresholdSegmenter.register()

viewer = napari.Viewer()
parent_dir = (
    r"D:\images\tnia-python-images\imagesc\2025_09_29_gray_scale_3d_test_set"
)

# Add the NDEasySegment widget to the viewer
nd_easy_segment_widget = NDEasySegment(viewer)
viewer.window.add_dock_widget(nd_easy_segment_widget)

nd_easy_segment_widget.load_image_directory(parent_dir)

napari.run()
