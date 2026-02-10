import napari

from napari_ai_lab.apps.nd_easy_segment import NDEasySegment
from napari_ai_lab.models import ImageDataModel
from napari_ai_lab.nd_sequence_viewer import NDSequenceViewer
from napari_ai_lab.nd_stacked_sequence_viewer import NDStackedSequenceViewer
from napari_ai_lab.Segmenters.GlobalSegmenters import (
    CellposeSegmenter,
    MicroSamSegmenter,
    StardistSegmenter,
    ThresholdSegmenter,
)

# Flag to control viewer type
stacked = True

# Register all global segmenters
CellposeSegmenter.register()
StardistSegmenter.register()
ThresholdSegmenter.register()
MicroSamSegmenter.register()

viewer = napari.Viewer()
parent_dir = (
    # r"D:\images\tnia-python-images\imagesc\2025_09_29_gray_scale_3d_test_set"
    # r"D:\images\tnia-python-images\imagesc\2025_10_16_grayscale_subset2"
    # r"D:\deep-learning\test\dx4"
    r"/home/bnorthan/dplexbio/images/dx4/"
    # r"D:\dplexbio\Nov 2025\model_o_data\testing"
    # r'D:\images\tnia-python-images\imagesc\2025_12_08_ND_Segmentation'
)

model = ImageDataModel(parent_dir)

# Configure annotation and prediction writer types based on stacked flag
if stacked:
    model.set_annotation_io_type("stacked_sequence")
    model.set_prediction_io_type("stacked_sequence")

# Add the NDEasySegment widget to the viewer
nd_easy_segment_widget = NDEasySegment(viewer, model)
viewer.window.add_dock_widget(nd_easy_segment_widget)

nd_easy_segment_widget.automatic_mode_btn.setChecked(True)
nd_easy_segment_widget.segmenter_combo.setCurrentText("StardistSegmenter")

segmenter = nd_easy_segment_widget.segmenter_cache["StardistSegmenter"]
print(f"Using segmenter: {segmenter}")

if segmenter.are_dependencies_available():
    segmenter._on_model_path_changed(r"D:\deep-learning\models\test_256_16")
    nd_easy_segment_widget._update_parameter_form(segmenter)
    nd_easy_segment_widget.parameter_form.set_selected_axis("ZYX")
    # nd_easy_segment_widget.load_image_directory(parent_dir)

# Add the appropriate sequence viewer widget based on stacked flag
if stacked:
    nd_sequence_viewer_widget = NDStackedSequenceViewer(viewer)
else:
    nd_sequence_viewer_widget = NDSequenceViewer(viewer)

viewer.window.add_dock_widget(
    nd_sequence_viewer_widget, name="Sequence Viewer", area="bottom"
)

# Connect sequence viewer to easy segment for automatic layer updates
nd_easy_segment_widget.connect_sequence_viewer(nd_sequence_viewer_widget)

# Automatically load images from the parent directory into sequence viewer
nd_sequence_viewer_widget.set_image_data_model(model)

napari.run()
