import napari

from napari_ai_lab.apps.nd_easy_segment import NDEasySegment
from napari_ai_lab.models import ImageDataModel
from napari_ai_lab.nd_sequence_viewer import NDSequenceViewer
from napari_ai_lab.nd_stacked_sequence_viewer import NDStackedSequenceViewer
from napari_ai_lab.Segmenters.GlobalSegmenters import (
    CellCastStardistSegmenter,
    CellposeSegmenter,
    MicroSamSegmenter,
    MonaiUNetSegmenter,
    SkImageWatershedSegmenter,
    StardistSegmenter,
    ThresholdSegmenter,
)

# Flag to control viewer type
viewer_type = "none"  # Options: "none", "stacked", "sequence"

# Register all global segmenters
CellposeSegmenter.register()
StardistSegmenter.register()
CellCastStardistSegmenter.register()
ThresholdSegmenter.register()
MicroSamSegmenter.register()
MonaiUNetSegmenter.register()
if SkImageWatershedSegmenter is not None:
    SkImageWatershedSegmenter.register()

viewer = napari.Viewer()

parent_dir = (
    # r"D:\images\tnia-python-images\imagesc\2025_09_29_gray_scale_3d_test_set"
    # r"D:\images\tnia-python-images\imagesc\2025_10_16_grayscale_subset2"
    # r"D:\deep-learning\test\dx4"
    # r"/home/bnorthan/dplexbio/images/dx4/"
    # r"D:\dplexbio\Nov 2025\model_o_data\testing"
    # r'D:\images\tnia-python-images\imagesc\2025_12_08_ND_Segmentation'
    # r"/home/bnorthan/images/tnia-python-images/imagesc/2026_02_07_vessels_czi/",
    r"/home/bnorthan/code/i2k/tnia/napari-ai-lab/tests/test_images/vessels_project"
    # r"/home/bnorthan/code/i2k/tnia/napari-ai-lab/tests/test_images/vessels_project"
)

model = ImageDataModel(parent_dir)

# Configure annotation and prediction writer types based on stacked flag
if viewer_type == "stacked":
    model.set_annotation_io_type("stacked_sequence")
    model.set_prediction_io_type("stacked_sequence")

model.set_prediction_io_type("zarr", axis_slice="YX")

# Add the NDEasySegment widget to the viewer
# Use "dialog" mode for standalone app (classic popup training dialog)
nd_easy_segment_widget = NDEasySegment(
    viewer, model, training_widget_mode="dialog"
)
viewer.window.add_dock_widget(nd_easy_segment_widget)

nd_easy_segment_widget.automatic_mode_btn.setChecked(True)

segmenter_name = "MonaiUNetSegmenter"

nd_easy_segment_widget.segmenter_combo.setCurrentText(segmenter_name)
segmenter = nd_easy_segment_widget.segmenter_cache[segmenter_name]

model_path = "/home/bnorthan/code/i2k/tnia/napari-ai-lab/tests/test_images/vessels_project/models/monai_unet_test.pth"

print(f"Using segmenter: {segmenter}")

segmenter.load_model(model_path)
nd_easy_segment_widget._update_segmenter_parameter_form(segmenter)
nd_easy_segment_widget.segmenter_parameter_form.set_selected_axis("YX")
# nd_easy_segment_widget.load_image_directory(parent_dir)

# Add the appropriate sequence viewer widget based on stacked flag
if viewer_type == "stacked":
    nd_sequence_viewer_widget = NDStackedSequenceViewer(viewer)
elif viewer_type == "sequence":
    nd_sequence_viewer_widget = NDSequenceViewer(viewer)

if viewer_type in ["stacked", "sequence"]:
    viewer.window.add_dock_widget(
        nd_sequence_viewer_widget, name="Sequence Viewer", area="bottom"
    )

    # Connect sequence viewer to easy segment for automatic layer updates
    nd_easy_segment_widget.connect_sequence_viewer(nd_sequence_viewer_widget)

    # Automatically load images from the parent directory into sequence viewer
    nd_sequence_viewer_widget.set_image_data_model(model)
else:
    image_data = model.load_image(0)
    image_layer = viewer.add_image(image_data, name="Image")
    nd_easy_segment_widget._set_image_layer(image_layer)

napari.run()
