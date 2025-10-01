import napari

from napari_ai_lab.nd_sequence_viewer import NDSequenceViewer

viewer = napari.Viewer()

parent_dir = (
    r"D:\images\tnia-python-images\imagesc\2025_09_29_gray_scale_3d_test_set"
)

# Add the NDEasyLabel widget to the viewer
nd_series_viewer = NDSequenceViewer(viewer)
viewer.window.add_dock_widget(
    nd_series_viewer, name="ND Series Viewer", area="bottom"
)

# Automatically load images from the parent directory
# nd_easy_label_widget.load_image_directory(parent_dir)

napari.run()
