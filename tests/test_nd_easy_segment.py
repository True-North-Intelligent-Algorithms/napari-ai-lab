import shutil
from pathlib import Path

from napari_ai_lab.apps.nd_easy_segment import NDEasySegment
from napari_ai_lab.models import ImageDataModel
from napari_ai_lab.Segmenters.GlobalSegmenters import MonaiUNetSegmenter


def test_nd_easy_segment_creates_prediction():
    """Test that NDEasySegment creates prediction file in predictions directory."""
    # Setup original and temporary directories
    original_parent_dir = (
        Path(__file__).parent / "test_images" / "vessels_project"
    )
    temp_parent_dir = original_parent_dir.parent / "vessels_project_temp"

    try:
        # Register all global segmenters
        MonaiUNetSegmenter.register()

        # Create temp directory and copy test file
        temp_parent_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            original_parent_dir / "Test lightsheet.tif",
            temp_parent_dir / "Test lightsheet.tif",
        )

        # Use the temporary directory for the model
        model = ImageDataModel(str(temp_parent_dir))
        model.set_prediction_io_type("tiff_slice", axis_slice="YX")

        # Create viewer without showing it
        import napari

        viewer = napari.Viewer(show=False)

        # Add the NDEasySegment widget to the viewer
        nd_easy_segment_widget = NDEasySegment(viewer, model)
        # viewer.window.add_dock_widget(nd_easy_segment_widget)
        nd_easy_segment_widget.automatic_mode_btn.setChecked(True)

        segmenter_name = "MonaiUNetSegmenter"
        nd_easy_segment_widget.segmenter_combo.setCurrentText(segmenter_name)
        segmenter = nd_easy_segment_widget.segmenter_cache[segmenter_name]
        model_path = str(
            original_parent_dir / "models" / "monai_unet_test.pth"
        )

        segmenter.load_model(model_path)
        nd_easy_segment_widget._update_segmenter_parameter_form(segmenter)
        nd_easy_segment_widget.segmenter_parameter_form.set_selected_axis("YX")

        image_data = model.load_image(0)
        image_layer = viewer.add_image(image_data, name="Image")
        nd_easy_segment_widget._set_image_layer(image_layer)

        # Programmatically segment the current image
        nd_easy_segment_widget._on_segment_current()

        # Assert that prediction file was created
        predictions_dir = model.get_predictions_directory(
            algorithm=segmenter_name
        )
        prediction_file = predictions_dir / "Test lightsheet_3.tif"

        assert (
            prediction_file.exists()
        ), f"Prediction file not found: {prediction_file}"
        print(f"✅ Test passed: Prediction file created at {prediction_file}")

        # Close viewer
        # viewer.close()

    finally:
        # Cleanup temporary directory
        if temp_parent_dir.exists():
            shutil.rmtree(temp_parent_dir)


if __name__ == "__main__":
    test_nd_easy_segment_creates_prediction()
