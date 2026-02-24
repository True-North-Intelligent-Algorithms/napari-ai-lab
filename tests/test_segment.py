"""
Test segmentation logic directly using ImageDataModel without NDEasySegment widget.

This test validates the core segmentation workflow:
1. Load image via ImageDataModel
2. Create and configure segmenter
3. Segment image using model.segment()
4. Save predictions using model.save_predictions()
"""

import shutil
from pathlib import Path

import numpy as np

from napari_ai_lab.models import ImageDataModel
from napari_ai_lab.Segmenters.GlobalSegmenters import MonaiUNetSegmenter
from napari_ai_lab.utility import get_current_slice_indices


def test_segment_logic_without_widget():
    """Test segmentation logic directly using model and segmenter without widget."""
    # Setup original and temporary directories
    original_parent_dir = Path(
        "/home/bnorthan/code/i2k/tnia/napari-ai-lab/tests/test_images/vessels_project"
    )
    temp_parent_dir = Path(str(original_parent_dir) + "_segment_test")

    try:
        # Register all global segmenters
        MonaiUNetSegmenter.register()

        # Create temp directory and copy test file
        temp_parent_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            original_parent_dir / "Test lightsheet.tif",
            temp_parent_dir / "Test lightsheet.tif",
        )

        # Create model with temporary directory
        model = ImageDataModel(str(temp_parent_dir))
        model.set_prediction_io_type("tiff_slice", axis_slice="YX")

        # Load image data
        image_index = 0
        image_data = model.load_image(image_index)
        print(f"Loaded image shape: {image_data.shape}")

        # Create and configure segmenter
        segmenter = MonaiUNetSegmenter()
        model_path = str(
            original_parent_dir / "models" / "monai_unet_test.pth"
        )
        segmenter._on_model_path_changed(model_path)
        print(f"Loaded model from: {model_path}")

        # Define selected axis and current step
        selected_axis = "YX"
        # For a 3D image (ZYX), current_step would be like (3, 0, 0)
        # We want slice at Z=3
        # The current_step should match the image dimensionality
        current_step = (3, 0, 0)

        # Extract current slice using same logic as nd_easy_segment
        indices = get_current_slice_indices(current_step, selected_axis)
        current_slice = image_data[indices]
        print(f"Extracted slice shape: {current_slice.shape}")
        print(f"Slice indices: {indices}")

        # Perform segmentation using model's segment method
        mask = model.segment(
            segmenter,
            current_slice,
            points=None,
            shapes=None,
        )
        print(f"Segmentation mask shape: {mask.shape}")
        print(f"Mask unique values: {np.unique(mask)}")

        # Get segmentation axis (handles axis transformations)
        segmentation_axis = segmenter.get_segmentation_axis(selected_axis)
        print(f"Segmentation axis: {segmentation_axis}")

        # Save predictions using model's save_predictions method
        model.save_predictions(
            mask,
            image_index,
            current_step=current_step,
            selected_axis=segmentation_axis,
        )

        # Assert that prediction file was created
        predictions_dir = temp_parent_dir / "predictions"
        prediction_file = predictions_dir / "Test lightsheet_3.tif"

        assert (
            prediction_file.exists()
        ), f"Prediction file not found: {prediction_file}"
        print(f"✅ Test passed: Prediction file created at {prediction_file}")

        # Validate the saved prediction
        from skimage.io import imread

        saved_mask = imread(prediction_file)
        print(f"Saved mask shape: {saved_mask.shape}")
        assert np.array_equal(
            mask, saved_mask
        ), "Saved mask doesn't match original mask"
        print("✅ Saved mask matches original mask")

    finally:
        # Cleanup temporary directory
        if temp_parent_dir.exists():
            shutil.rmtree(temp_parent_dir)


def test_segment_all_slices_logic():
    """Test segmentation of all slices logic without widget."""
    # Setup original and temporary directories
    original_parent_dir = Path(
        "/home/bnorthan/code/i2k/tnia/napari-ai-lab/tests/test_images/vessels_project"
    )
    temp_parent_dir = Path(str(original_parent_dir) + "_segment_all_test")

    try:
        # Register all global segmenters
        MonaiUNetSegmenter.register()

        # Create temp directory and copy test file
        temp_parent_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            original_parent_dir / "Test lightsheet.tif",
            temp_parent_dir / "Test lightsheet.tif",
        )

        # Create model with temporary directory
        model = ImageDataModel(str(temp_parent_dir))
        model.set_prediction_io_type("tiff_slice", axis_slice="YX")

        # Load image data
        image_index = 0
        image_data = model.load_image(image_index)
        image_shape = image_data.shape
        print(f"Loaded image shape: {image_shape}")

        # Create and configure segmenter
        segmenter = MonaiUNetSegmenter()
        model_path = str(
            original_parent_dir / "models" / "monai_unet_test.pth"
        )
        segmenter._on_model_path_changed(model_path)

        # Define selected axis
        selected_axis = "YX"

        # Calculate non-spatial dimensions (same logic as _on_segment_all)
        num_spatial = 2  # YX has 2 spatial dimensions
        num_non_spatial = len(image_shape) - num_spatial
        non_spatial_shape = image_shape[:num_non_spatial]

        # Calculate total number of slices to process
        total_slices = (
            int(np.prod(non_spatial_shape)) if non_spatial_shape else 1
        )
        print(f"Total slices to segment: {total_slices}")
        print(f"Non-spatial shape: {non_spatial_shape}")

        # Segment only first 3 slices for testing (to keep test fast)
        import itertools

        num_test_slices = min(3, total_slices)

        for idx, non_spatial_indices in enumerate(
            itertools.product(*[range(dim) for dim in non_spatial_shape])
        ):
            if idx >= num_test_slices:
                break

            # Build current_step tuple
            current_step = non_spatial_indices + (0,) * num_spatial

            # Extract slice
            indices = get_current_slice_indices(current_step, selected_axis)
            current_slice = image_data[indices]

            # Segment
            mask = model.segment(
                segmenter,
                current_slice,
                points=None,
                shapes=None,
            )

            # Get segmentation axis
            segmentation_axis = segmenter.get_segmentation_axis(selected_axis)

            # Save predictions
            model.save_predictions(
                mask,
                image_index,
                current_step=current_step,
                selected_axis=segmentation_axis,
            )

            print(
                f"Segmented slice {idx + 1}/{num_test_slices}: {current_step}"
            )

        # Verify prediction files were created
        predictions_dir = temp_parent_dir / "predictions"
        assert predictions_dir.exists(), "Predictions directory not found"

        # Check that at least the test slices were created
        prediction_files = list(predictions_dir.glob("*.tif"))
        assert len(prediction_files) >= num_test_slices, (
            f"Expected at least {num_test_slices} prediction files, "
            f"found {len(prediction_files)}"
        )
        print(
            f"✅ Test passed: Created {len(prediction_files)} prediction files"
        )
        for pf in prediction_files[:num_test_slices]:
            print(f"  - {pf.name}")

    finally:
        # Cleanup temporary directory
        if temp_parent_dir.exists():
            shutil.rmtree(temp_parent_dir)


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Segment single slice")
    print("=" * 60)
    test_segment_logic_without_widget()

    print("\n" + "=" * 60)
    print("Test 2: Segment multiple slices")
    print("=" * 60)
    test_segment_all_slices_logic()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
