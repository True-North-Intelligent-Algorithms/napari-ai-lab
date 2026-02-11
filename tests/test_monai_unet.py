"""
Test MONAI UNet Segmenter functionality.
"""

from pathlib import Path

import numpy as np
import pytest


def test_monai_unet_segmentation():
    """Test MONAI UNet segmentation on vessels_small.tif."""
    # Paths
    test_dir = Path(__file__).parent / "test_images" / "vessels_small"
    image_path = test_dir / "vessels_small.tif"
    model_path = test_dir / "vessels_plus.pth"
    output_path = test_dir / "vessels_small_segmentation.tif"

    # Check if required files exist
    if not image_path.exists():
        pytest.skip(f"Test image not found: {image_path}")

    if not model_path.exists():
        pytest.skip(f"Model file not found: {model_path}")

    # Import required modules
    try:
        import tifffile

        from napari_ai_lab.Segmenters.GlobalSegmenters.MonaiUNetSegmenter import (
            MonaiUNetSegmenter,
        )
    except ImportError as e:
        pytest.skip(f"Required package not installed: {e}")

    # Load the image
    print(f"\nLoading image: {image_path}")
    image = tifffile.imread(str(image_path))
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image range: [{image.min()}, {image.max()}]")

    # Check if dependencies are available
    segmenter = MonaiUNetSegmenter()
    if not segmenter.are_dependencies_available():
        pytest.skip("MONAI dependencies not available")

    # Load the model
    print(f"\nLoading model: {model_path}")
    segmenter.load_model(str(model_path))
    print(f"Model loaded: {segmenter.model_name}")

    # Check if output already exists
    if output_path.exists():
        print(f"\nOutput already exists: {output_path}")
        existing_segmentation = tifffile.imread(str(output_path))
        print(f"Existing segmentation shape: {existing_segmentation.shape}")
        print(f"Existing segmentation dtype: {existing_segmentation.dtype}")
        print(
            f"Existing segmentation range: [{existing_segmentation.min()}, {existing_segmentation.max()}]"
        )
        print(f"Unique labels in existing: {np.unique(existing_segmentation)}")

        # Perform segmentation
        print("\nPerforming segmentation...")
        segmentation = segmenter.segment(image)
        print(f"Segmentation shape: {segmentation.shape}")
        print(f"Segmentation dtype: {segmentation.dtype}")
        print(
            f"Segmentation range: [{segmentation.min()}, {segmentation.max()}]"
        )
        print(f"Unique labels: {np.unique(segmentation)}")

        # Compare with existing
        print("\nComparing with existing segmentation...")
        if segmentation.shape == existing_segmentation.shape:
            if np.array_equal(segmentation, existing_segmentation):
                print("✓ Segmentation matches existing output")
            else:
                diff_pixels = np.sum(segmentation != existing_segmentation)
                total_pixels = segmentation.size
                diff_percent = (diff_pixels / total_pixels) * 100
                print(
                    f"⚠ Segmentation differs: {diff_pixels}/{total_pixels} pixels ({diff_percent:.2f}%)"
                )
        else:
            print(
                f"⚠ Shape mismatch: {segmentation.shape} vs {existing_segmentation.shape}"
            )

        # Basic assertions
        assert segmentation is not None, "Segmentation should not be None"
        assert (
            segmentation.shape == image.shape
        ), "Segmentation shape should match image shape"
        assert segmentation.dtype == np.uint16, "Segmentation should be uint16"

    else:
        # Perform segmentation
        print("\nPerforming segmentation (first run)...")
        segmentation = segmenter.segment(image)
        print(f"Segmentation shape: {segmentation.shape}")
        print(f"Segmentation dtype: {segmentation.dtype}")
        print(
            f"Segmentation range: [{segmentation.min()}, {segmentation.max()}]"
        )
        print(f"Unique labels: {np.unique(segmentation)}")

        # Save the output
        print(f"\nSaving segmentation to: {output_path}")
        tifffile.imwrite(str(output_path), segmentation)
        print("✓ Segmentation saved successfully")

        # Basic assertions
        assert segmentation is not None, "Segmentation should not be None"
        assert (
            segmentation.shape == image.shape
        ), "Segmentation shape should match image shape"
        assert segmentation.dtype == np.uint16, "Segmentation should be uint16"
        assert output_path.exists(), "Output file should be created"


if __name__ == "__main__":
    # Run the test directly
    test_monai_unet_segmentation()
