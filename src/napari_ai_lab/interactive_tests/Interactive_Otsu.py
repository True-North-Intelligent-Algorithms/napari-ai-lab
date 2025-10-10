"""
Interactive Test for Otsu Segmentation.

This script provides an interactive test environment for validating
Otsu segmentation functionality with real data using matplotlib.

This is NOT a unit test - it's meant for manual testing and validation.
Demonstrates that segmenters work independently of napari.
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage import data

# Import segmenter directly to avoid SAM dependencies
from napari_ai_lab.Segmenters.GlobalSegmenters.OtsuSegmenter import (
    OtsuSegmenter,
)


def test_otsu_interactive():
    """Interactive test for Otsu segmentation using matplotlib."""

    # Create some test data
    print("Loading test image...")
    image = data.coins()  # Example image
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Image range: {image.min():.1f} - {image.max():.1f}")

    # Create Otsu segmenter
    print("\nCreating Otsu segmenter...")
    otsu_segmenter = OtsuSegmenter(invert_mask=False)
    print(f"Segmenter: {otsu_segmenter}")
    print(f"Supported axes: {otsu_segmenter.supported_axes}")
    print(f"Parameters: {otsu_segmenter.get_parameters_dict()}")

    # Perform segmentation
    print("\nPerforming segmentation...")
    mask = otsu_segmenter.segment(image)
    print("Segmentation complete!")
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"Mask unique values: {np.unique(mask)}")

    # Display results using matplotlib
    print("\nDisplaying results with matplotlib...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Segmentation mask
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Otsu Segmentation Mask")
    axes[1].axis("off")

    # Overlay (original image with mask outline)
    axes[2].imshow(image, cmap="gray")
    # Create mask outline by finding edges
    from skimage import segmentation

    boundaries = segmentation.find_boundaries(mask, mode="outer")
    axes[2].imshow(
        np.ma.masked_where(~boundaries, boundaries), cmap="Reds", alpha=0.7
    )
    axes[2].set_title("Overlay: Image + Mask Boundaries")
    axes[2].axis("off")

    plt.tight_layout()
    plt.suptitle("Otsu Segmentation Results", fontsize=16, y=1.02)

    print("Close the matplotlib window to continue...")
    plt.show()

    print("\n" + "=" * 50)
    print("Interactive test completed successfully!")
    print("This demonstrates that segmenters work independently of napari.")
    print("=" * 50)


if __name__ == "__main__":
    test_otsu_interactive()
