"""
Interactive Test for Threshold Segmentation.

This script provides an interactive test environment for validating
threshold segmentation functionality with real data using matplotlib.

This is NOT a unit test - it's meant for manual testing and validation.
Demonstrates that segmenters work independently of napari.
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage import data

# Import your segmenters
from napari_ai_lab.Segmenters import ThresholdSegmenter


def test_threshold_interactive():
    """Interactive test for Threshold segmentation using matplotlib."""

    # Create some test data
    print("Loading test image...")
    image = data.camera()  # Example grayscale image
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Image range: {image.min():.1f} - {image.max():.1f}")

    # Test different threshold values
    threshold_values = [100, 128, 150, 200]

    print(f"\nTesting threshold segmentation with values: {threshold_values}")

    # Create figure for comparison
    fig, axes = plt.subplots(2, len(threshold_values) + 1, figsize=(20, 8))

    # Show original image
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")  # Empty bottom-left cell

    for i, threshold in enumerate(threshold_values):
        col = i + 1

        print(f"\nTesting threshold = {threshold}")

        # Create threshold segmenter
        threshold_segmenter = ThresholdSegmenter(
            threshold=threshold, invert_mask=False
        )

        print(f"Segmenter: {threshold_segmenter}")
        print(f"Parameters: {threshold_segmenter.get_parameters_dict()}")

        # Perform segmentation
        mask = threshold_segmenter.segment(image)

        # Calculate statistics
        foreground_pixels = np.sum(mask)
        total_pixels = mask.size
        foreground_percentage = (foreground_pixels / total_pixels) * 100

        print(
            f"Foreground pixels: {foreground_pixels}/{total_pixels} ({foreground_percentage:.1f}%)"
        )

        # Display segmentation mask
        axes[0, col].imshow(mask, cmap="gray")
        axes[0, col].set_title(f"Threshold = {threshold}")
        axes[0, col].axis("off")

        # Display overlay
        axes[1, col].imshow(image, cmap="gray")
        # Create colored overlay where mask is True
        colored_mask = np.zeros((*mask.shape, 3))
        colored_mask[mask == 1] = [1, 0, 0]  # Red for foreground
        axes[1, col].imshow(colored_mask, alpha=0.3)
        axes[1, col].set_title(f"Overlay ({foreground_percentage:.1f}% fg)")
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.suptitle("Threshold Segmentation Comparison", fontsize=16, y=0.98)

    print("\nClose the matplotlib window to continue...")
    plt.show()

    # Test with inverted mask
    print("\n" + "=" * 50)
    print("Testing with inverted mask...")

    inverted_segmenter = ThresholdSegmenter(threshold=128, invert_mask=True)
    inverted_mask = inverted_segmenter.segment(image)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Normal mask
    normal_segmenter = ThresholdSegmenter(threshold=128, invert_mask=False)
    normal_mask = normal_segmenter.segment(image)
    axes[1].imshow(normal_mask, cmap="gray")
    axes[1].set_title("Normal Mask (threshold=128)")
    axes[1].axis("off")

    # Inverted mask
    axes[2].imshow(inverted_mask, cmap="gray")
    axes[2].set_title("Inverted Mask (threshold=128)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.suptitle("Normal vs Inverted Threshold Masks", fontsize=16, y=1.02)

    print("Close the matplotlib window to finish...")
    plt.show()

    print("\n" + "=" * 50)
    print("Interactive threshold test completed successfully!")
    print("This demonstrates that segmenters work independently of napari.")
    print("You can easily adjust threshold values and see immediate results.")
    print("=" * 50)


if __name__ == "__main__":
    test_threshold_interactive()
