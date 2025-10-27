"""
Interactive StarDist Test - Local and Remote Execution.

This script demonstrates StarDist segmentation in two modes:
1. Local execution (if stardist available)
2. Remote execution via appose (if stardist not available locally)

Provides visual feedback and concise status messages.
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage import data

from napari_ai_lab.Segmenters.execute_appose import execute_appose
from napari_ai_lab.Segmenters.GlobalSegmenters.StardistSegmenter import (
    StardistSegmenter,
)

# Remote stardist environment path
STARDIST_ENV_PATH = r"C:\\Users\\bnort\\miniconda3\\envs\\stardist_env"
STARDIST_ENV_PATH = r"/home/bnorthan/mambaforge/envs/easy_augment_stardist_085"


def test_stardist_interactive():
    """Interactive test showing local vs remote StarDist execution."""

    print("🧪 Interactive StarDist Test")
    print("=" * 50)

    # Load test image
    image = data.coins()
    print(f"📸 Image: {image.shape}, range: [{image.min()}-{image.max()}]")

    # Simple StarDist configuration
    segmenter = StardistSegmenter(
        model_preset="2D_versatile_fluo", prob_thresh=0.5, nms_thresh=0.4
    )
    run_in_local_environment = segmenter.are_dependencies_available()

    print(
        f"🔍 StarDist local: {'✅ Available' if run_in_local_environment else '❌ Not found'}"
    )
    print(f"🌐 Remote path: {STARDIST_ENV_PATH}")
    print("⚙️  Model: 2D_versatile_fluo")

    # Create simple figure - original and result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Show original image
    ax1.imshow(image, cmap="gray")
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Execute StarDist
    try:
        if run_in_local_environment:
            # Local execution
            print("🏠 Running StarDist locally...")
            result = segmenter.segment(image)

            num_objects = len(np.unique(result)) - 1
            ax2.imshow(result, cmap="nipy_spectral")
            ax2.set_title(f"✅ Found {num_objects} objects")
            print(f"✅ Success: Found {num_objects} objects")

        else:
            # Remote execution via appose
            print("🌐 Running StarDist with Appose...")
            result = execute_appose(image, segmenter, STARDIST_ENV_PATH)

            mask_array = result.ndarray()
            num_objects = len(np.unique(mask_array)) - 1
            ax2.imshow(mask_array, cmap="nipy_spectral")
            ax2.set_title(f"🌐 Found {num_objects} objects")
            print(f"✅ Remote success: Found {num_objects} objects")

    except (ValueError, AttributeError, RuntimeError, TypeError) as e:
        print(f"❌ Error: {e}")
        ax2.text(
            0.5,
            0.5,
            f"Error:\n{str(e)[:50]}...",
            transform=ax2.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="red",
        )
        ax2.set_title("❌ Error")
        ax2.axis("off")

    # Simple summary
    mode = (
        "Local Environment"
        if run_in_local_environment
        else "Appose Environment"
    )
    fig.suptitle(
        f"StarDist Segmentation Test run with {mode}", fontsize=14, y=0.95
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    print(f"\n📊 Tested StarDist using {mode} execution")
    print("👀 Close window to finish...")

    plt.show()

    print("\n✅ Interactive StarDist test completed!")


if __name__ == "__main__":
    test_stardist_interactive()
