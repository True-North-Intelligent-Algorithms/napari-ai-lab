"""
Interactive Cellpose Test - Local and Remote Execution.

This script demonstrates Cellpose segmentation in two modes:
1. Local execution (if cellpose available)
2. Remote execution via appose (if cellpose not available locally)

Provides visual feedback and concise status messages.
"""

import matplotlib.pyplot as plt
import numpy as np
from skimage import data

from napari_ai_lab.Segmenters.execute_appose import execute_appose
from napari_ai_lab.Segmenters.GlobalSegmenters.CellposeSegmenter import (
    CellposeSegmenter,
)

# Remote cellpose environment path
CELLPOSE_ENV_PATH = "C:\\Users\\bnort\\miniconda3\\envs\\microsam_cellpose"


def test_cellpose_interactive():
    """Interactive test showing local vs remote cellpose execution."""

    print("🧪 Interactive Cellpose Test")
    print("=" * 50)

    # Load test image
    image = data.coins()
    print(f"📸 Image: {image.shape}, range: [{image.min()}-{image.max()}]")

    # Simple cellpose configuration
    segmenter = CellposeSegmenter(
        model_type="cyto2", diameter=30, use_gpu=False
    )
    run_in_local_environment = segmenter.are_dependencies_available()

    print(
        f"🔍 Cellpose local: {'✅ Available' if run_in_local_environment else '❌ Not found'}"
    )
    print(f"🌐 Remote path: {CELLPOSE_ENV_PATH}")
    print("⚙️  Model: cyto2, diameter: 30")

    # Create simple figure - original and result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Show original image
    ax1.imshow(image, cmap="gray")
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Execute cellpose
    try:
        if run_in_local_environment:
            # Local execution
            print("🏠 Running cellpose locally...")
            result = segmenter.segment(image)

            num_cells = len(np.unique(result)) - 1
            ax2.imshow(result, cmap="nipy_spectral")
            ax2.set_title(f"✅ Found {num_cells} cells")
            print(f"✅ Success: Found {num_cells} cells")

        else:
            # Remote execution via appose
            print("🌐 Running cellpose with Appose...")
            result = execute_appose(image, segmenter, CELLPOSE_ENV_PATH)

            mask_array = result.ndarray()
            num_cells = len(np.unique(mask_array)) - 1
            ax2.imshow(mask_array, cmap="nipy_spectral")
            ax2.set_title(f"🌐 Found {num_cells} cells")
            print(f"✅ Remote success: Found {num_cells} cells")

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
        f"Cellpose Segmentation Test run with {mode}", fontsize=14, y=0.95
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    print(f"\n📊 Tested cellpose using {mode} execution")
    print("👀 Close window to finish...")

    plt.show()

    print("\n✅ Interactive Cellpose test completed!")


if __name__ == "__main__":
    test_cellpose_interactive()
