"""
Interactive MicroSAM Test - Local and Remote Execution.

This script demonstrates SAM3D segmentation in two modes:
1. Local execution (if micro_sam available)
2. Remote execution via appose (if micro_sam not available locally)

Provides visual feedback and concise status messages.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from skimage import data

from napari_ai_lab.Segmenters.execute_appose import execute_appose
from napari_ai_lab.Segmenters.InteractiveSegmenters.SAM3D import SAM3D

# Remote micro_sam environment path
MICROSAM_ENV_PATH = "C:\\Users\\bnort\\miniconda3\\envs\\microsam_cellpose"
MICROSAM_ENV_PATH = (
    r"/home/bnorthan/mambaforge/envs/microsam_cellpose_instanseg"
)


def test_microsam_interactive():
    """Interactive test showing local vs remote SAM3D execution."""

    print("🧪 Interactive MicroSAM Test")
    print("=" * 50)

    # Load test image
    image = data.coins()
    print(f"📸 Image: {image.shape}, range: [{image.min()}-{image.max()}]")

    # Simple SAM3D configuration
    segmenter = SAM3D()

    segmenter.model_type = "vit_b"
    segmenter.iou_threshold = 0.8
    segmenter.stability_score_threshold = 0.95

    run_in_local_environment = segmenter.are_dependencies_available()

    print(
        f"🔍 MicroSAM local: {'✅ Available' if run_in_local_environment else '❌ Not found'}"
    )
    print(f"🌐 Remote path: {MICROSAM_ENV_PATH}")
    print("⚙️  Model: vit_b, iou_threshold: 0.8")

    # Create simple figure - original and result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Show original image
    ax1.imshow(image, cmap="gray")

    # For interactive segmenter, we need to provide some test points
    # Try to hit a coin - moving up and left from center
    test_points = [[60, 100]]

    # Show test points on the original image
    ax1.scatter(
        np.array(test_points)[:, 1],
        np.array(test_points)[:, 0],
        c="red",
        s=50,
        marker="x",
    )
    ax1.set_title("Original Image + Test Points")
    ax1.axis("off")

    temp_path = os.path.join(os.path.dirname(__file__), "temp")

    # Execute SAM3D
    try:
        if run_in_local_environment:
            # Local execution
            print("🏠 Running SAM3D locally...")
            # Initialize predictor first (SAM3D needs this)
            segmenter.initialize_predictor(image, temp_path, "test_image")
            result = segmenter.segment(image, points=np.array(test_points))

            num_objects = len(np.unique(result)) - 1
            ax2.imshow(result, cmap="nipy_spectral")
            ax2.set_title(f"✅ Found {num_objects} objects")
            print(f"✅ Success: Found {num_objects} objects")

        else:
            # Remote execution via appose
            print("🌐 Running SAM3D with Appose...")
            # Pack test inputs for remote execution
            test_inputs = {"test_points": test_points}
            segmenter.initialize_embedding_save_path(temp_path, "test_image")
            result = execute_appose(
                image, segmenter, MICROSAM_ENV_PATH, test_inputs
            )

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
        f"SAM3D Segmentation Test run with {mode}", fontsize=14, y=0.95
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    print(f"\n📊 Tested SAM3D using {mode} execution")
    print("👀 Close window to finish...")

    plt.show()

    print("\n✅ Interactive MicroSAM test completed!")


if __name__ == "__main__":
    test_microsam_interactive()
