"""Simple test for training functionality."""

from pathlib import Path

from napari_ai_lab.Augmenters import SimpleAugmenter
from napari_ai_lab.models import ImageDataModel
from napari_ai_lab.Segmenters.GlobalSegmenters.MonaiUNetSegmenter import (
    MonaiUNetSegmenter,
)


def simple_updater(message, progress=None):
    """Simple updater that prints to console."""
    if progress is not None:
        print(f"[{progress}%] {message}")
    else:
        print(f"{message}")


def create_patches_if_needed(image_data_model, patches_dir, num_patches=40):
    """Create YX patches if they don't exist."""

    # Check if patches and info.json already exist
    info_json_path = patches_dir / "info.json"
    if patches_dir.exists() and info_json_path.exists():
        existing_patches = (
            list((patches_dir / "input0").glob("*.tif"))
            if (patches_dir / "input0").exists()
            else []
        )
        if len(existing_patches) >= num_patches:
            print(
                f"Found {len(existing_patches)} existing patches and info.json, skipping creation"
            )
            return

    print(f"Creating {num_patches} YX patches...")

    # Load image and annotations
    image = image_data_model.load_image(0)
    annotations = image_data_model.load_existing_annotations(
        image_shape=image.shape,
        image_index=0,
        subdirectory="class_0",
    )

    print(f"Image shape: {image.shape}")
    print(f"Annotations shape: {annotations.shape}")

    # Create augmenter with global normalization
    augmenter = SimpleAugmenter(seed=42, normalize=True, use_global_stats=True)

    # Compute global normalization statistics from full image
    print("\nComputing global normalization statistics from full image...")
    augmenter.compute_global_normalization_stats(
        image, percentile_low=1, percentile_high=99
    )

    axes = "YX"

    # Determine YX patch size based on image dimensions
    if len(image.shape) == 3:
        # 3D image: take single Z slice with YX dimensions
        patch_size_yx = (1, 128, 128)
        print(f"Generating 2D patches of size {patch_size_yx} (from 3D)...")
    elif len(image.shape) == 2:
        # 2D image: just YX
        patch_size_yx = (128, 128)
        print(f"Generating 2D patches of size {patch_size_yx}...")
    else:
        raise ValueError(f"Unexpected image dimensions: {image.shape}")

    # Check if image is large enough
    if not all(
        img_dim >= patch_dim
        for img_dim, patch_dim in zip(image.shape, patch_size_yx, strict=False)
    ):
        raise ValueError(
            f"Image shape {image.shape} is too small for {patch_size_yx} patches"
        )

    # Pre-compute valid coordinates
    print("Computing valid coordinates for YX patches...")
    augmenter.create_valid_coordinates(
        annotations, image.shape, patch_size_yx, axis=None
    )
    print(f"Found {len(augmenter.valid_coordinates)} valid positions")

    # Generate patches
    for i in range(num_patches):
        patch_base_name = "yx_patch"
        im_path, mask_path = augmenter.augment_and_save(
            image,
            annotations,
            str(patches_dir),
            patch_base_name,
            patch_size_yx,
            axis=None,
        )
        print(f"  Patch {i+1}/{num_patches} saved")

    # Write info.json with num_inputs=1, num_truths=1
    print("\nWriting info.json...")
    augmenter.write_info(
        patch_path=str(patches_dir),
        axes=axes,
        num_inputs=1,  # 1 input channel
        num_truths=1,  # 1 ground truth class
        sub_sample=1,
    )

    print(f"✓ Created {num_patches} patches in {patches_dir}")


def test_monai_training():
    """Test basic training workflow with existing patches."""
    # Create image data model pointing to project with patches
    project_path = Path(
        "/home/bnorthan/code/i2k/tnia/napari-ai-lab/tests/test_images/vessels_project_test/"
    )
    image_data_model = ImageDataModel(parent_directory=project_path)

    patches_dir = image_data_model.get_patches_directory(axis="yx")

    # Create patches if they don't exist
    create_patches_if_needed(image_data_model, patches_dir, num_patches=20)

    # Get patch directory
    print(f"\nPatch directory: {patches_dir}")

    # Create segmenter
    segmenter = MonaiUNetSegmenter()

    # Set training parameters
    segmenter.num_epochs = 10  # Set to 200 epochs for training

    segmenter.tile_size = (128, 128)  # Set tile size to match patch size

    # Set simple updater for progress output
    segmenter.updater = simple_updater

    # Set patch path
    segmenter.patch_path = str(patches_dir)
    print(f"Set patch path: {segmenter.patch_path}")

    # Set model path to models directory
    models_dir = image_data_model.get_models_directory()
    segmenter.model_path = str(models_dir)
    segmenter.model_name = "monai_unet_test.pth"  # Use a test-specific model name to avoid overwriting any real models
    print(f"Set model path: {segmenter.model_path}")

    # Run training
    result = segmenter.train()

    # Store result for assertions later
    print(f"\nTraining result: {result}")
    print(f"CUDA present: {result.get('cuda_present')}")
    print(f"Number of devices: {result.get('ndevices')}")
    print(f"Device: {result.get('device')}")

    # Get info for later verification
    num_inputs = result.get("num_inputs")
    num_truths = result.get("num_truths")
    print("\nPatch info:")
    print(f"Number of input channels: {num_inputs}")
    print(f"Number of truth classes: {num_truths}")

    # Get file lists for predictions
    X = result.get("X")
    Y = result.get("Y")
    X_val = result.get("X_val")
    Y_val = result.get("Y_val")

    print("\nFile lists:")
    print(f"Training images (X): {len(X)}")
    print(f"Training labels (Y): {len(Y)} classes")
    if len(Y) > 0:
        print(f"  - Class 0: {len(Y[0])} files")
    print(f"Validation images (X_val): {len(X_val)}")
    print(f"Validation labels (Y_val): {len(Y_val)} classes")
    if len(Y_val) > 0:
        print(f"  - Class 0: {len(Y_val[0])} files")

    # Now run all assertions at the end
    print("\n=== Running verification assertions ===")
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "success" in result, "Result should contain 'success' key"
    print("✓ Result structure is correct")

    assert num_inputs == 1, f"Expected 1 input channel, got {num_inputs}"
    assert num_truths == 1, f"Expected 1 truth class, got {num_truths}"
    print("✓ Verified 1 input channel and 1 truth class")

    assert len(X) == 20, f"Expected 20 training images, got {len(X)}"
    assert len(Y) == 1, f"Expected 1 ground truth class list, got {len(Y)}"
    assert (
        len(Y[0]) == 20
    ), f"Expected 20 training label files for class 0, got {len(Y[0])}"
    print("✓ Verified 20 training patches and 0 validation patches")

    print("✅ All assertions passed!")


if __name__ == "__main__":
    test_monai_training()
    print("✅ Test completed!")
