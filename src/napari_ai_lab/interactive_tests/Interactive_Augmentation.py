"""
Headless augmentation script - no GUI, no viewer.

This script demonstrates how to:
1. Create an ImageDataModel from a directory
2. Load images and print their dimensions
3. Load corresponding annotations and analyze them
4. Prepare for augmentation (future step)
"""

import numpy as np

from napari_ai_lab.Augmenters import SimpleAugmenter
from napari_ai_lab.models import ImageDataModel


def main():
    """Main function for headless augmentation processing."""

    # Set up the directory containing images and annotations
    parent_dir = "/home/bnorthan/images/tnia-python-images/imagesc/2026_02_07_vessels_czi/"

    print("=" * 80)
    print("Headless Augmentation Script")
    print("=" * 80)
    print(f"\nParent directory: {parent_dir}\n")

    # Step 1: Create the image data model
    print("Step 1: Creating ImageDataModel...")
    model = ImageDataModel(parent_dir)

    # Get information about available images
    image_count = model.get_image_count()
    print(f"Found {image_count} images in directory")

    if image_count == 0:
        print("No images found. Exiting.")
        return

    # List all image paths
    image_paths = model.get_image_paths()
    print("\nAvailable images:")
    for i, path in enumerate(image_paths):
        print(f"  [{i}] {path.name}")

    # Step 2: Load the first image and print dimensions
    print("\n" + "=" * 80)
    print("Step 2: Loading first image...")
    print("=" * 80)

    image_index = 0
    image = model.load_image(image_index)

    print(f"\nImage dimensions: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Axis types: {model.axis_types}")
    print(f"Number of pixels: {image.size:,}")
    print(f"Min value: {image.min()}")
    print(f"Max value: {image.max()}")
    print(f"Mean value: {image.mean():.2f}")

    # Step 3: Load corresponding annotations and analyze
    print("\n" + "=" * 80)
    print("Step 3: Loading annotations...")
    print("=" * 80)

    try:
        # Try to load existing annotations
        annotations = model.load_existing_annotations(
            image_shape=image.shape,
            image_index=image_index,
            subdirectory="class_0",
        )

        print(f"\nAnnotations dimensions: {annotations.shape}")
        print(f"Annotations dtype: {annotations.dtype}")

        # Verify shapes match
        if annotations.shape == image.shape:
            print("✓ Annotation shape matches image shape")
        else:
            print(
                f"✗ Shape mismatch! Image: {image.shape}, Annotations: {annotations.shape}"
            )

        # Count labeled pixels (non-zero pixels)
        labeled_pixels = np.count_nonzero(annotations)
        total_pixels = annotations.size
        labeled_percentage = (labeled_pixels / total_pixels) * 100

        print("\nAnnotation statistics:")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Labeled pixels (non-zero): {labeled_pixels:,}")
        print(f"  Unlabeled pixels (zero): {total_pixels - labeled_pixels:,}")
        print(f"  Labeled percentage: {labeled_percentage:.2f}%")

        # Get unique labels
        unique_labels = np.unique(annotations)
        print(f"\nUnique labels found: {len(unique_labels)}")
        print(f"  Labels: {unique_labels}")

        # Count pixels per label
        if len(unique_labels) > 1:
            print("\nPixels per label:")
            for label in unique_labels:
                if label == 0:
                    continue  # Skip background
                count = np.count_nonzero(annotations == label)
                percentage = (count / total_pixels) * 100
                print(f"  Label {label}: {count:,} pixels ({percentage:.2f}%)")

    except Exception as e:  # noqa
        print(f"Error loading annotations: {e}")
        print("Annotations may not exist yet.")
        annotations = None

    # Step 4: Augmentation using SimpleAugmenter
    print("\n" + "=" * 80)
    print("Step 4: Augmentation with SimpleAugmenter")
    print("=" * 80)

    if annotations is None:
        print("No annotations available. Skipping augmentation.")
    else:
        # Create a SimpleAugmenter with a random seed for reproducibility
        augmenter = SimpleAugmenter(seed=42)
        print("Created SimpleAugmenter with seed=42")

        # 4a: Generate 5 ZYX patches (128x128x16)
        print("\n--- ZYX Augmentation (3D patches) ---")

        # Get patches directory for ZYX (assuming Z is axis 0)
        # For ZYX, we want to crop along all axes, so axis=None
        patches_dir_zyx = model.get_patches_directory(axis="zyx")
        print(f"Patches directory (ZYX): {patches_dir_zyx}")

        patch_size_zyx = (16, 128, 128)  # Z, Y, X
        num_patches_zyx = 5

        print(
            f"Generating {num_patches_zyx} patches of size {patch_size_zyx}..."
        )

        # Check if the image is large enough for 3D patches
        if len(image.shape) >= 3 and all(
            img_dim >= patch_dim
            for img_dim, patch_dim in zip(
                image.shape, patch_size_zyx, strict=False
            )
        ):
            # Pre-compute valid coordinates for ZYX patches
            print("Computing valid coordinates for ZYX patches...")
            augmenter.create_valid_coordinates(
                annotations, image.shape, patch_size_zyx, axis=None
            )
            print(f"Found {len(augmenter.valid_coordinates)} valid positions")

            for i in range(num_patches_zyx):
                patch_base_name = "zyx_patch"
                im_path, mask_path = augmenter.augment_and_save(
                    image,
                    annotations,
                    str(patches_dir_zyx),
                    patch_base_name,
                    patch_size_zyx,
                    axis=None,  # Crop along all axes
                )
                print(f"  Patch {i+1}/{num_patches_zyx} saved:")
                print(f"    Image: {im_path}")
                print(f"    Mask:  {mask_path}")
        else:
            print(
                f"⚠ Image shape {image.shape} is too small for {patch_size_zyx} patches"
            )

        # 4b: Generate 5 YX patches (128x128)
        print("\n--- YX Augmentation (2D patches) ---")

        # Get patches directory for YX (we'll extract 2D slices)
        # For 2D from 3D, we might want to specify an axis or handle differently
        # Here we'll use axis=None for the directory but specify 2D patch size
        patches_dir_yx = model.get_patches_directory(axis="yx")
        print(f"Patches directory (YX): {patches_dir_yx}")

        num_patches_yx = 5

        # Determine the YX patch size based on image dimensions
        if len(image.shape) == 3:
            # 3D image: take single Z slice with YX dimensions
            patch_size_yx = (1, 128, 128)  # Single Z slice
            print(
                f"Generating {num_patches_yx} 2D patches of size {patch_size_yx} (from 3D)..."
            )
        elif len(image.shape) == 2:
            # 2D image: just YX
            patch_size_yx = (128, 128)
            print(
                f"Generating {num_patches_yx} 2D patches of size {patch_size_yx}..."
            )
        else:
            print(f"⚠ Unexpected image dimensions: {image.shape}")
            patch_size_yx = None

        if patch_size_yx and all(
            img_dim >= patch_dim
            for img_dim, patch_dim in zip(
                image.shape, patch_size_yx, strict=False
            )
        ):
            # Pre-compute valid coordinates for YX patches
            print("Computing valid coordinates for YX patches...")
            augmenter.create_valid_coordinates(
                annotations, image.shape, patch_size_yx, axis=None
            )
            print(f"Found {len(augmenter.valid_coordinates)} valid positions")

            for i in range(num_patches_yx):
                patch_base_name = "yx_patch"
                im_path, mask_path = augmenter.augment_and_save(
                    image,
                    annotations,
                    str(patches_dir_yx),
                    patch_base_name,
                    patch_size_yx,
                    axis=None,  # Random crop across all dimensions
                )

                # If we extracted a 3D patch with Z=1, squeeze it for display
                print(f"  Patch {i+1}/{num_patches_yx} saved:")
                print(f"    Image: {im_path}")
                print(f"    Mask:  {mask_path}")
        elif patch_size_yx:
            print(
                f"⚠ Image shape {image.shape} is too small for {patch_size_yx} patches"
            )

        print("\n✓ Augmentation completed!")
        print(f"  Total patches generated: {num_patches_zyx + num_patches_yx}")
        print(f"  Saved to: {patches_dir_zyx}")

    print("\n" + "=" * 80)
    print("Script completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
