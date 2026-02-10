"""
Headless augmentation script - no GUI, no viewer.

This script demonstrates how to:
1. Create an ImageDataModel from a directory
2. Load images and print their dimensions
3. Load corresponding annotations and analyze them
4. Prepare for augmentation (future step)
"""

import numpy as np

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

    # Step 4: Future - Augmentation will go here
    print("\n" + "=" * 80)
    print("Step 4: Augmentation (to be implemented)")
    print("=" * 80)
    print("\nNext steps:")
    print("  - Apply SimpleAugmenter to create random crops")
    print("  - Save augmented patches for training")
    print("  - Generate multiple augmented versions per image")

    print("\n" + "=" * 80)
    print("Script completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
