"""
Interactive test for array slicing operations with multi-dimensional image data.
"""

from pathlib import Path

import napari

from napari_ai_lab.utility import (
    create_artifact_name,
    get_current_slice_indices,
)


def main():
    """Open a CZI file and display it in napari."""
    # Path to test CZI file (relative to this file's location)
    test_image_path = (
        Path(__file__).parent.parent.parent.parent
        / "tests"
        / "test_images"
        / "czi"
        / "Image 6_Subset-pos02_t1-35.czi"
    )

    # Check if file exists
    if not test_image_path.exists():
        print(f"Error: Test image not found: {test_image_path}")
        return

    try:
        from czifile import CziFile

        with CziFile(str(test_image_path)) as czi:
            image_data = czi.asarray()
            image_data = (
                image_data.squeeze()
            )  # Remove singleton dimensions for better viewing
            axes = czi.axes

            # Print information
            print(f"\nCZI file: {test_image_path.name}")
            print(f"Shape: {image_data.shape}")
            print(f"Axes: {axes}")
            print(f"Data type: {image_data.dtype}")

            # Create napari viewer and add image
            viewer = napari.Viewer()
            viewer.add_image(image_data, name=test_image_path.name)

            image_name = test_image_path.name

            # Connect event handler for dimension changes
            def on_dims_changed(event):
                selected_axis = "ZYX"
                """Print current step when user scrolls through dimensions."""
                print(f"Current step: {viewer.dims.current_step}")
                indices = get_current_slice_indices(
                    viewer.dims.current_step, selected_axis
                )
                print(f"Current slice indices: {indices}")
                slice_data = image_data[indices]
                print(f"Current slice shape: {slice_data.shape}")
                print(f"Current slice mean: {slice_data.mean()}")
                image_step_name = create_artifact_name(
                    image_name, viewer.dims.current_step, "ZYX"
                )
                print(f"Current slice artifact name: {image_step_name}")

                print()

            viewer.dims.events.current_step.connect(on_dims_changed)

            # Display axes info in viewer console
            print("\nDisplaying in napari...")
            print("Use the slider to navigate through dimensions")
            print("Current step will be printed as you scroll")

            # Run napari (blocks until viewer is closed)
            napari.run()

    except ImportError as e:
        print(f"Error: Required package not installed: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"Error: Failed to open CZI file: {e}")


if __name__ == "__main__":
    main()
