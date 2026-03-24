"""
Test script for ProgressLogger functionality.

Tests both NapariProgressLogger and ConsoleProgressLogger.
"""

import pytest

from napari_ai_lab.utilities import (
    ConsoleProgressLogger,
    NapariProgressLogger,
)


def test_console_logger():
    """Test console-based progress logger."""
    print("\n" + "=" * 60)
    print("Testing ConsoleProgressLogger")
    print("=" * 60)

    logger = ConsoleProgressLogger()

    # Test progress updates
    logger.log_info("Starting test operation...")

    for i in range(1, 6):
        logger.update_progress(i, 5, "Processing items")

    # Test different log levels
    logger.log_info("✅ Operation completed successfully")
    logger.log_warning("⚠️  This is a warning message")
    logger.log_error("❌ This is an error message")

    print()


def test_napari_logger_without_viewer():
    """Test NapariProgressLogger without a viewer (should fallback to console)."""
    print("\n" + "=" * 60)
    print("Testing NapariProgressLogger (No Viewer - Fallback Mode)")
    print("=" * 60)

    logger = NapariProgressLogger(viewer=None)

    # Test progress updates
    logger.log_info("Starting test operation...")

    for i in range(1, 6):
        logger.update_progress(i, 5, "Generating patches")

    # Test different log levels
    logger.log_info("✅ Patches generated successfully")
    logger.log_warning("⚠️  Some patches were skipped")
    logger.log_error("❌ Failed to write info.json")

    print()


@pytest.mark.skip(reason="Interactive test - disrupts automated testing")
def test_napari_logger_with_viewer():
    """Test NapariProgressLogger with actual napari viewer."""
    print("\n" + "=" * 60)
    print("Testing NapariProgressLogger (With Napari Viewer)")
    print("=" * 60)
    print("This will open a napari window with notifications.")
    print("Close the napari window to continue.")
    print("=" * 60 + "\n")

    import napari

    viewer = napari.Viewer()
    logger = NapariProgressLogger(viewer=viewer)

    # Test progress updates
    logger.log_info("🎨 Starting patch generation...")

    import time

    for i in range(1, 11):
        logger.update_progress(i, 10, "Generating patches")
        time.sleep(0.3)  # Simulate work

    # Test different log levels
    logger.log_info("✅ Successfully generated 10 patches!")
    time.sleep(1)

    logger.log_warning("⚠️  Some augmentations may have overlapping regions")
    time.sleep(1)

    # This won't actually show error since we're not simulating a real error
    # but it demonstrates the API
    # logger.log_error("❌ This would show an error notification")

    print("\nCheck napari window for notifications and status bar updates!")
    print("Close the napari window to exit.\n")

    napari.run()


if __name__ == "__main__":
    # Test console logger
    test_console_logger()

    # Test napari logger without viewer (fallback)
    test_napari_logger_without_viewer()

    # Test napari logger with viewer (interactive)
    # Uncomment to test with actual napari window:
    test_napari_logger_with_viewer()
