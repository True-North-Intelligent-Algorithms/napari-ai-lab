"""
Launch script for ND AI Lab - Combined tabbed interface.

This script launches the combined AI Lab plugin with Label, Augment,
and Segment tabs in a single interface.
"""

import napari

from napari_ai_lab.apps.nd_ai_lab import NDAILab

# Create viewer
viewer = napari.Viewer()

# Create combined widget (no model yet)
nd_ai_lab_widget = NDAILab(viewer)
viewer.window.add_dock_widget(nd_ai_lab_widget, area="right", name="AI Lab")

# TODO Phase 2: Create model when user loads directory
# Example of setting model later:
# from pathlib import Path
# from napari_ai_lab.models import ImageDataModel
# parent_dir = Path("tests/test_images/vessels_small")
# model = ImageDataModel(parent_dir)
# nd_ai_lab_widget.set_image_data_model(model)

# TODO Phase 2: Add sequence viewer

print("✨ ND AI Lab launched successfully!")
print("   Tabs: Label | Augment | Segment")
print("   Phase 1: Basic structure - no model yet")
print("   📝 Model can be set later via set_image_data_model()")

napari.run()
