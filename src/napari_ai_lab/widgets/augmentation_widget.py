from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class AugmentationParametersGroup(QGroupBox):
    """Reusable group of augmentation parameters and controls."""

    def __init__(self, parent=None):
        super().__init__("Augmentation Parameters", parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QGridLayout()
        self.setLayout(layout)

        # add horizontal flip check box
        self.horizontal_flip_check_box = QCheckBox("Horizontal Flip")
        self.horizontal_flip_check_box.setChecked(True)
        layout.addWidget(self.horizontal_flip_check_box, 0, 0)

        # add vertical flip check box
        self.vertical_flip_check_box = QCheckBox("Vertical Flip")
        self.vertical_flip_check_box.setChecked(True)
        layout.addWidget(self.vertical_flip_check_box, 0, 1)

        # add rotate check box
        self.random_rotate_check_box = QCheckBox("Random Rotate")
        self.random_rotate_check_box.setChecked(True)
        layout.addWidget(self.random_rotate_check_box, 1, 0)

        # add random resize check box
        self.random_resize_check_box = QCheckBox("Random Resize")
        self.random_resize_check_box.setChecked(True)
        layout.addWidget(self.random_resize_check_box, 1, 1)

        # add random brightness contrast check box
        self.random_brightness_contrast_check_box = QCheckBox(
            "Random Brightness/Contrast"
        )
        self.random_brightness_contrast_check_box.setChecked(True)
        layout.addWidget(self.random_brightness_contrast_check_box, 2, 0)

        # add random gamma check box
        self.random_gamma_check_box = QCheckBox("Random Gamma")
        self.random_gamma_check_box.setChecked(False)
        layout.addWidget(self.random_gamma_check_box, 2, 1)

        # add random adjust color check box
        self.random_adjust_color_check_box = QCheckBox("Random Adjust Color")
        self.random_adjust_color_check_box.setChecked(False)
        layout.addWidget(self.random_adjust_color_check_box, 3, 0)

        # add elastic deformation check box
        self.elastic_deformation_check_box = QCheckBox("Elastic Deformation")
        self.elastic_deformation_check_box.setChecked(False)
        layout.addWidget(self.elastic_deformation_check_box, 3, 1)

        # add number patches spin box and label
        num_patches_layout = QHBoxLayout()
        num_patches_label = QLabel("Patches per ROI:")
        self.number_patches_spin_box = QSpinBox()
        self.number_patches_spin_box.setRange(1, 1000)
        self.number_patches_spin_box.setValue(100)
        num_patches_layout.addWidget(num_patches_label)
        num_patches_layout.addWidget(self.number_patches_spin_box)
        layout.addLayout(num_patches_layout, 4, 0)

        # add patch size spin box and label
        patch_size_layout = QHBoxLayout()
        patch_size_label = QLabel("Patch size:")
        self.patch_size_spin_box = QSpinBox()
        self.patch_size_spin_box.setRange(1, 4096)
        self.patch_size_spin_box.setValue(256)
        patch_size_layout.addWidget(patch_size_label)
        patch_size_layout.addWidget(self.patch_size_spin_box)
        layout.addLayout(patch_size_layout, 4, 1)

        # add augment current button
        self.augment_current_button = QPushButton("Augment current image")
        layout.addWidget(self.augment_current_button, 5, 0)

        # add perform augmentation button
        self.perform_augmentation_button = QPushButton("Augment all images")
        layout.addWidget(self.perform_augmentation_button, 5, 1)

        # add delete augmentations button
        self.delete_augmentations_button = QPushButton("Delete augmentations")
        layout.addWidget(self.delete_augmentations_button, 6, 0)

        # add advanced augmentation settings button
        self.augmentation_settings_button = QPushButton("Settings...")
        layout.addWidget(self.augmentation_settings_button, 6, 1)

    def get_parameters(self):
        """Get current augmentation parameters as a dictionary."""
        return {
            "horizontal_flip": self.horizontal_flip_check_box.isChecked(),
            "vertical_flip": self.vertical_flip_check_box.isChecked(),
            "random_rotate": self.random_rotate_check_box.isChecked(),
            "random_resize": self.random_resize_check_box.isChecked(),
            "random_brightness_contrast": self.random_brightness_contrast_check_box.isChecked(),
            "random_gamma": self.random_gamma_check_box.isChecked(),
            "random_adjust_color": self.random_adjust_color_check_box.isChecked(),
            "elastic_deformation": self.elastic_deformation_check_box.isChecked(),
            "num_patches": self.number_patches_spin_box.value(),
            "patch_size": self.patch_size_spin_box.value(),
        }


if __name__ == "__main__":
    import sys

    from qtpy.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Create test dialog
    dialog = QDialog()
    dialog.setWindowTitle("Augmentation Parameters Test")
    dialog_layout = QVBoxLayout()
    dialog.setLayout(dialog_layout)

    # Add augmentation group
    augmentation_group = AugmentationParametersGroup()
    dialog_layout.addWidget(augmentation_group)

    # Add OK button to close
    ok_button = QPushButton("OK")
    ok_button.clicked.connect(dialog.accept)
    dialog_layout.addWidget(ok_button)

    dialog.exec_()
