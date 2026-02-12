"""
Training Dialog for ND Operations.

This module provides a dialog for configuring training parameters
for ND Operations (like segmenters) that support training.
"""

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
)

from .nd_operation_widget import NDOperationWidget


class TrainDialog(QDialog):
    """
    Dialog for configuring training parameters.

    This dialog creates an NDOperationWidget filtered to show only
    training parameters (fields with metadata['param_type'] == 'training').
    """

    def __init__(self, nd_operation, parent=None):
        """
        Initialize the training dialog.

        Args:
            nd_operation: The ND Operation instance to configure training parameters for.
            parent: Parent widget.
        """
        super().__init__(parent)

        self.nd_operation = nd_operation

        # Set dialog properties
        self.setWindowTitle("Training Parameters")
        self.setMinimumWidth(400)

        # Main layout
        main_layout = QVBoxLayout(self)

        # Create parameter widget filtered for training parameters only
        self.parameter_widget = NDOperationWidget(
            nd_operation=nd_operation,
            parent=self,
            param_type_to_parse="training",
        )
        main_layout.addWidget(self.parameter_widget)

        # Add OK and Cancel buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        self.setLayout(main_layout)

    def get_training_parameters(self):
        """
        Get the training parameter values from the widget.

        Returns:
            dict: Dictionary of parameter names to values.
        """
        return self.parameter_widget.parameter_values
