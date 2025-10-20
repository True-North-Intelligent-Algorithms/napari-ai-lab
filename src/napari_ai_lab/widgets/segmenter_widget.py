"""
Segmenter Widget for Interactive and Global Segmenters.

This module provides a widget that automatically generates form elements
for parameters defined in Segmenter dataclasses.
"""

import dataclasses
from typing import Any

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class SegmenterWidget(QWidget):
    """
    A widget that automatically generates form elements from dataclass parameters.

    This widget inspects the dataclass fields of a Segmenter (Interactive or Global)
    and creates appropriate Qt input widgets based on the field metadata.
    """

    # Signal emitted when any parameter value changes
    parameters_changed = Signal(dict)

    def __init__(self, segmenter=None, parent=None):
        """
        Initialize the segmenter widget.

        Args:
            segmenter: The Segmenter instance (dataclass) to create parameter widgets for.
            parent: Parent widget.
        """
        super().__init__(parent)

        self.segmenter = segmenter
        self.parameter_widgets = {}  # Maps field names to widget instances
        self.parameter_values = {}  # Current parameter values
        self._instructions_text = None  # Instructions label widget
        self._axis_combo = None  # Axis selection combo box
        self.selected_axis = None  # Currently selected axis

        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)

        # Form layout for parameters
        self.form_layout = QFormLayout()
        self.main_layout.addLayout(self.form_layout)

        # Parse and create widgets if segmenter is provided
        if segmenter is not None:
            self.parse_parameters()

    def set_segmenter(self, segmenter):
        """
        Set the segmenter instance and rebuild the parameter form.

        Args:
            segmenter: The Segmenter instance (dataclass) to create parameter widgets for.
        """
        self.segmenter = segmenter
        self.clear_form()
        self.parse_parameters()

    def clear_form(self):
        """Clear all parameter widgets from the form."""
        # Clear instructions widget if it exists
        if self._instructions_text is not None:
            self._instructions_text.deleteLater()
            self._instructions_text = None

        # Clear axis combo if it exists
        if self._axis_combo is not None:
            self._axis_combo.deleteLater()
            self._axis_combo = None

        # Clear existing widgets
        while self.form_layout.count():
            child = self.form_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.parameter_widgets.clear()
        self.parameter_values.clear()
        self.selected_axis = None

    def parse_parameters(self):
        """
        Parse dataclass fields and create appropriate Qt widgets.
        """
        if not self.segmenter or not dataclasses.is_dataclass(self.segmenter):
            return

        # Check if the segmenter has instructions
        self._add_instructions_if_present()

        # Add axis selection if segmenter supports multiple axes
        self._add_axis_selection_if_present()

        # Get dataclass fields
        fields = dataclasses.fields(self.segmenter)

        for field in fields:
            self._create_widget_for_field(field)

    def _add_instructions_if_present(self):
        """
        Add instructions label if the segmenter has instructions.
        """
        if hasattr(self.segmenter, "instructions"):
            instructions_text = self.segmenter.instructions
            if instructions_text and isinstance(instructions_text, str):
                # Create instructions label with similar styling as nd_easy_label
                self._instructions_text = QLabel(instructions_text)
                self._instructions_text.setStyleSheet(
                    "QLabel { font-size: 14px; color: #357; }"
                )
                self._instructions_text.setWordWrap(True)

                # Add instructions at the top of the form
                self.main_layout.addWidget(self._instructions_text)

                print(
                    f"Added instructions for {self.segmenter.__class__.__name__}"
                )

    def _add_axis_selection_if_present(self):
        """
        Add axis selection combo box if the segmenter supports multiple axes.
        """
        if hasattr(self.segmenter, "supported_axes"):
            try:
                # Get supported axes directly from the segmenter instance
                supported_axes = self.segmenter.supported_axes

                if supported_axes and len(supported_axes) > 1:
                    from qtpy.QtWidgets import QComboBox

                    # Create axis selection combo box
                    self._axis_combo = QComboBox()

                    for axis in supported_axes:
                        self._axis_combo.addItem(axis)

                    # Set default to first axis
                    self.selected_axis = supported_axes[0]

                    # Connect value changes
                    self._axis_combo.currentTextChanged.connect(
                        self._on_axis_changed
                    )

                    # Add to form with label
                    axis_label = QLabel("Axis to Process:")
                    self.form_layout.addRow(axis_label, self._axis_combo)

                    print(
                        f"Added axis selection for {self.segmenter.__class__.__name__}: {supported_axes}"
                    )

            except (TypeError, ValueError, AttributeError, RuntimeError) as e:
                print(
                    f"Could not get supported axes for {self.segmenter.__class__.__name__}: {e}"
                )

    def _on_axis_changed(self, axis_text):
        """Handle axis selection changes."""
        self.selected_axis = axis_text
        self.segmenter.selected_axis = axis_text
        print(f"Selected axis: {axis_text}")

    def _create_widget_for_field(self, field):
        """
        Create an appropriate Qt widget for a dataclass field.

        Args:
            field: The dataclass field to create a widget for.
        """
        metadata = field.metadata
        field_name = field.name
        field_type = metadata.get("type", "str")

        # Get constraints from metadata
        min_val = metadata.get("min", None)
        max_val = metadata.get("max", None)
        step = metadata.get("step", 1)

        # Get current value from segmenter instance, fallback to metadata default
        if self.segmenter and hasattr(self.segmenter, field_name):
            current_val = getattr(self.segmenter, field_name)
        else:
            current_val = metadata.get("default", None)

        # Store current value
        self.parameter_values[field_name] = current_val

        # Create label
        label = QLabel(field_name.replace("_", " ").title())

        # Create appropriate widget based on field type
        if field_type == "int":
            widget = self._create_int_widget(
                field_name, min_val, max_val, current_val, step
            )
        elif field_type == "float":
            widget = self._create_float_widget(
                field_name, min_val, max_val, current_val, step
            )
        elif field_type == "bool":
            widget = self._create_bool_widget(field_name, current_val)
        elif field_type == "str" and "choices" in metadata:
            widget = self._create_choice_widget(
                field_name, metadata["choices"], current_val
            )
        else:
            # Default to a generic widget (could be extended for strings, etc.)
            widget = QLabel(f"Unsupported type: {field_type}")

        # Store widget reference
        self.parameter_widgets[field_name] = widget

        # Add to form layout
        self.form_layout.addRow(label, widget)

    def _create_int_widget(
        self,
        field_name: str,
        min_val: int | None,
        max_val: int | None,
        default_val: int | None,
        step: int,
    ):
        """Create an integer input widget (spinbox with optional slider)."""
        # Create horizontal layout for spinbox and optional slider
        widget_layout = QHBoxLayout()
        container = QWidget()
        container.setLayout(widget_layout)

        # Create spinbox
        spinbox = QSpinBox()
        if min_val is not None:
            spinbox.setMinimum(min_val)
        if max_val is not None:
            spinbox.setMaximum(max_val)
        if default_val is not None:
            spinbox.setValue(default_val)
        spinbox.setSingleStep(step)

        widget_layout.addWidget(spinbox)

        # Add slider if range is reasonable (not too large)
        slider = None
        if (
            min_val is not None
            and max_val is not None
            and (max_val - min_val) <= 1000
        ):
            slider = QSlider()
            slider.setOrientation(1)  # Horizontal
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            if default_val is not None:
                slider.setValue(default_val)
            slider.setSingleStep(step)

            widget_layout.addWidget(slider)

            # Connect spinbox and slider
            spinbox.valueChanged.connect(slider.setValue)
            slider.valueChanged.connect(spinbox.setValue)

        # Connect value changes
        spinbox.valueChanged.connect(
            lambda value, name=field_name: self._on_parameter_changed(
                name, value
            )
        )

        # Store both widgets for easy access
        container.spinbox = spinbox
        container.slider = slider

        return container

    def _create_float_widget(
        self,
        field_name: str,
        min_val: float | None,
        max_val: float | None,
        default_val: float | None,
        step: float,
    ):
        """Create a float input widget (double spinbox)."""
        spinbox = QDoubleSpinBox()

        if min_val is not None:
            spinbox.setMinimum(min_val)
        if max_val is not None:
            spinbox.setMaximum(max_val)
        if default_val is not None:
            spinbox.setValue(default_val)
        spinbox.setSingleStep(step)

        # Set reasonable decimal places based on step size
        if step < 0.01:
            spinbox.setDecimals(3)
        elif step < 0.1:
            spinbox.setDecimals(2)
        else:
            spinbox.setDecimals(1)

        # Connect value changes
        spinbox.valueChanged.connect(
            lambda value, name=field_name: self._on_parameter_changed(
                name, value
            )
        )

        return spinbox

    def _create_bool_widget(self, field_name: str, default_val: bool | None):
        """Create a boolean input widget (checkbox)."""
        checkbox = QCheckBox()

        if default_val is not None:
            checkbox.setChecked(default_val)

        # Connect value changes
        checkbox.toggled.connect(
            lambda checked, name=field_name: self._on_parameter_changed(
                name, checked
            )
        )

        return checkbox

    def _create_choice_widget(
        self, field_name: str, choices: list, default_val: str | None
    ):
        """Create a choice widget (combobox) for string parameters with predefined options."""
        from qtpy.QtWidgets import QComboBox

        combobox = QComboBox()

        for choice in choices:
            combobox.addItem(choice)

        if default_val is not None and default_val in choices:
            combobox.setCurrentText(default_val)

        # Connect value changes
        combobox.currentTextChanged.connect(
            lambda text, name=field_name: self._on_parameter_changed(
                name, text
            )
        )

        return combobox

    def _on_parameter_changed(self, field_name: str, value: Any):
        """
        Handle parameter value changes.

        Args:
            field_name: Name of the parameter that changed.
            value: New value of the parameter.
        """
        self.parameter_values[field_name] = value
        self.parameters_changed.emit(self.parameter_values.copy())

    def get_parameters(self) -> dict[str, Any]:
        """
        Get current parameter values.

        Returns:
            dict: Dictionary of parameter names to current values.
        """
        return self.parameter_values.copy()

    def set_parameters(self, parameters: dict[str, Any]):
        """
        Set parameter values programmatically.

        Args:
            parameters: Dictionary of parameter names to values.
        """
        for field_name, value in parameters.items():
            if field_name in self.parameter_widgets:
                widget = self.parameter_widgets[field_name]

                # Set value based on widget type
                if hasattr(widget, "spinbox"):  # Integer widget with spinbox
                    widget.spinbox.setValue(value)
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(value)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(value)
                elif hasattr(widget, "setCurrentText"):  # ComboBox
                    widget.setCurrentText(str(value))

                # Update stored value
                self.parameter_values[field_name] = value

    def update_segmenter_parameters(self):
        """
        Update the segmenter instance with current parameter values from the widget.

        Updates the segmenter instance in-place with the current form values.
        """
        if not self.segmenter:
            raise ValueError("No segmenter instance set")

        # Update segmenter parameters with current form values
        for field_name, value in self.parameter_values.items():
            if hasattr(self.segmenter, field_name):
                setattr(self.segmenter, field_name, value)

        return self.segmenter

    def sync_segmenter_instance(self, segmenter_instance):
        """
        Sync an existing segmenter instance with current parameter values from the widget.

        Args:
            segmenter_instance: Existing segmenter instance to update with current widget values.

        Returns:
            The same segmenter instance with updated parameter values.
        """
        if segmenter_instance is None:
            return None

        # Update each parameter in the segmenter instance
        for field_name, value in self.parameter_values.items():
            if hasattr(segmenter_instance, field_name):
                setattr(segmenter_instance, field_name, value)
            else:
                print(
                    f"Warning: Segmenter instance has no attribute '{field_name}'"
                )

        return segmenter_instance

    def get_selected_axis(self):
        """
        Get the currently selected axis configuration.

        Returns:
            str: The selected axis configuration (e.g., "YX", "ZYX", etc.) or None if no axis selected.
        """
        return self.selected_axis

    def set_selected_axis(self, axis):
        """
        Set the selected axis configuration.

        Args:
            axis (str): The axis configuration to select.
        """
        if self._axis_combo is not None:
            index = self._axis_combo.findText(axis)
            if index >= 0:
                self._axis_combo.setCurrentIndex(index)
                self.selected_axis = axis
            else:
                print(f"Warning: Axis '{axis}' not found in available options")
