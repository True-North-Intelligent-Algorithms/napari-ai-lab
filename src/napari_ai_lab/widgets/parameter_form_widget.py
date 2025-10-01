"""
Parameter Form Widget for Interactive Segmenters.

This module provides a widget that automatically generates form elements
for parameters defined in Interactive Segmenter dataclasses.
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


class ParameterFormWidget(QWidget):
    """
    A widget that automatically generates form elements from dataclass parameters.

    This widget inspects the dataclass fields of an Interactive Segmenter
    and creates appropriate Qt input widgets based on the field metadata.
    """

    # Signal emitted when any parameter value changes
    parameters_changed = Signal(dict)

    def __init__(self, segmenter_class=None, parent=None):
        """
        Initialize the parameter form widget.

        Args:
            segmenter_class: The Interactive Segmenter class (dataclass) to parse parameters from.
            parent: Parent widget.
        """
        super().__init__(parent)

        self.segmenter_class = segmenter_class
        self.parameter_widgets = {}  # Maps field names to widget instances
        self.parameter_values = {}  # Current parameter values

        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)

        # Form layout for parameters
        self.form_layout = QFormLayout()
        self.main_layout.addLayout(self.form_layout)

        # Parse and create widgets if segmenter class is provided
        if segmenter_class is not None:
            self.parse_parameters()

    def set_segmenter_class(self, segmenter_class):
        """
        Set the segmenter class and rebuild the parameter form.

        Args:
            segmenter_class: The Interactive Segmenter class (dataclass) to parse parameters from.
        """
        self.segmenter_class = segmenter_class
        self.clear_form()
        self.parse_parameters()

    def clear_form(self):
        """Clear all parameter widgets from the form."""
        # Clear existing widgets
        while self.form_layout.count():
            child = self.form_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.parameter_widgets.clear()
        self.parameter_values.clear()

    def parse_parameters(self):
        """
        Parse dataclass fields and create appropriate Qt widgets.
        """
        if not self.segmenter_class or not dataclasses.is_dataclass(
            self.segmenter_class
        ):
            return

        # Get dataclass fields
        fields = dataclasses.fields(self.segmenter_class)

        for field in fields:
            self._create_widget_for_field(field)

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
        default_val = metadata.get("default", None)
        step = metadata.get("step", 1)

        # Store default value
        self.parameter_values[field_name] = default_val

        # Create label
        label = QLabel(field_name.replace("_", " ").title())

        # Create appropriate widget based on field type
        if field_type == "int":
            widget = self._create_int_widget(
                field_name, min_val, max_val, default_val, step
            )
        elif field_type == "float":
            widget = self._create_float_widget(
                field_name, min_val, max_val, default_val, step
            )
        elif field_type == "bool":
            widget = self._create_bool_widget(field_name, default_val)
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

                # Update stored value
                self.parameter_values[field_name] = value

    def create_segmenter_instance(self, **additional_kwargs):
        """
        Create an instance of the segmenter class with current parameter values.

        Args:
            **additional_kwargs: Additional keyword arguments to pass to constructor.

        Returns:
            Instance of the segmenter class with current parameter values.
        """
        if not self.segmenter_class:
            raise ValueError("No segmenter class set")

        # Combine parameter values with additional kwargs
        kwargs = {**self.parameter_values, **additional_kwargs}

        return self.segmenter_class(**kwargs)
