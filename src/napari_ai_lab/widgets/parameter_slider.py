from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QSlider, QWidget


class ParameterSlider(QWidget):
    def __init__(
        self,
        label: str,
        min_value: float = 0.0,
        max_value: float = 1.0,
        default_value: float = 0.5,
        step: int = 100,
        parent: QWidget = None,
    ):
        super().__init__(parent)

        self.min_value = min_value
        self.max_value = max_value
        self.step = step

        layout = QHBoxLayout()

        # Label
        self.label = QLabel(label)
        self.label.setFixedWidth(120)
        layout.addWidget(self.label)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, step)
        self.slider.setValue(
            int((default_value - min_value) / (max_value - min_value) * step)
        )
        layout.addWidget(self.slider)

        # Value display
        self.value_label = QLineEdit(f"{default_value:.3f}")
        self.value_label.setFixedWidth(60)
        layout.addWidget(self.value_label)

        # Connect signals
        self.slider.valueChanged.connect(self.update_value_label)
        self.value_label.editingFinished.connect(self.update_slider)

        self.setLayout(layout)

    def update_value_label(self, slider_value: int):
        actual_value = self.min_value + (slider_value / self.step) * (
            self.max_value - self.min_value
        )
        self.value_label.setText(f"{actual_value:.3f}")

    def update_slider(self):
        try:
            value = float(self.value_label.text())
            if self.min_value <= value <= self.max_value:
                slider_value = int(
                    (value - self.min_value)
                    / (self.max_value - self.min_value)
                    * self.step
                )
                self.slider.blockSignals(True)
                self.slider.setValue(slider_value)
                self.slider.blockSignals(False)
            else:
                # Reset to current slider value if out of range
                self.update_value_label(self.slider.value())
        except ValueError:
            # Reset to current slider value if invalid input
            self.update_value_label(self.slider.value())

    def get_value(self) -> float:
        return self.min_value + (self.slider.value() / self.step) * (
            self.max_value - self.min_value
        )
