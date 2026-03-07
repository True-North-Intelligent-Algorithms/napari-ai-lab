"""
QtPy-based progress logger with visible progress bar and log output.

Provides a ProgressLogger implementation using QtPy widgets that can be
embedded in any Qt-based GUI.
"""

from qtpy.QtWidgets import QProgressBar, QTextBrowser, QVBoxLayout, QWidget


class QtProgressLogger:
    """
    Qt-based progress logger with progress bar and text log.

    Creates a widget containing:
    - QProgressBar for visual progress tracking
    - QTextBrowser for log messages

    Can be embedded in any Qt GUI by calling get_widget().
    """

    def __init__(self):
        """Initialize Qt progress logger with widget components."""
        # Create container widget
        self._widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        layout.setSpacing(5)  # Small spacing between widgets
        self._widget.setLayout(layout)

        # Progress bar (fixed height, doesn't stretch)
        self.progressBar = QProgressBar()
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(100)
        self.progressBar.setValue(0)
        self.progressBar.setFixedHeight(25)  # Fixed height
        layout.addWidget(self.progressBar, 0)  # Stretch factor 0 = no stretch

        # Text log browser (stretches to fill space)
        self.textBrowser_log = QTextBrowser()
        self.textBrowser_log.setMinimumHeight(100)  # Minimum height
        layout.addWidget(
            self.textBrowser_log, 1
        )  # Stretch factor 1 = stretches

        self._total = 0

    def get_widget(self):
        """
        Get the Qt widget containing progress bar and log.

        Returns:
            QWidget: Widget that can be added to layouts or docked.
        """
        return self._widget

    def update_progress(self, current: int, total: int, message: str = ""):
        """
        Update progress bar and optionally log message.

        Args:
            current: Current progress value
            total: Total items to process
            message: Optional message to log (only logs on milestones)
        """
        self._total = total

        # Calculate percentage
        if total > 0:
            percentage = int((current / total) * 100)
            self.progressBar.setValue(percentage)

        # Log milestone messages (every 10% or completion)
        if message and (
            current % max(1, total // 10) == 0 or current == total
        ):
            status = f"{message} ({current}/{total})"
            self.textBrowser_log.append(status)

    def log_info(self, message: str):
        """Log informational message."""
        self.textBrowser_log.append(f"ℹ️  {message}")

    def log_warning(self, message: str):
        """Log warning message."""
        self.textBrowser_log.append(f"⚠️  {message}")

    def log_error(self, message: str):
        """Log error message."""
        self.textBrowser_log.append(f"❌ {message}")

    def clear(self):
        """Clear log and reset progress bar."""
        self.textBrowser_log.clear()
        self.progressBar.setValue(0)
        self._total = 0
