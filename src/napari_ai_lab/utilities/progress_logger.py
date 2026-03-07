"""
Progress and logging utilities for napari-ai-lab.

Provides a generic interface for progress tracking and logging that can work
with different backends (Napari, tqdm, console, etc.).
"""

from typing import Protocol


class ProgressLogger(Protocol):
    """
    Protocol for progress tracking and logging.

    This protocol defines a generic interface that allows different implementations
    (Napari notifications, tqdm, console prints, etc.) without coupling code to
    specific frameworks.
    """

    def update_progress(
        self, current: int, total: int, message: str = ""
    ) -> None:
        """
        Update progress indicator.

        Args:
            current: Current progress value (e.g., completed items)
            total: Total items to process
            message: Optional message describing the operation
        """
        ...

    def log_info(self, message: str) -> None:
        """Log informational message."""
        ...

    def log_warning(self, message: str) -> None:
        """Log warning message."""
        ...

    def log_error(self, message: str) -> None:
        """Log error message."""
        ...


class NapariProgressLogger:
    """
    Napari-specific implementation of ProgressLogger.

    Uses Napari's built-in progress tracking (napari.utils.progress) and
    notification system. Falls back to console printing if viewer is not available.
    """

    def __init__(self, viewer=None):
        """
        Initialize with optional napari viewer.

        Args:
            viewer: Napari viewer instance. If None, falls back to console output.
        """
        self.viewer = viewer
        self._progress_bar = None
        self._current_total = 0

    def update_progress(self, current: int, total: int, message: str = ""):
        """Update progress using napari.utils.progress."""
        # Create progress bar on first call
        if self._progress_bar is None and total > 0:
            try:
                from napari.utils import progress

                self._progress_bar = progress(total=total, desc=message)
                self._current_total = total
            except (ImportError, RuntimeError):
                # Fallback if napari progress not available
                self._progress_bar = None

        # Update progress bar
        if self._progress_bar is not None:
            # Calculate increment since last update
            increment = current - (
                self._progress_bar.n if hasattr(self._progress_bar, "n") else 0
            )
            if increment > 0:
                self._progress_bar.update(increment)

            # Update description if message changed
            if message and hasattr(self._progress_bar, "set_description"):
                self._progress_bar.set_description(message)
        else:
            # Console fallback
            if current % 10 == 0 or current == total:
                status = (
                    f"{message} ({current}/{total})"
                    if message
                    else f"{current}/{total}"
                )
                print(f"  {status}")

        # Close progress bar when complete
        if current >= total and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None

    def log_info(self, message: str):
        """Show info notification in napari or print to console."""
        if (
            self.viewer
            and hasattr(self.viewer, "window")
            and self.viewer.window
        ):
            try:
                self.viewer.window.notification_manager.show_info(message)
            except (AttributeError, RuntimeError):
                print(f"ℹ️  {message}")
        else:
            print(f"ℹ️  {message}")

    def log_warning(self, message: str):
        """Show warning notification in napari or print to console."""
        if (
            self.viewer
            and hasattr(self.viewer, "window")
            and self.viewer.window
        ):
            try:
                self.viewer.window.notification_manager.show_warning(message)
            except (AttributeError, RuntimeError):
                print(f"⚠️  {message}")
        else:
            print(f"⚠️  {message}")

    def log_error(self, message: str):
        """Show error notification in napari or print to console."""
        if (
            self.viewer
            and hasattr(self.viewer, "window")
            and self.viewer.window
        ):
            try:
                self.viewer.window.notification_manager.show_error(message)
            except (AttributeError, RuntimeError):
                print(f"❌ {message}")
        else:
            print(f"❌ {message}")


class ConsoleProgressLogger:
    """
    Simple console-based progress logger.

    Fallback implementation that uses print statements.
    Useful for testing or when no viewer is available.
    """

    def update_progress(self, current: int, total: int, message: str = ""):
        """Print progress to console."""
        status = f"{message} " if message else ""
        print(f"Progress: {status}{current}/{total}")

    def log_info(self, message: str):
        """Print info message to console."""
        print(f"ℹ️  {message}")

    def log_warning(self, message: str):
        """Print warning message to console."""
        print(f"⚠️  {message}")

    def log_error(self, message: str):
        """Print error message to console."""
        print(f"❌ {message}")
