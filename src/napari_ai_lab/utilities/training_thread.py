"""
TrainingThread: manages running model training operations in a background thread.

Provides a QThread wrapper that forwards progress updates from training loops
to the main/GUI thread via Qt signals for safe widget updates.
"""

from qtpy.QtCore import QObject, QThread, Signal


class _TrainingWorker(QObject):
    """QObject that runs a segmenter's ``train`` method in a worker thread.

    The worker forwards updater(step, total, message) calls from the
    training loop to the main/GUI thread via the ``progress`` signal so
    Qt widgets can be updated safely.
    """

    progress = Signal(int, int, str)  # (step, total, message)
    log_info = Signal(str)
    log_warning = Signal(str)
    log_error = Signal(str)
    finished = Signal(object)  # result dict (or None on error)
    error = Signal(str)

    def __init__(self, train_fn):
        """Args:
        train_fn: callable accepting a single ``updater`` kwarg.
        """
        super().__init__()
        self.train_fn = train_fn

    def _updater(self, step, total, message):
        # Called on the worker thread; emit a signal for the GUI thread.
        self.progress.emit(int(step), int(total), str(message))

    def run(self):
        """Execute training; called on the worker thread."""
        result = None
        try:
            result = self.train_fn(updater=self._updater)
        except (
            RuntimeError,
            ValueError,
            TypeError,
            OSError,
            IndexError,
            AttributeError,
        ) as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit(result)


class TrainingThread:
    """Convenience wrapper that manages QThread + _TrainingWorker lifecycle.

    Usage::

        tt = TrainingThread(segmenter.train)
        tt.progress.connect(progress_logger.update_progress)
        tt.finished.connect(on_training_finished)  # receives result dict
        tt.error.connect(on_training_error)
        tt.start()

    The caller must keep a reference to this object until ``finished`` fires.
    """

    def __init__(self, train_fn):
        self.thread = QThread()
        self.worker = _TrainingWorker(train_fn)
        self.worker.moveToThread(self.thread)

        # Wire lifecycle
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Expose signals for external connection
        self.progress = self.worker.progress
        self.log_info = self.worker.log_info
        self.log_warning = self.worker.log_warning
        self.log_error = self.worker.log_error
        self.finished = self.worker.finished
        self.error = self.worker.error

    def start(self):
        """Start training on the worker thread."""
        self.thread.start()
