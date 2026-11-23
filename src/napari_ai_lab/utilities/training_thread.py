"""
TrainingThread: manages running model training operations in a background thread.

Provides a QThread wrapper that forwards progress updates from training loops
to the main/GUI thread via Qt signals for safe widget updates.
"""

from qtpy.QtCore import QObject, QThread, Signal

from .processor_thread import relay_signals


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
        except Exception as e:  # noqa: BLE001
            # Anything else -- a worker process that crashed, say -- is still
            # a failed training, said in a line. Left uncaught, napari shows
            # it with every local variable: every image path, for YOLO.
            import traceback

            traceback.print_exc()
            self.error.emit(f"{type(e).__name__}: {e}")
        finally:
            self.finished.emit(result)


class _TrainingRelay(QObject):
    """_TrainingWorker's signals, on the GUI thread; see relay_signals."""

    progress = Signal(int, int, str)
    log_info = Signal(str)
    log_warning = Signal(str)
    log_error = Signal(str)
    finished = Signal(object)
    error = Signal(str)


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

        # Expose signals for external connection, from the GUI thread
        names = (
            "progress",
            "log_info",
            "log_warning",
            "log_error",
            "finished",
            "error",
        )
        self.relay = _TrainingRelay()
        relay_signals(self.worker, self.relay, names)
        for name in names:
            setattr(self, name, getattr(self.relay, name))

    def start(self):
        """Start training on the worker thread."""
        self.thread.start()
