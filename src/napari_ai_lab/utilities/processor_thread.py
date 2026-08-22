"""
ProcessorThread: runs a processor's ``process_all`` on a worker thread.

It only ever calls ``process_all``, so it drives any processor offering that
method -- SliceProcessor and SequenceProcessor today, augmentation as well.
The only module here that imports Qt.
"""

import traceback
from typing import ClassVar

from qtpy.QtCore import QObject, QThread, Signal


class _ProcessorWorker(QObject):
    """QObject that runs a processor's process_all in a worker thread.

    Emits signals so the main/GUI thread can safely update widgets and layers.
    """

    progress = Signal(int, int)  # (current, total)
    slice_done = Signal(tuple, object)  # (current_step, result)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self, processor, operation_fn, start_index=None, end_index=None
    ):
        super().__init__()
        self.processor = processor
        self.operation_fn = operation_fn
        self.start_index = start_index
        self.end_index = end_index

    def run(self):
        """Execute process_all; called on the worker thread."""
        try:
            self.processor.process_all(
                self.operation_fn,
                on_slice_done=lambda step, result: self.slice_done.emit(
                    step, result
                ),
                on_progress=lambda cur, tot: self.progress.emit(cur, tot),
                start_index=self.start_index,
                end_index=self.end_index,
            )
        # Broad on purpose. An exception escaping a Qt slot does not raise --
        # it aborts the process, and the user sees the window vanish with no
        # message. Anything unexpected becomes an error signal instead.
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.error.emit(f"{type(e).__name__}: {e}")
        finally:
            self.finished.emit()


class ProcessorThread:
    """Convenience wrapper that manages QThread + _ProcessorWorker lifecycle.

    Usage::

        spt = ProcessorThread(processor, operation_fn)
        spt.progress.connect(my_progress_handler)
        spt.slice_done.connect(my_slice_done_handler)
        spt.finished.connect(my_finished_handler)
        spt.start()

    ``finished`` is the QThread's, so it means the thread has really stopped.
    """

    #: Alive wrappers, so Python cannot collect one mid-run and abort Qt.
    _running: ClassVar[set] = set()

    def __init__(
        self, processor, operation_fn, start_index=None, end_index=None
    ):
        self.thread = QThread()
        self.processor = processor
        self.worker = _ProcessorWorker(
            processor,
            operation_fn,
            start_index=start_index,
            end_index=end_index,
        )
        self.worker.moveToThread(self.thread)

        ProcessorThread._running.add(self)
        self.thread.finished.connect(
            lambda: ProcessorThread._running.discard(self)
        )

        # Wire lifecycle
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Expose signals for external connection
        self.progress = self.worker.progress
        self.slice_done = self.worker.slice_done
        self.finished = self.thread.finished
        self.error = self.worker.error

    def start(self):
        """Start processing on the worker thread."""
        self.thread.start()
