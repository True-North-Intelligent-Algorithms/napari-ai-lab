"""
ProcessorThread: runs a processor's ``process_all`` on a worker thread.

It only ever calls ``process_all``, so it drives any processor offering that
method -- SliceProcessor and SequenceProcessor today, augmentation as well.
The only module here that imports Qt.
"""

import traceback
from typing import ClassVar

from qtpy.QtCore import QObject, QThread, QTimer, Signal

# Why the wrappers below hand out a relay's signals, not the worker's: under
# PySide6 (napari's bundled app) a lambda or plain function connected to a
# worker's signal runs on the worker thread, since it has no thread of its
# own. PyQt sends it to the GUI thread instead. Run off the GUI thread, a
# callback that adds a layer corrupts Qt: garbage drawn, a window that hangs.
# A relay is a QObject made on the GUI thread; a signal-to-signal connection
# to it is queued, so it re-emits there and every slot runs there.


def relay_signals(source, relay, names):
    """Forward each named signal of *source* to *relay*'s signal of that name."""
    for name in names:
        getattr(source, name).connect(getattr(relay, name))


def _release_when_finished(wrapper, running):
    """Drop *wrapper* from *running* once its relay's finished has fired.

    On the next turn of the GUI loop, so the relay outlives every slot
    connected to its finished, the caller's included.
    """
    wrapper.relay.finished.connect(
        lambda: QTimer.singleShot(0, lambda: running.discard(wrapper))
    )


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


class _ProcessorRelay(QObject):
    """_ProcessorWorker's signals, on the GUI thread; see relay_signals."""

    progress = Signal(int, int)
    slice_done = Signal(tuple, object)
    error = Signal(str)
    finished = Signal()


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

        # Wire lifecycle
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Expose signals for external connection, from the GUI thread
        self.relay = _ProcessorRelay()
        relay_signals(
            self.worker, self.relay, ("progress", "slice_done", "error")
        )
        relay_signals(self.thread, self.relay, ("finished",))
        self.progress = self.relay.progress
        self.slice_done = self.relay.slice_done
        self.finished = self.relay.finished
        self.error = self.relay.error
        _release_when_finished(self, ProcessorThread._running)

    def start(self):
        """Start processing on the worker thread."""
        self.thread.start()


class _CallWorker(QObject):
    """Runs one callable on a worker thread; see CallThread."""

    done = Signal(object)  # the callable's return value
    error = Signal(str)
    finished = Signal()

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        # Broad for the same reason as _ProcessorWorker.run.
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.error.emit(f"{type(e).__name__}: {e}")
        finally:
            self.finished.emit()


class _CallRelay(QObject):
    """_CallWorker's signals, on the GUI thread; see relay_signals."""

    done = Signal(object)
    error = Signal(str)
    finished = Signal()


class CallThread:
    """Run one callable off the GUI thread, ProcessorThread's way.

    For a single long call -- segmenting the current image, which may first
    build an environment for minutes -- so the window keeps painting and a
    progress log can show what is happening. ``done`` carries the return
    value to the GUI thread; touch layers and widgets there, not in *fn*.
    """

    _running: ClassVar[set] = set()

    def __init__(self, fn):
        self.thread = QThread()
        self.worker = _CallWorker(fn)
        self.worker.moveToThread(self.thread)

        CallThread._running.add(self)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.relay = _CallRelay()
        relay_signals(self.worker, self.relay, ("done", "error"))
        relay_signals(self.thread, self.relay, ("finished",))
        self.done = self.relay.done
        self.error = self.relay.error
        self.finished = self.relay.finished
        _release_when_finished(self, CallThread._running)

    def start(self):
        self.thread.start()


class BuildRelay(QObject):
    """Carries environment-build news from a build's thread to the GUI's.

    Pass ``relay.message.emit`` as the listener; a Qt signal emitted from
    another thread is delivered on the receiver's own thread.
    """

    message = Signal(str)
