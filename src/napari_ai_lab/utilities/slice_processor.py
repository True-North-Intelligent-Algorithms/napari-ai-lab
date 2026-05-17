"""
SliceProcessor: iterates over non-spatial slices of ND data and applies
an operation to each slice.

Supports both single-slice (process_slice) and full-volume (process_all) modes.
SliceProcessorWorker wraps process_all in a QThread for non-blocking GUI updates.
"""

import itertools

import numpy as np
from qtpy.QtCore import QObject, QThread, Signal


class SliceProcessor:
    """Processes operations over non-spatial slices of ND data.

    Given an image shape, a selected spatial axis string (e.g. "YX", "ZYX"),
    and optionally axes to collapse, this class computes the iteration space
    and provides methods to process individual slices or all slices.

    Args:
        image_shape: Shape of the full ND image array.
        selected_axis: Spatial axis string, e.g. "YX", "ZYX", "YXC".
        axes_to_collapse: Optional list of axis names that are collapsed
            (not iterated over, not spatial).
    """

    def __init__(self, image_shape, selected_axis, axes_to_collapse=None):
        self.image_shape = image_shape
        self.selected_axis = selected_axis
        self.axes_to_collapse = axes_to_collapse

        # Determine number of spatial dimensions from the axis string
        if selected_axis.endswith("ZYX"):
            self.num_spatial = 3
        elif selected_axis.endswith("YX"):
            self.num_spatial = 2
        else:
            self.num_spatial = 2

        num_non_spatial = len(image_shape) - self.num_spatial
        num_collapsed = len(axes_to_collapse) if axes_to_collapse else 0
        self.num_for_loop = (
            num_non_spatial - num_collapsed
            if axes_to_collapse
            else num_non_spatial
        )

        # Dimensions to iterate over are assumed to be the leading dims
        self.loop_shape = image_shape[: self.num_for_loop]
        self.total_slices = (
            int(np.prod(self.loop_shape)) if self.loop_shape else 1
        )

    def _make_step(self, non_spatial_indices):
        """Build a current_step tuple from non-spatial indices."""
        return non_spatial_indices + (0,) * self.num_spatial

    def iter_steps(self):
        """Yield (index, current_step) for all non-spatial slices."""
        for idx, non_spatial_indices in enumerate(
            itertools.product(*[range(d) for d in self.loop_shape])
        ):
            yield idx, self._make_step(non_spatial_indices)

    def process_slice(self, current_step, operation_fn, on_slice_done=None):
        """Process a single slice.

        Args:
            current_step: Tuple of indices identifying the slice position.
            operation_fn: Callable(current_step) -> result.
            on_slice_done: Optional callable(current_step, result) called
                after operation_fn completes.

        Returns:
            The result of operation_fn.
        """
        result = operation_fn(current_step)
        if on_slice_done:
            on_slice_done(current_step, result)
        return result

    def process_all(
        self,
        operation_fn,
        on_slice_done=None,
        on_progress=None,
        start_index=None,
        end_index=None,
    ):
        """Process non-spatial slices in the (optional) flat index range.

        Args:
            operation_fn: Callable(current_step) -> result.
            on_slice_done: Optional callable(current_step, result) called
                after each slice.
            on_progress: Optional callable(current_index, total_slices)
                called before each slice for progress reporting.
            start_index: Optional inclusive flat index of the first slice
                to process.  Defaults to 0.
            end_index: Optional inclusive flat index of the last slice to
                process.  Defaults to ``total_slices - 1``.
        """
        first = 0 if start_index is None else max(0, int(start_index))
        last = (
            self.total_slices - 1
            if end_index is None
            else min(self.total_slices - 1, int(end_index))
        )
        if first > last:
            return
        total = last - first + 1
        report_idx = 0
        for idx, current_step in self.iter_steps():
            if idx < first:
                continue
            if idx > last:
                break
            report_idx += 1
            if on_progress:
                on_progress(report_idx, total)
            self.process_slice(current_step, operation_fn, on_slice_done)


class _SliceWorker(QObject):
    """QObject that runs SliceProcessor.process_all in a worker thread.

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
            self.finished.emit()


class SliceProcessorThread:
    """Convenience wrapper that manages QThread + _SliceWorker lifecycle.

    Usage::

        spt = SliceProcessorThread(processor, operation_fn)
        spt.progress.connect(my_progress_handler)
        spt.slice_done.connect(my_slice_done_handler)
        spt.finished.connect(my_finished_handler)
        spt.start()

    The caller must keep a reference to this object until ``finished`` fires.
    """

    def __init__(
        self, processor, operation_fn, start_index=None, end_index=None
    ):
        self.thread = QThread()
        self.worker = _SliceWorker(
            processor,
            operation_fn,
            start_index=start_index,
            end_index=end_index,
        )
        self.worker.moveToThread(self.thread)

        # Wire lifecycle
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Expose signals for external connection
        self.progress = self.worker.progress
        self.slice_done = self.worker.slice_done
        self.finished = self.worker.finished
        self.error = self.worker.error

    def start(self):
        """Start processing on the worker thread."""
        self.thread.start()


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
