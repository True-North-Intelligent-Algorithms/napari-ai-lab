"""
SequenceProcessor: runs an operation over every image in a sequence, one
SliceProcessor per image.

Offers the same ``process_all`` as SliceProcessor, so ProcessorThread drives
either.  See docs/spec/0006-batch-segmentation-over-a-sequence.md.
"""

from .slice_processor import SliceProcessor, unsupported_iteration


def why_not(segmenter, axis_types, selected_axis, axes_to_collapse=None):
    """Why this image cannot be processed, or None if it can.

    Two questions, kept apart on purpose: whether the segmenter takes these
    axes, and whether the slicer can build the iteration for what is left.
    The first is the segmenter's to answer and permanent; the second is this
    package's own limitation and should shrink.
    """
    ok, reason = segmenter.can_process(axis_types, selected_axis)
    if not ok:
        return reason
    return unsupported_iteration(axis_types, selected_axis, axes_to_collapse)


class SequenceProcessor:
    """Runs an operation over every image in a sequence, slice by slice.

    Same interface as SliceProcessor -- one ``process_all`` -- so
    ProcessorThread drives either.  Per image it loads, asks the segmenter
    whether it can run there, and either skips with a reason or hands the
    slices to a SliceProcessor.

    Saving happens here, where the image index is known.  That is what makes a
    headless run just a call with no callbacks: artifacts are written, nothing
    is drawn.

    ``source`` is anything offering ``get_image_count()``, ``get_image_paths()``,
    ``load_image(i)``, ``image_data``, ``axis_types`` and ``save_predictions``.
    ImageDataModel today, and nothing here should need more than that.

    Args:
        source: The image sequence, as described above.
        segmenter: Asked ``can_process`` per image and
            ``get_segmentation_axis`` once; nothing else is called.
        selected_axis: Spatial axis string, e.g. "YX", "ZYX".
        axes_to_collapse: Optional axis names collapsed rather than iterated.
        current_index: The image the viewer is showing.  It is reloaded when
            the batch ends so the model matches the viewer again, and it is
            the only image whose slices reach ``on_slice_done`` -- results for
            the others are written to disk and picked up when the user
            scrolls to them.
    """

    def __init__(
        self,
        source,
        segmenter,
        selected_axis,
        axes_to_collapse=None,
        current_index=None,
    ):
        self.source = source
        self.segmenter = segmenter
        self.selected_axis = selected_axis
        self.axes_to_collapse = axes_to_collapse
        self.current_index = current_index
        # Predictions are saved against the axes the segmenter *outputs*,
        # which may drop a channel: YXC in, YX out.
        self.segmentation_axis = segmenter.get_segmentation_axis(selected_axis)
        self.total_slices = source.get_image_count()
        self.processed = []
        self.skipped = []

    def process_all(
        self,
        operation_fn,
        on_slice_done=None,
        on_progress=None,
        start_index=None,
        end_index=None,
    ):
        """Process every image in the (optional) inclusive range.

        ``start_index`` and ``end_index`` are *image* indices here, not the
        flat slice indices SliceProcessor takes.  Progress is likewise
        reported per image, so it runs monotonically across the batch rather
        than resetting at every image.

        Args:
            operation_fn: Callable(current_step) -> result, reading whichever
                image is currently loaded.
            on_slice_done: Optional callable(current_step, result), called
                only for ``current_index``.
            on_progress: Optional callable(current_image, total_images).
            start_index: Optional inclusive index of the first image.
            end_index: Optional inclusive index of the last image.
        """
        count = self.source.get_image_count()
        first = 0 if start_index is None else max(0, int(start_index))
        last = (
            count - 1 if end_index is None else min(count - 1, int(end_index))
        )
        if first > last:
            return
        total = last - first + 1
        self.processed = []
        self.skipped = []

        try:
            for i in range(first, last + 1):
                if on_progress:
                    on_progress(i - first + 1, total)

                self.source.load_image(i)
                reason = why_not(
                    self.segmenter,
                    self.source.axis_types,
                    self.selected_axis,
                    self.axes_to_collapse,
                )
                if reason:
                    self.skipped.append((i, reason))
                    print(f"Skipping {self._name(i)}: {reason}")
                    continue

                processor = SliceProcessor(
                    self.source.image_data.shape,
                    self.selected_axis,
                    self.axes_to_collapse,
                )
                processor.process_all(
                    operation_fn,
                    on_slice_done=self._save_then(
                        i, on_slice_done if i == self.current_index else None
                    ),
                )
                self.processed.append(i)
        finally:
            if self.current_index is not None:
                self.source.load_image(self.current_index)

    def _name(self, image_index):
        """The file name for an image index, for logging."""
        return self.source.get_image_paths()[image_index].name

    def _save_then(self, image_index, on_slice_done):
        """Wrap a slice callback so the prediction is saved first."""

        def save(current_step, result):
            self.source.save_predictions(
                result,
                image_index,
                current_step=current_step,
                selected_axis=self.segmentation_axis,
                axes_to_collapse=self.axes_to_collapse,
            )
            if on_slice_done:
                on_slice_done(current_step, result)

        return save

    def summary(self):
        """What ran and what did not, one line per skipped image."""
        lines = [
            f"{len(self.processed)} processed, {len(self.skipped)} skipped"
        ]
        lines += [f"  {self._name(i)}: {reason}" for i, reason in self.skipped]
        return "\n".join(lines)
