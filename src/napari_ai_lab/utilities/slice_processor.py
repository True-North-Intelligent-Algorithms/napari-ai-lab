"""
SliceProcessor: iterates over non-spatial slices of ND data and applies
an operation to each slice.

Supports both single-slice (process_slice) and full-volume (process_all) modes.
"""

import itertools

import numpy as np


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

    def process_all(self, operation_fn, on_slice_done=None, on_progress=None):
        """Process all non-spatial slices.

        Args:
            operation_fn: Callable(current_step) -> result.
            on_slice_done: Optional callable(current_step, result) called
                after each slice.
            on_progress: Optional callable(current_index, total_slices)
                called before each slice for progress reporting.
        """
        for idx, current_step in self.iter_steps():
            if on_progress:
                on_progress(idx + 1, self.total_slices)
            self.process_slice(current_step, operation_fn, on_slice_done)
