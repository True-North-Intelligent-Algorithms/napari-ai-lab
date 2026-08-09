"""Measure how long the sequence viewer takes to switch position.

Switching position tears down the current layers and rebuilds them for the
next image, which is slow enough to be felt. This script puts a number on it
so the effect of any fix can be compared against a baseline.

It is deliberately the smallest thing that reproduces the cost: one dataset,
no segmenters, no augmenters. Registering those pulls in torch and friends and
has nothing to do with layer switching.

Run it directly:

    python benchmarks/bench_sequence_switch.py

Position 0 is loaded during launch and is not measured -- it carries a cold
file cache and one-time setup. Two switches are timed, 0 -> 1 and 1 -> 2,
because at several seconds each that is enough signal without a long wait.
"""

import cProfile
import pstats
import subprocess
import sys
import time
from pathlib import Path

import napari
from qtpy.QtWidgets import QApplication

from napari_ai_lab.apps.nd_ai_lab_launcher import launch_nd_ai_lab
from napari_ai_lab.Augmenters import SimpleAugmenter

PROJECT_ROOT = Path(__file__).parent.parent
DATASET = PROJECT_ROOT / "tests" / "test_images" / "neurips blood cells"

VIEWER_TYPE = "sequence"
AXES_TO_COLLAPSE = "C"
AXIS_TYPES = "NYXC"

#: Leave the viewer open when the run finishes, to poke at it by hand.
INTERACTIVE = False

#: Pass --profile to rank what the switch actually spends its time in. Only the
#: switches are profiled, not launch. Note that profiling inflates the reported
#: seconds -- read the ranking, not the totals, on a profiled run.
PROFILE = "--profile" in sys.argv


def context() -> str:
    """One line identifying this run, so two runs can be compared."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        commit = "unknown"
    return f"commit {commit}, napari {napari.__version__}"


def time_switch(sequence_viewer, index: int) -> float:
    """Move the sequence viewer to `index` and return the seconds it took.

    Driving the scrollbar rather than calling the handler directly keeps this
    on the same path the user takes: valueChanged -> _on_scroll_changed ->
    _load_current_image, which removes the old layers and adds new ones.
    processEvents is inside the timed region so that repaints the switch
    triggers are counted too, rather than landing after the clock stops.
    """
    start = time.perf_counter()
    sequence_viewer.image_scrollbar.setValue(index)
    QApplication.processEvents()
    return time.perf_counter() - start


def main() -> None:
    if not DATASET.is_dir():
        raise SystemExit(f"Dataset not found: {DATASET}")

    # The augment tab assumes at least one augmenter is registered -- with none,
    # NDEasyAugment never assigns self.augmenter and the first image change
    # fails. SimpleAugmenter is numpy-only, so registering it keeps this on the
    # same path as launch_nd_ai_lab.py without pulling in albumentations.
    SimpleAugmenter.register()

    viewer = napari.Viewer()

    launch_start = time.perf_counter()
    _, sequence_viewer, _ = launch_nd_ai_lab(
        viewer,
        DATASET,
        viewer_type=VIEWER_TYPE,
        axes_to_collapse=AXES_TO_COLLAPSE,
        axis_types=AXIS_TYPES,
    )
    launch_seconds = time.perf_counter() - launch_start

    if sequence_viewer is None:
        raise SystemExit(f"No sequence viewer for viewer_type={VIEWER_TYPE!r}")

    # set_image_data_model already loaded position 0 during launch.
    QApplication.processEvents()
    assert (
        sequence_viewer.current_index == 0
    ), "expected to start at position 0"

    profiler = cProfile.Profile() if PROFILE else None
    if profiler:
        profiler.enable()

    first = time_switch(sequence_viewer, 1)
    second = time_switch(sequence_viewer, 2)

    if profiler:
        profiler.disable()

    print()
    print("=" * 56)
    print("Sequence viewer switch timing")
    print(context())
    print(f"dataset: {DATASET.name}")
    print("-" * 56)
    print(f"launch (includes loading position 0) : {launch_seconds:7.2f} s")
    print(f"switch 0 -> 1                        : {first:7.2f} s")
    print(f"switch 1 -> 2                        : {second:7.2f} s")
    print("=" * 56)

    if profiler:
        print("\nTop 25 by cumulative time (two switches, profiled):")
        print("Times are inflated by profiling -- compare the ranking, not")
        print("the totals, against the unprofiled run above.\n")
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative").print_stats(25)

    if INTERACTIVE:
        napari.run()


if __name__ == "__main__":
    main()
