# 0006 — Segmenting a whole sequence

"Segment Range" segments every image in stacked mode and crashes in sequence
mode. This describes the loop that is missing, where it goes, and what to do
when the images in a folder are not all the same shape.

The short version: a sequence is the outer loop and a stacked sequence is a
sequence of one, so there is one loop rather than two modes.

## What happens today

`_on_segment_range` (`nd_easy_segment.py:557`) hands
`image_data_model.image_data` to a `SliceProcessor`, which counts
`total_slices` from the non-spatial axes of that one array
(`slice_processor.py:52`) and walks flat indices over them.

That is correct in stacked mode by accident. `load_image(0, stacked=True)`
sets `image_data` to the whole stack with a leading sequence axis
(`image_data_model.py:229`), so the sequence *is* a non-spatial axis and
iterating slices iterates images for free.

In sequence mode `load_image(i)` sets `image_data` to one squeezed image
(`image_data_model.py:275,287`). There is no sequence axis, so nothing in the
range machinery can reach the other images. The loop it needs does not exist.

This is the same split 0005 describes: `self.image_data` is *either* one image
*or* the whole stack, and everything downstream has to know which.

## A sequence is the outer loop

The segment tab should not ask which mode it is in. It asks the model for the
images to process and gets either N entries or one, and the inner
`SliceProcessor` walks whatever axes that entry has:

- **sequence** — N entries, one per file, each with its own axes
- **stacked** — one entry, whose array carries the sequence axis

Stacking stays what 0005 calls it: an IO strategy, not a mode. It earns its
place for many small same-shaped files, where one contiguous read beats N
opens.

## Where the loop goes

`_SliceWorker` only ever calls `processor.process_all(...)`
(`slice_processor.py:149`) — one call, one method, nothing else on the
processor is touched. It is duck-typed. So a batcher exposing the same
interface as `SliceProcessor` reuses the existing thread wrapper unchanged —
threading, progress, `slice_done` and error handling all keep working, and no
second thread class is needed.

Which makes the wrapper's name wrong. `SliceProcessorThread` driving a
`SequenceProcessor` reads like a bug every time someone meets it, and the
class was never slice-specific — only its name was. It becomes
`ProcessorThread`, and `_SliceWorker` becomes `_ProcessorWorker`. Six lines
outside the module (`utilities/__init__.py:32,62`, `nd_easy_augment.py:22,398`,
`image_data_model.py:2477,2490`) plus two docstring mentions at
`image_data_model.py:2469,2474`. No back-compat alias: the only consumers are
in this repo, and an alias would preserve exactly the confusing name being
removed. `SliceProcessor` keeps its name — it really does process slices.

So: a `SequenceProcessor` beside `SliceProcessor` in
`utilities/slice_processor.py`. It is a loop, not state, which is why it is
not on `ImageDataModel`; and it imports no Qt, which is what makes it usable
headless.

```python
class SequenceProcessor:
    """Runs an operation over every image in the sequence.

    Same interface as SliceProcessor, so ProcessorThread drives either.
    """
    def __init__(self, model, selected_axis, axes_to_collapse=None): ...

    def process_all(self, operation_fn, on_slice_done=None, on_progress=None,
                    start_index=None, end_index=None):
        for i in range(first, last + 1):
            self.image_index = i
            self.model.load_image(i)
            SliceProcessor(self.model.image_data.shape, ...).process_all(...)
```

`ImageDataModel` gains `segment_sequence(...)` mirroring `segment_range`
(`:2441`) — same arguments, same `(processor, thread)` return — so the call
site changes by one word.

## Headless is the absence of a callback, not a flag

Saving currently happens in the widget's `_on_segment_slice_done`
(`nd_easy_segment.py:704–711`), which also updates the viewer. Headless cannot
call that, so the `save_predictions` call moves into the batcher, where the
image index is known. The widget callback keeps only the layer update.

After that, "headless" needs no flag anywhere: construct a `SequenceProcessor`
and call `process_all` with no `on_slice_done`. Artifacts are written; nothing
is drawn.

For the GUI the view stays where it is while the batch runs. Results for other
images appear when the user scrolls to them, through the prediction-loading
path that already exists. The viewer never scrolls itself.

## Heterogeneous sequences

Nothing stops a folder holding a 7D czi, an RGB png and a 3D greyscale stack.
The axis chosen from the image on screen may be meaningless for the next one.

`selected_axis` names the axes the segmenter consumes; the rest are iterated.
So the test per image is a set test against `axis_types`, which `load_image`
already populates (`image_data_model.py:277,281`):

- axes **present** → run. Extra non-spatial axes are more iterations, not a
  problem: `YX` against a `ZYX` image is 40 slices.
- axes **missing** → skip. `ZYX` against a `TYX` image cannot run; the image
  has no Z.

Letters alone are not sufficient. `axes_to_collapse` is chosen once from the
current image, so an RGB `YXC` and a greyscale `YX` both pass "is YX present"
and then one of them gets its channel axis iterated as if it were time — three
greyscale segmentations of one photo. The test is therefore *requested axes
present, and the leftover axes ones this configuration knows how to iterate*.

On what to do about a failure:

- If the **currently viewed** image fails, stop with an error before starting.
  That is a mistake in the request, not a heterogeneous folder, and there is
  nothing worth batching.
- Otherwise skip, log the reason per image (`image_042.czi: axes TYX, need
  Z`), and finish with a summary. A twenty-minute batch should not die at
  image 73.

A pre-flight pass that classifies the sequence and puts the count in front of
the user — "84 of 100 will be processed, 16 skipped" — is the version of this
worth having, since discovering it at the end is the same information too
late. Whether it is affordable depends on reading shapes without a full load;
tiff and czi headers both carry it, but the IO layer does not expose it yet.

## Model state after the batch

`load_image(i)` mutates `image_data`, `axis_types` and `scale`. Left alone,
the batch ends with the model pointing at the last image while the viewer
shows another. The batcher reloads the starting index in a `finally`.

This leans on exactly the ambient state 0005 wants gone. It is deliberate and
temporary: when callers pass the array they mean instead of reading
`self.image_data`, the reload disappears and the batcher gets shorter. Nothing
here should make that harder.

## Decisions

Both settled 2026-08-20.

- **The compatibility test is a `segmenter.can_process()` hook**, not a set
  test in the batcher. The letters alone cannot express what some segmenters
  need — a tracker needs `T` specifically, not merely three axes — and putting
  the test on the segmenter is where that knowledge already lives. The default
  implementation on `SegmenterBase` *is* the set test, so no existing
  segmenter needs editing; one that wants to refuse something the letters
  allow overrides it.

- **Reporting is post-hoc**, not pre-flight. Pre-flight needs image shapes
  without a full load, and the IO layer does not expose headers yet. Each skip
  is logged with its reason as the batch runs, so a long run is never silent,
  and the summary lands at the end. Pre-flight counts stay worth having and
  move to `OPEN.md` rather than blocking this.

A note for later: the `can_process` hook and the memory declaration in
scikit-ops [design 0017] are plausibly one declaration read two ways — axes
for whether an op applies, memory for whether it fits. Nothing here should
make merging them harder.

[design 0017]: ../../../scikit-ops/docs/design/0017-memory-and-tiled-processing.md

## Plan

Two commits. The rename is pure and lands first, so neither diff has to be
read through the other.

**1. Rename the thread wrapper.** `SliceProcessorThread` → `ProcessorThread`,
`_SliceWorker` → `_ProcessorWorker`, and the six call sites and two docstrings
listed above. No behaviour change.

**2. The sequence batcher.**

- `Segmenters/SegmenterBase.py` — add
  `can_process(axis_types, selected_axis) -> (bool, str)`, returning the set
  test (requested axes present, leftover axes ones this configuration knows
  how to iterate) and a reason string on failure. Inherited by every
  segmenter; none are edited.
- `utilities/slice_processor.py` — add `SequenceProcessor` as sketched above.
  Per image: `load_image(i)`, `can_process`, then either skip-and-log or build
  a `SliceProcessor` and run it. It owns `save_predictions`. A `finally`
  reloads the starting index so the model matches the viewer. Returns a
  summary of processed and skipped counts with reasons.
- `models/image_data_model.py` — add `segment_sequence(...)` mirroring
  `segment_range` (`:2441`).
- `apps/nd_easy_segment.py` — drop `save_predictions` from
  `_on_segment_slice_done`; the batcher does it. Check the currently viewed
  image up front and fail before starting if it cannot run. Show the summary
  when the batch finishes.

Two things the nesting changes, both easy to get wrong:

- **`start_index` / `end_index` mean image indices** in the outer processor,
  not flat slice indices. Stacked mode passes them today, so check
  `nd_easy_segment` is not feeding slice indices into a sequence run.
- **`on_progress` must report a flat count across the batch**, or the
  progress bar resets on every image.

Untouched: stacked mode, which runs as a sequence of one; every individual
segmenter; and the layer wiring in `nd_ai_lab.py`.

## Built differently than planned

- `can_process` answers only "are my axes here". Whether the slicer can build
  the iteration is `unsupported_iteration` in `slice_processor.py`, kept apart
  so a temporary limit does not become a rule on every segmenter.
- A leading channel axis iterates fine; a trailing one cannot, because the
  slicer iterates leading axes. So `CYX` works and `YXC` needs `C` collapsed.
  See OPEN.md, "Segmenting channels separately".
- `_on_segment_slice_done` was split rather than stripped -- single-slice mode
  shares it. The sequence path connects only the update half.
- `SequenceProcessor` takes `segmenter` and `current_index` as well as
  `source`, and lives in its own `sequence_processor.py`.
- `ProcessorThread` moved to `processor_thread.py` and now publishes the
  QThread's `finished`, not the worker's.

## What this is not

Not a change to what a segmenter does to one image, and not a rework of
stacked mode. Stacked keeps working through the same path as everything else,
as a sequence of one. Removing it, if that ever happens, is 0005's business.
