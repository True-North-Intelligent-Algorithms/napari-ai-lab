# 0005 — Cleaning up ND AI Lab

**Status:** proposed. One symptom fixed (the pushed `current_image_index`);
nothing else here is built.

Not a rewrite. A list of the structural problems that keep producing bugs,
with the smallest change that would stop each — written down now because the
same shapes keep recurring, and because the migration to scikit-ops
([0001](0001-what-moves-to-scikit-ops.md)) will be easier against a smaller
surface.

## The bug that prompted this

Segmenting the second image in a sequence overwrote the first image's saved
prediction. The cause was one line that did not exist: nothing set
`current_image_index` on the sub-widgets.

Every sub-widget inherits `current_image_index = 0` from `BaseNDApp`, and each
uses it to decide which image to save to and load from. It is assigned in
`BaseNDApp`'s own image-change handler — which the combined app does not call,
because `NDAILab` distributes layers by direct attribute assignment instead.
So the copies stayed at 0 for the whole session while the app moved through
the sequence.

The display symptom (seeing the previous image's segmentation) was the
harmless half. The saving symptom was data loss.

It is fixed by pushing the index in `_distribute_layers_to_sub_apps`. That is
a patch, not a cure: nothing prevents the next piece of duplicated state from
doing the same thing.

## The pattern: state copied rather than read

`NDAILab` owns state and hands copies to four sub-widgets. Layers, the model,
the axes, the index. Every copy is a chance for one holder to move on while
another does not.

**The cure is to stop copying.** One owner per piece of state, and sub-widgets
read it rather than holding it. `current_image_index` belongs to the model —
it is a property of the project, not of a tab — and `ImageDataModel` is
already passed to everyone.

Worth checking first whether `ImageDataModel` already tracks a current index;
if it does, this is deletion rather than addition.

The same argument applies to `predictions_layers`, which is shared by
reference in one direction (`self.predictions_layers = self.segment_widget.predictions_layers`)
and rebound in the other (`_cleanup_layers` sets it to `{}`). It currently
re-syncs by accident on the next distribution. That is not a bug today and
will be one eventually.

## The model is a sequence; `self.image_data` is the exception

`ImageDataModel` holds no per-sequence-number state, and that is right. The
view knows which image is current and tells the model "do this at index 8":
`load_image(index)`, `save_annotations(data, image_index, …)`,
`load_existing_predictions(image_index=…)`, `save_boxes(boxes, image_index)`.
Index to filename to disk, computed per call. Save on leave, load on arrive.

The exception is **`self.image_data`**, which caches the currently loaded
array. That is "which image am I looking at" — view state, living in the
model — and it is the direct source of the stacked leak, because
`self.image_data` is *either* one image *or* the whole stack depending on the
IO type. Anything reading it has to ask which, so the question spreads:
`load_existing_predictions` calls `io.set_shape_total(self.image_data.shape)`.

That splits the model's 57 mentions of "stacked" in two:

- **Stacked as an IO strategy** — the `stacked_sequence` artifact IO, writing
  one array rather than N files. Legitimate. Writer choices belong here.
- **Stacked as a view** — the `_is_stacked_sequence()` branches in
  `save_boxes`, `load_existing_boxes` and `get_boxes_for_image`. These exist
  only because in stacked mode the layer geometry carries the frame index,
  which is a fact about the viewer.

Removing `self.image_data` as ambient state — callers know their index and can
pass the shape they mean — takes most of the second group with it.

## Save-on-switch never ran at all *(fixed)*

`_process_image_change` guarded its save with:

```python
if (active_widget_name == "Label"
    and hasattr(self, "labels_layer")     # never true
    and self.annotations_layer ...):
```

`labels_layer` does not exist on `NDAILab` — a rename that missed three
string references — so the block never executed and **nothing was ever saved
on an image change**: not annotations, not boxes, not label patches. Work was
persisted only by the explicit save button, and by the nag on shutdown, which
is why the gap went unnoticed while everything was built in stacked mode,
where you never leave an image.

The tab test was a deduplication guard from when each sub-app saved its own
layers; picking the visible tab made exactly one of them win. `NDAILab` owns
the layers centrally now, so there is one thing to save and nothing to
deduplicate — and which tab is on screen should never decide whether work is
persisted. Both are gone.

Two other references to the same dead attribute, in `_cleanup_layers`, made a
fallback branch unreachable and a reference-clearing line a no-op. Also fixed.

## Two viewers, and the reason for each

`nd_sequence_viewer` switches per image; `nd_stacked_sequence_viewer` loads
everything into one padded array so scrolling is a napari slider move.

The stacked one exists because switching is slow —
`benchmarks/bench_sequence_switch.py` measures ~1s, of which ~0.8s is
`gc.collect` called inside napari's layer removal, once per layer, seven
layers per switch.

That cost is avoidable: reuse the layers and assign `.data` rather than
removing and re-adding them. Nothing about these layers is image-specific
except their data and scale.

**But stacking should not be deleted even if switching gets fast**, because
for genuinely ND data — a 7-D CZI — the stack *is* the data, and per-image
switching would be the workaround. The awkward parts of `pad_to_largest`
(padding, 8-bit coercion, RGB conversion, mixed 2D/3D) all come from the
other case: manufacturing an ND array out of a heterogeneous folder.

So the line to aim for:

| Data | Viewer |
| --- | --- |
| genuinely one ND array | stacked, with no coercion needed |
| folder of differing images | switching, once it is fast |

That keeps the mode that is right and lets the coercion code go, rather than
maintaining both indefinitely.

Note also that the coercions are a **view** concern — processing reads the
original files. The thing to protect is that the view stays derived and never
authoritative, and that the mapping back to file coordinates
(`_original_shapes`) stays correct.

### One dataset, two presentations

The two modes are not two kinds of data. A stacked folder is an ND dataset,
no different in kind from a 7-D CZI; a sequence is the same dataset shown one
image at a time. Anything saved should therefore be identical in both, and
read correctly in both.

The modes disagree about exactly one thing: **whether the sequence index is a
coordinate or a filter.**

| | Sequence index is | On load | On save |
| --- | --- | --- | --- |
| **Stacked** | a napari dimension | it becomes the leading coordinate, so napari shows the item on its own frame | read the leading coordinate to get the index |
| **Sequence** | our own slider, outside napari | rows for other images are skipped | the current index names the image |

Everything else — the file, its columns, its meaning — is shared. Where the
code branches on mode for any other reason, that is a bug waiting to happen.

**Boxes were the worked example.** `boxes.csv` is one project-wide file keyed
by `file_name`, and both halves of this rule had been commented out:
`save_boxes` rewrote the whole file rather than preserving other images' rows,
and `_load_existing_boxes` drew every row on every image. Together they made a
box drawn on one image appear on all of them and then be re-stamped under
another image's name, deleting the original row. Both are restored, and the
mode branch is now the single coordinate-vs-filter decision above.

The same rule should be applied wherever else the code asks which mode it is
in, and the writer asymmetry is worth stating separately: **one input may
produce many outputs.** A stacked dataset can be written back as a sequence of
files, writing only the steps that have data rather than a mostly-empty array
— the same tension `design/notes_02_23_2026.md` records between the sparse
writers (`TiffSliceIO`, `ZarrArtifactIO`) and the whole-array ones
(`TiffArtifactIO`, `NumpyArtifactIO`). A CZI has it too. It is a property of
the writer, not of the viewer.

## Scale, and where metadata should come from

`get_scale()` returns something, but only sometimes something real. `self.scale`
is derived inside `load_image` from `scale_by_axis` when the axis types line up,
and the comment there is honest about the rest: *"Axes without a known physical
scale (e.g. T, C, or any non-CZI image) get 1.0."* A second pass then replaces
the whole list with ones whenever the last dimension looks like RGB. So today
it is real for CZI and `1.0` for nearly everything else.

It is also computed as a **side effect of loading**, on `self.image_data` --
the ambient state described above. "What is the spacing of image 7" is
therefore not a question the model can answer without loading image 7.

Two consequences, one immediate:

- **Layer reuse does not update scale.** Assigning `.data` leaves the previous
  image's scale in place. Invisible when every scale is 1.0, wrong for a
  project mixing images with different spacing. Not worth patching in the
  reuse path, because the value it would copy is mostly not real.

- **The fix is upstream of all of this.** Reading images is not our job.
  napari has built-in readers and an IO plugin system, and formats like CZI
  come with readers that already parse spacing, units and axis order. The work
  is to **survey what is available and how reliably each one reports
  metadata**, then use those readers rather than growing our own. The model
  would then hold spacing that is only ever as good as the reader it came
  from -- which is the right dependency, and an honest one.

That survey is its own task. Nothing here should be patched around until it
happens; a scale that is quietly wrong is worse than one that is obviously 1.0.

## Two methods asking one question

`availability()` and `are_dependencies_available()` both exist, which is
transitional and should not stay.

Today the **bool is the source of truth** and the rich value is derived from
it, except in `SkopSegmenter`, which overrides it the other way round. That is
backwards. The end state is `availability()` as the only implementation, with
`are_dependencies_available()` a thin `bool(self.availability())` on the base
class until its callers move, and then gone.

It survives because it has real users: `ImageDataModel.segment` uses it to
choose between local execution and the `execute_appose` path, the banner falls
back to it, three interactive tests call it, and every hand-written segmenter
implements it. So flipping the direction touches every segmenter and belongs
in its own commit.

Note the appose call site disappears anyway: `execute_appose.py` is the
machinery scikit-ops replaces, and goes once ops cover what it does
([0001](0001-what-moves-to-scikit-ops.md) schedules it first). Nothing to do
now; the two methods coexist until then.

## Smaller things worth doing while nearby

- **Reading predictions changes where they get written.**
  `NDEasySegment._load_existing_prediction_layers` walks the prediction
  subdirectories and calls `set_current_segmenter_name(name)` before each read,
  rather than passing the `subdirectory` argument `load_existing_predictions`
  already accepts. That value is also the default target for
  `save_predictions`, so after the loop the model points at whichever
  directory came last. Harmless with one segmenter, a coin flip with several.
  Fixed in the reuse path; this one still has it.
- **`pad_to_largest` exists twice** — `utility.py:199` and
  `utilities/image_util.py:234`. One of them is dead or they have drifted;
  either way it should be one.
- **`NDEasySegment` overrides `_update_segmenter_parameter_form`** and, before
  0004, silently dropped a base-class change. Overrides that must call `super()`
  to stay correct are a trap; the base should offer a hook instead.
- **The training combo lists every segmenter**, including ones that cannot
  train (`_populate_segmenter_combo` adds all frameworks to
  `training_segmenter_combo`). It should filter on capability.
- **`Segmenters/GlobalSegmenters/` is not PEP 8** — noted in 0001 as the cheap
  moment to rename is when directories move anyway.

## What this is not

Not a rewrite, and not a plan to make the apps generic. The UX is the product
here ([0001](0001-what-moves-to-scikit-ops.md)); these are the parts that are
accidental rather than designed.

The ordering that makes sense: fix state ownership first, because it is where
the data-loss bugs live; then layer reuse, because it decides the viewer
question; then the rest as they are passed.
