# Open items

Things known to be wrong or undecided, too small or too unresolved to deserve
their own numbered spec. One heading each, newest at the top.

**Status** is one of:

- `open` — known, not yet decided what to do.
- `decided: <what>` — the call has been made, not yet built.
- `done` — built. Move the item to the *Resolved* section at the bottom with
  a one-line note on what happened, rather than deleting it: the next person
  to hit the same symptom should find the answer, not the question.

Ask "what else to do" in any session and this file is the answer.

---

## Starting empty and choosing a directory from the GUI

**Status:** decided: this is the user's path — needs testing, soon.

`launch_nd_ai_lab.py` picks its data by an `if/elif` chain over a hardcoded
`test_set` variable (lines 127-237), one branch per folder under
`tests/test_images/`. That is the developer harness and it stays one: a
dataset is chosen by editing the file, which nobody but us can do.

The user's path is the opposite — the lab opens with nothing loaded and the
user points it at their own directory. That already exists but has not been
exercised much, and it is the thing a new user meets first, so it is the thing
most worth getting right. Command-line selection of a built-in set stays
alongside it for our own use.

---

## Users need pixi on the command line, and appose's copy is not it

**Status:** open — a documentation gap, not a bug.

Appose downloads and manages its own pixi at
`~/.local/share/appose/.pixi/bin/pixi` (`appose/tool/pixi.py`, invoked from
`PixiBuilder.build`). It covers all six platforms, it is never put on PATH,
and it is enough to build every op environment without the user having pixi.

It cannot bootstrap the lab itself. `pixi run lab`
(`pixi/pytorch_napari/pixi.toml:158`) has to run before any Python exists, so
the user needs their own pixi — `winget install prefix-dev.pixi` on Windows,
the pixi.sh installer elsewhere. The two copies share `~/.cache/rattler`, so
having both costs one 58 MB binary and nothing more.

What is missing is saying so. A user who installs napari-ai-lab, wants pixi at
a terminal, and finds nothing there has no way to learn that a pixi already
came with it, or that installing a second one is harmless. Install docs should
say both.

Separately, there is no `[project.scripts]` entry point, and
`launch_nd_ai_lab.py` is a flat script with no `main()`, so a pip install
leaves nothing to type. Only worth doing if a pip route is wanted alongside
the pixi one.

---

## Segmenting channels separately

**Status:** open — a real use case the slicing cannot express.

`SliceProcessor` iterates *leading* axes. A channel axis is trailing, so YXC
asked to iterate loops over Y instead — 780 "slices" of nonsense. The only way
through is to collapse C, and `can_process` refuses the alternative rather
than let it happen quietly.

That refusal is a limitation, not a judgement about channels. Three distinct
channels — nuclei, mito, CY3 — are often exactly what one wants segmented
separately, and the collapse is right sometimes and wrong other times. Nothing
today lets the user say which.

Fixing it means either moving the iterated axis to the front before slicing,
or teaching `SliceProcessor` to iterate an arbitrary axis rather than assuming
the leading ones.

---

## Cellpose reloads its model once per image

**Status:** open — performance only, found running spec 0006's batcher.

`Using CellposeSAM model: cpsam` prints once per image in a sequence run: ten
images, ten initialisations. VRAM is flat across them (2.77 GB reserved), so
nothing leaks, but the load is most of the wall clock on small images.

A batch has one segmenter instance throughout, so the model could be loaded
once. Whether that belongs in `CellposeSegmenter` or in a general
"segmenter is about to see many images" hook is undecided — the same question
applies to every model-backed segmenter.

---

## Which pollen images suit instance segmentation

**Status:** open — the decision exists only as folder membership.

`tests/test_images/pollen_count` and `pollen_morphology` are a 10/10 split of
the 20-image pollen set, made in an earlier session: many grains per field
versus one to three large ones. The criterion is written down nowhere.
`pollen SOURCES.md` covers provenance and licences and does not mention the
split, and `tests/test_images/` is gitignored, so there is no commit history
to recover it from either.

If the folders are ever lost the judgement goes with them. `pollen SOURCES.md`
lives outside the project folders and is the natural place to record it.

## Reading image shapes without a full load

**Status:** open — deferred out of spec 0006, not a blocker.

Spec 0006 wanted a pre-flight pass telling the user "84 of 100 will be
processed, 16 skipped" before a batch starts, rather than discovering it in a
summary twenty minutes later. It settled for post-hoc reporting because the
count needs every image's `axis_types`, and getting those today means
`load_image` on all of them — which is the batch itself.

Both tiff and czi headers carry shape and axes; the IO layer does not expose a
way to read them without the pixels. If it did, pre-flight is cheap and the
decision in 0006 flips.

Wider than 0006: anything wanting to describe a folder before working through
it has the same problem.

---

## Annotations are a collection; labels and patches are still singular

**Status:** open — deliberately deferred. The immediate bug is fixed; the
generalisation is not.

`annotations/` became a collection — one `annotations/<layer name>/` directory
per labels layer, so several classes can be annotated separately. Everything
downstream stayed singular:

```
annotations/<name>/          a collection, one per labels layer
labels/input0, labels/truth0 one pair, whichever collection was active
patches/patches_axis_*/      one set, from that one pair
```

So `truth0` holds whichever collection the active-layer combo happened to
select when Save Project or Augment ran. Switching the combo and re-running
silently rewrites it with a different class. `labels/info.json` now records
which collection was used, which makes that visible after the fact but does not
make it correct.

The generalisation is that `labels/` and `patches/` become per-collection too,
and it reaches further than it first looks. Consumers of the `input0` /
`ground_truth0` convention:

- `generate_patches_from_labels` and both crop methods in `image_data_model.py`
- `dl_util.py:102` (`load_patches`), `io_util.py:37`
- `nd_easy_augment.py:500` (the "show patches" viewer)
- training in `MonaiUNetSegmenter`, `MonaiUNetSegmenter3D`, `MicroSamSegmenter`
- `TrainingBase.py`, which already documents `ground_truth0, ground_truth1, …`
  as a numbered series — so the convention anticipated this and nothing uses it

Worth deciding before building: whether a collection maps to a **numbered**
truth directory beside one input set (`truth0`, `truth1`, sharing `input0` —
which is what the numbering above was for, and is the multi-class training
shape), or to a **separate patch tree** per collection. The first is smaller
and matches what training frameworks expect; the second generalises to
collections that do not share crop geometry.

Not urgent while there is one collection. It becomes urgent the day there are
two, and the symptom will be silently mixed-up training data rather than an
error.

Prompted by the empty-truth bug: writes had moved to `subdirectory=<layer
name>` while two readers still defaulted to `class_0`, so `truth0` was cropped
from an empty `annotations/class_0/`. Fixed by threading the active layer's
name through; `annotations/class_0/` is left on disk here and is stale.

---

## Does appose still report environment-build progress, and under what name?

**Status:** open — found incidentally, not chased.

`pixi/pytorch_napari/pixi.toml` took appose from a sibling checkout rather than
PyPI because `PixiInstallMonitor` was in no release, and without it the first
run of an op is a silent multi-minute hang while an environment builds.

That symbol is no longer in the checkout either. It landed in appose 3e97f55
(2026-06-24), which *is* an ancestor of the checkout's HEAD, so it was removed
or renamed upstream some time after. So the stated reason for preferring a
checkout no longer holds, and nobody has checked what replaced it.

Three things to find out, in order:

1. What happened to `PixiInstallMonitor` — renamed, folded into `Service`, or
   dropped. `git log -S PixiInstallMonitor` in `../appose-python` answers it.
2. Whether build progress reaches the user today at all. The symptom to look
   for is the original one: trigger a first-run environment build from the
   segmenter list and see whether anything appears before it finishes.
3. Whether the checkout is still needed, or PyPI would now do.

Not urgent — nothing regressed, this was always the state. It surfaced because
adding albumentations forced a full re-resolve, which is also how the appose
source came to be declared explicitly in that file; the comment there has the
detail.

---

## Stacked mode writes boxes.csv in a different format than sequence mode

**Status:** open — fix or deprecate, undecided.

Stacked mode is a *view* over the same images, so it should not write the
underlying project data differently. It does.

Same two boxes, same project, written by the two modes:

```
file_name,xstart,ystart,xend,yend,m3pos
cell_00176.png,388,967,1012,1475,        <- sequence mode
cell_00173.png,526,846,1497,1774,        <- sequence mode
cell_00176.tif,388,967,1012,1475,4       <- stacked mode
cell_00173.tif,526,846,1497,1774,1       <- stacked mode
```

Two divergences, not one:

1. **File name.** Stacked mode names the row after the `.tif` it built the
   stack from; sequence mode names it after the `.png` it loaded directly.
   The same image ends up in `boxes.csv` twice under two names, so a
   whole-project pass such as `crop_and_save_all_label_patches` treats it as
   two images and crops it twice.
2. **`m3pos`.** Stacked mode writes the image's index within the stack into a
   middle-position column. Sequence mode has no such axis and leaves the cell
   empty. `load_existing_boxes` (`image_data_model.py:1288`) assumes every row
   has a value once the header carries the column, so a mixed file raises
   `ValueError: could not convert string to float: ''`.

The second one is a crash and would need handling whichever way this goes; the
first is the actual design question.

**The two ways out.**

*Fix.* Stacked mode writes rows in the same shape sequence mode does — named
after the source image, with middle positions describing axes the image
genuinely has, not the stack's N. The N index is a property of the view and
belongs nowhere in `boxes.csv`. Stacked mode then keeps earning its place as a
test that the indexing works for general ND (`NYXC`) and not just `N` separate
`YXC`.

*Deprecate.* If sequence mode is fast enough for the datasets that matter,
stacked mode is a second code path carrying complex indexing for no user-facing
gain, and removing it deletes this problem along with a good deal else.

Undecided because the value of stacked mode is not really about this dataset —
it is about whether the ND indexing gets exercised anywhere. Decide that first.

Reproduce with the same images through both paths:

```sh
cd pixi/pytorch_napari
pixi run lab-sequence
pixi run lab-stacked
```

---

## Two StarDist segmenters, and duplicated model-map code

**Status:** open — deliberate, deferred until the skop one can train.

`StardistSkopSegmenter` carries its own copy of `BUILTIN_MODEL_MAP`,
`build_pretrained_model_map`, `get_custom_model_from_path` and
`get_model_axis_map` rather than sharing them with `StardistSegmenter`.
Around 80 lines of `os.listdir` plus `json.load` on `config.json`, duplicated
on purpose so the commit adding the skop segmenter does not also refactor the
working one.

The real question underneath is whether `StardistSkopSegmenter` eventually
*replaces* `StardistSegmenter`. It cannot yet — the direct one trains and the
skop one does not, and inference through skop needs `stardist2d_custom` in
scikit-ops before a user-trained model is reachable at all.

- If the skop one replaces it, the duplicate leaves with the old class and no
  refactor is needed.
- If both stay, the four move to a `StardistModelMap` mixin in `mixins/`,
  matching `TrainingBase`.

Decide once the skop segmenter can train. Until then the risk is a scan fix
landing in one copy and not the other.

## Resolved

## scikit-ops is a copy, not an editable install, in pytorch_napari

**Status:** done — the environment no longer depends on skop-napari.

Both declarations said `editable = true`; uv drops that flag when applying a
path dependency's sources transitively, so the transitive one arrived
non-editable and collided with the direct one. Declaring it in only one place
was the fix, and the choice of place was made for us: nothing in napari-ai-lab
has imported skop-napari since 7b36e20 removed the generated parameter form,
so it left the environment and took the second declaration with it.

The `pip install -e` workaround this item used to prescribe is gone with it,
and would not have worked on a rebuilt environment anyway: that environment
has no pip.
