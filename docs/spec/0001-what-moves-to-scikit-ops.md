# 0001 — What moves to scikit-ops, and what stays

**Status:** proposed. Nothing here has been built. No code has moved.

This is the first cut of the split between `napari-ai-lab`, [`scikit-ops`]
(the `skop` package) and [`skop-napari`]. It is deliberately coarse. Expect it
to grow several times its size as each row turns out to have its own
complications.

Sibling checkouts, for now:

```
../scikit-ops     src/skop/          op-independent machinery + ops
../skop-napari    src/skop_napari/   the napari front end for those ops
```

## The rule

The split is by **lifetime**, not by "is it AI".

| Lifetime | Owner |
| --- | --- |
| One call — image in, arrays out | `scikit-ops` |
| One session — an embedding, a warm model, a prompt loop | **nobody yet.** See "No home yet" |
| One project — which images, which labels, which patches, what is on disk | `napari-ai-lab` |

Corollary, and the thing to guard: **scikit-ops must never learn what a project
is.** Every addition to it should be justifiable to someone who has never heard
of napari-ai-lab. A path/artifact role passes that test. A `TrainingProject`
type would not.

## Replace

These have a better-tested twin next door already.

| Part of ai-lab | Files | Replaced by | Note |
| --- | --- | --- | --- |
| Global segmenters | `Segmenters/GlobalSegmenters/` — Cellpose (771), Stardist (888), MicroSam (749), Otsu, Threshold, Watershed | `skop.ops.segment.*`, `skop.ops.mask.*`, `skop.ops.threshold` | Same algorithms, already written as `@op` functions |
| Remote-environment machinery | `Segmenters/execute_appose.py`, `apps/remote_env_dialog.py`, `~/.napari_ai_lab/environments.json` | skop named environments — `envs/<id>/pixi.toml`, `worker.py`, `runner.py` | **Do this one first.** Most duplicated, and skop's version is the one with tests |
| Hand-built parameter forms | `widgets/nd_operation_widget.py` (854) — `_create_int_widget`, `_create_float_widget`, `_create_choice_widget`, … | `skop_napari/_widget.py` | It takes `annotation_for` as a parameter precisely so a non-napari host can drive it |
| Parameter declaration | per-segmenter dataclass fields, `param_type: "inference"`, `get_parameters_dict()` | `Annotated[float, {"widget_type": ..., "min": ...}]` in the op signature | The signature *is* the declaration |
| Axis selection | axis combo + `axis_map` in `nd_operation_widget.py` | `skop.Axes(...)`, `skop_napari/_axes.py`, `_plans.py` | See scikit-ops design 0006 |
| Progress and cancel | `utilities/progress_logger.py`, `utilities/qt_progress_logger.py`, `utilities/training_thread.py` | `skop.progress()`, `skop.cancel_requested()`, `skop_napari/_run.py` | |

Roughly 3–4k lines.

## Stays

| Part of ai-lab | Files | Why |
| --- | --- | --- |
| ND project / data model | `models/image_data_model.py` (2385), `models/local_image_data_model.py` | Project lifetime. skop is stateless-function-shaped |
| Artifact I/O | `artifact_io/` — tiff, zarr, stacked sequence, tiff slice | Same |
| Label editing | `apps/nd_easy_label.py` (2289), `apps/edit_masks.py` | Stateful, interactive, project-aware |
| The ND apps | `apps/base_nd_app.py` (764), `nd_ai_lab.py`, `nd_easy_segment.py`, `nd_easy_augment.py` | The UX is the product here |
| ND slicing / looping | `utilities/slice_processor.py`, per-plane iteration over a project | `Axes` tells an op what a *plane* looks like. The loop over the project is ours. **Do not push this down** |
| Training and augmentation | `Augmenters/`, `datasets/`, `mixins/TrainingBase.py`, `widgets/train_dialog.py` | Orchestrator role — see below |
| Vendored bbox layer | `vendored/napari_bbox/` | Unrelated to any of this |

## Split down the middle

**Interactive segmenters** (`Segmenters/InteractiveSegmenters/`) are two
families wearing one name.

| Segmenter | Shape | Verdict |
| --- | --- | --- |
| `Otsu2D`, `Square2D`, `RegionGrow3D`, `FeatureRegionGrow3D`, `HoughSphereFit3D`, `AnisotropicSphereFit3D` | `(image, prompt) → mask`, no expensive state | **Movable today.** skop's `Role` enum already has `points` and `shapes`, so the prompt is expressible with no new machinery |
| `SAM3D`, `SAMSphere3D`, microSAM prompting | embed once, then many cheap prompt→mask calls | Needs the session concept. See below |

So the problem is not "interactive does not fit". It is "embedding-backed does
not fit", which is a much smaller problem.

## No home yet

Two gaps, in the order I would close them.

### 1. A path / artifact role — cheap, high leverage

An op today takes arrays and returns arrays. Several things we need are
neither:

- run Cellpose or Stardist with **my own trained model** (a file or directory)
- training's **input** is a dataset on disk, not one array
- training's **output** is a model artifact, not an array

One role meaning "the value that crosses the wire is a filesystem location, not
data" covers all three. It unblocks custom-model inference immediately — which
we need regardless of training — and is a prerequisite for the rest. Cheapest
useful thing on this list.

### 2. Worker-side session handles — real design work

For the embedding-backed segmenters. Two candidate shapes:

1. **Handle.** `embed(image) -> Session`, where `Session` is opaque and never
   serializes, then `predict(session, points) -> LabelsData`. skop already has
   `exclusive=True` for a sticky worker, so the open questions are the role
   itself, who frees the handle, and what happens when a worker dies.
2. **Stateful op object**, with setup/call.

Preference is (1). Shape (2) breaks the thesis in the scikit-ops README — "an
op is an ordinary Python function" — and once one op is a class, every front
end has to learn two shapes.

### Training

Progress and cancel are already solved. Duration is not really skop's problem —
a worker running for two hours is fine; surviving a restart is ours.

Training stays in ai-lab as the **orchestrator** — it knows the project, picks
the images, builds the patches — and calls down for the pieces (augment,
normalize, tile). Eventually the train loop itself can be one long op: paths
in, path out. That needs gap 1 first.

See also scikit-ops `docs/design/0011-deep-learning-training-ops.md`.

## How to actually do it

1. **Adapter, not rewrite.** Keep `GlobalSegmenterBase` as a thin shim whose
   `segment()` body becomes a skop call. The ND apps keep calling
   `segmenter.segment(image)` and never notice. One segmenter per commit
   instead of one large frightening PR.
2. **`execute_appose.py` first.** Once one segmenter round-trips through
   skop's environment keying, the registry and the dialog simply delete.
3. **Then the widget.** Swap the `_create_*_widget` guts for
   `skop_napari/_widget.py`; keep the ai-lab-specific rows (remote env,
   patches combo, model combo) as chrome around the generated form.
4. **Leave interactive and training alone** until the two gaps above are
   settled.

## Open questions

- **Dependency direction.** Hard dependency on `scikit-ops` (headless, no
  napari) seems clear. `skop-napari` is less clear — leaning
  optional/loose, since the ND apps have their own UX and should not be
  routed through the generic Ops panel. Two plugins both registering dock
  widgets is fine, but worth deciding on purpose.
- **Naming.** `Segmenters/GlobalSegmenters/` is not PEP 8. If directories are
  moving anyway, that is the cheap moment.
- What happens to `Segmenters/SegmenterBase.py`'s registry once ops are
  discovered by `skop.discovery` instead?

[`scikit-ops`]: ../../../scikit-ops
[`skop-napari`]: ../../../skop-napari
