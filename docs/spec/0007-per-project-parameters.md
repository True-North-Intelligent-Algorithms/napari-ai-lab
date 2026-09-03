# 0007 — Parameters that belong to a project

**Status:** proposed. Nothing built.

A project remembers its images, its annotations, its patches and its models. It
does not remember the settings that produced any of them. Every parameter in
the app resets to a hardcoded default on restart, and nothing says so.

## The failure this is written after

Patch size was set to 256 during a session of experiments on the bee project.
The session ended. On restart the field read 128 again — `nd_easy_augment.py`
does `patch_size_xy_spinbox.setValue(128)` and nothing overrides it — and
patches were regenerated at that size without anyone noticing.

The largest object in that project is a bee **304 px** long. In a 128 px patch
the biggest thing that can exist is about 150. So every training patch showed
bees at half size or less, the model learned that bees are small, and on the
full image it drew a scatter of small bees across each big one. The more it
trained, the more confident it became.

Hours went into the model instead: receptive field, architecture depth, grid
size, scale augmentation, the training-data size distribution. All of it was
sound work on the wrong question. The cause was a spinbox that forgot.

Three properties made it expensive, and they are what this spec is really
about:

- **Silent.** No error, no warning, no visible difference. Patches at 128
  look exactly like patches at 256 until you measure them.
- **Plausible.** Every downstream symptom had a believable explanation that
  was not the real one, and each took time to rule out.
- **Repeatable.** It will happen again to anyone who changes a setting,
  restarts, and reasonably assumes the project kept it.

## What needs to persist

Not one kind of thing. Three, and they live in different places today, which
is why there is no single obvious fix.

**Widget defaults hardcoded in app code.** Patch size, number of patches, the
Z patch size. These are `setValue(...)` calls in `nd_easy_augment.py` and
friends. They are not fields on any object and nothing reads or writes them
outside the widget. This is the category that caused the failure.

**Dataclass fields on augmenters and segmenters.** Everything carrying
`param_type` metadata: `num_epochs`, `prob_thresh`, `min_long_axis`,
`unet_n_depth`. These at least exist as attributes, but they live on an
instance that is constructed fresh each session.

**Selections.** Which segmenter, which model, which patch directory, which
axis, which annotation collection was active. These decide what the other two
categories even apply to, and a restored parameter set attached to the wrong
segmenter is worse than no restoration.

## Questions to settle before building

**Where does it live?** A project is a directory of images plus `annotations/`,
`labels/`, `patches/`, `models/`. A `settings.json` beside those is the obvious
answer and matches how `info.json` already records patch metadata. It should be
readable and hand-editable, because the first thing anyone will do when it
misbehaves is open it.

**Per class or flat?** Parameters belong to a specific segmenter or augmenter,
and two segmenters both have `num_epochs` meaning different things. Namespacing
by class name — `{"AlbumentationsAugmenter": {...}}` — is the smaller decision
to make now and hard to add later.

**When is it written?** On every change, or on an explicit Save Project? Saving
on change is what people expect from a settings panel and is what would have
prevented this. Saving on demand is more predictable but relies on a habit, and
the failure above is precisely a habit not happening.

**What happens to parameters that no longer exist?** Today `size_factor` was
replaced by `min_long_axis` and `max_long_axis`. A settings file written last
week names a field that is gone. Unknown keys must be ignored rather than
raising, and missing keys must fall back to the field default — otherwise every
refactor breaks every existing project.

**Which settings are genuinely global?** Window geometry, the last project
opened, perhaps a preferred segmenter. These do not belong in a project
directory. The split should be decided once rather than case by case, and the
default should be per-project — a setting in the wrong place is a setting that
silently applies to work it was not chosen for.

**Does a project record what produced its artifacts?** Separate from restoring
the UI: a `patches/` directory could record the settings that generated it, and
a model could record the settings it was trained with. That is a stronger
guarantee than remembering the last value, and it is what would let the app say
"these patches were made at 128 px, your setting is now 512" instead of leaving
you to measure the tifs. `info.json` already does a little of this, and
`history.csv` in a model directory does more.

## Deliberately not in scope

Restoring parameters is not the same as validating them. A patch size of 128
against a 304 px object is wrong whether or not it was remembered, and a
separate check should say so at generation time — the same shape as the
receptive-field check. Persistence removes the silent reset; it does not remove
the need to notice a bad value.

## Why this is urgent

There is a workshop in a month. Attendees will change a setting, restart, and
get different results with no indication why — and unlike us, they will not
have spent the morning learning to suspect it. A tool that quietly forgets what
you told it teaches people not to trust it, and they are right.
