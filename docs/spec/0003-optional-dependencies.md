# 0003 — Optional dependencies and the minimal install

**Status:** proposed. The pattern below has been applied to two modules
(`artifact_io/zarr_artifact_io.py`, `Augmenters/albumentations_augmenter.py`)
as a stopgap. Nothing else here is built.

## The problem, concretely

`v0.1.0a1` cannot be imported without **zarr** *and* **albumentations**, neither
of which is a declared dependency. So `pip install napari-ai-lab` produces a
package that fails on `import napari_ai_lab`. It works in the model-stack
environments only because micro-sam and friends happen to pull those in.

This was not one bug. It is an unmade decision showing up in a different place
each time a smaller environment is tried, and the repo currently holds **three
different answers** to the same question:

| Package | Pattern today | Behaviour when the dependency is missing |
| --- | --- | --- |
| `Segmenters/*` | `__getattr__` + `_try_import_segmenter` | Returns `None`, prints a warning. Caller must check |
| `artifact_io` | try/except at import, raise in `__init__` | Package imports; construction raises with install instructions |
| `Augmenters` | eager `import albumentations` | **Fatal at import** — took down the whole package |

Every caller pays for this. `launch_nd_ai_lab.py` carries thirty lines of
`if CellposeSegmenter is not None:` purely to service the first row.

## The rule

**Import must never fail. Construction may.**

Concretely, for every optional dependency:

```python
try:
    import zarr
except ImportError:          # optional dependency
    zarr = None
```

and at the point of use — `__init__`, or `__post_init__` for a dataclass:

```python
if zarr is None:
    raise ImportError(
        "Zarr artifact I/O requires zarr, which is not installed. "
        "Install it with: pip install zarr"
    )
```

**The guard alone is not enough if the package appears in a type annotation.**
Annotations are evaluated when the `def` runs — that is, while the class body
is being built at import time — so a return type of `-> A.Compose` is
`None.Compose` on a machine without albumentations, and the module dies at
import despite the try/except. This is what actually happened in
`albumentations_augmenter.py`, and the traceback is confusing because it points
at a `def` line inside the class body, which reads like a call.

The fix is one line at the top of any module that names an optional dependency
in an annotation:

```python
from __future__ import annotations
```

That makes every annotation in the file a string, evaluated only if something
asks for it (`typing.get_type_hints`), which nothing here does. It must be the
first statement in the file — comments may precede it, code may not. Prefer it
over quoting the one annotation (`-> "A.Compose"`), since it also protects any
annotation added later.

So the pattern is three parts: **guard the import, defer the annotations, raise
at construction.**

Two properties this has that the `None`-returning approach does not:

- **The error is actionable.** It names the package and the command. A caller
  who never touches the feature never sees it.
- **No caller-side checks.** Nothing has to test `is not None` before use, so
  the thirty-line registration block in the launcher can eventually go.

The segmenter packages should converge on this. That is a bigger change than
the two modules already converted, because callers depend on the `None`
behaviour today — hence a spec rather than a commit.

### Laziness is a separate question

A package `__init__.py` that eagerly imports its concrete implementations —
`Augmenters/__init__.py` imports `AlbumentationsAugmenter` on line 1 — looks
like part of this problem, and is not. Once a module is guarded as above, it
imports cleanly whether or not its dependency is present, so an eager
`__init__` is safe.

The remaining reason to import lazily is **cost, not failure**. That is why
`GlobalSegmenters/__init__.py` uses `__getattr__`: its modules pull in torch
and tensorflow, seconds and gigabytes. Albumentations is a few hundred
milliseconds, so `Augmenters/__init__.py` should stay eager and keep the
convenience of `from napari_ai_lab.Augmenters import AlbumentationsAugmenter`.

Decide laziness on measured import time, not on principle.

## Tiers

The point of the rule is to make a promise that can be tested.

| Tier | Needs | What must work |
| --- | --- | --- |
| **Minimal** | numpy, magicgui, qtpy, scikit-image, napari | Load a project, view a sequence, annotate, edit masks, threshold/Otsu/watershed segmenters, tiff and numpy I/O |
| **Plus formats** | `[zarr]` | Zarr artifact I/O |
| **Plus augmentation** | `[albumentations]` | AlbumentationsAugmenter. `SimpleAugmenter` is numpy-only and belongs in Minimal |
| **Plus models** | `[cellpose]`, `[stardist]`, `[sam]`, `[monai]` | The corresponding segmenters, training, and the interactive SAM tools |

The minimal tier is the promise: **a laptop with no GPU can annotate and
threshold.** It is also the tier most likely to break silently, because none of
the development environments resemble it.

## Extras

Once the guards exist, `[project.optional-dependencies]` is mechanical:

```toml
zarr = ["zarr"]
albumentations = ["albumentations"]
cellpose = ["cellpose"]
# … and an `all` that includes them
```

Note this is a convenience, not the mechanism. The try/except is what makes a
dependency optional; the extra just gives it a name to install by.

## How the UI should degrade

Unsettled, and it matters more than the plumbing.

A feature that is missing because a dependency is absent should say so, not
vanish. If Cellpose is silently absent from the segmenter list, a user
concludes the app cannot do Cellpose. If it is listed and disabled with
"requires cellpose — pip install napari-ai-lab[cellpose]", they know what to do.

Against that: a long list of things you cannot use is its own kind of bad. A
reasonable compromise is to show unavailable items grouped and dimmed rather
than interleaved, but this needs to be tried rather than argued.

One thing is already decided, in `artifact_io/__init__.py`: `"zarr"` stays in
`AVAILABLE_ARTIFACT_IO` even when zarr is missing, because that registry is the
error path as well as the menu. Removing it would turn a helpful ImportError
into `ValueError: Artifact I/O 'zarr' not available`, which tells the user the
format does not exist rather than that it is not installed.

## How this gets verified

**`pixi/sandbox`** is the minimal-tier environment: python, napari, PyQt5,
napari-ai-lab, nothing else. It found both the zarr and albumentations failures
within minutes of first being built, which is the argument for keeping it.

What would make it a real check rather than a lucky canary:

1. A smoke test that imports every public module in a minimal environment. Most
   of these failures are import-time, so most would be caught by import alone.
2. Running the existing test suite there, to see what assumes a model stack.
3. CI on the minimal environment. Today nothing tests this configuration, which
   is why a broken release was publishable.

## Consequences

- **`v0.1.0a2` is needed before anyone can `pip install` this.** The two
  stopgap fixes may be enough for import; a minimal-environment smoke test
  would tell us.
- The article's environments are unaffected — they have every optional
  dependency — but a reader who tries `pip install napari-ai-lab==0.1.0a1`
  outside pixi hits this immediately.

## Open questions

- **Do the segmenters convert to raise-on-construction, or stay `None`?**
  Converting is better and touches every launcher script.
- **Is `napari` itself optional?** The runtime dependencies today do not include
  it (only `[all]` pulls `napari[all]`), yet `apps/` imports it unconditionally.
  So a "headless" install is already nominally possible and certainly broken.
  Either commit to headless use or make napari a hard dependency.
- **Where does the tier promise get written down for users?** README, or a
  table in the docs. It is worth stating publicly, because it is the thing that
  makes the minimal case feel deliberate rather than accidental.
