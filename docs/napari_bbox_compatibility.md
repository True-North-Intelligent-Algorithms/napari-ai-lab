# Keeping the vendored 3D bounding box layer working across napari releases

napari has no 3D bounding box layer, so a plugin that needs one has to bring
its own. There is no third-party package tracking napari releases for this
either, so `src/napari_ai_lab/vendored/napari_bbox/` is a vendored copy that
has to be patched by hand whenever napari changes the layer API.

This file is the running record of that work. It exists for two reasons:

1. So the next person hitting the same break — quite possibly the same person,
   a year later — can see what changed and why.
2. As evidence for the napari team and for funders. "Third-party 3D bounding
   boxes cost N patches across M releases, each found only at runtime, by one
   person" is a much more useful argument than "this is hard to maintain". It
   also shows which API changes are expensive downstream, which is information
   the core team cannot easily collect: most people who hit this give up
   quietly rather than reporting it.

The goal here is **not** to maintain a general-purpose 3D bounding box layer.
It is to keep one plugin working, and to document the cost of doing so.

## The shape of the problem

A napari layer is two classes, and both have to track the API:

- the **model** — `BoundingBoxLayer(Layer)`
- the **visual** — `VispyBoundingBoxLayer(VispyBaseLayer)`

Both breaks recorded so far were **runtime-only**. Nothing static catches
them — not a type checker, not an import test, not `pip check`. The model
failure needs the layer to be constructed; the visual failure needs it to be
added to a viewer with a canvas. That is the core reason this cannot be
automated away, and the argument for a smoke test that actually builds a
viewer and adds a layer.

### How the version directories work

`_utils.parse_package_by_napari_version` lists the `napari_*` directories and
picks the **highest one whose version is <= the installed napari**. The
directories are not independent copies: each is a *delta* on the one below,
subclassing it and overriding only what changed. `napari_0_6_0/bounding_boxes.py`
is a single line re-exporting the 0.5.0 class.

```
napari_0_4_15   the full vendored copy, thousands of lines
napari_0_4_18   \
napari_0_5_0     >  deltas, each subclassing the previous
napari_0_6_0    /
napari_0_8_0    added here -- see the incidents below
```

**A fix belongs in a new directory for the napari version that required it**,
not in an older one. Two reasons: older directories stay byte-identical to what
was vendored, so they are provably untouched; and the diff for a release's
worth of breakage is one self-contained directory, which is what makes this
readable to anyone else.

The chains differ between the model and the visual, which is worth knowing
before assuming where a class comes from:

```
model:  0_8_0 -> 0_6_0 -> 0_5_0 -> 0_4_18 -> 0_4_15
visual: 0_8_0 -> 0_5_0 -> 0_4_15            (0_6_0 only re-exports;
                                             0_5_0 overrides __init__)
```

An earlier attempt at the 0.8 work patched `napari_0_4_15` in place. The model
fix was inherited and worked; the visual fix was silently dead, because
`napari_0_5_0` overrides that `__init__`. Both were reverted in favour of the
directory below.

### The wildcard import trap

`_bounding_boxes_key_bindings.py` has no `__all__` and imports
`BoundingBoxLayer` itself. So this, at the end of a version directory's
`__init__.py`:

```python
from .bounding_boxes import BoundingBoxLayer          # ours
from ._bounding_boxes_key_bindings import *           # silently overwrites it
```

re-exports the *older* class and clobbers the one just imported. It raises
nothing; the package imports cleanly and the wrong class is used. The symptom
is `BoundingBoxLayer.__module__` naming a directory you did not expect.

Put the wildcard import **first** so the explicit imports win. Worth checking
with:

```python
from napari_ai_lab.vendored.napari_bbox import BoundingBoxLayer
print(BoundingBoxLayer.__module__)
```

## Incidents

### napari 0.8.0 — `_get_layer_slicing_state` became abstract

**Symptom**

```
TypeError: Can't instantiate abstract class BoundingBoxLayer without an
implementation for abstract method '_get_layer_slicing_state'
```

**Cause.** napari moved layer slicing behind a state object. `Layer` gained a
new `@abstractmethod`, `_get_layer_slicing_state`, which returns a
`_LayerSlicingState`. Because it is abstract, adding it is a breaking change
for every existing subclass — an existing layer cannot be instantiated until
it implements the method.

**Fix.** `napari_0_8_0/bounding_boxes.py`, 10 lines: a
`_BoundingBoxSlicingState` routing `_set_view_slice` back to the layer's own
existing method — the same two-line adapter napari's `Shapes` uses — and
`_get_layer_slicing_state` returning it.

Because the directory only loads on napari >= 0.8, `_LayerSlicingState` can be
imported unconditionally. No version guards are needed anywhere in it, which
is the main practical benefit of putting the work in its own directory.

**Notes.** The traceback points at a `def` line inside the class body, which
reads like a call and is not; it is the annotation being evaluated. Worth
knowing when reading the next one of these.

### napari 0.8.0 — `VispyBaseLayer.__init__` gained `font_info`

**Symptom**

```
TypeError: VispyBoundingBoxLayer.__init__() got an unexpected keyword
argument 'font_info'
```

**Cause.** `VispyBaseLayer.__init__` now takes a required `font_info`
argument, and `_qt/qt_viewer.py` passes `font_info=self.canvas.font_info()`
when it constructs the visual for every layer. The vendored visual's
`__init__` predates that and does not accept it.

**Fix.** `napari_0_8_0/vispy_bounding_box_layer.py`. Accept `**kwargs` and
forward them to the base `__init__`, taking kwargs rather than naming
`font_info` so whatever the next release adds passes through without another
patch.

The subclass has to repeat the whole constructor body rather than calling
`super().__init__`, because the 0.5.0 version it inherits from builds the
visual and calls `VispyBaseLayer.__init__` itself, passing neither. Everything
after those two lines is copied unchanged.

### napari 0.8.0 — `ClippingPlanesMixin` requires `font_info` too

**Symptom**

```
TypeError: ClippingPlanesMixin.__init__() missing 1 required keyword-only
argument: 'font_info'
```

**Cause.** The same change, one level deeper. `font_info` has to reach not only
the vispy *layer* but the compound visual *node* it builds:
`ClippingPlanesMixin.__init__(self, *args, font_info: FontInfo, **kwargs)` is
keyword-only and required.

**Fix.** `napari_0_8_0/vispy_bounding_box_visual.py`, a subclass taking
`**kwargs` and forwarding them, plus `node = BoundingBoxVisual(**kwargs)` in
the layer above. napari's own `VispyShapesLayer` does exactly this — passes
`font_info` to both the visual and the base layer — so the shape of the fix is
copied from upstream rather than invented.

Three separate constructor sites for one added argument, each discovered by a
separate crash, is worth noting: the cost of an added required argument on a
base class is paid once per subclass *per level of the hierarchy*.

## For the napari conversation

Points worth carrying forward:

- All three breaks were **additive from napari's side** — a new abstract
  method, a new constructor argument — and none is additive from a subclass's
  point of view. Adding an `@abstractmethod`, or a required argument to a base
  `__init__`, is a breaking change for every downstream layer.
- One added argument (`font_info`) required **three** separate fixes, because
  it has to be threaded through every level of the hierarchy that constructs
  something. The cost of such a change scales with the depth of downstream
  class hierarchies, which is invisible from inside napari.
- None of it is detectable without running a GUI. A downstream maintainer finds
  out when a user reports a crash.
- The fixes are individually small — 10 to 20 lines. The cost is the
  discovery: each one took a fresh environment, a failing run, and a read
  through napari's source to work out what was expected. Three crash-fix-rerun
  cycles for one release.
- A documented, supported extension point for custom layers — or a 3D bounding
  box layer in napari itself — would remove all of this.
