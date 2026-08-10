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

### Which copy is actually running

There are four vendored copies — `napari_0_4_15`, `napari_0_4_18`,
`napari_0_5_0`, `napari_0_6_0` — and `_utils.parse_package_by_napari_version`
picks the **highest directory whose version is <= the installed napari**. On
napari 0.8.0 that is `napari_0_6_0`, and there is no 0.8 directory.

Each copy subclasses the one below it, but **not uniformly**, and this is the
trap when patching:

```
model:  0_6_0 -> 0_5_0 -> 0_4_18 -> 0_4_15      (a fix in 0_4_15 is inherited)
visual: 0_6_0 -> 0_5_0 -> 0_4_15                (0_5_0 overrides __init__,
                                                 so a fix in 0_4_15 is NOT used)
```

So before fixing anything, find the class that is actually in the chain for
the napari version you are on. A fix in the wrong copy compiles, imports and
does nothing.

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

**Fix.** 12 lines in
`boundingbox/napari_0_4_15/bounding_boxes.py`: a `_BoundingBoxSlicingState`
that routes `_set_view_slice` back to the layer's own existing method — the
same two-line adapter `Shapes` uses — plus the method returning it. The import
of `_LayerSlicingState` is guarded so the file still works on napari < 0.8,
where the class does not exist.

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

**Fix.** Accept `**kwargs` and forward them to the base `__init__`. This is
version-agnostic: on older napari nothing extra is passed, on 0.8 `font_info`
flows through, and whatever the next release adds flows through too without
another patch.

Applied in **two** places, because the visual class is overridden partway up
the chain:

- `boundingbox/napari_0_4_15/vispy_bounding_box_layer.py` — the base
- `boundingbox/napari_0_5_0/vispy_bounding_box_layer.py` — **the one that
  actually runs on napari 0.8**, which calls `VispyBaseLayer.__init__`
  directly rather than going through its parent

The first fix alone changed nothing at runtime. See "Which copy is actually
running" above — this cost a wasted edit and is the single most useful thing
in this file for whoever hits the next one.

## For the napari conversation

Points worth carrying forward:

- Both breaks were **additive from napari's side** — a new abstract method, a
  new constructor argument — and neither is additive from a subclass's point
  of view. Adding an `@abstractmethod` to a public base class is a breaking
  change for every downstream layer.
- Neither break is detectable without running a GUI. A downstream maintainer
  finds out when a user reports a crash.
- The fixes are individually small. The cost is not the code, it is the
  discovery: each one took a fresh environment, a failing run, and a read
  through napari's source to work out what was expected.
- A documented, supported extension point for custom layers — or a 3D bounding
  box layer in napari itself — would remove all of this.
