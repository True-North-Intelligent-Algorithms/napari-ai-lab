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

## Resolved

Nothing yet.
