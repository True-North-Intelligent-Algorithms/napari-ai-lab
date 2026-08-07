# napari-ai-lab — notes for Claude

A collection of napari plugins and utilities for ND segmentation with Cellpose,
StarDist, SAM and friends. Generated from the napari-plugin-template (copier,
npe2), published on PyPI as `napari-ai-lab`.

App-shaped, where skop-napari is one generic panel driven by op signatures:
this repo is several purpose-built apps with their own UIs.

## Where this is heading

Ongoing iterations of ND AI Lab aim to **build on scikit-ops**, migrating code
there where that is the right home and deliberately not where it is not. The
split being aimed at:

- **scikit-ops** owns the deep-learning **ops** — training and inference
  wrapped so that many callers can consume them, of which this repo is one.
  An op that others would want belongs there, not here.
- **napari-ai-lab** owns **annotation, inference and training as widgets and
  workflows** that are friendly and enjoyable to use. The user experience is
  the product here, and it is not something an op signature can express.

So when adding something, ask which side of that line it falls on. A model
wrapper, a training loop, a pre/post-processing step with no UI in it: likely
scikit-ops. Anything about how a person draws, corrects, reviews or launches:
here. Some things are genuinely UI-shaped and should not be forced into an op
— avoiding that mistake matters as much as making the migration.

## Layout

```
src/napari_ai_lab/napari.yaml   npe2 manifest
src/napari_ai_lab/apps/         the apps: nd_ai_lab, easy_label, easy_segment,
                                easy_augment, edit_masks, interactive local
                                learning, launcher and dialogs
src/napari_ai_lab/Augmenters/   augmentation
src/napari_ai_lab/artifact_io/  saving and loading
src/napari_ai_lab/datasets/     dataset handling
src/napari_ai_lab/vendored/     vendored third-party code (napari_bbox)
src/launch_*.py                 top-level launchers, one per app
pixi/<name>/pixi.toml           environments: stardist, stardist8,
                                microsam_cellposesam_czi, cellcast_test
docs/, design/                  loose notes plus docs/spec/
notebooks/, experiments/, scripts/, temp_scripts/
```

## Commands

```sh
python src/launch_nd_ai_lab.py      # and the other launch_*.py, one per app
pytest                              # dependency group is `testing`, not `dev`
pytest -m "not bioio"               # skip the slow bioio tests
```

Environments are pixi projects under `pixi/`, one per model stack. Several are
registered as Jupyter kernels user-wide (`stardist`, `microsam_cellposesam`,
`microsam_cellposesam_czi`, `segment_everything`), so they show up in every
notebook kernel picker on this machine, including other projects'.

## Facts worth not rediscovering

- Runtime dependencies are deliberately light — numpy, magicgui, qtpy,
  scikit-image — with `napari-ai-lab[all]` pulling `napari[all]`. The model
  stacks are not runtime dependencies.
- When a heavy dependency is missing, the code can fall back to running it
  through appose in a separate environment, and there is a dialog for choosing
  a remote environment (see the `remote_env_dialog` app module and recent
  history).
- The working tree is usually dirty: notebooks and `pixi/*/pixi.lock` pick up
  incidental changes. Do not sweep them into unrelated commits.
- `src/` holds both the package and loose launcher scripts, including some
  copies (`launch_nd_ai_lab copy.py`, `launch_temp.py`). Not all of it is live.

## Related repositories (siblings)

- `../scikit-ops` — ops plus the machinery that runs them in isolated
  environments; the appose approach here is the same idea, and the intended
  home for the deep-learning ops this repo builds on. See *Where this is
  heading* above.
- `../skop-napari` — the generic op panel for napari.
- `../tnia-python` — utilities; an
  [install rework](../tnia-python/docs/specs/0001-improve-installation.md) is pending.
