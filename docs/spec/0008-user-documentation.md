# 0008 — Documentation a user can read

There is none. `docs/` holds nine developer notes, `docs/spec/` holds plans,
and `README.md` is the copier template plus an alpha-status note. Nothing
tells a person what the apps are, how a project is laid out, or what any
parameter does.

The short version: **use mkdocs with the Material theme**, deployed with
`mkdocs gh-deploy`, the same as the predecessor repo — plus one page type that
repo never had, a parameter reference that says what a control *does* rather
than which button to press.

## Recommendation: mkdocs-material

`../napari-easy-augment-batch-dl` already does this and it works.
Its `mkdocs.yml` is Material theme with an explicit `nav`, `admonition` /
`superfences` / `emoji` extensions, fifteen pages under `docs/`, screenshots
in `docs/images/`, and a `how_to_make_these_docs.md` recording the workflow:
`mkdocs serve` to preview with live reload, `mkdocs gh-deploy` to build and
push to GitHub Pages. Copy that setup rather than designing one. There is no
reason to evaluate sphinx here.

Two things to fix while copying: its `repo_url` is still the placeholder
`https://github.com/myrepo`, and its built `site/` directory is committed —
that should be git-ignored.

mkdocs is not a runtime dependency. It belongs in a docs feature of
`pixi/pytorch_napari/pixi.toml`, or its own small environment.

## Why a site is needed at all, and not just better panel text

The panel already carries an `instructions` string per operation, and it is
the right place for a line of orientation. It cannot carry an explanation. It
is a fixed block of text in a narrow dock, read while the user is trying to do
something else, and it has to cover every control at once — so each control
gets a phrase.

Colour augmentation is the first example of what that phrase leaves out, and
there will be many more. The panel can say "colour jitter, off by default".
What it cannot say is that `hue_limit` sets how far hue *can* move rather than
how far it does — the shift is a uniform draw, so 0.5 is the whole colour
circle appearing occasionally, not every patch coming out green. A user
reading 0.5 the wrong way leaves the slider alone and never gets the variation
they needed. That explanation is three sentences and currently lives only in a
code comment in `albumentations_augmenter.py`.

[0007](0007-per-project-parameters.md) is the same shape from the other
direction: patch size silently reverting to 128 ruined a training run, and
nobody could have caught it because nothing anywhere says what patch size has
to be relative to the objects in the image.

So the controls worth documenting first are the ones where a wrong mental
model produces a plausible-looking model that is quietly worse — before any
screenshot tour.

## Two surfaces, one rule

There are two places user-facing text can live, and they should not say the
same thing:

- **The panel.** Every operation has an `instructions` string, rendered by
  `nd_operation_widget.py:284`. It is in front of the user at the moment they
  are looking at the control. It gets *what the control does*, in a line.
- **The site.** Gets the reasoning — why a value matters, what goes wrong at
  the extremes, how the parameters interact.

The rule: if a user would be stuck without it, it goes in the panel. If they
would be *wrong* without it, it goes on the site and the panel links there.

## The first pages

Ordered by what a new user hits first, not by what is easy to write:

1. **index** — what ND AI Lab is and what the apps are for
2. **install** — pixi is the route; `pip install napari-ai-lab` in the README
   today is misleading. Fold in the pixi-on-PATH item from
   [OPEN.md](OPEN.md), which is a documentation gap by its own admission.
3. **project layout** — a project directory, what appears in it, what the app
   remembers and (until 0007) what it does not
4. **annotate → augment → train → predict** — the actual loop, one page each
5. **parameter reference** — the page this spec exists for. Colour jitter
   first, then patch size and `min_long_axis`/`max_long_axis`, then the
   training parameters.

Screenshots matter and are the slow part. Text without them is still worth
shipping.

## Where the files go

`docs/` currently mixes user-facing nothing with developer notes
(`progress_logger_*.md`, `axes_collapse_guide.md`, `pixi_environment_notes.md`
and friends). mkdocs builds every markdown file under its `docs_dir`, so
those would land on the public site unless moved or excluded.

Proposed: `docs/` becomes the user site, existing flat notes move to
`docs/dev/`, `docs/spec/` stays where it is, and both are kept off the nav.
Alternative is a separate `site_docs/` tree, which avoids moving anything but
splits documentation across two directories forever.

## Open questions

- **Move the dev notes or use a second tree?** Moving is cleaner and is the
  recommendation; it touches paths that other notes may reference.
- **Does this repo get its own GitHub Pages site, or a section of an existing
  one?** The predecessor has its own. Two sites for two generations of the
  same tool may confuse more than it helps.
- **How much of the parameter reference belongs in scikit-ops instead?** Once
  augmentation is an op ([0002](0002-augmentation-as-an-op.md)), the meaning
  of `hue_limit` travels with the op, not with this UI. The explanation of
  *when you would want it* stays here either way.
