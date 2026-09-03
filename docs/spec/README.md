# napari-ai-lab specs

Proposed work, numbered. Same convention as scikit-ops' `docs/spec/` — a file
here describes something that does **not** exist yet, and either gets deleted
or graduates into a design document once it does.

The flat notes in `docs/` are a different thing: after-the-fact notes on code
that already exists.

[OPEN.md](OPEN.md) is a third thing: a register of known-broken and undecided
items that are too small or too unresolved to have a plan yet. It is what
"what else to do" resolves to.

| | |
| --- | --- |
| [0001](0001-what-moves-to-scikit-ops.md) | What moves to scikit-ops and skop-napari, and what stays |
| [0002](0002-augmentation-as-an-op.md) | Augmentation becomes a skop op; the loop and the patch directory stay |
| [0003](0003-optional-dependencies.md) | Import never fails, construction may; the minimal install as a testable promise |
| [0004](0004-first-scikit-ops-segmenter.md) | One skop op in the segmenter list, magicgui panel, and what a first-run environment build owes the user |
| [0005](0005-ai-lab-cleanup.md) | State copied rather than read, save-on-switch, two viewers, and what to do about each |
| [0006](0006-batch-segmentation-over-a-sequence.md) | Segment Range over every image; a stacked sequence is a sequence of one |
| [0007](0007-per-project-parameters.md) | Parameters that belong to a project, not to a hardcoded default |
| [0008](0008-user-documentation.md) | mkdocs for users; a parameter reference for the controls a wrong guess ruins |
