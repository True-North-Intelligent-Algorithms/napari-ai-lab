"""Read and plot a model's training history.

``history.csv`` is written into a model directory by the training op, one row
per epoch, and appended to when training continues. Continuing copies the
parent model's directory first, so the file arrives carrying its ancestors'
rows -- a model trained from ``my_model`` already knows what ``my_model`` did.

What the columns are for:

``run``      increments once per training call, so restarting training on the
             same model is a boundary
``model``    the name trained into. It changes when a run writes to a new
             name -- which is a new model descended from the last, not a
             rename, though "renamed" is the shorter word for it
``dataset``  identifies the patch set, so regenerating patches is a boundary

Marking a boundary is all this claims. A changed ``dataset`` says the training
data is not what it was; it does not say whether labels were edited, patches
re-augmented, or both. That is a deliberate limit -- a full record of how a
dataset changed is a versioned-storage problem, and this is a note in the
margin, not a provenance system.
"""

import csv
import os

#: Columns whose value changing between rows marks a boundary, and what to
#: call it on the plot.
BOUNDARIES = {
    "run": "trained again",
    "model": "new name",
    "dataset": "patches changed",
}


def load_history(model_dir):
    """Rows of ``history.csv``, oldest first. Empty when there is none."""
    path = os.path.join(model_dir, "history.csv")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def find_boundaries(rows):
    """Where something other than the epoch changed.

    Returns ``[(epoch, [label, ...]), ...]``. The first row is not a boundary:
    everything has just changed at the start, which is not news.
    """
    found = []
    for previous, row in zip(rows, rows[1:], strict=False):
        # A blank in the earlier row means the column did not exist when it
        # was written -- rows migrated from an older history. Nothing is known
        # about them, and not knowing is not a change.
        changed = [
            label
            for column, label in BOUNDARIES.items()
            if previous.get(column) and row.get(column) != previous.get(column)
        ]
        # Every rename or dataset change is also a new run, so saying so
        # adds nothing beside them. It is only worth reporting alone.
        if len(changed) > 1:
            changed.remove(BOUNDARIES["run"])
        if changed:
            found.append((float(row["epoch"]), changed))
    return found


def plot_history(model_dir, ax=None):
    """Plot loss and val_loss, with a dotted line at each boundary.

    Returns the matplotlib axes, or None when the model has no history --
    a model trained before this file existed, or never trained at all.
    """
    import matplotlib.pyplot as plt

    rows = load_history(model_dir)
    if not rows:
        return None

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    epochs = [float(r["epoch"]) for r in rows]
    for column, style in (("loss", "-"), ("val_loss", "--")):
        ax.plot(
            epochs,
            [float(r[column]) for r in rows],
            style,
            label=column,
        )

    for epoch, changed in find_boundaries(rows):
        ax.axvline(epoch, color="grey", linestyle=":", linewidth=1)
        ax.annotate(
            ", ".join(changed),
            xy=(epoch, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -10),
            textcoords="offset points",
            fontsize=8,
            color="grey",
            rotation=90,
            va="top",
        )

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(os.path.basename(os.path.normpath(model_dir)))
    ax.legend()
    ax.figure.text(
        0.01,
        0.01,
        "Boundaries mark that something changed, not what: a patch set is "
        "identified by a counter, not by its contents.",
        fontsize=7,
        color="grey",
    )
    return ax
