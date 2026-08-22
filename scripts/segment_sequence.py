"""Segment every image in a folder, headless.

The sequence path from spec 0006 with no viewer attached.  Predictions land
under ``predictions/<SegmenterName>/``, one file per image; nothing is drawn.
Images the segmenter cannot handle are skipped and listed at the end rather
than stopping the run.

Set VIEW to open napari on the same model afterwards -- the sequence
scrollbar and two layers, no AI Lab UI -- so the predictions just written can
be stepped through.

Edit the settings below and run it.  Paths are this machine's; change them.
"""

from napari_ai_lab.models.image_data_model import ImageDataModel
from napari_ai_lab.Segmenters.GlobalSegmenters import CellposeSegmenter

# --- settings -------------------------------------------------------------

# Three RGB pollen images, copied from the folder `pixi run lab` uses.
FOLDER = "tests/test_images/pollen_small"
# The class itself, not its name: segmenters register only when something
# calls Cls.register(), which the app launcher does and a script need not.
SEGMENTER = CellposeSegmenter

# Axes the segmenter consumes, and the ones collapsed rather than iterated.
# RGB tiffs are YXC: without COLLAPSE every image is refused with
# "axes YXC, cannot iterate C", which is deliberate -- iterating C would
# segment each colour separately.
AXIS = "YX"
COLLAPSE = ["C"]  # None for greyscale

# Image indices, inclusive.  None for the whole folder.
START = None
END = None

VIEW = True

# --------------------------------------------------------------------------


def main():
    model = ImageDataModel(FOLDER)
    print(f"{model.get_image_count()} images in {model.parent_directory}")

    segmenter = SEGMENTER()

    first = START or 0
    model.load_image(first)

    processor = model.segment_sequence(
        segmenter=segmenter,
        selected_axis=AXIS,
        axes_to_collapse=COLLAPSE,
        start_index=START,
        end_index=END,
        current_index=first,
        on_progress=lambda cur, tot: print(f"image {cur}/{tot}"),
    )

    print(processor.summary())

    if VIEW:
        view_sequence(model)


def view_sequence(model):
    """Browse the images and their predictions -- no AI Lab UI, just the data.

    Half headless: the same model the batch ran on, the sequence scrollbar to
    step through it, and a labels layer reloaded from disk for each image.
    """
    import napari

    from napari_ai_lab.nd_sequence_viewer import NDSequenceViewer

    viewer = napari.Viewer()
    sequence_viewer = NDSequenceViewer(viewer)
    viewer.window.add_dock_widget(
        sequence_viewer, area="bottom", name="Sequence"
    )

    # NDSequenceViewer adds the image layer; the predictions layer is ours.
    shown = {}

    def show_predictions(image_layer, image_index):
        if shown.get("layer") in viewer.layers:
            viewer.layers.remove(shown["layer"])
        predictions = model.load_existing_predictions(
            image_index,
            image_shape=image_layer.data.shape,
            subdirectory=SEGMENTER.__name__,
            axes_to_collapse=COLLAPSE,
        )
        shown["layer"] = viewer.add_labels(
            predictions, name=f"predictions: {SEGMENTER.__name__}"
        )

    sequence_viewer.image_changed.connect(show_predictions)
    sequence_viewer.set_image_data_model(model)
    napari.run()  # blocks until the window is closed


if __name__ == "__main__":
    main()
