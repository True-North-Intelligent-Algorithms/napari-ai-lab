import os


def zero_pad_index(index, width=5):
    return f"{index:0{width}d}"


def generate_next_name(image_path, name, ext=".tif"):

    index = 0
    image_name = image_path + "/" + name + "_" + zero_pad_index(index) + ext

    while os.path.exists(image_name):
        index = index + 1
        image_name = (
            image_path + "/" + name + "_" + zero_pad_index(index) + ext
        )

    base_name = os.path.basename(image_name)
    base_name = os.path.splitext(base_name)[0]

    return base_name


def generate_patch_names(image_path, mask_path, data_name, ext=".tif"):
    """
    Find the next unused zero-padded filename pair for an image patch and its
    corresponding mask/label patch.

    Scans *image_path* for files matching ``<data_name>_NNNNN<ext>`` and
    returns the first pair of full paths where the image file does **not** yet
    exist.  Both paths share the same index so the two files always have
    matching names.

    Example::

        image_path = "patches/input0"
        mask_path  = "patches/truth0"
        data_name  = "patch"

        # If patch_00000.tif and patch_00001.tif already exist in image_path:
        generate_patch_names(image_path, mask_path, data_name)
        # → ("patches/input0/patch_00002.tif", "patches/truth0/patch_00002.tif")

    Args:
        image_path: Directory where input image patches are stored.
        mask_path:  Directory where mask/label patches are stored.
        data_name:  Base name prefix for the patch files (e.g. ``"patch"``).
        ext:        File extension including the dot (default ``".tif"``).

    Returns:
        tuple[str, str]: ``(image_name, mask_name)`` — full paths to the next
        available image patch file and its paired mask file.
    """
    index = 0
    image_name = (
        image_path + "/" + data_name + "_" + zero_pad_index(index) + ext
    )
    mask_name = mask_path + "/" + data_name + "_" + zero_pad_index(index) + ext

    while os.path.exists(image_name):
        index += 1
        image_name = (
            image_path + "/" + data_name + "_" + zero_pad_index(index) + ext
        )
        mask_name = (
            mask_path + "/" + data_name + "_" + zero_pad_index(index) + ext
        )

    return image_name, mask_name


def generate_next_patch_name(image_path, name, ext=".tif"):
    """
    Thin alias for ``generate_next_name``.  Returns only the base name
    (no path, no extension) of the next unused zero-padded patch file.

    Args:
        image_path: Directory to scan for existing patch files.
        name:       Base name prefix (e.g. ``"patch"``).
        ext:        File extension (default ``".tif"``).

    Returns:
        str: Next available base name, e.g. ``"patch_00003"``.
    """
    return generate_next_name(image_path, name, ext)
