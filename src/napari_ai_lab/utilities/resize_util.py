"""
Shared resize helpers for downsizing images and labels by a YX factor.

Used by segmenters (e.g. StardistSegmenter, MonaiUNetSegmenter) and datasets
(e.g. PyTorchSemanticDataset) so that downsize_factor logic is consistent.
"""

import numpy as np
from skimage.transform import resize


def downsize_yx(
    img: np.ndarray, factor: int, is_label: bool = False
) -> np.ndarray:
    """
    Downsize image (or label) by an integer factor along the Y and X axes only.

    For 3D arrays the last two axes (Y, X) are downsized; the leading Z axis
    is preserved. For 2D arrays the first two axes (Y, X) are downsized.

    Parameters
    ----------
    img : np.ndarray
        Input image or label array.
    factor : int
        Integer downsize factor. ``factor == 1`` returns ``img`` unchanged.
    is_label : bool, optional
        If True, use nearest-neighbor (order=0) interpolation with
        ``anti_aliasing=False`` and preserve the input dtype. If False,
        use bilinear (order=1) with ``anti_aliasing=True``.
    """
    if factor == 1:
        return img

    new_shape = list(img.shape)
    if img.ndim >= 3:
        # 3D (or higher): downsize last two axes (Y, X), preserve Z (and channels)
        new_shape[-2] = img.shape[-2] // factor
        new_shape[-1] = img.shape[-1] // factor
    else:
        # 2D: downsize Y, X
        new_shape[0] = img.shape[0] // factor
        new_shape[1] = img.shape[1] // factor

    if is_label:
        return resize(
            img,
            new_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(img.dtype)
    return resize(
        img,
        new_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )


def upsize_to_shape(
    arr: np.ndarray, target_shape: tuple, is_label: bool = True
) -> np.ndarray:
    """
    Upsize an array back to ``target_shape``.

    Used to restore a downsized prediction/label to the original image size.

    Parameters
    ----------
    arr : np.ndarray
        Array to upsize.
    target_shape : tuple
        Desired output shape.
    is_label : bool, optional
        If True (default), use nearest-neighbor (order=0) and preserve dtype.
        If False, use bilinear (order=1) with anti-aliasing.
    """
    if tuple(arr.shape) == tuple(target_shape):
        return arr
    if is_label:
        return resize(
            arr,
            target_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(arr.dtype)
    return resize(
        arr,
        target_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )
