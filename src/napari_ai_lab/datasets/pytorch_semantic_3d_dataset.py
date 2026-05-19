"""3D variant of PyTorchSemanticDataset for ZYX / ZYXC volumetric patches."""

import numpy as np
from tifffile import imread
from tqdm import tqdm


class PyTorchSemantic3DDataset:
    """
    3D semantic-segmentation dataset.

    Loads volumetric image / label patches into memory and prepares them
    for a 3D PyTorch / MONAI UNet (spatial_dims=3).

    Image shapes accepted per file:
        (Z, Y, X)        -> stored as (1, Z, Y, X)
        (Z, Y, X, C)     -> stored as (C, Z, Y, X)

    Label shapes accepted per file:
        (Z, Y, X)        -> stored as (1, Z, Y, X)

    Files whose shape does not match ``target_shape`` are silently skipped.
    """

    def __init__(
        self,
        image_files,
        label_files_list,
        target_shape,
        downsize_factor: int = 1,
    ):
        assert len(image_files) == len(label_files_list[0])
        assert all(
            x.name == y.name
            for x, y in zip(image_files, label_files_list[0], strict=False)
        )

        if downsize_factor != 1:
            print(
                "⚠️  PyTorchSemantic3DDataset: downsize_factor != 1 is not "
                "supported for 3D data; ignoring."
            )
        self.downsize_factor = 1

        self.images = []
        self.labels = []

        for idx in tqdm(range(len(image_files))):
            image = imread(image_files[idx])
            image = image.astype(np.float32)

            labels = []
            for label_files in label_files_list:
                labels.append(imread(label_files[idx]))

            # Add / move channel dim
            if image.ndim == 3:  # (Z, Y, X)
                image = np.expand_dims(image, axis=0)  # (1, Z, Y, X)
            elif image.ndim == 4:  # (Z, Y, X, C)
                image = np.transpose(image, (3, 0, 1, 2))  # (C, Z, Y, X)
            else:
                raise ValueError(
                    f"Unsupported 3D image shape: {image.shape}. "
                    "Expected (Z,Y,X) or (Z,Y,X,C)."
                )

            label = np.expand_dims(labels[0], axis=0)  # (1, Z, Y, X)

            self.images.append(image)
            self.labels.append(label)

        self.images = np.stack(self.images)
        self.labels = np.stack(self.labels).astype(np.int64)

        self.max_label_index = int(np.max(self.labels))

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)
