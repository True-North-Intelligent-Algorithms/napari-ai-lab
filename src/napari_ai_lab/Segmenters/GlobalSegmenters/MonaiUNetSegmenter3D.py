"""
MONAI 3D UNet Global Segmenter.

3D variant of :class:`MonaiUNetSegmenter`. Trains on volumetric (Z,Y,X) or
(Z,Y,X,C) patches and runs 3D sliding-window inference (no slice-by-slice).
"""

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tifffile import imread
from torch.utils.data import DataLoader

from ...datasets.pytorch_semantic_3d_dataset import PyTorchSemantic3DDataset
from ...utilities.dl_util import normalize_percentile
from .GlobalSegmenterBase import GlobalSegmenterBase
from .MonaiUNetSegmenter import MonaiUNetSegmenter

try:
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import UNet

    _is_monai_available = True
except ImportError:
    _is_monai_available = False


@dataclass
class MonaiUNetSegmenter3D(MonaiUNetSegmenter):
    """
    MONAI 3D UNet global segmenter with training support.

    Operates exclusively on 3D volumes (axes ``ZYX`` or ``ZYXC``). Uses a
    UNet with ``spatial_dims=3`` and 3D sliding-window inference.
    """

    instructions = """
MONAI 3D UNet Automatic Segmentation:
• Trains a 3D UNet on volumetric (Z,Y,X) / (Z,Y,X,C) patches
• Tile Size (XY) and Tile Size Z control 3D sliding-window ROI
• Predicts full 3D label volumes (no slice-by-slice fallback)
• Best for: 3D semantic segmentation with custom trained models
    """

    name: str = field(default="MonaiUNetSegmenter3D", init=False, repr=False)

    tile_size_z: int = field(
        default=64,
        metadata={
            "type": "int",
            "param_type": "inference",
            "min": 8,
            "max": 1024,
            "step": 8,
            "default": 64,
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # Restrict to volumetric axes only
        self._supported_axes = ["ZYX", "ZYXC"]
        self._potential_axes = ["ZYX", "ZYXC"]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def segment(self, image, normalize=True, **kwargs):
        """Run 3D MONAI UNet segmentation on a volumetric image."""

        if self.model_file_path is None:
            raise ValueError(
                "Model file path is not set. Please set model_file_path to "
                "load a trained model."
            )

        self.load_model(self.model_file_path)

        if self.model is None:
            raise ValueError(
                "Model not loaded. Set model_file_path or call load_model()."
            )

        if image.ndim < 3:
            raise ValueError(
                f"MonaiUNetSegmenter3D requires 3D images. Got shape: "
                f"{image.shape}"
            )

        # Normalize
        if normalize:
            image_norm = normalize_percentile(image)
        else:
            image_norm = image.astype(np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        # Build (1, C, Z, Y, X)
        if image_norm.ndim == 3:  # ZYX
            image_tensor = torch.from_numpy(image_norm[None, None])
        elif image_norm.ndim == 4:  # ZYXC -> (C, Z, Y, X)
            czyx = np.transpose(image_norm, (3, 0, 1, 2))
            image_tensor = torch.from_numpy(czyx[None])
        else:
            raise ValueError(f"Unsupported 3D image shape: {image.shape}")

        image_tensor = image_tensor.float().to(device)

        import time

        t0 = time.perf_counter()
        with torch.no_grad():
            if self.use_tiles:
                roi_size = (
                    self.tile_size_z,
                    self.tile_size,
                    self.tile_size,
                )
                y = sliding_window_inference(
                    image_tensor,
                    roi_size,
                    sw_batch_size=1,
                    predictor=self.model,
                    mode="gaussian",
                    overlap=0.125,
                )
            else:
                # Pad last 3 spatial dims to multiples of 2**depth
                stride = 2**self.depth
                d, h, w = image_tensor.shape[-3:]
                pad_d = (stride - d % stride) % stride
                pad_h = (stride - h % stride) % stride
                pad_w = (stride - w % stride) % stride
                if pad_d or pad_h or pad_w:
                    # F.pad order for 3D: (Wl,Wr, Hl,Hr, Dl,Dr)
                    image_tensor = F.pad(
                        image_tensor,
                        (0, pad_w, 0, pad_h, 0, pad_d),
                    )
                y = self.model(image_tensor)
                if pad_d or pad_h or pad_w:
                    y = y[..., :d, :h, :w]
        elapsed = time.perf_counter() - t0
        mode_str = (
            f"3D tiled ({self.tile_size_z}x{self.tile_size}x{self.tile_size})"
            if self.use_tiles
            else "3D full volume"
        )
        print(f"   Inference ({mode_str}): {elapsed:.2f}s")

        probabilities = F.softmax(y, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

        result = predicted_classes.cpu().numpy().squeeze()
        if not self.show_background_class:
            result = result - 1
        result = (result + 1).astype(np.uint16)

        return result

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, updater=None, use_tqdm=False):
        """Train a 3D MONAI UNet on volumetric patches."""

        if not _is_monai_available:
            return {
                "success": False,
                "error": "MONAI is not available. Cannot train model.",
            }

        if updater is not None:
            use_tqdm = False
            print(
                "🚀 Starting MONAI 3D UNet training (with progress updater)..."
            )
        elif use_tqdm:
            print(
                "🚀 Starting MONAI 3D UNet training (with tqdm progress bar)..."
            )
        else:
            print("🚀 Starting MONAI 3D UNet training...")

        print(f"   Sparse: {self.sparse}")
        print(f"   Num Classes: {self.num_classes}")
        print(f"   Depth: {self.depth}")
        print(f"   Features Level 1: {self.features_level_1}")
        print(
            f"   Class Weights: [{self.weight_c1}, {self.weight_c2}, "
            f"{self.weight_c3}]"
        )
        print(f"   Epochs: {self.num_epochs}")
        print(f"   Learning Rate: {self.learning_rate}")
        print(f"   Dropout: {self.dropout}")
        print(f"   Save Interval: {self.save_interval}")

        patch_path = Path(self.patch_path)

        if updater is not None:
            updater(0, 0, "Training MONAI 3D Semantic model")

        cuda_present = torch.cuda.is_available()
        ndevices = torch.cuda.device_count()
        use_cuda = cuda_present and ndevices > 0
        device = torch.device("cuda" if use_cuda else "cpu")

        with open(patch_path / "info.json") as json_file:
            data = json.load(json_file)
            sub_sample = data.get("sub_sample", 1)
            print("sub_sample", sub_sample)
            axes = data["axes"]
            print("axes", axes)
            num_inputs = data["num_inputs"]
            print("num_inputs", num_inputs)
            num_truths = data["num_truths"]
            print("num_truths", num_truths)

        if axes not in ("ZYX", "ZYXC"):
            return {
                "success": False,
                "error": (
                    f"MonaiUNetSegmenter3D requires 3D patches (axes ZYX or "
                    f"ZYXC); got axes={axes!r}."
                ),
            }

        image_patch_path = patch_path / "input0"
        tif_files = glob(str(image_patch_path / "*.tif"))
        if not tif_files:
            return {
                "success": False,
                "error": f"No training patches found in {image_patch_path}",
            }
        first_im = imread(tif_files[0])
        target_shape = first_im.shape

        # axes ZYX -> 1 input channel; ZYXC -> trailing-dim channels
        if axes == "ZYX":
            num_in_channels = 1
        else:  # ZYXC
            num_in_channels = first_im.shape[-1] if first_im.ndim == 4 else 3

        assert (
            patch_path.exists()
        ), f"root directory with images and masks {patch_path} does not exist"

        train_input_str = "input0"
        train_ground_truth_str = "ground_truth"
        X, Y = self.get_image_label_files(
            patch_path, train_input_str, train_ground_truth_str, num_truths
        )

        validation_input_str = "input_validation0"
        validation_ground_truth_str = "ground truth_validation"
        X_val, Y_val = self.get_image_label_files(
            patch_path,
            validation_input_str,
            validation_ground_truth_str,
            num_truths,
        )

        train_data = PyTorchSemantic3DDataset(
            image_files=X,
            label_files_list=Y,
            target_shape=target_shape,
            downsize_factor=self.downsize_factor,
        )
        print(f"Training data size: {len(train_data)}")

        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

        validation_data = None
        validation_loader = None
        if len(X_val) > 0:
            validation_data = PyTorchSemantic3DDataset(
                image_files=X_val,
                label_files_list=Y_val,
                target_shape=target_shape,
                downsize_factor=self.downsize_factor,
            )
            print(f"Validation data size: {len(validation_data)}")
            validation_loader = DataLoader(
                validation_data, batch_size=1, shuffle=False
            )
        else:
            print("No validation data found - training without validation")

        self.num_classes_auto = True
        if self.num_classes_auto:
            if self.sparse:
                self.num_classes = train_data.max_label_index
            else:
                self.num_classes = train_data.max_label_index + 1

        if self.model is None:
            channels = tuple(
                self.features_level_1 * (2 ** (i - 1) if i > 1 else 1)
                for i in range(self.depth + 1)
            )
            strides = tuple(2 for _ in range(self.depth))

            self.model = UNet(
                spatial_dims=3,
                in_channels=num_in_channels,
                out_channels=self.num_classes,
                channels=channels,
                strides=strides,
                num_res_units=2,
                act=("LeakyReLU", {"negative_slope": 0.01, "inplace": True}),
                norm="batch",
                dropout=self.dropout,
            )

        self.model = self.model.to(device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.model.train(True)

        weights = torch.ones(self.num_classes, dtype=torch.float32)
        weights[0] = self.weight_c1
        if self.num_classes > 1:
            weights[1] = self.weight_c2
        if self.num_classes > 2:
            weights[2] = self.weight_c3

        loss_function = torch.nn.CrossEntropyLoss(
            weight=weights, ignore_index=-1
        ).to(device)
        dtype = torch.LongTensor

        print("\n🔍 Training diagnostics:")
        print(
            f"   Model expects: in_channels={num_in_channels}, "
            f"out_channels={self.num_classes}"
        )
        print(
            f"   Data shape from dataset: "
            f"{train_data.images.shape if hasattr(train_data, 'images') else 'N/A'}"
        )
        print(f"   Target shape: {target_shape}, Axes: {axes}")

        self.train_loop(
            train_loader,
            self.model,
            loss_function,
            optimizer,
            dtype,
            self.num_epochs,
            device,
            validation_loader=validation_loader,
            sparse=self.sparse,
            use_tqdm=use_tqdm,
            updater=updater,
        )

        torch.save(
            self.model, Path(self.model_save_dir) / self.training_model_name
        )
        self.model_file_path = str(
            Path(self.model_save_dir) / self.training_model_name
        )
        self.inference_model_name = Path(self.training_model_name).stem

        metadata_path = Path(self.model_file_path).with_suffix(".json")
        self._save_downsize_factor_to_json(metadata_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = Path(self.training_model_name).stem + f"_{timestamp}.csv"
        csv_path = Path(self.model_save_dir) / csv_name
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Epoch", "Training Loss", "Validation Loss"])
            for epoch, (train_loss, val_loss) in enumerate(
                zip(
                    self.train_loss_list,
                    self.validation_loss_list,
                    strict=False,
                )
            ):
                writer.writerow([epoch, train_loss, val_loss])
        print(f"Saved loss history to {csv_path}")

        return {
            "success": False,
            "message": "Training finished",
            "cuda_present": cuda_present,
            "ndevices": ndevices,
            "device": str(device),
            "num_inputs": num_inputs,
            "num_truths": num_truths,
            "X": X,
            "Y": Y,
            "X_val": X_val,
            "Y_val": Y_val,
        }

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return GlobalSegmenterBase.register_framework(
            "MonaiUNetSegmenter3D", cls
        )
