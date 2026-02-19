"""
MONAI UNet Global Segmenter.

This module provides a MONAI UNet segmenter for automatic semantic segmentation
of entire images without user prompts.
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
from tqdm.auto import tqdm

from ...datasets.pytorch_semantic_dataset import PyTorchSemanticDataset
from ...mixins import TrainingBase
from ...utilities.dl_util import normalize_image
from .GlobalSegmenterBase import GlobalSegmenterBase

# Try to import MONAI dependencies
try:
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import UNet

    _is_monai_available = True
except ImportError:
    _is_monai_available = False


@dataclass
class MonaiUNetSegmenter(GlobalSegmenterBase, TrainingBase):
    """
    MONAI UNet global segmenter with training support.

    This segmenter uses a trained MONAI UNet model for automatic semantic
    segmentation on entire images using sliding window inference.
    """

    instructions = """
MONAI UNet Automatic Segmentation:
• Uses trained UNet model for semantic segmentation
• Tile Size: Size of patches for sliding window inference
• Show Background Class: Include background in output
• Requires pre-trained model loaded from disk
• Best for: semantic segmentation with custom trained models
• Works with 2D and 3D images
    """

    # Segmenter name
    name: str = field(default="MonaiUNetSegmenter", init=False, repr=False)

    # Parameters for MONAI UNet segmentation
    tile_size: int = field(
        default=1024,
        metadata={
            "type": "int",
            "param_type": "inference",
            "min": 128,
            "max": 100000,
            "step": 128,
            "default": 1024,
        },
    )

    show_background_class: bool = field(
        default=True,
        metadata={
            "type": "bool",
            "param_type": "inference",
            "default": True,
        },
    )

    model_name: str = field(
        default="",
        metadata={
            "type": "file",
            "param_type": "inference",
            "file_type": "file",
            "default": "",
        },
    )

    sparse: bool = field(
        default=True,
        metadata={
            "type": "bool",
            "param_type": "training",
            "default": True,
        },
    )

    num_classes: int = field(
        default=2,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 1,
            "max": 10,
            "step": 1,
            "default": 2,
        },
    )

    depth: int = field(
        default=5,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 4,
            "max": 6,
            "step": 1,
            "default": 5,
        },
    )

    features_level_1: int = field(
        default=32,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 8,
            "max": 64,
            "step": 1,
            "default": 32,
        },
    )

    weight_c1: int = field(
        default=1,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 1,
            "max": 100,
            "step": 1,
            "default": 1,
        },
    )

    weight_c2: int = field(
        default=1,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 1,
            "max": 100,
            "step": 1,
            "default": 1,
        },
    )

    weight_c3: int = field(
        default=1,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 1,
            "max": 100,
            "step": 1,
            "default": 1,
        },
    )

    num_epochs: int = field(
        default=100,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 0,
            "max": 100000,
            "step": 1,
            "default": 100,
        },
    )

    save_interval: int = field(
        default=10,
        metadata={
            "type": "int",
            "param_type": "training",
            "min": 1,
            "max": 10000,
            "step": 1,
            "default": 10,
        },
    )

    learning_rate: float = field(
        default=0.0001,
        metadata={
            "type": "float",
            "param_type": "training",
            "min": 0.0,
            "max": 1.0,
            "step": 0.001,
            "default": 0.0001,
        },
    )

    dropout: float = field(
        default=0.2,
        metadata={
            "type": "float",
            "param_type": "training",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "default": 0.2,
        },
    )

    def __post_init__(self):
        """Initialize the segmenter after dataclass initialization."""
        # Initialize TrainingBase to set up loss tracking lists
        TrainingBase.__init__(self)

        # Model will be loaded from disk
        self.model = None

        # Set up a simple default updater function
        # This will be used if no updater is passed to training/inference methods
        self.updater = lambda message, progress: print(
            f"[{progress}%] {message}"
        )

        # Set supported axes
        self._supported_axes = ["YX", "YXC", "ZYX", "ZYXC"]
        self._potential_axes = ["YX", "YXC", "ZYX", "ZYXC"]

    def __setattr__(self, name, value):
        """Override setattr to detect model_name changes."""
        # Get old value if it exists
        old_value = getattr(self, name, None) if hasattr(self, name) else None

        # Set the new value
        super().__setattr__(name, value)

        # Check if this is model_name and value changed
        if name == "model_name" and old_value != value and value:
            print(f"🔄 Model name changed from '{old_value}' to '{value}'")
            self._on_model_path_changed(value)

    def _on_model_path_changed(self, model_path: str):
        """Handle model name changes - load the PyTorch model."""
        print(f"📁 Loading MONAI UNet model from: {model_path}")

        try:
            self.model = torch.load(model_path, weights_only=False)
            print(
                f"✅ Successfully loaded MONAI UNet model from: {model_path}"
            )
            print(f"   Model type: {type(self.model)}")
        except Exception as e:  # noqa
            print(f"❌ Failed to load model from {model_path}: {e}")
            self.model = None

    def are_dependencies_available(self):
        """
        Check if required dependencies are available.

        Returns:
            bool: True if MONAI and torch can be imported, False otherwise.
        """
        return _is_monai_available

    def get_version(self):
        """Get the MONAI version."""
        if _is_monai_available:
            try:
                import monai

                return f"monai-{monai.__version__}"
            except (ImportError, AttributeError):
                return "monai-unknown"
        return None

    def load_model(self, model_path: str):
        """
        Load a trained model from disk.

        Args:
            model_path (str): Path to the saved model file (.pth)
        """
        if not _is_monai_available:
            raise ImportError("MONAI is not available. Cannot load model.")

        # Setting model_name will trigger __setattr__ which loads the model
        self.model_name = model_path

    def segment(self, image, normalize=True, **kwargs):
        """
        Perform MONAI UNet segmentation on entire image.

        Args:
            image (numpy.ndarray): Input image to segment.
            **kwargs: Additional keyword arguments.

        Returns:
            numpy.ndarray: Labeled segmentation mask.

        Raises:
            ValueError: If image dimensions are not supported or model not loaded.
        """
        if self.model is None:
            raise ValueError(
                "Model not loaded. Call load_model() before segmentation."
            )

        if len(image.shape) < 2:
            raise ValueError(
                f"MonaiUNetSegmenter requires at least 2D images. Got shape: {image.shape}"
            )

        # Prepare image for inference
        if normalize:
            image_norm = normalize_image(image)
        else:
            image_norm = image.astype(np.float32)

        # Prepare tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        # Add channel and batch dimensions if needed
        if len(image_norm.shape) == 2:  # YX -> 1, 1, Y, X
            image_tensor = torch.from_numpy(image_norm[None, None, :, :])
        elif len(image_norm.shape) == 3:  # ZYX or YXC
            # Assume ZYX for 3D
            image_tensor = torch.from_numpy(image_norm[None, None, :, :, :])
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        image_tensor = image_tensor.float().to(device)

        # Perform sliding window inference
        with torch.no_grad():
            if len(image.shape) == 2:
                roi_size = (self.tile_size, self.tile_size)
            else:
                roi_size = (self.tile_size, self.tile_size)

            y = sliding_window_inference(
                image_tensor,
                roi_size,
                sw_batch_size=1,
                predictor=self.model,
                mode="gaussian",
                overlap=0.125,
            )

        # Apply softmax and get predicted classes
        probabilities = F.softmax(y, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

        # Handle background class
        result = predicted_classes.cpu().numpy().squeeze()
        if not self.show_background_class:
            result = result - 1

        return (result + 1).astype(np.uint16)

    def predict(self, image):
        device = torch.device("cuda")
        self.model.to(device)

        image_ = image  # quantile_normalization(image.astype(np.float32))

        # move channel position to first axis if data has channel
        if len(image_.shape) == 3:
            features = image_.transpose(2, 0, 1)
        else:
            # add trivial channel axis
            features = np.expand_dims(image_, axis=0)

        # make into tensor and add trivial batch dimension
        x = torch.from_numpy(features).unsqueeze(0).to(device)

        # move into evaluation mode
        self.model.eval()

        with torch.no_grad():
            # perform sliding window inference to avoid running out of memory on smaller GPUS
            y = sliding_window_inference(
                x,  # Input tensor
                (self.tile_size, self.tile_size),  # Patch size
                1,  # Batch size during inference
                self.model,  # Model for inference
                mode="gaussian",  # Inference mode
                overlap=0.125,  # Overlap factor
            )

        # Apply softmax along the class dimension (dim=1)
        probabilities = F.softmax(y, dim=1)
        # now predicted classes are max of probabilities along the class dimension
        predicted_classes = torch.argmax(probabilities, dim=1)

        if not self.show_background_class:
            predicted_classes = predicted_classes - 1

        return predicted_classes.cpu().detach().numpy().squeeze() + 1

    @classmethod
    def register(cls):
        """Register this segmenter with the framework."""
        return GlobalSegmenterBase.register_framework(
            "MonaiUNetSegmenter", cls
        )

    def get_execution_string(self, image, **kwargs):
        """
        Generate execution string for remote processing.

        Args:
            image (numpy.ndarray): Input image
            **kwargs: Additional keyword arguments

        Returns:
            str: Python code string for remote execution
        """
        return f"""
# MONAI UNet Segmentation
import torch
from monai.inferers import sliding_window_inference
import torch.nn.functional as F

# Load model
model = torch.load('{self.model_name}', weights_only=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Normalize and prepare image
image_norm = (image - image.min()) / (image.max() - image.min())
image_tensor = torch.from_numpy(image_norm[None, None, :, :]).float().to(device)

# Inference
with torch.no_grad():
    y = sliding_window_inference(
        image_tensor, ({self.tile_size}, {self.tile_size}),
        sw_batch_size=1, predictor=model, mode='gaussian', overlap=0.125
    )

probabilities = F.softmax(y, dim=1)
result = torch.argmax(probabilities, dim=1).cpu().numpy().squeeze()
"""

    def train_loop(
        self,
        train_loader,
        net,
        loss_fn,
        optimizer,
        dtype,
        num_epochs,
        device,
        validation_loader=None,
        steps_per_update=-1,
        sparse=False,
        use_tqdm=False,
    ):

        # set train flags, initialize step
        net.train()
        loss_fn.train()
        epoch = 0

        while epoch < num_epochs:
            # reset data loader to get random augmentations
            np.random.seed()

            # zero gradients
            total_loss = 0.0  # To track the sum of losses for averaging

            if steps_per_update == -1:
                total_steps = len(train_loader.dataset)
            else:
                total_steps = steps_per_update

            # Create progress bar context manager conditionally
            if use_tqdm:
                pbar = tqdm(
                    total=total_steps, desc=f"Epoch {epoch}", leave=True
                )
            else:
                # Create a dummy context manager that does nothing
                from contextlib import nullcontext

                pbar = nullcontext()

            with pbar if use_tqdm else nullcontext() as progress:

                for feature, label in train_loader:

                    optimizer.zero_grad()

                    label = label.type(dtype)

                    if sparse:
                        label = label - 1
                    label = label.to(device)
                    feature = feature.to(device)

                    # forward
                    predicted = net(feature)
                    label = torch.squeeze(label, 1)
                    loss_value = loss_fn(input=predicted, target=label)

                    # Accumulate loss for averaging
                    total_loss += loss_value.item()

                    # pass through loss
                    loss_value.backward()

                    if use_tqdm and progress is not None:
                        progress.update(label.shape[0])

                    optimizer.step()

            # Compute the average loss over all steps
            average_loss = total_loss / len(train_loader)
            self.train_loss_list.append(average_loss)

            # Calculate validation loss
            val_loss_str = ""
            if validation_loader is not None:
                net.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for val_feature, val_label in validation_loader:
                        val_label = val_label.type(dtype)
                        if sparse:
                            val_label = val_label - 1
                        val_label = val_label.to(device)
                        val_feature = val_feature.to(device)
                        val_predicted = net(val_feature)
                        val_label = torch.squeeze(val_label, 1)
                        val_loss_value = loss_fn(
                            input=val_predicted, target=val_label
                        )
                        total_val_loss += val_loss_value.item()
                average_val_loss = total_val_loss / len(validation_loader)
                self.validation_loss_list.append(average_val_loss)
                val_loss_str = f", validation loss: {average_val_loss:.4f}"
                net.train()

            print(
                f"Epoch {epoch} - training loss: {average_loss:.4f}{val_loss_str}"
            )

            if epoch % self.save_interval == 0 and epoch > 0:
                # Insert 'checkpoint' before the file extension
                model_path_obj = Path(self.model_name)
                checkpoint_name = (
                    model_path_obj.stem + "_checkpoint" + model_path_obj.suffix
                )
                torch.save(net, Path(self.model_path) / Path(checkpoint_name))

            if self.updater is not None:
                progress = int(epoch / self.num_epochs * 100)
                self.updater(
                    f"Epoch {epoch} - training loss: {average_loss:.4f}{val_loss_str}",
                    progress,
                )

            epoch += 1

    def train(self, updater=None):
        """
        Train the MONAI UNet model.

        This is a simple placeholder implementation. Full training logic
        will be implemented later with proper data loading, training loops,
        validation, etc.

        Args:
            updater (callable, optional): A callback function for progress updates.
                Example: updater(epoch=10, loss=0.5, status="Training...")

        Returns:
            dict: Training results containing success status and metrics.
        """
        if not _is_monai_available:
            return {
                "success": False,
                "error": "MONAI is not available. Cannot train model.",
            }

        print("🚀 Starting MONAI UNet training...")
        print(f"   Sparse: {self.sparse}")
        print(f"   Num Classes: {self.num_classes}")
        print(f"   Depth: {self.depth}")
        print(f"   Features Level 1: {self.features_level_1}")
        print(
            f"   Class Weights: [{self.weight_c1}, {self.weight_c2}, {self.weight_c3}]"
        )
        print(f"   Epochs: {self.num_epochs}")
        print(f"   Learning Rate: {self.learning_rate}")
        print(f"   Dropout: {self.dropout}")
        print(f"   Save Interval: {self.save_interval}")

        # TODO: Implement actual training logic here
        # - Load training data
        # - Create MONAI UNet model with specified parameters
        # - Set up optimizer and loss function
        # - Training loop with validation
        # - Save model checkpoints
        # - Return training metrics

        patch_path = Path(self.patch_path)

        if updater is None:
            updater = self.updater

        if updater is not None:
            updater("Training Monai Semantic model", 0)

        cuda_present = torch.cuda.is_available()
        ndevices = torch.cuda.device_count()
        use_cuda = cuda_present and ndevices > 0
        device = torch.device(
            "cuda" if use_cuda else "cpu"
        )  # "cuda:0" ... default device, "cuda:1" would be GPU index 1, "cuda:2" etc

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

        image_patch_path = patch_path / "input0"

        tif_files = glob(str(image_patch_path / "*.tif"))
        first_im = imread(tif_files[0])
        target_shape = first_im.shape

        num_in_channels = 1 if axes == "YX" else 3

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

        train_data = PyTorchSemanticDataset(
            image_files=X, label_files_list=Y, target_shape=target_shape
        )

        # NOTE: the length of the dataset might not be the same as n_samples
        #       because files not having the target shape will be discarded
        print(f"Training data size: {len(train_data)}")

        train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

        # Create validation dataset and loader only if validation data exists
        validation_data = None
        validation_loader = None
        if len(X_val) > 0:
            validation_data = PyTorchSemanticDataset(
                image_files=X_val,
                label_files_list=Y_val,
                target_shape=target_shape,
            )
            print(f"Validation data size: {len(validation_data)}")
            validation_loader = DataLoader(
                validation_data, batch_size=8, shuffle=False
            )
        else:
            print("No validation data found - training without validation")

        self.num_classes_auto = True

        if self.num_classes_auto:

            if self.sparse:
                # if sparse background will be label 1 so number of classes is the max label indexes
                # ie if the max label index is 3 then there are 3 classes, 1, 2, 3 and 0 is unlabeled
                # (we subtract 1 at later step so 1 (background) becomes 0 and 0 (not labeled) becomes -1)
                self.num_classes = train_data.max_label_index
            else:
                # if not sparse background will be label 0 so number of classes is the max label indexes + 1
                # ie if there are 3 classes the indexes are 0, 1, 2, so need to add 1 to the max index to get number of classes
                self.num_classes = train_data.max_label_index + 1

        # there is an inconstency in how different classes can be defined
        # 1. every class has it's own label image (one-hot encoded)
        # 2. every class has a unique value in the label image
        # When I wrote a lot of this code I was thinking of the first case, but now see the second may be easier for the user
        # so number of output channels is the max of the truth image
        # use monai to create a model, note we don't use an activation function because
        # we use CrossEntropyLoss that includes a softmax, and our prediction will include the softmax
        if self.model is None:

            channels = tuple(
                self.features_level_1 * (2 ** (i - 1) if i > 1 else 1)
                for i in range(self.depth + 1)
            )
            strides = tuple(2 for i in range(self.depth))
            # channels = (self.features_level_1, self.features)

            self.model = UNet(
                spatial_dims=2,
                in_channels=num_in_channels,
                out_channels=self.num_classes,
                channels=channels,  #
                strides=strides,
                num_res_units=2,  # BasicUNet has no residual blocks
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

        weights[0] = 0.33
        weights[1] = 0.33
        weights[2] = 0.33

        # weights = weights*0.01

        # loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=weights).to(device)
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1).to(device)
        dtype = torch.LongTensor

        # Diagnostic prints for troubleshooting input/model mismatch
        print("\n🔍 Training diagnostics:")
        print(
            f"   Model expects: in_channels={num_in_channels}, out_channels={self.num_classes}"
        )
        print(
            f"   Data shape from dataset: {train_data.images.shape if hasattr(train_data, 'images') else 'N/A'}"
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
        )

        torch.save(self.model, Path(self.model_path) / self.model_name)

        # Save training and validation losses to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = Path(self.model_name).stem + f"_{timestamp}.csv"
        csv_path = Path(self.model_path) / csv_name
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
            "message": "Training implementation coming soon",
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
