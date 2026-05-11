"""
Base class for global segmenters with registry system.

This module provides a base class that global segmenters can inherit from,
along with a registry system for automatic discovery and registration of
segmenter implementations.

Global segmenters perform automatic segmentation on entire images without
requiring user prompts (points/shapes).
"""

import json

import numpy as np

from ...utilities.resize_util import downsize_yx, upsize_to_shape
from ..SegmenterBase import SegmenterBase


class GlobalSegmenterBase(SegmenterBase):
    """
    Base class for global segmenters with registry functionality.

    This class provides global segmentation capabilities where segmenters
    process entire images automatically without requiring user prompts.

    Global segmenters are designed for automatic segmentation workflows
    where no user interaction (points/shapes) is required and the entire
    image is processed globally.

    Example usage:
        # In a derived segmenter class (e.g., OtsuSegmenter):
        from .GlobalSegmenterBase import GlobalSegmenterBase

        class OtsuSegmenter(GlobalSegmenterBase):
            # Implementation here
            pass

        # Register the segmenter when the module is imported:
        GlobalSegmenterBase.register_framework('OtsuSegmenter', OtsuSegmenter)
    """

    # Registry for global segmenter frameworks - separate from interactive segmenters
    registry = {}

    @classmethod
    def register_framework(cls, name, framework):
        """
        Add a global framework to the registry.

        Args:
            name (str): The name of the framework.
            framework (GlobalSegmenterBase): The framework class to register.
        """
        cls.registry[name] = framework
        print(f"Registered global segmenter: {name}")

    def segment(self, image, **kwargs):
        """
        Perform global segmentation on the given image.

        This is the main method that should be overridden by derived classes.
        Global segmenters work on the entire image without user prompts.

        Args:
            image (numpy.ndarray): Input image to segment.
            **kwargs: Additional keyword arguments specific to the segmenter.

        Returns:
            numpy.ndarray: Segmentation mask or labeled image.

        Raises:
            NotImplementedError: If not implemented by derived class.
        """
        raise NotImplementedError(
            "Derived global segmenter classes must implement the segment method"
        )

    # ------------------------------------------------------------------
    # Shared downsize/upsize helpers
    #
    # Many global segmenters expose a ``downsize_factor`` training/inference
    # parameter. The helpers below provide a single source of truth so that
    # both training-time and inference-time downsizing use identical logic
    # across segmenters (StarDist, MONAI UNet, ...).
    # ------------------------------------------------------------------

    def _downsize_image(
        self, img: np.ndarray, factor: int, is_label: bool = False
    ) -> np.ndarray:
        """Downsize ``img`` by ``factor`` along Y and X. See ``resize_util``."""
        return downsize_yx(img, factor, is_label=is_label)

    def _upsize_to_shape(
        self, arr: np.ndarray, target_shape: tuple, is_label: bool = True
    ) -> np.ndarray:
        """Upsize ``arr`` back to ``target_shape``. See ``resize_util``."""
        return upsize_to_shape(arr, target_shape, is_label=is_label)

    def _load_downsize_factor_from_json(self, json_path) -> bool:
        """
        Load ``downsize_factor`` from a JSON file (if present) and set it
        on ``self``. Returns True if loaded, False otherwise. Errors are
        swallowed with a warning so that missing/corrupt metadata does not
        break inference.
        """
        import os

        if not os.path.exists(str(json_path)):
            return False
        try:
            with open(str(json_path)) as f:
                params = json.load(f)
            if "downsize_factor" in params:
                self.downsize_factor = params["downsize_factor"]
                print(
                    f"   Loaded downsize_factor={self.downsize_factor} from {os.path.basename(str(json_path))}"
                )
                return True
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            print(
                f"   Warning: Could not load downsize_factor from {os.path.basename(str(json_path))}: {e}"
            )
        return False

    def _save_downsize_factor_to_json(
        self, json_path, extra: dict | None = None
    ) -> None:
        """Save ``downsize_factor`` (and optional extra fields) to JSON."""
        import os

        params = {"downsize_factor": getattr(self, "downsize_factor", 1)}
        if extra:
            params.update(extra)
        os.makedirs(os.path.dirname(str(json_path)), exist_ok=True)
        with open(str(json_path), "w") as f:
            json.dump(params, f, indent=2)
        print(
            f"   Saved downsize_factor={params['downsize_factor']} to {os.path.basename(str(json_path))}"
        )
