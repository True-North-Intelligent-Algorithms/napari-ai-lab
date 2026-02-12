"""
Training Base Mixin.

This module provides a base class for segmenters that support training.
Segmenters can inherit from this class to add training capabilities.
"""

from abc import ABC, abstractmethod


class TrainingBase(ABC):
    """
    Base class for segmenters that support training.

    This class provides a simple interface for training segmenters.
    Subclasses should implement the train() method with their specific
    training logic.

    Attributes:
        train_loss_list (list): List to track training loss per epoch.
        validation_loss_list (list): List to track validation loss per epoch.
        patch_path (str): Path to the directory containing training patches.
    """

    def __init__(self):
        """Initialize training base with loss tracking lists."""
        # Loss tracking lists - accessible by subclasses
        self.train_loss_list = []
        self.validation_loss_list = []
        # Path to training patches directory
        self._patch_path = None

    @property
    def patch_path(self):
        """Get the path to training patches directory."""
        return self._patch_path

    @patch_path.setter
    def patch_path(self, value):
        """Set the path to training patches directory."""
        self._patch_path = value

    @abstractmethod
    def train(self, updater=None):
        """
        Train the segmenter model.

        Args:
            updater (callable, optional): A callback function that can be called
                during training to update progress. The callback should accept
                keyword arguments like epoch, loss, etc.
                Example: updater(epoch=10, loss=0.5, status="Training...")

        Returns:
            dict: A dictionary containing training results/metrics.
                  Example: {"success": True, "final_loss": 0.1, "epochs": 100}

        Raises:
            NotImplementedError: If the subclass doesn't implement this method.
        """
        raise NotImplementedError(
            "Subclasses must implement the train() method"
        )

    def can_train(self):
        """
        Check if this segmenter supports training.

        Returns:
            bool: True if training is supported, False otherwise.
        """
        # By default, if a class inherits from TrainingBase, it can train
        return True
