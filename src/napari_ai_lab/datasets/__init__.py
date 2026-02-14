"""
Datasets module for napari-ai-lab.

This module contains dataset classes for training machine learning models.
"""

from .pytorch_semantic_dataset import PyTorchSemanticDataset

__all__ = ["PyTorchSemanticDataset"]
