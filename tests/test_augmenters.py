import os
import tempfile

import numpy as np
import pytest

from napari_ai_lab.Augmenters import AugmenterBase, SimpleAugmenter


class TestSimpleAugmenter:
    """Test suite for SimpleAugmenter class."""

    def test_simple_augmenter_initialization(self):
        """Test that SimpleAugmenter can be initialized."""
        augmenter = SimpleAugmenter()
        assert isinstance(augmenter, AugmenterBase)

    def test_simple_augmenter_with_seed(self):
        """Test that SimpleAugmenter respects random seed."""
        augmenter1 = SimpleAugmenter(seed=42)
        augmenter2 = SimpleAugmenter(seed=42)

        im = np.random.rand(100, 100)
        mask = np.random.randint(0, 2, (100, 100))
        patch_size = (50, 50)

        crop1_im, crop1_mask = augmenter1.augment(im, mask, patch_size)
        crop2_im, crop2_mask = augmenter2.augment(im, mask, patch_size)

        np.testing.assert_array_equal(crop1_im, crop2_im)
        np.testing.assert_array_equal(crop1_mask, crop2_mask)

    def test_random_crop_2d(self):
        """Test random cropping on 2D images."""
        augmenter = SimpleAugmenter(seed=42)
        im = np.random.rand(100, 100)
        mask = np.random.randint(0, 2, (100, 100))
        patch_size = (50, 50)

        cropped_im, cropped_mask = augmenter.augment(im, mask, patch_size)

        assert cropped_im.shape == patch_size
        assert cropped_mask.shape == patch_size

    def test_random_crop_3d(self):
        """Test random cropping on 3D images."""
        augmenter = SimpleAugmenter(seed=42)
        im = np.random.rand(50, 100, 100)
        mask = np.random.randint(0, 2, (50, 100, 100))
        patch_size = (25, 50, 50)

        cropped_im, cropped_mask = augmenter.augment(im, mask, patch_size)

        assert cropped_im.shape == patch_size
        assert cropped_mask.shape == patch_size

    def test_random_crop_with_axis(self):
        """Test random cropping along a specific axis."""
        augmenter = SimpleAugmenter(seed=42)
        im = np.random.rand(50, 100, 100)
        mask = np.random.randint(0, 2, (50, 100, 100))
        patch_size = (25, 100, 100)  # Only crop along first axis

        cropped_im, cropped_mask = augmenter.augment(
            im, mask, patch_size, axis=0
        )

        assert cropped_im.shape == patch_size
        assert cropped_mask.shape == patch_size

    def test_shape_mismatch_raises_error(self):
        """Test that mismatched image and mask shapes raise an error."""
        augmenter = SimpleAugmenter()
        im = np.random.rand(100, 100)
        mask = np.random.randint(0, 2, (50, 50))
        patch_size = (50, 50)

        with pytest.raises(
            ValueError, match="Image and mask shapes must match"
        ):
            augmenter.augment(im, mask, patch_size)

    def test_patch_larger_than_image_raises_error(self):
        """Test that patch size larger than image raises an error."""
        augmenter = SimpleAugmenter()
        im = np.random.rand(50, 50)
        mask = np.random.randint(0, 2, (50, 50))
        patch_size = (100, 100)

        with pytest.raises(
            ValueError, match="Patch size .* is larger than image size"
        ):
            augmenter.augment(im, mask, patch_size)

    def test_patch_dimension_mismatch_raises_error(self):
        """Test that patch dimension mismatch raises an error."""
        augmenter = SimpleAugmenter()
        im = np.random.rand(50, 100, 100)
        mask = np.random.randint(0, 2, (50, 100, 100))
        patch_size = (50, 50)  # 2D patch for 3D image

        with pytest.raises(
            ValueError,
            match="patch_size dimensions .* must match image dimensions",
        ):
            augmenter.augment(im, mask, patch_size)

    def test_augment_and_save(self):
        """Test augmenting and saving patches to disk."""
        augmenter = SimpleAugmenter(seed=42)
        im = np.random.rand(100, 100)
        mask = np.random.randint(0, 2, (100, 100))
        patch_size = (50, 50)

        with tempfile.TemporaryDirectory() as tmpdir:
            patch_path = tmpdir
            patch_base_name = "test_patch"

            im_path, mask_path = augmenter.augment_and_save(
                im, mask, patch_path, patch_base_name, patch_size
            )

            # Check that files were created
            assert os.path.exists(im_path)
            assert os.path.exists(mask_path)

            # Check that files have expected naming pattern
            assert "test_patch_im" in os.path.basename(im_path)
            assert "test_patch_mask" in os.path.basename(mask_path)

    def test_augment_and_save_multiple_calls(self):
        """Test that multiple calls to augment_and_save create unique filenames."""
        augmenter = SimpleAugmenter(seed=42)
        im = np.random.rand(100, 100)
        mask = np.random.randint(0, 2, (100, 100))
        patch_size = (50, 50)

        with tempfile.TemporaryDirectory() as tmpdir:
            patch_path = tmpdir
            patch_base_name = "test_patch"

            # Save multiple patches
            paths1 = augmenter.augment_and_save(
                im, mask, patch_path, patch_base_name, patch_size
            )
            paths2 = augmenter.augment_and_save(
                im, mask, patch_path, patch_base_name, patch_size
            )

            # Check that different files were created
            assert paths1[0] != paths2[0]
            assert paths1[1] != paths2[1]

            # Check that all files exist
            assert os.path.exists(paths1[0])
            assert os.path.exists(paths1[1])
            assert os.path.exists(paths2[0])
            assert os.path.exists(paths2[1])
