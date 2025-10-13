# Interactive Tests for Napari AI Lab Segmenters

This directory contains interactive test scripts for manually testing
segmentation functionality. These are **NOT unit tests** and will **NOT** be
picked up by pytest.

## Purpose

Files in this directory are meant to be run manually for:
- Interactive testing of segmenters
- Visual validation of segmentation results
- Manual testing with real data
- Development and debugging workflows
- **Demonstrating segmenters work independently of napari**

## Key Design Philosophy

These tests use **matplotlib** instead of napari to prove that the segmentation
algorithms work independently of any specific visualization framework. This ensures:

- ✅ **Framework Independence**: Segmenters work without napari
- ✅ **Portability**: Users can integrate with any visualization library
- ✅ **Lightweight Testing**: No GUI dependencies for basic validation
- ✅ **Universal Compatibility**: Works in any Python environment

## Usage

```bash
# Run individual tests
python Interactive_Otsu.py
python Interactive_Threshold.py
python Interactive_Cellpose.py      # Requires: pip install cellpose
python Interactive_Appose_Cellpose.py  # Tests appose execution string generation

# Or run from project root
python src/napari_ai_lab/interactive_tests/Interactive_Otsu.py
python src/napari_ai_lab/interactive_tests/Interactive_Cellpose.py
python src/napari_ai_lab/interactive_tests/Interactive_Appose_Cellpose.py
```

## Dependencies

These tests only require:
- `matplotlib` (for visualization)
- `scikit-image` (for test images and utilities)
- The segmenter classes themselves

**No napari dependency required!**

## Test Structure

Each test follows this pattern:
1. Load test data (using `skimage.data`)
2. Create segmenter instance
3. Display segmenter information
4. Perform segmentation
5. Show results with matplotlib
6. Print summary statistics
