# CellCast StarDist Segmenter Integration

## Overview
Added a new global segmenter for CellCast StarDist that uses the `predict_versatile_fluo` model for automatic cell segmentation in fluorescence microscopy images.

## Files Created/Modified

### New Files:
1. **`src/napari_ai_lab/Segmenters/GlobalSegmenters/CellCastStardistSegmenter.py`**
   - Simple segmenter with no parameters (for now)
   - Uses `cellcast.models.stardist_2d.predict_versatile_fluo()`
   - Automatic GPU detection and usage
   - Multi-channel to grayscale conversion (if needed)
   - Automatic image normalization

2. **`tests/test_cellcast_segmenter.py`**
   - Basic registration test
   - Simple prediction test with synthetic image

### Modified Files:
1. **`src/napari_ai_lab/Segmenters/GlobalSegmenters/__init__.py`**
   - Added `"CellCastStardistSegmenter": ".CellCastStardistSegmenter"` to `_OPTIONAL_SEGMENTERS`

2. **`src/launch_nd_ai_lab.py`**
   - Added import: `CellCastStardistSegmenter`
   - Added registration: `if CellCastStardistSegmenter is not None: CellCastStardistSegmenter.register()`

3. **`src/launch_nd_easy_segment.py`**
   - Added import: `CellCastStardistSegmenter`
   - Added registration: `CellCastStardistSegmenter.register()`
   - Fixed pre-existing syntax error (missing colon on line 81)

## Usage

### From the UI:
1. Launch napari-ai-lab
2. Go to Segment tab
3. Select "CellCastStardistSegmenter" from the segmenter dropdown
4. Click "Segment" button

### Programmatic:
```python
from napari_ai_lab.Segmenters.GlobalSegmenters import CellCastStardistSegmenter

# Create segmenter
segmenter = CellCastStardistSegmenter()

# Predict on image
labels = segmenter.predict(image)
```

## Features
- ✅ No parameters required (simple to use)
- ✅ Automatic GPU detection (falls back to CPU if no GPU)
- ✅ Handles RGB/multi-channel images (converts to grayscale)
- ✅ Automatic image normalization
- ✅ Follows same pattern as other global segmenters
- ✅ Graceful degradation if cellcast not installed

## Installation
To use this segmenter, install cellcast:
```bash
pip install cellcast
```

Or add to your pixi.toml:
```toml
[pypi-dependencies]
cellcast = "*"
```

## Future Enhancements
Future versions could add parameters like:
- `gpu: bool` - Manual GPU on/off control
- `threshold: float` - Probability threshold
- `normalize: bool` - Control normalization
- Other CellCast model options if available
