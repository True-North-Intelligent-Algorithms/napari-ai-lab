# StarDist Model-Axis Auto-Update Feature

## Overview
The StarDist segmenter now automatically updates the axis selection when you change the model preset. It also validates that the selected model is compatible with the current image.

## Model-Axis Mapping

| Model Preset | Recommended Axis | Image Type Required |
|-------------|------------------|---------------------|
| `2D_versatile_fluo` | YX | Grayscale 2D |
| `2D_versatile_he` | YXC | RGB/Multi-channel 2D |
| `3D_demo` | ZYX | Grayscale 3D |

## Behavior

### 1. Compatible Model Selection
When you select a model that is compatible with your current image:
- ✅ The axis combo **automatically updates** to the recommended axis
- ✅ You can proceed with segmentation immediately

**Example**:
- Image: Grayscale 2D (shape: 512×512)
- Change model to `2D_versatile_fluo`
- Result: Axis automatically set to "YX" ✅

### 2. Incompatible Model Selection
When you select a model that requires dimensions not present in your image:
- ⚠️ A **warning dialog** appears explaining the incompatibility
- 🔄 The model combo **reverts** to the previous selection
- The axis combo remains unchanged

**Example**:
- Image: Grayscale 2D (shape: 512×512, axis: YX)
- Change model to `2D_versatile_he` (requires RGB with YXC axis)
- Result: Warning shown, model reverts to previous selection ❌

Warning message:
```
⚠️ Model '2D_versatile_he' requires a channel dimension (axis: YXC)

Your current image does not have a channel dimension.
Please load an RGB/multi-channel image to use this model.

Keeping previous model: 2D_versatile_fluo
```

## Why Does This Happen?

The system filters available axes based on your **current image dimensions**:

1. **Image Analysis**: When you load an image, the system determines what dimensions it has:
   - Grayscale 2D: `(H, W)` → axis info "YX" → `has_c=False, has_z=False`
   - RGB 2D: `(H, W, 3)` → axis info "YXC" → `has_c=True, has_z=False`
   - Grayscale 3D: `(D, H, W)` → axis info "ZYX" → `has_c=False, has_z=True`

2. **Axis Filtering**: Only axes compatible with the image are shown in the combo:
   - If image has no C dimension → "YXC", "ZYXC" hidden
   - If image has no Z dimension → "ZYX", "ZYXC" hidden

3. **Model Validation**: When you select a model:
   - System checks if recommended axis is in the filtered list
   - If not available → warning + revert
   - If available → auto-update axis

## Testing

### Test Case 1: Fluo Model (Always Works with 2D)
```python
# Load any grayscale 2D image
image = np.random.rand(512, 512)

# Select 2D_versatile_fluo
# Expected: Axis automatically set to "YX" ✅
```

### Test Case 2: 3D Demo (Requires 3D Image)
```python
# Load grayscale 3D image
image = np.random.rand(32, 512, 512)

# Select 3D_demo
# Expected: Axis automatically set to "ZYX" ✅

# But with 2D image:
image = np.random.rand(512, 512)
# Select 3D_demo
# Expected: Warning shown, model reverted ⚠️
```

### Test Case 3: HE Model (Requires RGB)
```python
# Load RGB image
image = np.random.rand(512, 512, 3)

# Select 2D_versatile_he
# Expected: Axis automatically set to "YXC" ✅

# But with grayscale image:
image = np.random.rand(512, 512)
# Select 2D_versatile_he
# Expected: Warning shown, model reverted ⚠️
```

## Implementation Details

### Files Modified
1. **base_nd_app.py** - `_on_segmenter_parameters_changed()`
   - Added model validation logic
   - Added automatic axis update
   - Added revert on incompatibility

2. **StardistSegmenter.py**
   - Added `MODEL_AXIS_MAP` dictionary
   - Added `get_recommended_axis()` method
   - Added `__setattr__` override for debug output

### Key Functions
- `get_recommended_axis()` - Returns axis for current model preset
- `get_supported_axes_from_shape()` - Filters axes based on image
- `set_parameter()` - Used to revert combo to previous value
- `set_selected_axis()` - Updates axis combo programmatically

## Debug Output

The system prints helpful debug messages:

```
Model changed to 2D_versatile_he, recommended axis: YXC
❌ Cannot use model '2D_versatile_he': requires YXC, but only ['YX'] available
```

Or on success:
```
Model changed to 2D_versatile_fluo, recommended axis: YX
Synced segmenter instance with new parameters
```
