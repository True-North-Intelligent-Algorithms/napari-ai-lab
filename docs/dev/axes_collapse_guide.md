# Axis Collapsing for Annotations and Predictions

## Overview

Simple, flexible system for handling different dimensionality between images and their annotations/predictions.

**Key Principle**: Caller tells the model which axes to collapse. No need to anticipate all future scenarios.

## Common Use Case: Collapse Channels

### Problem
- Image: ZYXC (10, 512, 512, 3) - 3D image with 3 channels
- Want: ZYX annotations (10, 512, 512) - 3D labels without channels

### Solution
```python
# Load annotations with collapsed C axis
labels = model.load_existing_annotations(
    image_shape=(10, 512, 512, 3),
    image_index=0,
    axes_to_collapse="C"  # <-- Simply specify which axis to collapse
)
# Returns: (10, 512, 512) array

# Save annotations (same parameter for consistency)
model.save_annotations(
    labels_array=labels,  # Shape: (10, 512, 512)
    image_index=0,
    axes_to_collapse="C"  # <-- Document what was collapsed
)
```

### Works for Predictions Too
```python
# Same logic for predictions
predictions = model.load_existing_predictions(
    image_shape=(10, 512, 512, 3),
    image_index=0,
    axes_to_collapse="C"
)
# Returns: (10, 512, 512) array

model.save_predictions(
    predictions_array=predictions,
    image_index=0,
    axes_to_collapse="C"
)
```

## Future Flexibility

### Collapse Multiple Axes
```python
# Image: TZYXC (5, 10, 512, 512, 3) - time series with channels
# Want: ZYX (10, 512, 512) - just spatial dimensions

labels = model.load_existing_annotations(
    image_shape=(5, 10, 512, 512, 3),
    image_index=0,
    axes_to_collapse=["T", "C"]  # <-- List of axes
)
# Returns: (10, 512, 512) array
```

### Other Scenarios (Future)
```python
# Collapse time only
axes_to_collapse="T"  # TYXC -> YXC

# Collapse depth
axes_to_collapse="Z"  # ZYXC -> YXC

# Any combination
axes_to_collapse=["T", "Z", "C"]  # Whatever makes sense
```

## How It Works Internally

```python
def _compute_annotation_shape(image_shape, axes_to_collapse):
    """
    Simple algorithm:
    1. Look at image axis types (from self.axis_types: "ZYXC")
    2. Remove axes specified in axes_to_collapse
    3. Return corresponding dimensions from image_shape

    Example:
        axis_types = "ZYXC"
        image_shape = (10, 512, 512, 3)
        axes_to_collapse = "C"

        Process:
        - Z: keep dim 0 -> 10
        - Y: keep dim 1 -> 512
        - X: keep dim 2 -> 512
        - C: collapse (skip dim 3)

        Result: (10, 512, 512)
    """
```

## Integration Points

### Where to Pass `axes_to_collapse`

Wherever you load or save annotations/predictions from the model:

```python
# In segmenter classes
class MySegmenter:
    def segment(self, image):
        # If segmenter produces ZYX output from ZYXC input
        labels = self.model.load_existing_annotations(
            image_shape=image.shape,
            image_index=self.current_index,
            axes_to_collapse="C"  # <-- Segmenter decides
        )

        # Do segmentation...
        result = self.run_segmentation(image)  # Returns ZYX

        # Save with same parameter
        self.model.save_annotations(
            labels_array=result,
            image_index=self.current_index,
            axes_to_collapse="C"
        )
```

## Benefits

1. **Simple**: Just a string or list parameter
2. **Flexible**: Works for any axis combination
3. **Future-proof**: No code changes needed for new scenarios
4. **Explicit**: Caller knows their requirements
5. **Consistent**: Same parameter for load and save

## No Special Cases

The system doesn't need to know about:
- What each axis means
- Which operations collapse which axes
- Model-specific behaviors

It just does what you tell it: "collapse these axes, keep the rest."
