# Progress Logger for Segmentation UI

## Overview
Added a visual progress logger to the Automatic Segmentation UI that displays real-time progress when segmenting images.

## Changes Made

### 1. Added Progress Logger Widget to UI
**File**: `src/napari_ai_lab/apps/nd_easy_segment.py`

```python
# In _setup_ui() method, inside auto_controls_group:
self.segment_progress_logger = QtProgressLogger()
auto_layout.addWidget(self.segment_progress_logger.get_widget())
```

The progress logger widget is now visible in the "Automatic Segmentation" group, below the buttons.

### 2. Updated `_on_segment_current()` Method
Shows progress when segmenting the current image:

```python
def _on_segment_current(self):
    # Clear previous logs
    self.segment_progress_logger.clear()
    self.segment_progress_logger.log_info("Segmenting current image...")

    # ... segmentation code ...

    # Update progress: 1 out of 2 steps
    self.segment_progress_logger.update_progress(1, 2, "Processing...")
    self._segment_nd_slice(current_step=self.viewer.dims.current_step)

    # Update progress: 2 out of 2 steps (100%)
    self.segment_progress_logger.update_progress(2, 2, "✅ Complete")
    self.segment_progress_logger.log_info("✅ Segmentation complete")
```

### 3. Updated `_on_segment_all()` Method
Shows detailed progress when segmenting all slices:

```python
def _on_segment_all(self):
    # Clear and initialize
    self.segment_progress_logger.clear()
    self.segment_progress_logger.log_info("Starting segmentation of all slices...")

    # Log configuration
    self.segment_progress_logger.log_info(f"Selected axis: {selected_axis}")
    self.segment_progress_logger.log_info(f"Image shape: {image_shape}")
    self.segment_progress_logger.log_info(f"Total slices to segment: {total_slices}")

    # Update progress for each slice
    for idx, non_spatial_indices in enumerate(...):
        # Pass current (idx+1), total (total_slices), and message
        self.segment_progress_logger.update_progress(
            idx + 1,
            total_slices,
            f"Processing slice {idx + 1}/{total_slices}"
        )

        # Segment the slice
        self._segment_nd_slice(current_step=current_step)

    # Completion message
    self.segment_progress_logger.log_info(f"✅ Completed segmentation of all {total_slices} slices")
```

## UI Appearance

The progress logger appears in the "Automatic Segmentation" group:

```
┌─────────────────────────────────────────────┐
│   Automatic Segmentation                    │
├─────────────────────────────────────────────┤
│  [Segment Current Image]                    │
│  [Segment All Images]                       │
│  [Train]                                    │
│                                             │
│  ┌───────────────────────────────────────┐  │
│  │ Progress Logger                       │  │
│  ├───────────────────────────────────────┤  │
│  │ Starting segmentation of all slices...│  │
│  │ Selected axis: YX                     │  │
│  │ Image shape: (10, 512, 512)          │  │
│  │ Total slices to segment: 10           │  │
│  │ ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░ 50%            │  │
│  │ Processing slice 5/10                 │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## Progress Logger API

The `QtProgressLogger` provides these methods:

### `clear()`
Clears all previous log messages and resets the progress bar.

```python
self.segment_progress_logger.clear()
```

### `log_info(message: str)`
Adds an informational message to the log.

```python
self.segment_progress_logger.log_info("Starting segmentation...")
```

### `update_progress(current: int, total: int, message: str = "")`
Updates the progress bar based on current/total ratio and displays a status message.

**Important**: Pass `current` and `total` as integers, not a percentage!

```python
# Correct ✅
self.segment_progress_logger.update_progress(5, 10, "Processing slice 5/10")

# Wrong ❌
self.segment_progress_logger.update_progress(50, "Processing...")  # 50% is not valid!
```

The method automatically calculates the percentage: `(current / total) * 100`

### `get_widget()`
Returns the Qt widget to add to the layout.

```python
layout.addWidget(self.segment_progress_logger.get_widget())
```

## Example Progress Messages

### Segment Current Image:
```
Segmenting current image...
Processing...
✅ Segmentation complete
```

### Segment All Images (10 slices):
```
Starting segmentation of all slices...
Selected axis: YX
Image shape: (10, 512, 512)
Total slices to segment: 10
[Progress: 10%] Processing slice 1/10
[Progress: 20%] Processing slice 2/10
[Progress: 30%] Processing slice 3/10
...
[Progress: 100%] Processing slice 10/10
✅ Completed segmentation of all 10 slices
```

## Benefits

1. **Visual Feedback**: Users can see segmentation is running (not frozen)
2. **Progress Tracking**: Shows exactly which slice is being processed
3. **Time Estimation**: Progress percentage helps estimate remaining time
4. **Configuration Logging**: Shows axis mode and image shape for debugging
5. **Completion Confirmation**: Clear success message when done

## Consistency with Training

The segmentation progress logger follows the same pattern as the training progress logger already in use:

- Training logger: Shows in Training tab for model training
- Segmentation logger: Shows in Automatic Segmentation group for inference

Both use the same `QtProgressLogger` class for consistent UX.
