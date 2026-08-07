# Progress Logger API Fix

## The Problem

The initial implementation had an incorrect understanding of the `update_progress()` API signature:

```python
# ❌ WRONG - What was implemented initially
self.segment_progress_logger.update_progress(50, "Processing...")
# This tried to divide by the string "Processing..." causing a TypeError
```

**Error Message:**
```python
TypeError: unsupported operand type(s) for /: 'int' and 'str'
    if total > 0:
        total = 'Processing...'
        percentage = int((current / total) * 100)  # Can't divide by string!
```

## The Correct API

The `QtProgressLogger.update_progress()` method signature is:

```python
def update_progress(self, current: int, total: int, message: str = "")
```

**Parameters:**
- `current`: Current item/step number (e.g., 5)
- `total`: Total items/steps (e.g., 10)
- `message`: Optional status message

**The method calculates the percentage automatically:** `(current / total) * 100`

## The Fix

### For `_on_segment_current()`:

**Before (Wrong):**
```python
self.segment_progress_logger.update_progress(50, "Processing...")
self._segment_nd_slice(current_step=self.viewer.dims.current_step)
self.segment_progress_logger.update_progress(100, "✅ Complete")
```

**After (Correct):**
```python
self.segment_progress_logger.update_progress(1, 2, "Processing...")
self._segment_nd_slice(current_step=self.viewer.dims.current_step)
self.segment_progress_logger.update_progress(2, 2, "✅ Complete")
```

- `1, 2` = 50% (step 1 of 2)
- `2, 2` = 100% (step 2 of 2)

### For `_on_segment_all()`:

**Before (Wrong):**
```python
for idx, non_spatial_indices in enumerate(...):
    progress_percent = int((idx + 1) / total_slices * 100)
    self.segment_progress_logger.update_progress(
        progress_percent,  # ❌ Passing percentage (e.g., 50)
        f"Processing slice {idx + 1}/{total_slices}"
    )
```

**After (Correct):**
```python
for idx, non_spatial_indices in enumerate(...):
    self.segment_progress_logger.update_progress(
        idx + 1,        # ✅ Current slice number (e.g., 5)
        total_slices,   # ✅ Total slices (e.g., 10)
        f"Processing slice {idx + 1}/{total_slices}"
    )
```

## Examples

### Example 1: Segmenting 10 slices

```python
total_slices = 10

# Slice 1
update_progress(1, 10, "Processing slice 1/10")   # Shows 10%

# Slice 5
update_progress(5, 10, "Processing slice 5/10")   # Shows 50%

# Slice 10
update_progress(10, 10, "Processing slice 10/10") # Shows 100%
```

### Example 2: Simple 2-step process

```python
# Step 1 (start)
update_progress(0, 2, "Starting...")    # Shows 0%

# Step 2 (halfway)
update_progress(1, 2, "Processing...")  # Shows 50%

# Step 3 (complete)
update_progress(2, 2, "Complete!")      # Shows 100%
```

## Key Takeaway

**Never calculate percentages manually!** Let the progress logger do it:

```python
# ✅ GOOD: Pass current and total, let it calculate percentage
update_progress(current_item, total_items, message)

# ❌ BAD: Calculate percentage yourself
percentage = int((current_item / total_items) * 100)
update_progress(percentage, message)  # Wrong signature!
```

## Related Files Changed

1. `/src/napari_ai_lab/apps/nd_easy_segment.py`
   - Fixed `_on_segment_current()` method
   - Fixed `_on_segment_all()` method

2. `/docs/segmentation_progress_logger.md`
   - Updated documentation with correct API usage
   - Added warning about common mistake

3. `/docs/progress_logger_fix.md` (this file)
   - Detailed explanation of the fix
