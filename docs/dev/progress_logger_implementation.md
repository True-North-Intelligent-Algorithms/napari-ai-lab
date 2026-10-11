# Progress Logger Implementation - Phase 1

## Overview

Implemented a generic progress tracking and logging system for `napari-ai-lab` that allows operations like `generate_patches()` to report progress and log messages through different backends (Napari notifications, console, etc.) without tight coupling.

## What Was Implemented

### 1. Core Progress Logger System

**File**: `src/napari_ai_lab/utilities/progress_logger.py`

Three components:

#### `ProgressLogger` (Protocol)
- Defines generic interface for progress tracking and logging
- Methods:
  - `update_progress(current, total, message)` - Update progress indicator
  - `log_info(message)` - Log informational message
  - `log_warning(message)` - Log warning message
  - `log_error(message)` - Log error message

#### `NapariProgressLogger` (Implementation)
- Uses Napari's notification system and status bar
- Falls back to console printing if viewer unavailable
- Methods:
  - `update_progress()` → Updates `viewer.window._status_bar`
  - `log_info()` → `viewer.window.notification_manager.show_info()`
  - `log_warning()` → `viewer.window.notification_manager.show_warning()`
  - `log_error()` → `viewer.window.notification_manager.show_error()`

#### `ConsoleProgressLogger` (Fallback)
- Simple console-based implementation using print statements
- Useful for testing, CLI usage, or when no viewer available

### 2. Updated `generate_patches()` Method

**File**: `src/napari_ai_lab/models/image_data_model.py`

**Changes**:
- Added `progress_logger=None` parameter (optional)
- Uses logger for progress updates if provided, otherwise falls back to print
- Progress tracking:
  ```python
  for i in range(self.num_patches):
      # ... generate patch ...
      if progress_logger:
          progress_logger.update_progress(i + 1, self.num_patches, "Generating patches")
      else:
          print(f"  Created {i+1}/{self.num_patches} patches")
  ```

**Benefits**:
- ✅ Backward compatible (works without logger)
- ✅ Generic interface (not tied to Napari)
- ✅ Clean separation of concerns

### 3. Updated `nd_easy_augment` Widget

**File**: `src/napari_ai_lab/apps/nd_easy_augment.py`

**Changes**:
- Imports `NapariProgressLogger`
- Creates logger in `_on_generate_patches()`:
  ```python
  progress_logger = NapariProgressLogger(self.viewer)
  patches_dir = self.image_data_model.generate_patches(
      image=image,
      annotations=annotations,
      axis="yx",
      axes_string="YX",
      progress_logger=progress_logger,  # ✅ Pass logger
  )
  ```
- Error handling uses logger for notifications

### 4. Utilities Module Export

**File**: `src/napari_ai_lab/utilities/__init__.py` (NEW)

Exports:
- `ProgressLogger` (Protocol)
- `NapariProgressLogger` (Napari implementation)
- `ConsoleProgressLogger` (Console fallback)

### 5. Test Suite

**File**: `tests/test_progress_logger.py`

Three test functions:
- `test_console_logger()` - Tests console fallback
- `test_napari_logger_without_viewer()` - Tests Napari logger without viewer (fallback mode)
- `test_napari_logger_with_viewer()` - Interactive test with real napari window

## How It Works

### Flow for Patch Generation

```
User clicks "Generate Patches" in nd_easy_augment
    ↓
nd_easy_augment._on_generate_patches()
    ↓
Creates NapariProgressLogger(viewer)
    ↓
Calls image_data_model.generate_patches(..., progress_logger=logger)
    ↓
generate_patches() uses logger for:
    - log_info("🎨 Generating patches...")
    - update_progress(i, total, "Generating patches") in loop
    - log_info("📝 Writing info.json...")
    - log_info("✅ Created N patches...")
    ↓
User sees:
    - Status bar updates: "Generating patches (5/100)"
    - Toast notifications: "✅ Created 100 patches..."
```

### Napari Integration Points

| Feature | Napari API | Our Usage |
|---------|-----------|-----------|
| **Notifications** | `viewer.window.notification_manager.show_info/warning/error()` | Toast popups for events |
| **Status Bar** | `viewer.window._status_bar.showMessage()` | Progress updates |
| **Fallback** | N/A | Print to console if viewer unavailable |

## Testing

Run the test suite:
```bash
cd /home/bnorthan/code/i2k/tnia/napari-ai-lab
python tests/test_progress_logger.py
```

Expected output:
1. Console logger tests (print to terminal)
2. Napari logger without viewer (print to terminal)
3. Napari logger with viewer (opens napari window with notifications)

## Usage Examples

### Example 1: Using in nd_easy_augment (Already Implemented)

```python
from napari_ai_lab.utilities import NapariProgressLogger

# In widget class:
progress_logger = NapariProgressLogger(self.viewer)
patches_dir = self.image_data_model.generate_patches(
    image=image,
    annotations=annotations,
    progress_logger=progress_logger,
)
```

### Example 2: Using Console Logger (No Viewer)

```python
from napari_ai_lab.utilities import ConsoleProgressLogger

logger = ConsoleProgressLogger()
patches_dir = model.generate_patches(
    image=image,
    annotations=annotations,
    progress_logger=logger,
)
```

### Example 3: Custom Logger Implementation

```python
class CustomLogger:
    """Custom logger that writes to a file."""

    def update_progress(self, current, total, message=""):
        with open("progress.log", "a") as f:
            f.write(f"{message} {current}/{total}\n")

    def log_info(self, message):
        with open("progress.log", "a") as f:
            f.write(f"INFO: {message}\n")

    # ... etc ...

# Use it:
logger = CustomLogger()
patches_dir = model.generate_patches(..., progress_logger=logger)
```

## Architecture Benefits

### 1. **Decoupling**
- `image_data_model` doesn't import napari
- Can be used in non-napari contexts (CLI, notebooks, batch scripts)

### 2. **Testability**
- Easy to mock logger in unit tests
- Can verify logging calls without actual napari viewer

### 3. **Extensibility**
- Future: Add `TqdmProgressLogger` for notebooks
- Future: Add `QtProgressLogger` for custom dialogs
- Future: Add centralized logging to files

### 4. **Backward Compatibility**
- Old code still works (logger is optional)
- Gradual migration path

## Future Enhancements (Phase 2)

### Potential Additions:

1. **TqdmProgressLogger** (for Jupyter notebooks)
   ```python
   class TqdmProgressLogger:
       def __init__(self):
           self.pbar = None

       def update_progress(self, current, total, message=""):
           if self.pbar is None:
               from tqdm.auto import tqdm
               self.pbar = tqdm(total=total, desc=message)
           self.pbar.update(1)
   ```

2. **Qt Progress Dialog**
   ```python
   class QtProgressLogger:
       def __init__(self, parent=None):
           from qtpy.QtWidgets import QProgressDialog
           self.dialog = QProgressDialog(parent)

       def update_progress(self, current, total, message=""):
           self.dialog.setValue(current)
           self.dialog.setMaximum(total)
   ```

3. **Centralized Logging**
   - Add file logging support
   - Add log level filtering (DEBUG, INFO, WARNING, ERROR)
   - Add timestamps

4. **Context Manager Support**
   ```python
   with NapariProgressLogger(viewer) as logger:
       model.generate_patches(..., progress_logger=logger)
       # Auto-cleanup on exit
   ```

## Files Changed

1. ✅ **NEW**: `src/napari_ai_lab/utilities/progress_logger.py` (143 lines)
2. ✅ **NEW**: `src/napari_ai_lab/utilities/__init__.py` (17 lines)
3. ✅ **NEW**: `tests/test_progress_logger.py` (110 lines)
4. ✅ **MODIFIED**: `src/napari_ai_lab/models/image_data_model.py`
   - Added `progress_logger` parameter to `generate_patches()`
   - Updated progress tracking logic (44 lines changed)
5. ✅ **MODIFIED**: `src/napari_ai_lab/apps/nd_easy_augment.py`
   - Import `NapariProgressLogger`
   - Use logger in `_on_generate_patches()` (15 lines changed)

## Summary

Phase 1 implementation provides:
- ✅ Generic progress/logging interface (`ProgressLogger` protocol)
- ✅ Napari-specific implementation with notifications
- ✅ Console fallback for non-Napari usage
- ✅ Integration with `generate_patches()` method
- ✅ Working in `nd_easy_augment` widget
- ✅ Test suite for verification
- ✅ Backward compatible (optional parameter)
- ✅ Extensible for future backends (tqdm, Qt, file logging)

**Ready for production use!** 🎉
