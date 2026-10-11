# Launch

## From napari

- **Plugins → Napari AI Lab → 2D Instance AI Lab**
- **Open Project**
  - **Browse...** to your folder
  - **Viewer type:** sequence
  - RGB images: **Axes to collapse** `C`, **Axis types** `NYXC`
- **OK**

> **SCREENSHOT TODO:** Open Project dialog

## From Python

Profiles are used to switch between different option sets.  For example 2d-instance-skop, will only populate the options relevant to 2d instance segmentation and choose the scikit-ops versions.

```python
import napari
from napari_ai_lab.apps.nd_ai_lab_launcher import launch_nd_ai_lab

from napari_ai_lab.apps.profiles import list_profiles, get_profile

list_profiles()
```

```python
viewer = napari.Viewer()

ai_lab, sequence_viewer, model = launch_nd_ai_lab(
    viewer,
    str(project_path),
    viewer_type="sequence",
    axes_to_collapse="C",
    axis_types="NYXC",
    profile="2d-instance-skop",
)
```

## Why those arguments

| argument | why |
|---|---|
| `viewer_type="sequence"` | all three sets hold images of different shapes, so they cannot be one stacked array |
| `axes_to_collapse="C"` | colour is not an axis to annotate along; labels are 2D per image |
| `axis_types="NYXC"` | N images, each Y by X with a colour axis |

A project of equal-sized images could use `viewer_type="stacked"` instead.

## Using it

The **Sequence Viewer** at the bottom moves between images. The **AI Lab**
dock on the right has Label, Augment and Segment.

![AI Lab](images/ai_lab.png)
