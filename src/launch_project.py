"""Open ND AI Lab on a project folder, from the command line.

    python src/launch_project.py FOLDER [--profile 2d-detect]

The same settings as docs/launch.md -- sequence viewer, colour collapsed,
NYXC -- which suit a folder of photos that may differ in size.
"""

import argparse

import napari

from napari_ai_lab.apps.nd_ai_lab_launcher import launch_nd_ai_lab
from napari_ai_lab.apps.profiles import list_profiles

parser = argparse.ArgumentParser(description="Open ND AI Lab on a folder")
parser.add_argument("folder", help="project folder: a folder of images")
parser.add_argument(
    "--profile",
    default="2d-instance",
    choices=list_profiles(),
    help="what AI Lab offers (default: %(default)s)",
)
args = parser.parse_args()

viewer = napari.Viewer()
launch_nd_ai_lab(
    viewer,
    args.folder,
    viewer_type="sequence",
    axes_to_collapse="C",
    axis_types="NYXC",
    profile=args.profile,
)
print(f"ND AI Lab on {args.folder}, profile {args.profile}")
napari.run()
