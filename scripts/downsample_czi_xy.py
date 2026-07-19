r"""
Downsample a CZI file by 2x in X and Y only (Z unchanged).

Reads 'Test lightsheet.czi' from tests/test_images/vessels_large/,
downsamples each Z plane using 2x2 block averaging, doubles the
X and Y physical scale values in the metadata, and writes a new CZI
to tests/test_images/vessels_ds2/.

Run with the microsam_cellposesam_czi pixi Python:
  pixi\microsam_cellposesam_czi\.pixi\envs\default\python.exe scripts\downsample_czi_xy.py
"""

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from czifile import CziFile
from pylibCZIrw import czi as pyczi
from skimage.transform import downscale_local_mean

# ── paths ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
INPUT = (
    REPO / "tests" / "test_images" / "vessels_large" / "Test lightsheet.czi"
)
OUT_DIR = REPO / "tests" / "test_images" / "vessels_ds2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUT_DIR / "Test lightsheet ds2.czi"

# ── read ─────────────────────────────────────────────────────────────────────
print(f"Reading {INPUT}  ({INPUT.stat().st_size / 1e9:.2f} GB) …")
with CziFile(str(INPUT)) as czi:
    axes = czi.axes  # e.g. 'VBHIRSTCZYX0'
    shape = czi.shape
    data = czi.asarray()  # numpy array with all axes, dtype uint16/float32
    metadata_xml = czi.metadata()

print(f"  axes : {axes}")
print(f"  shape: {shape}")

# ── find Y and X axis positions ───────────────────────────────────────────────
y_idx = axes.index("Y")
x_idx = axes.index("X")
z_idx = axes.index("Z") if "Z" in axes else None
c_idx = axes.index("C") if "C" in axes else None
print(f"  axis positions  Z={z_idx}  Y={y_idx}  X={x_idx}  C={c_idx}")

# ── downsample Y and X by 2 ───────────────────────────────────────────────────
# Build downscale factor tuple: 1 everywhere except 2 at Y and X.
factors = [1] * len(axes)
factors[y_idx] = 2
factors[x_idx] = 2
factors = tuple(factors)

print(f"Downsampling (factors {factors}) …")
ds_data = downscale_local_mean(data.astype(np.float32), factors).astype(
    data.dtype
)
print(f"  downsampled shape: {ds_data.shape}")

# ── update metadata: double X and Y Distance/Value ────────────────────────────
root = ET.fromstring(metadata_xml)
changed = []
for axis_id in ("X", "Y"):
    node = root.find(f".//Scaling/Items/Distance[@Id='{axis_id}']/Value")
    if node is not None and node.text:
        old_val = float(node.text)
        new_val = old_val * 2.0
        node.text = repr(new_val)  # keeps full float precision
        changed.append(f"{axis_id}: {old_val:.4e} → {new_val:.4e} m")

print("  Updated scale:", "  ".join(changed) if changed else "(none found)")


# Extract the raw Distance/Value from the (already-doubled) XML.
# pylibCZIrw.write_metadata() claims to take µm but stores values verbatim in
# the XML without any unit conversion.  The original CZI stores in meters and
# image_data_model.py reads back assuming meters (× 1e6 → µm).  So we pass the
# raw meter values here so the round-trip works correctly.
def _scale_raw(et_root, axis_id):
    node = et_root.find(f".//Scaling/Items/Distance[@Id='{axis_id}']/Value")
    if node is not None and node.text:
        try:
            return float(node.text)  # meters, stored verbatim by pylibCZIrw
        except ValueError:
            pass
    return 0.0


scale_x_raw = _scale_raw(root, "X")
scale_y_raw = _scale_raw(root, "Y")
scale_z_raw = _scale_raw(root, "Z")

# ── write new CZI plane by plane ─────────────────────────────────────────────
# pylibCZIrw.CziWriter.write() expects a 2D (H, W) or (H, W, C) uint8/uint16
# array per call.  We iterate over all leading dims (S, T, C, Z …) and write
# each Y-X plane individually.

# Identify how many Z and C slices exist.
n_z = ds_data.shape[z_idx] if z_idx is not None else 1
n_c = ds_data.shape[c_idx] if c_idx is not None else 1

# Squeeze the sample axis ('0' in VBHIRSTCZYX0) which is always 1 for
# single-sample (grayscale) CZIs, so the Y-X plane is 2D.
sample_idx = axes.index("0") if "0" in axes else None

print(f"Writing {OUTPUT} …")
print(f"  Z planes: {n_z}  channels: {n_c}")

if OUTPUT.exists():
    OUTPUT.unlink()

with pyczi.create_czi(str(OUTPUT), exist_ok=True) as writer:
    # write_metadata stores values verbatim — pass raw meter values so they
    # match the original CZI format (Distance/Value in meters).
    writer.write_metadata(
        scale_x=scale_x_raw, scale_y=scale_y_raw, scale_z=scale_z_raw
    )

    for z in range(n_z):
        for c in range(n_c):
            # Build index tuple: use slice(None) for Y and X, scalar ints for everything else.
            idx = []
            for i, ax in enumerate(axes):
                if ax == "Y" or ax == "X":
                    idx.append(slice(None))
                elif ax == "Z":
                    idx.append(z)
                elif ax == "C":
                    idx.append(c)
                else:
                    idx.append(
                        0
                    )  # all other singleton axes (V, B, H, I, R, S, T, sample)

            plane_data = ds_data[tuple(idx)]  # shape: (Y, X) or (Y, X, 1)

            # Drop trailing sample dim if present
            if plane_data.ndim == 3 and plane_data.shape[2] == 1:
                plane_data = plane_data[:, :, 0]

            # pylibCZIrw expects shape (H, W) with uint16 or float32
            plane_data = np.ascontiguousarray(plane_data)

            writer.write(
                data=plane_data,
                plane={"C": c, "Z": z},
                location=(0, 0),
                scene=0,
            )

        if z % 50 == 0 or z == n_z - 1:
            print(f"  wrote Z {z + 1}/{n_z}")

print(f"\nDone → {OUTPUT}  ({OUTPUT.stat().st_size / 1e9:.2f} GB)")
