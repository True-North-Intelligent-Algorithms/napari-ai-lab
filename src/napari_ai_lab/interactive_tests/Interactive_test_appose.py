"""
Interactive Appose Test — headless (no napari GUI).

Runs three global segmenters entirely via ``execute_appose`` against
per-segmenter pixi environments shipped with this repo, then plots the
input image and each mask overlaid in a 2x2 matplotlib figure.

Environments (in this repo):
    pixi/microsam_cellposesam_czi   → CellposeSAM (cpsam) + MicroSAM
    pixi/stardist                   → StarDist 2D

Note:
    This script does not need any of the segmentation packages installed
    locally — the segmenter classes only produce execution strings, and
    all real work happens in the pixi environments listed above. The
    only local requirement is ``appose`` (plus numpy, scikit-image, and
    matplotlib for the plot).

Test image:
    ``skimage.data.hubble_deep_field()`` — RGB Hubble Deep Field image.
    We sum the channels to a single-channel grayscale before segmenting.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import data

from napari_ai_lab.Segmenters.execute_appose import execute_appose
from napari_ai_lab.Segmenters.GlobalSegmenters.CellposeSegmenter import (
    CellposeSegmenter,
)
from napari_ai_lab.Segmenters.GlobalSegmenters.StardistSegmenter import (
    StardistSegmenter,
)

# ---------------------------------------------------------------------------
# Pixi environments in this repo
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
CELLPOSE_ENV = REPO_ROOT / "pixi" / "microsam_cellposesam_czi"
STARDIST_ENV = REPO_ROOT / "pixi" / "stardist"


# ---------------------------------------------------------------------------
# Test-image selection
# ---------------------------------------------------------------------------
# "hubble" — Hubble deep field (RGB, 872x1024). Slow but pretty.
# "coins"  — skimage.data.coins() (grayscale, 303x384). Small + quick.
IMAGE = "hubble"

# Per-image cellpose diameter hint (pixels).
_DIAMETER = {"hubble": 30.0, "coins": 30.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_test_image(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (display, gray_uint8) for the chosen test image.

    ``display`` is what we plot in the top-left panel (RGB for hubble,
    grayscale for coins).  ``gray_uint8`` is what we feed to every
    segmenter.
    """
    if name == "hubble":
        rgb = data.hubble_deep_field()
        gray = rgb.astype(np.float32).sum(axis=-1)
        gray = (gray - gray.min()) / max(gray.max() - gray.min(), 1e-6)
        gray = (gray * 255.0).astype(np.uint8)
        return rgb, gray
    if name == "coins":
        gray = data.coins()
        return gray, gray.astype(np.uint8)
    raise ValueError(f"Unknown IMAGE {name!r} (expected 'hubble' or 'coins')")


def _overlay(ax, image_gray, mask, title):
    ax.imshow(image_gray, cmap="gray")
    if mask is not None and np.any(mask):
        overlay = np.ma.masked_where(mask == 0, mask)
        ax.imshow(overlay, cmap="nipy_spectral", alpha=0.5)
    ax.set_title(title)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("🧪 Interactive Appose Test")
    print("=" * 60)
    print(f"CellposeSAM env : {CELLPOSE_ENV}")
    print(f"Cellpose3  env  : {CELLPOSE_ENV}  (same env, model=cyto3)")
    print(f"StarDist env    : {STARDIST_ENV}")
    print("=" * 60)

    rgb, gray = load_test_image(IMAGE)
    print(f"📸 {IMAGE}: display={rgb.shape}, gray={gray.shape} {gray.dtype}")

    # -- Segmenters -----------------------------------------------------

    diameter = _DIAMETER[IMAGE]

    cpsam = CellposeSegmenter()
    cpsam.inference_model_name = "cpsam"
    cpsam.diameter = diameter

    cp3 = CellposeSegmenter()
    cp3.inference_model_name = "cyto3"
    cp3.diameter = diameter

    stardist = StardistSegmenter()
    stardist.inference_model_name = "2D_versatile_fluo"
    stardist.prob_thresh = 0.5
    stardist.nms_thresh = 0.4

    # -- Run remotely via appose ---------------------------------------

    print("▶️  Cellpose3 (cyto3) ...")
    m_cp3 = execute_appose(gray, cp3, CELLPOSE_ENV)
    print(f"   {int(np.max(m_cp3))} objects")

    print("▶️  CellposeSAM (cpsam) ...")
    m_cpsam = execute_appose(gray, cpsam, CELLPOSE_ENV)
    print(f"   {int(np.max(m_cpsam))} objects")

    print("▶️  StarDist (2D_versatile_fluo) ...")
    m_sd = execute_appose(gray, stardist, STARDIST_ENV)
    print(f"   {int(np.max(m_sd))} objects")

    # -- Plot 2x2 ------------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))

    axes[0, 0].imshow(rgb, cmap="gray" if rgb.ndim == 2 else None)
    axes[0, 0].set_title(f"{IMAGE} (input)")
    axes[0, 0].axis("off")

    _overlay(
        axes[0, 1],
        gray,
        m_cpsam,
        f"CellposeSAM — {int(np.max(m_cpsam))} objects",
    )
    _overlay(
        axes[1, 0],
        gray,
        m_cp3,
        f"Cellpose3 (cyto3) — {int(np.max(m_cp3))} objects",
    )
    _overlay(axes[1, 1], gray, m_sd, f"StarDist — {int(np.max(m_sd))} objects")

    fig.suptitle(f"Appose remote-segmenter test ({IMAGE})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show(block=True)


if __name__ == "__main__":
    main()
