# Install

- NVIDIA GPU: good test coverage.
- Mac M-series: good reports.
- AMD GPUs: not tested.
- CPU: should work, but can be slow.

Models (Cellpose, StarDist, ...) are not installed up front. AI Lab runs them
through [scikit-ops](https://github.com/apposed/scikit-ops), which builds each
model's environment the first time you use it. That first run takes several
minutes.

## 1. napari hub

!!! note "Coming soon"
    Waiting on scikit-ops being published to PyPI.

## 2. pip and conda

scikit-ops is not on PyPI yet, so it is installed from GitHub.

```sh
conda create -n napari-ai-lab -c conda-forge python=3.12 napari pyqt
conda activate napari-ai-lab
pip install "scikit-ops @ git+https://github.com/apposed/scikit-ops@69771a9"
pip install napari-ai-lab albumentations
```

For interactive SAM, also install micro-sam (several GB):

```sh
conda install -c conda-forge micro_sam
```

## 3. pixi

[Install pixi](https://pixi.sh/latest/installation/), then reopen the terminal
so `pixi` is on the path.

Pick one environment and download its `pixi.toml` into an empty folder:

- **microsam**: includes interactive SAM. Several GB, mostly torch and CUDA.
- **lite**: no SAM. Much smaller.

```sh
mkdir napari-ai-lab
cd napari-ai-lab
curl -O https://raw.githubusercontent.com/True-North-Intelligent-Algorithms/napari-ai-lab/main/pixi/install/microsam/pixi.toml
pixi run napari
```

For lite, replace `microsam` with `lite` in the URL. The first
`pixi run` installs everything; later runs start straight away.

## Check it worked

In napari: **Plugins → Napari AI Lab** lists ND AI Lab, 2D Instance AI Lab and
2D Detect AI Lab.

## Learn more: the I2K 2026 tutorial

AI Lab is built on scikit-ops. The
[I2K 2026 tutorial](https://github.com/bnorthan/i2k-2026) teaches both through
notebooks, and installs AI Lab along the way.
